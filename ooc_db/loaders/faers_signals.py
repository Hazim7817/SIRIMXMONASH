"""Compute FAERS disproportionality signals for drugs in the database.

For each drug, counts openFDA adverse event reports that mention the drug,
the event (a set of MedDRA preferred terms, e.g. liver injury), both, and
neither, then computes PRR and ROR. Results go to `faers_signal`, and a
summary row goes to `reference_outcome` (source 'FAERS', method
'statistical_signal', use_for_scoring = false until you decide otherwise).

A signal means the event is reported unusually often with the drug. It is a
hypothesis, not proof: reporting is voluntary, publicity drives reports, and
patients' diseases can cause the event. Use it as supporting evidence.

Run from the ooc_db folder:

    python -m loaders.faers_signals                        # every drug in the database
    python -m loaders.faers_signals --drug acetaminophen   # specific drugs
    python -m loaders.faers_signals --source DILIrank      # drugs with a DILIrank row

Get a free openFDA API key (https://open.fda.gov/apis/authentication/) and
set OPENFDA_API_KEY: without one openFDA allows only 1,000 requests a day,
and each drug needs 2 requests.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import sys
from dataclasses import dataclass

from loaders import db
from loaders.disproportionality import CRITERIA, Stats, Table2x2, compute, is_signal
from loaders.openfda import OpenFDA, OpenFDAError, any_of

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_TERMS = HERE / "event_terms" / "dili_narrow.txt"
EVENT_FIELD = "patient.reaction.reactionmeddrapt.exact"
DRUG_FIELDS = {
    # openFDA-harmonised generic name. A phrase search also matches
    # combination products that contain the drug.
    "generic_name": "patient.drug.openfda.generic_name",
    # Active ingredient as harmonised by openFDA (e.g. salt forms).
    "substance_name": "patient.drug.openfda.substance_name",
}
EVENT_SCOPE = {"dili_narrow": ("toxicity", "liver")}


def read_terms(path: pathlib.Path) -> list[str]:
    terms = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            terms.append(line.upper())
    if not terms:
        raise ValueError(f"{path}: no event terms")
    if len(set(terms)) != len(terms):
        raise ValueError(f"{path}: duplicate event terms")
    return terms


@dataclass
class DrugResult:
    drug_id: int
    name: str
    query_names: list[str]
    table: Table2x2 | None = None
    stats: Stats | None = None
    signal: bool | None = None
    note: str = ""


def select_drugs(conn, names: list[str] | None, source: str | None) -> list[tuple[int, str]]:
    if names:
        rows = []
        for n in names:
            drug_id = db.find_drug(conn, n)
            if drug_id is None:
                sys.exit(f"Drug not in database: {n}")
            rows.append((drug_id, conn.execute("SELECT name FROM drug WHERE drug_id = %s",
                                               (drug_id,)).fetchone()[0]))
        return rows
    if source:
        return conn.execute(
            """SELECT DISTINCT d.drug_id, d.name FROM drug d
               JOIN reference_outcome r USING (drug_id)
               WHERE r.source = %s ORDER BY d.name""", (source,)).fetchall()
    return conn.execute("SELECT drug_id, name FROM drug ORDER BY name").fetchall()


def names_for(conn, drug_id: int, name: str) -> list[str]:
    aliases = [r[0] for r in conn.execute(
        "SELECT alias FROM drug_alias WHERE drug_id = %s ORDER BY alias", (drug_id,))]
    seen, out = set(), []
    for n in [name, *aliases]:
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def check_terms(client: OpenFDA, terms: list[str]) -> list[str]:
    """Event terms that match no reports at all (likely misspelled)."""
    return [t for t in terms if client.count(any_of(EVENT_FIELD, [t])) == 0]


def analyse(client: OpenFDA, drug_id: int, name: str, query_names: list[str], *,
            drug_field: str, event_search: str, n_total: int, n_event: int,
            criterion: str) -> DrugResult:
    res = DrugResult(drug_id, name, query_names)
    drug_search = any_of(drug_field, query_names)
    n_drug = client.count(drug_search)
    if n_drug == 0:
        res.note = "no FAERS reports found under this name"
        return res
    n_both = client.count(f"{drug_search} AND {event_search}")
    res.table = Table2x2.from_totals(n_total, n_drug, n_event, n_both)
    res.stats = compute(res.table)
    res.signal = is_signal(res.table, res.stats, criterion)
    return res


def verdict(res: DrugResult, min_drug_reports: int) -> str:
    """positive = signal; negative = no signal despite enough reports."""
    if res.signal:
        return "positive"
    if res.table.a + res.table.b >= min_drug_reports:
        return "negative"
    return "ambiguous"


def save(conn, res: DrugResult, *, event_definition: str, terms: list[str], criterion: str,
         min_drug_reports: int, last_updated: str | None, today: dt.date) -> None:
    t, s = res.table, res.stats
    conn.execute(
        """
        INSERT INTO faers_signal
            (drug_id, event_definition, event_terms, drug_query, a, b, c, d,
             prr, prr_lower95, prr_upper95, ror, ror_lower95, ror_upper95,
             chi2_yates, is_signal, criteria, queried_on)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (drug_id, event_definition) DO UPDATE SET
            event_terms = EXCLUDED.event_terms, drug_query = EXCLUDED.drug_query,
            a = EXCLUDED.a, b = EXCLUDED.b, c = EXCLUDED.c, d = EXCLUDED.d,
            prr = EXCLUDED.prr, prr_lower95 = EXCLUDED.prr_lower95, prr_upper95 = EXCLUDED.prr_upper95,
            ror = EXCLUDED.ror, ror_lower95 = EXCLUDED.ror_lower95, ror_upper95 = EXCLUDED.ror_upper95,
            chi2_yates = EXCLUDED.chi2_yates, is_signal = EXCLUDED.is_signal,
            criteria = EXCLUDED.criteria, queried_on = EXCLUDED.queried_on
        """,
        (res.drug_id, event_definition, terms, " OR ".join(res.query_names), t.a, t.b, t.c, t.d,
         s.prr, s.prr_lower95, s.prr_upper95, s.ror, s.ror_lower95, s.ror_upper95,
         s.chi2_yates, res.signal, CRITERIA[criterion], today),
    )
    if event_definition not in EVENT_SCOPE:
        return  # unknown organ/endpoint: keep the statistics only
    endpoint, organ = EVENT_SCOPE[event_definition]
    ror_text = (f"ROR {s.ror:.2f} (95% CI {s.ror_lower95:.2f}-{s.ror_upper95:.2f})"
                if s.ror is not None else "ROR not estimable")
    db.upsert_reference(conn, {
        "drug_id": res.drug_id,
        "endpoint": endpoint,
        "organ": organ,
        "species": "human",
        "evidence_type": "spontaneous adverse event reports",
        "source": "FAERS",
        "finding": (f"{'Signal' if res.signal else 'No signal'} ({CRITERIA[criterion]}); "
                    f"{t.a} of {t.a + t.b} reports mention {event_definition} terms; {ror_text}"),
        "verdict": verdict(res, min_drug_reports),
        "citation": f"openFDA drug adverse event API, data updated {last_updated or 'unknown'}",
        "use_for_scoring": False,
        "method": "statistical_signal",
        "source_record_id": None,
        "retrieved_on": today,
    })


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dsn", help=f"database URL (default: ${db.DSN_ENV})")
    p.add_argument("--drug", action="append", dest="drugs", help="drug name (repeatable); default all")
    p.add_argument("--source", help="only drugs with a reference_outcome row from this source, e.g. DILIrank")
    p.add_argument("--terms", type=pathlib.Path, default=DEFAULT_TERMS,
                   help="file of MedDRA preferred terms, one per line (default: DILI narrow)")
    p.add_argument("--drug-field", choices=sorted(DRUG_FIELDS), default="generic_name")
    p.add_argument("--criterion", choices=sorted(CRITERIA), default="ror",
                   help="signal rule: " + "; ".join(f"{k} = {v}" for k, v in CRITERIA.items()))
    p.add_argument("--min-drug-reports", type=int, default=100,
                   help="below this many reports for the drug, 'no signal' is recorded as ambiguous")
    p.add_argument("--api-key", default=os.environ.get("OPENFDA_API_KEY"))
    p.add_argument("--cache", type=pathlib.Path, default=pathlib.Path(".faers_cache.json"),
                   help="file that stores counts so an interrupted run can resume")
    args = p.parse_args(argv)

    terms = read_terms(args.terms)
    event_definition = args.terms.stem
    drug_field = DRUG_FIELDS[args.drug_field]
    client = OpenFDA(api_key=args.api_key, cache_path=args.cache)
    today = dt.date.today()

    with db.connect(args.dsn) as conn:
        drugs = select_drugs(conn, args.drugs, args.source)
        if not drugs:
            sys.exit("No drugs selected.")
        needed = 2 * len(drugs) + len(terms) + 2
        if not args.api_key and needed > 1000:
            print(f"warning: about {needed} requests needed but openFDA allows 1,000 a day without "
                  "an API key. Set OPENFDA_API_KEY, or re-run tomorrow (the cache resumes).",
                  file=sys.stderr)

        try:
            n_total = client.count(None)
            unknown = check_terms(client, terms)
            if unknown:
                print("warning: these event terms match no reports (check spelling): "
                      + ", ".join(unknown), file=sys.stderr)
            event_search = any_of(EVENT_FIELD, terms)
            n_event = client.count(event_search)
            print(f"FAERS data updated {client.last_updated}: {n_total:,} reports, "
                  f"{n_event:,} with {event_definition} terms")

            signals = 0
            for i, (drug_id, name) in enumerate(drugs, 1):
                res = analyse(client, drug_id, name, names_for(conn, drug_id, name),
                              drug_field=drug_field, event_search=event_search,
                              n_total=n_total, n_event=n_event, criterion=args.criterion)
                if res.table is None:
                    print(f"[{i}/{len(drugs)}] {name}: {res.note}")
                    continue
                save(conn, res, event_definition=event_definition, terms=terms,
                     criterion=args.criterion, min_drug_reports=args.min_drug_reports,
                     last_updated=client.last_updated, today=today)
                conn.commit()  # keep finished drugs if a later one fails
                signals += bool(res.signal)
                ror = f"{res.stats.ror:.2f}" if res.stats.ror is not None else "n/a"
                print(f"[{i}/{len(drugs)}] {name}: {res.table.a}/{res.table.a + res.table.b} "
                      f"reports, ROR {ror}, {'SIGNAL' if res.signal else 'no signal'}")
        except OpenFDAError as e:
            sys.exit(f"openFDA error: {e}\nFinished drugs are saved; re-run to continue.")

    print(f"Done: {signals} signals among {len(drugs)} drugs ({client.requests_made} API requests). "
          "Compare with DILIrank: SELECT * FROM ooc.v_faers_vs_dilirank;")


if __name__ == "__main__":
    main()
