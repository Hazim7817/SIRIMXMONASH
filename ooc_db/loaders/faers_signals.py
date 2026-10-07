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
# .exact matches the whole preferred term; the plain field would also match
# longer terms ('HEPATITIS' inside 'HEPATITIS B').
EVENT_FIELD = "patient.reaction.reactionmeddrapt.exact"
# Drug name fields, searched as phrases (case-insensitive, word-based, so
# 'troglitazone' also matches 'REZULIN (TROGLITAZONE)' and combination
# products). No one field is complete: openFDA's harmonised names are missing
# for many withdrawn drugs, and the reported product name is free text.
DRUG_FIELDS = {
    "generic_name": "patient.drug.openfda.generic_name",
    "medicinalproduct": "patient.drug.medicinalproduct",
    "activesubstance": "patient.drug.activesubstance.activesubstancename",
    # Whole-name matches on openFDA's harmonised names (--exact): 'ESTRADIOL'
    # does not match 'ETHINYL ESTRADIOL', but salt forms ('ESTRADIOL
    # VALERATE') need their own drug_alias, and drugs openFDA could not
    # harmonise (often withdrawn ones) are not found at all.
    "generic_name_exact": "patient.drug.openfda.generic_name.exact",
    "substance_name_exact": "patient.drug.openfda.substance_name.exact",
}
WORD_FIELDS = ["generic_name", "medicinalproduct", "activesubstance"]
EXACT_FIELDS = ["generic_name_exact", "substance_name_exact"]
# Which reference_outcome row each event definition (term file name) feeds.
EVENT_SCOPE = {
    "dili_narrow": ("toxicity", "liver", "FAERS"),
    "dili_extended": ("toxicity", "liver", "FAERS (extended terms)"),
}
# Salt and hydrate words dropped to get the base drug name, so 'Abacavir
# sulfate' is also searched as 'abacavir'.
SALT_WORDS = {
    "acetate", "anhydrous", "besilate", "besylate", "bitartrate", "bromide", "calcium",
    "chloride", "citrate", "dihydrate", "dipropionate", "disodium", "fumarate", "gluconate",
    "hcl", "hemihydrate", "hyclate", "hydrobromide", "hydrochloride", "lactate", "magnesium",
    "maleate", "mesilate", "mesylate", "monohydrate", "napsylate", "nitrate", "pamoate",
    "phosphate", "potassium", "sodium", "succinate", "sulfate", "sulphate", "tartrate",
    "tosylate", "trihydrate",
}
# Elements and inorganic ions: a bare one would match unrelated products
# ('Silver nitrate' -> 'Silver' would match silver sulfadiazine), so a name
# is never reduced to one of these.
BARE_ELEMENTS = set("""
    actinium aluminium aluminum americium antimony argon arsenic astatine barium berkelium
    beryllium bismuth bohrium boron bromine cadmium caesium calcium californium carbon cerium
    cesium chlorine chromium cobalt copernicium copper curium darmstadtium dubnium dysprosium
    einsteinium erbium europium fermium flerovium fluorine francium gadolinium gallium germanium
    gold hafnium hassium helium holmium hydrogen indium iodine iridium iron krypton lanthanum
    lawrencium lead lithium livermorium lutetium magnesium manganese meitnerium mendelevium
    mercury molybdenum moscovium neodymium neon neptunium nickel nihonium niobium nitrogen
    nobelium oganesson osmium oxygen palladium phosphorus platinum plutonium polonium potassium
    praseodymium promethium protactinium radium radon rhenium rhodium roentgenium rubidium
    ruthenium rutherfordium samarium scandium seaborgium selenium silicon silver sodium
    strontium sulfur sulphur tantalum technetium tellurium tennessine terbium thallium thorium
    thulium tin titanium tungsten uranium vanadium xenon ytterbium yttrium zinc zirconium
    ammonium chromic cupric cuprous ferric ferrous mercuric mercurous stannic stannous
    thallous vanadyl
""".split())


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
    drug_query: str = ""
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


def base_name(name: str) -> str | None:
    """Name without trailing salt/hydrate words, or None if nothing to strip.

    Never reduces a name to a bare salt or element ('Potassium chloride',
    'Silver nitrate' stay as they are), which would match unrelated products.
    """
    words = name.split()
    while len(words) > 1 and words[-1].lower().strip(",") in SALT_WORDS:
        words.pop()
    if len(words) == len(name.split()) or words[-1].lower() in SALT_WORDS | BARE_ELEMENTS:
        return None
    return " ".join(words)


def names_for(conn, drug_id: int, name: str) -> list[str]:
    """The drug's name, its aliases, and their base names, without repeats."""
    aliases = [r[0] for r in conn.execute(
        "SELECT alias FROM drug_alias WHERE drug_id = %s ORDER BY alias", (drug_id,))]
    seen, out = set(), []
    for n in [name, *aliases]:
        for candidate in (n, base_name(n)):
            if candidate and candidate.lower() not in seen:
                seen.add(candidate.lower())
                out.append(candidate)
    return out


def drug_clause(fields: list[str], names: list[str]) -> str:
    """Reports where any of the name fields matches any of the names.

    Exact fields hold upper-case values, so names are upper-cased for them.
    """
    clauses = []
    for f in fields:
        field = DRUG_FIELDS[f]
        values = [n.upper() for n in names] if field.endswith(".exact") else names
        clauses.append(any_of(field, list(dict.fromkeys(values))))
    return "(" + " OR ".join(clauses) + ")"


def check_terms(client: OpenFDA, terms: list[str]) -> list[str]:
    """Event terms that match no reports at all (likely misspelled)."""
    return [t for t in terms if client.count(any_of(EVENT_FIELD, [t])) == 0]


def analyse(client: OpenFDA, drug_id: int, name: str, query_names: list[str], *,
            drug_fields: list[str], event_search: str, n_total: int, n_event: int,
            criterion: str) -> DrugResult:
    drug_search = drug_clause(drug_fields, query_names)
    res = DrugResult(drug_id, name, query_names, drug_query=drug_search)
    n_drug = client.count(drug_search)
    if n_drug == 0:
        res.note = "no FAERS reports found under this name"
        return res
    n_both = client.count(f"{drug_search} AND {event_search}")
    res.table = Table2x2.from_totals(n_total, n_drug, n_event, n_both)
    res.stats = compute(res.table)
    res.signal = is_signal(res.table, res.stats, criterion)
    return res


def expected_events(t: Table2x2) -> float:
    """Drug-event reports expected if the drug had the background event rate."""
    return (t.a + t.b) * (t.a + t.c) / t.n


def verdict(res: DrugResult, min_expected: float) -> str:
    """positive = signal.

    negative = the event was reported no more often than expected, and at
    least `min_expected` reports were expected, so an excess could have been
    seen. A drug with few reports is 'ambiguous': no signal is not evidence
    of safety when there was too little data to detect one.
    """
    if res.signal:
        return "positive"
    expected = expected_events(res.table)
    if expected >= min_expected and res.table.a <= expected:
        return "negative"
    return "ambiguous"


def clear(conn, res: DrugResult, event_definition: str, today: dt.date) -> None:
    """A drug that now returns no reports: drop its old statistics and mark
    its FAERS reference row ambiguous, keeping your use_for_scoring choice."""
    conn.execute("DELETE FROM faers_signal WHERE drug_id = %s AND event_definition = %s",
                 (res.drug_id, event_definition))
    if event_definition in EVENT_SCOPE:
        endpoint, organ, source = EVENT_SCOPE[event_definition]
        conn.execute(
            """UPDATE reference_outcome SET verdict = 'ambiguous', retrieved_on = %s,
                      finding = 'No FAERS reports found for: ' || %s
               WHERE drug_id = %s AND source = %s AND endpoint = %s AND organ = %s
                 AND species = 'human' AND method = 'statistical_signal'""",
            (today, res.drug_query, res.drug_id, source, endpoint, organ))


def save(conn, res: DrugResult, *, event_definition: str, terms: list[str], criterion: str,
         min_expected: float, last_updated: str | None, today: dt.date) -> None:
    t, s = res.table, res.stats
    expected = expected_events(t)
    res_verdict = verdict(res, min_expected)
    conn.execute(
        """
        INSERT INTO faers_signal
            (drug_id, event_definition, event_terms, drug_query, a, b, c, d, expected_a,
             prr, prr_lower95, prr_upper95, ror, ror_lower95, ror_upper95,
             chi2_yates, is_signal, criteria, verdict, min_expected, faers_updated, queried_on)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (drug_id, event_definition) DO UPDATE SET
            event_terms = EXCLUDED.event_terms, drug_query = EXCLUDED.drug_query,
            a = EXCLUDED.a, b = EXCLUDED.b, c = EXCLUDED.c, d = EXCLUDED.d,
            expected_a = EXCLUDED.expected_a,
            prr = EXCLUDED.prr, prr_lower95 = EXCLUDED.prr_lower95, prr_upper95 = EXCLUDED.prr_upper95,
            ror = EXCLUDED.ror, ror_lower95 = EXCLUDED.ror_lower95, ror_upper95 = EXCLUDED.ror_upper95,
            chi2_yates = EXCLUDED.chi2_yates, is_signal = EXCLUDED.is_signal,
            criteria = EXCLUDED.criteria, verdict = EXCLUDED.verdict,
            min_expected = EXCLUDED.min_expected, faers_updated = EXCLUDED.faers_updated,
            queried_on = EXCLUDED.queried_on
        """,
        (res.drug_id, event_definition, terms, res.drug_query, t.a, t.b, t.c, t.d, expected,
         s.prr, s.prr_lower95, s.prr_upper95, s.ror, s.ror_lower95, s.ror_upper95,
         s.chi2_yates, res.signal, CRITERIA[criterion], res_verdict, min_expected, last_updated,
         today),
    )
    if event_definition not in EVENT_SCOPE:
        return  # unknown organ/endpoint: keep the statistics only
    endpoint, organ, source = EVENT_SCOPE[event_definition]
    ror_text = (f"ROR {s.ror:.2f} (95% CI {s.ror_lower95:.2f}-{s.ror_upper95:.2f})"
                if s.ror is not None else "ROR not estimable")
    db.upsert_reference(conn, {
        "drug_id": res.drug_id,
        "endpoint": endpoint,
        "organ": organ,
        "species": "human",
        "evidence_type": "spontaneous adverse event reports",
        "source": source,
        "finding": (f"{'Signal' if res.signal else 'No signal'} ({CRITERIA[criterion]}); "
                    f"{t.a} of {t.a + t.b} reports mention {event_definition} terms "
                    f"({expected:.1f} expected); {ror_text}"),
        "verdict": res_verdict,
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
                   help="file of MedDRA preferred terms, one per line (default: dili_narrow.txt). "
                        "Only dili_narrow.txt and dili_extended.txt also write a reference_outcome "
                        "row; other files are stored in faers_signal only")
    p.add_argument("--drug-field", action="append", dest="drug_fields", choices=sorted(DRUG_FIELDS),
                   help="drug name field to search (repeatable; default: "
                        + ", ".join(WORD_FIELDS) + ")")
    p.add_argument("--exact", action="store_true",
                   help="match whole harmonised names only (" + ", ".join(EXACT_FIELDS) + "), for "
                        "drugs whose name is part of another's, e.g. estradiol / ethinyl estradiol")
    # argparse treats % in help text as a format character, hence the escaping.
    p.add_argument("--criterion", choices=sorted(CRITERIA), default="ror",
                   help="signal rule: " + "; ".join(f"{k} = {v}" for k, v in CRITERIA.items())
                   .replace("%", "%%"))
    p.add_argument("--min-expected", type=float, default=5,
                   help="a drug without a signal is recorded as negative only if at least this "
                        "many event reports were expected at the background rate and no more "
                        "than that were seen; otherwise ambiguous (default: 5)")
    p.add_argument("--api-key", default=os.environ.get("OPENFDA_API_KEY"))
    p.add_argument("--cache", type=pathlib.Path, default=pathlib.Path(".faers_cache.json"),
                   help="file that stores counts so an interrupted run can resume")
    args = p.parse_args(argv)

    terms = read_terms(args.terms)
    event_definition = args.terms.stem
    if event_definition not in EVENT_SCOPE:
        print(f"note: term file '{args.terms.name}' is not one of "
              f"{', '.join(n + '.txt' for n in EVENT_SCOPE)}; results go to faers_signal only "
              f"(as event_definition '{event_definition}'), with no reference_outcome row.",
              file=sys.stderr)
    drug_fields = args.drug_fields or (EXACT_FIELDS if args.exact else WORD_FIELDS)
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
                              drug_fields=drug_fields, event_search=event_search,
                              n_total=n_total, n_event=n_event, criterion=args.criterion)
                if res.table is None:
                    clear(conn, res, event_definition, today)
                    conn.commit()
                    print(f"[{i}/{len(drugs)}] {name}: {res.note}")
                    continue
                save(conn, res, event_definition=event_definition, terms=terms,
                     criterion=args.criterion, min_expected=args.min_expected,
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
