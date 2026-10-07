import contextlib

import pytest

from fake_openfda import FakeSession
from loaders import faers_signals as fs
from loaders import load_dilirank as ld
from loaders.openfda import OpenFDA, any_of


def test_read_terms(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("# comment\nHepatotoxicity  # inline\n\nLiver injury\n")
    assert fs.read_terms(p) == ["HEPATOTOXICITY", "LIVER INJURY"]
    p.write_text("A\na\n")
    with pytest.raises(ValueError, match="duplicate"):
        fs.read_terms(p)


def test_term_files_are_valid():
    narrow = fs.read_terms(fs.DEFAULT_TERMS)
    extended = fs.read_terms(fs.HERE / "event_terms" / "dili_extended.txt")
    assert "DRUG-INDUCED LIVER INJURY" in narrow
    assert set(narrow) < set(extended)
    for excluded in ("HEPATITIS B", "HEPATITIS C", "ALANINE AMINOTRANSFERASE INCREASED",
                     "HEPATIC ENZYME INCREASED"):
        assert excluded not in extended
    assert {"dili_narrow", "dili_extended"} <= set(fs.EVENT_SCOPE)


@pytest.mark.parametrize("name, base", [
    ("Abacavir sulfate", "Abacavir"),
    ("Fluphenazine hydrochloride dihydrate", "Fluphenazine"),
    ("Losartan potassium", "Losartan"),
    ("Potassium chloride", None),     # never reduce to a bare salt
    ("Magnesium sulfate", None),
    ("Sodium valproate", None),
    ("Aminosalicylic acid", None),
    ("Acetaminophen", None),
])
def test_base_name(name, base):
    assert fs.base_name(name) == base


def test_drug_clause_searches_every_field_with_or():
    clause = fs.drug_clause(ALL_FIELDS, ["Abacavir sulfate", "Abacavir"])
    assert clause == (
        '(patient.drug.openfda.generic_name:("Abacavir sulfate" OR "Abacavir")'
        ' OR patient.drug.medicinalproduct:("Abacavir sulfate" OR "Abacavir")'
        ' OR patient.drug.activesubstance.activesubstancename:("Abacavir sulfate" OR "Abacavir"))')
    assert "+" not in clause  # a literal + would be sent as %2B and break the query


# --- end to end against the database with a fake openFDA ---------------------

TERMS = ["HEPATOTOXICITY", "LIVER INJURY"]
EVENT = any_of(fs.EVENT_FIELD, TERMS)
ALL_FIELDS = list(fs.DRUG_FIELDS)


def drug(*names):
    return fs.drug_clause(ALL_FIELDS, list(names))


TOTALS = {
    None: 10_000_000,
    any_of(fs.EVENT_FIELD, ["HEPATOTOXICITY"]): 30_000,
    any_of(fs.EVENT_FIELD, ["LIVER INJURY"]): 0,          # misspelled/unused term -> warning
    EVENT: 30_000,
    drug("Troglitazone"): 5_000,
    f"{drug('Troglitazone')} AND {EVENT}": 400,         # strong signal
    drug("buspirone"): 20_000,
    f"{drug('buspirone')} AND {EVENT}": 50,             # below background rate
    drug("rarely used"): 40,
    f"{drug('rarely used')} AND {EVENT}": 0,            # too few reports to say
    # "not in faers" has no reports at all
}


@pytest.fixture
def run(conn, tmp_path, monkeypatch, capsys):
    session = FakeSession(TOTALS)
    monkeypatch.setattr(fs, "OpenFDA", lambda api_key, cache_path: OpenFDA(
        api_key=api_key, cache_path=cache_path, session=session, sleep=lambda s: None, min_interval=0))
    monkeypatch.setattr(fs.db, "connect", lambda dsn: contextlib.nullcontext(conn))
    terms = tmp_path / "dili_narrow.txt"
    terms.write_text("\n".join(TERMS))
    for name in ("Troglitazone", "buspirone", "rarely used", "not in faers"):
        conn.execute("INSERT INTO drug (name) VALUES (%s)", (name,))

    def go(*extra):
        fs.main(["--terms", str(terms), "--cache", str(tmp_path / "c.json"), *extra])
        return capsys.readouterr()
    go.session = session
    return go


def test_end_to_end(run, conn):
    out = run()
    assert "LIVER INJURY" in out.err  # term with no reports is flagged
    assert "not in faers: no FAERS reports found" in out.out

    rows = {r[0]: r[1:] for r in conn.execute(
        """SELECT d.name, f.a, f.b, f.c, f.d, f.is_signal, round(f.ror, 3)
           FROM faers_signal f JOIN drug d USING (drug_id)""")}
    a, b = 400, 4_600
    c, d = 30_000 - 400, 10_000_000 - 5_000 - 30_000 + 400
    assert rows["Troglitazone"][:5] == (a, b, c, d, True)
    assert float(rows["Troglitazone"][5]) == pytest.approx((a * d) / (b * c), abs=1e-3)
    assert rows["buspirone"][4] is False
    assert "not in faers" not in rows

    refs = dict(conn.execute(
        """SELECT d.name, r.verdict FROM reference_outcome r JOIN drug d USING (drug_id)
           WHERE r.source = 'FAERS' AND r.method = 'statistical_signal'
             AND r.use_for_scoring = false AND r.organ = 'liver'"""))
    assert refs == {"Troglitazone": "positive", "buspirone": "negative", "rarely used": "ambiguous"}


def test_rerun_uses_cache_and_does_not_duplicate(run, conn):
    run()
    calls_first = len(run.session.calls)
    run()
    assert len(run.session.calls) == calls_first + 1  # only the release check
    assert conn.execute("SELECT count(*) FROM faers_signal").fetchone()[0] == 3
    assert conn.execute("SELECT count(*) FROM reference_outcome WHERE source = 'FAERS'").fetchone()[0] == 3


def test_aliases_and_base_names_are_searched(run, conn):
    drug_id = conn.execute("SELECT drug_id FROM drug WHERE name = 'buspirone'").fetchone()[0]
    conn.execute("INSERT INTO drug_alias (drug_id, alias) VALUES (%s, 'Buspirone hydrochloride')",
                 (drug_id,))
    conn.execute("INSERT INTO drug_alias (drug_id, alias) VALUES (%s, 'Buspar')", (drug_id,))
    assert fs.names_for(conn, drug_id, "buspirone") == ["buspirone", "Buspar", "Buspirone hydrochloride"]
    run("--drug", "buspirone")
    searches = [c.get("search") for c in run.session.calls]
    assert drug("buspirone", "Buspar", "Buspirone hydrochloride") in searches


def test_single_drug_field(run, conn):
    run("--drug", "Troglitazone", "--drug-field", "generic_name")
    searches = [c.get("search") for c in run.session.calls]
    assert fs.drug_clause(["generic_name"], ["Troglitazone"]) in searches


def test_extended_terms_use_their_own_reference_row(conn, tmp_path, monkeypatch, capsys):
    terms = tmp_path / "dili_extended.txt"
    terms.write_text("HEPATOTOXICITY\n")
    event = any_of(fs.EVENT_FIELD, ["HEPATOTOXICITY"])
    session = FakeSession({None: 1_000_000, event: 3_000, drug("Troglitazone"): 500,
                           f"{drug('Troglitazone')} AND {event}": 60})
    monkeypatch.setattr(fs, "OpenFDA", lambda api_key, cache_path: OpenFDA(
        api_key=api_key, cache_path=cache_path, session=session, sleep=lambda s: None, min_interval=0))
    monkeypatch.setattr(fs.db, "connect", lambda dsn: contextlib.nullcontext(conn))
    conn.execute("INSERT INTO drug (name) VALUES ('Troglitazone')")
    fs.main(["--terms", str(terms), "--cache", str(tmp_path / "c.json")])
    assert conn.execute("SELECT source FROM reference_outcome").fetchall() == [("FAERS (extended terms)",)]


def test_compare_with_dilirank_view(run, conn, tmp_path):
    import pandas as pd
    path = tmp_path / "d.csv"
    pd.DataFrame([["Compound Name", "vDILIConcern"],
                  ["Troglitazone", "vMost-DILI-Concern"],
                  ["buspirone", "vMost-DILI-Concern"]]).to_csv(path, header=False, index=False)
    ld.load(conn, ld.parse(path))
    run("--source", "DILIrank")
    view = dict(conn.execute("SELECT drug, agreement FROM v_faers_vs_dilirank"))
    assert view == {"Troglitazone": "agree", "buspirone": "disagree"}
