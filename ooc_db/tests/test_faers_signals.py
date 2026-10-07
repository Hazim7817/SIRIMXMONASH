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


def test_default_terms_file_is_valid():
    terms = fs.read_terms(fs.DEFAULT_TERMS)
    assert terms and all(t == t.upper() for t in terms)


# --- end to end against the database with a fake openFDA ---------------------

TERMS = ["HEPATOTOXICITY", "LIVER INJURY"]
EVENT = any_of(fs.EVENT_FIELD, TERMS)
GN = fs.DRUG_FIELDS["generic_name"]


def drug(*names):
    return any_of(GN, list(names))


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


def test_aliases_are_searched(run, conn):
    drug_id = conn.execute("SELECT drug_id FROM drug WHERE name = 'buspirone'").fetchone()[0]
    conn.execute("INSERT INTO drug_alias (drug_id, alias) VALUES (%s, 'Buspar')", (drug_id,))
    run("--drug", "buspirone")
    searches = [c.get("search") for c in run.session.calls]
    assert drug("buspirone", "Buspar") in searches


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
