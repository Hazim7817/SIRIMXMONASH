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
ALL_FIELDS = fs.WORD_FIELDS


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
    session = FakeSession(dict(TOTALS))  # a copy: tests may change it
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


# --- review regressions --------------------------------------------------------

from loaders.disproportionality import Table2x2, compute, is_signal


def _res(a, b, c, d, criterion="ror"):
    t = Table2x2(a, b, c, d)
    s = compute(t)
    return fs.DrugResult(1, "x", ["x"], table=t, stats=s, signal=is_signal(t, s, criterion))


@pytest.mark.parametrize("cells, criterion, expected", [
    ((3, 97, 120_000, 9_879_900), "ror", "ambiguous"),        # ROR 2.5 (CI 0.8-8.0), 1.2 expected
    ((2, 148, 99_998, 19_899_852), "ror", "ambiguous"),       # elevated, a < 3: not a signal, not safe
    ((0, 120, 100_000, 19_899_880), "ror", "ambiguous"),      # only 0.6 reports expected
    ((190, 9_810, 120_000, 9_870_000), "evans", "ambiguous"), # significant excess but PRR < 2
    ((50, 19_950, 29_950, 9_950_050), "ror", "negative"),     # 50 seen, 60 expected
    ((400, 4_600, 29_600, 9_965_400), "ror", "positive"),
])
def test_verdict_needs_enough_expected_reports(cells, criterion, expected):
    assert fs.verdict(_res(*cells, criterion), min_expected=5) == expected


@pytest.mark.parametrize("name", ["Silver nitrate", "Zinc acetate", "Gallium nitrate",
                                  "Ammonium lactate", "Ferrous sulfate", "Lithium citrate"])
def test_base_name_never_returns_a_bare_element(name):
    assert fs.base_name(name) is None


def test_help_works(capsys):
    with pytest.raises(SystemExit) as e:
        fs.main(["--help"])
    assert e.value.code == 0
    assert "95%" in capsys.readouterr().out


def test_stale_results_removed_when_drug_has_no_reports(run, conn, tmp_path):
    run()
    assert conn.execute("SELECT count(*) FROM faers_signal").fetchone()[0] == 3
    for k in [k for k in run.session.totals if k and "Troglitazone" in k]:
        del run.session.totals[k]
    out = run("--cache", str(tmp_path / "fresh.json"))  # later option wins: no cached counts
    assert "Troglitazone: no FAERS reports found" in out.out
    left = {r[0] for r in conn.execute(
        "SELECT d.name FROM faers_signal JOIN drug d USING (drug_id)")}
    assert left == {"buspirone", "rarely used"}
    assert conn.execute(
        """SELECT r.verdict FROM reference_outcome r JOIN drug d USING (drug_id)
           WHERE d.name = 'Troglitazone' AND r.source = 'FAERS'""").fetchall() == [("ambiguous",)]


def test_signal_row_records_query_expected_count_and_verdict(run, conn):
    run()
    q, expected, verdict = conn.execute(
        """SELECT drug_query, expected_a, verdict FROM faers_signal JOIN drug d USING (drug_id)
           WHERE d.name = 'buspirone'""").fetchone()
    assert q == drug("buspirone")
    assert float(expected) == pytest.approx(20_000 * 30_000 / 10_000_000)
    assert verdict == "negative"


def test_unknown_term_file_is_announced(conn, tmp_path, monkeypatch, capsys):
    terms = tmp_path / "my_terms.txt"
    terms.write_text("HEPATOTOXICITY\n")
    session = FakeSession({None: 1_000})
    monkeypatch.setattr(fs, "OpenFDA", lambda api_key, cache_path: OpenFDA(
        api_key=api_key, cache_path=cache_path, session=session, sleep=lambda s: None, min_interval=0))
    monkeypatch.setattr(fs.db, "connect", lambda dsn: contextlib.nullcontext(conn))
    conn.execute("INSERT INTO drug (name) VALUES ('x')")
    fs.main(["--terms", str(terms), "--cache", str(tmp_path / "c.json")])
    assert "faers_signal only" in capsys.readouterr().err


def test_view_compares_signals_and_sets_aside_underpowered_drugs(run, conn, tmp_path):
    import pandas as pd
    path = tmp_path / "d.csv"
    pd.DataFrame([["Compound Name", "vDILIConcern"],
                  ["Troglitazone", "vMost-DILI-Concern"],
                  ["buspirone", "vLess-DILI-Concern"],
                  ["rarely used", "vNo-DILI-Concern"]]).to_csv(path, header=False, index=False)
    ld.load(conn, ld.parse(path))
    run("--source", "DILIrank")
    view = {r[0]: r[1:] for r in conn.execute(
        "SELECT drug, faers_signal, faers_verdict, dilirank_class, agreement FROM v_faers_vs_dilirank")}
    assert view == {
        "Troglitazone": (True, "positive", "most", "agree"),
        "buspirone": (False, "negative", "less", "disagree"),
        "rarely used": (False, "ambiguous", "no", "too few reports"),  # 0.1 reports expected
    }


def test_view_compares_well_powered_drugs_with_a_small_excess(conn):
    # 70 seen, 60 expected: no signal, not 'negative' (a > expected), but well
    # powered, so the DILIrank comparison still counts it as 'no signal'.
    conn.execute("INSERT INTO drug (name) VALUES ('d')")
    conn.execute("""INSERT INTO reference_outcome (drug_id, endpoint, organ, species, source, finding,
                    verdict, method, citation)
                    VALUES (1, 'toxicity', 'liver', 'human', 'DILIrank', 'vNo-DILI-concern', 'negative',
                            'database', %s)""", (ld.CITATION,))
    conn.execute("""INSERT INTO faers_signal (drug_id, event_definition, event_terms, drug_query, a, b, c, d,
                    expected_a, is_signal, criteria, verdict, min_expected)
                    VALUES (1, 'dili_narrow', '{X}', 'q', 70, 19930, 29930, 9950070, 60, false, 'ror',
                            'ambiguous', 5)""")
    assert conn.execute("SELECT faers_verdict, agreement FROM v_faers_vs_dilirank").fetchone() == (
        "ambiguous", "agree")


def test_hand_typed_dilirank_rows_are_compared(conn):
    conn.execute("INSERT INTO drug (name) VALUES ('Ketoconazole')")
    conn.execute("""INSERT INTO reference_outcome (drug_id, endpoint, organ, species, source, finding, verdict)
                    VALUES (1, 'toxicity', 'liver', 'human', 'DILIrank', 'Most-DILI-concern', 'positive')""")
    conn.execute("""INSERT INTO faers_signal (drug_id, event_definition, event_terms, drug_query, a, b, c, d,
                    expected_a, is_signal, criteria, verdict, min_expected)
                    VALUES (1, 'dili_narrow', '{X}', 'q', 50, 950, 1000, 98000, 10, true, 'ror', 'positive', 5)""")
    assert conn.execute("SELECT dilirank_class, agreement FROM v_faers_vs_dilirank").fetchone() == (
        "most", "agree")


def test_exact_option_matches_whole_names(run, conn):
    run("--drug", "Troglitazone", "--exact")
    searches = [c.get("search") for c in run.session.calls]
    expected = ('(patient.drug.openfda.generic_name.exact:("TROGLITAZONE")'
                ' OR patient.drug.openfda.substance_name.exact:("TROGLITAZONE"))')
    assert expected in searches


def test_zero_reports_keeps_scoring_choice(run, conn, tmp_path):
    run()
    conn.execute("UPDATE reference_outcome SET use_for_scoring = true WHERE source = 'FAERS'")
    for k in [k for k in run.session.totals if k and "Troglitazone" in k]:
        del run.session.totals[k]
    run("--cache", str(tmp_path / "fresh.json"))
    verdict, scoring, finding = conn.execute(
        """SELECT r.verdict, r.use_for_scoring, r.finding FROM reference_outcome r
           JOIN drug d USING (drug_id) WHERE d.name = 'Troglitazone' AND r.source = 'FAERS'""").fetchone()
    assert (verdict, scoring) == ("ambiguous", True)
    assert finding.startswith("No FAERS reports found")


def test_signal_row_records_cutoff_and_release(run, conn):
    run("--min-expected", "10")
    assert conn.execute(
        "SELECT DISTINCT min_expected, faers_updated FROM faers_signal").fetchall() == [(10, "2026-09-30")]


def test_scripts_refuse_an_outdated_schema(conn):
    from loaders import db
    conn.execute("COMMENT ON SCHEMA ooc IS NULL")
    conn.commit()
    with pytest.raises(SystemExit, match="older schema.sql"):
        db.check_schema(conn)


def test_exact_and_drug_field_cannot_be_combined(capsys):
    with pytest.raises(SystemExit):
        fs.main(["--exact", "--drug-field", "generic_name"])
    assert "not allowed with" in capsys.readouterr().err


def test_no_reports_row_cites_the_current_release(run, conn, tmp_path):
    run()
    for k in [k for k in run.session.totals if k and "Troglitazone" in k]:
        del run.session.totals[k]
    run.session.last_updated = "2026-12-31"
    run("--cache", str(tmp_path / "fresh.json"))
    assert conn.execute(
        """SELECT r.citation FROM reference_outcome r JOIN drug d USING (drug_id)
           WHERE d.name = 'Troglitazone' AND r.source = 'FAERS'""").fetchone()[0].endswith("2026-12-31")


def test_friendly_connection_errors():
    from loaders import db
    with pytest.raises(SystemExit, match="Could not connect"):
        db.connect("postgresql://nobody:wrong@localhost:1/none")
