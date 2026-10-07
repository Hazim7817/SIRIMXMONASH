import datetime as dt

import pandas as pd
import pytest

from loaders import load_dilirank as ld

ROWS = [
    ["DILIrank dataset (title row above the header)", None, None, None, None],
    [None, None, None, None, None],
    ["LTKBID", "Compound Name", "Severity Class", "Label Section", "vDILIConcern"],
    ["LT00001", "acetaminophen", "5", "Warnings and precautions", "vMost-DILI-Concern"],
    ["LT00002", "Troglitazone", "8", "Withdrawn", "vMost-DILI-Concern"],
    ["LT00003", "buspirone", "0", "No match", "vNo-DILI-Concern"],
    ["LT00004", "aspirin", "3", "Adverse reactions", "vLess-DILI-Concern "],
    ["LT00005", "drug  with   spaces", "", "", "Ambiguous DILI-concern"],
    [None, None, None, None, None],
    ["Footnote: see Chen et al. 2016", None, None, None, None],
]


@pytest.fixture
def xlsx(tmp_path):
    path = tmp_path / "DILIrank.xlsx"
    pd.DataFrame(ROWS).to_excel(path, header=False, index=False)
    return path


def test_parse_finds_header_and_categories(xlsx):
    entries = ld.parse(xlsx)
    assert [(e.name, e.category) for e in entries] == [
        ("acetaminophen", "most"),
        ("Troglitazone", "most"),
        ("buspirone", "no"),
        ("aspirin", "less"),
        ("drug with spaces", "ambiguous"),
    ]
    assert entries[0].ltkbid == "LT00001"
    assert entries[1].severity == "8"
    assert entries[0].label_section == "Warnings and precautions"


def test_parse_csv(tmp_path):
    path = tmp_path / "dilirank.csv"
    pd.DataFrame(ROWS[2:7]).to_csv(path, header=False, index=False)
    assert len(ld.parse(path)) == 4


def test_parse_rejects_unknown_category(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([["Compound Name", "vDILIConcern"], ["x", "Very concerning"]]).to_csv(
        path, header=False, index=False)
    with pytest.raises(ValueError, match="Unrecognised"):
        ld.parse(path)


def test_parse_requires_header(tmp_path):
    path = tmp_path / "noheader.csv"
    pd.DataFrame([["a", "b"], ["c", "d"]]).to_csv(path, header=False, index=False)
    with pytest.raises(ValueError, match="Compound Name"):
        ld.parse(path)


def test_category_spellings():
    assert ld._category("vMost-DILI-Concern") == "most"
    assert ld._category("Most DILI concern") == "most"
    assert ld._category("vNo-DILI-concern") == "no"
    assert ld._category("vLess-DILI-Concern") == "less"
    assert ld._category("Ambiguous DILI-concern") == "ambiguous"
    assert ld._category("") is None


def test_conflicting_duplicates_become_ambiguous():
    entries = [
        ld.Entry("Drug A", "most", "vMost-DILI-Concern"),
        ld.Entry("drug a", "no", "vNo-DILI-Concern"),
        ld.Entry("Drug B", "no", "vNo-DILI-Concern"),
        ld.Entry("DRUG B", "no", "vNo-DILI-Concern"),
    ]
    merged, warnings = ld.merge_duplicates(entries)
    assert [(e.name, e.category) for e in merged] == [("Drug A", "ambiguous"), ("Drug B", "no")]
    assert len(warnings) == 1


def test_less_concern_mapping():
    assert ld.verdict_for("less", "positive") == "positive"
    assert ld.verdict_for("less", "ambiguous") == "ambiguous"
    assert ld.verdict_for("most", "negative") == "positive"
    assert ld.verdict_for("no", "positive") == "negative"


# --- database tests (need OOC_TEST_DSN) -------------------------------------

def _refs(conn):
    return conn.execute(
        """SELECT d.name, r.verdict, r.source_record_id, r.use_for_scoring
           FROM reference_outcome r JOIN drug d USING (drug_id)
           WHERE r.source = 'DILIrank' ORDER BY d.name"""
    ).fetchall()


def test_load_creates_drugs_and_references(conn, xlsx):
    conn.execute("INSERT INTO drug (name, human_cmax_um) VALUES ('Acetaminophen', 139)")
    entries, _ = ld.merge_duplicates(ld.parse(xlsx))
    stats = ld.load(conn, entries, retrieved_on=dt.date(2026, 10, 7))

    assert stats["drugs created"] == 4  # acetaminophen already existed (case-insensitive match)
    assert _refs(conn) == [
        ("Acetaminophen", "positive", "LT00001", True),
        ("Troglitazone", "positive", "LT00002", True),
        ("aspirin", "positive", "LT00004", True),
        ("buspirone", "negative", "LT00003", True),
        ("drug with spaces", "ambiguous", "LT00005", True),
    ]
    # existing drug data untouched
    assert conn.execute("SELECT human_cmax_um FROM drug WHERE name = 'Acetaminophen'").fetchone()[0] == 139


def test_reload_updates_without_duplicating_and_keeps_scoring_choice(conn, xlsx):
    entries, _ = ld.merge_duplicates(ld.parse(xlsx))
    ld.load(conn, entries)
    conn.execute("UPDATE reference_outcome SET use_for_scoring = false WHERE source_record_id = 'LT00004'")

    ld.load(conn, entries, less_concern_as="ambiguous")

    rows = {r[0]: r for r in _refs(conn)}
    assert len(rows) == 5
    assert rows["aspirin"][1] == "ambiguous"     # verdict refreshed
    assert rows["aspirin"][3] is False           # your choice kept


def test_only_existing(conn, xlsx):
    conn.execute("INSERT INTO drug (name) VALUES ('Troglitazone')")
    entries, _ = ld.merge_duplicates(ld.parse(xlsx))
    stats = ld.load(conn, entries, only_existing=True)
    assert stats["skipped (drug not in database)"] == 4
    assert [r[0] for r in _refs(conn)] == ["Troglitazone"]
    assert conn.execute("SELECT count(*) FROM drug").fetchone()[0] == 1


def test_alias_match(conn, xlsx):
    drug_id = conn.execute("INSERT INTO drug (name) VALUES ('Paracetamol') RETURNING drug_id").fetchone()[0]
    conn.execute("INSERT INTO drug_alias (drug_id, alias) VALUES (%s, 'Acetaminophen')", (drug_id,))
    entries, _ = ld.merge_duplicates(ld.parse(xlsx))
    ld.load(conn, entries)
    assert "Paracetamol" in [r[0] for r in _refs(conn)]
    assert conn.execute("SELECT count(*) FROM drug WHERE lower(name) = 'acetaminophen'").fetchone()[0] == 0


def test_parse_dilirank_2_layout(tmp_path):
    # DILIrank 2.0 spells headers and categories differently, and some names
    # carry non-breaking spaces.
    path = tmp_path / "DILIrank2.xlsx"
    pd.DataFrame([
        ["LTKBID", "CompoundName", "SeverityClass", "LabelSection", "vDILI-Concern", "Comment"],
        ["LT00040", "Abacavir sulfate", "8", "Warnings & precautions", "vMOST-DILI-concern", "Unchanged"],
        ["LT00505", "Aminosalicylic acid\xa0", "3", "Adverse reactions", "vLess-DILI-concern", "New"],
        ["LT00917", "Atracurium", "0", "No match", "vNo-DILI-Concern", "Unchanged"],
    ]).to_excel(path, header=False, index=False)
    entries = ld.parse(path)
    assert [(e.name, e.category, e.ltkbid) for e in entries] == [
        ("Abacavir sulfate", "most", "LT00040"),
        ("Aminosalicylic acid", "less", "LT00505"),
        ("Atracurium", "no", "LT00917"),
    ]


def test_parse_version_1_without_v_prefix(tmp_path):
    path = tmp_path / "v1.csv"
    pd.DataFrame([
        ["LTKBID", "Compound Name", "Severity Class", "Label Section", "vDILIConcern", "Version"],
        ["LT00003", "mercaptopurine", "8", "Warnings and precautions", "Most-DILI-Concern", "1"],
    ]).to_csv(path, header=False, index=False)
    assert [(e.name, e.category) for e in ld.parse(path)] == [("mercaptopurine", "most")]
