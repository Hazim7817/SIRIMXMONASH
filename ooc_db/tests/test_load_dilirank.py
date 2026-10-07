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


def _two_sheet_workbook(path):
    v1 = pd.DataFrame([
        ["Drug Induced Liver Injury Rank (DILIrank) Dataset | FDA", None, None, None, None, None],
        ["LTKBID", "Compound Name", "Severity Class", "Label Section", "vDILIConcern", "Version"],
        ["LT00917", "Atracurium", "0", "No match", "No-DILI-Concern", "2"],
    ])
    v2 = pd.DataFrame([
        ["Drug Induced Liver Injury Rank (DILIrank) Dataset Ver 2.0 | FDA", None, None, None, None, None],
        ["LTKBID", "CompoundName", "SeverityClass", "LabelSection", "vDILI-Concern", "Comment"],
        ["LT00917", "Atracurium", "0", "No match", "vNo-DILI-Concern", "Unchanged"],
        ["LT00486", "Polidocanol", "0", "No match", "Ambiguous-DILI-concern", "New"],
    ])
    with pd.ExcelWriter(path) as w:  # 'version 1' first, to prove the choice is by name
        v1.to_excel(w, sheet_name="version 1", header=False, index=False)
        v2.to_excel(w, sheet_name="version 2", header=False, index=False)


def test_two_sheet_workbook_prefers_version_2(tmp_path):
    path = tmp_path / "Drug Induced Liver Injury Rank (DILIrank 2.0) Dataset  FDA.xlsx"
    _two_sheet_workbook(path)
    assert ld.resolve_sheet(path) == "version 2"
    entries = ld.parse(path)
    assert [(e.name, e.category, e.comment) for e in entries] == [
        ("Atracurium", "no", "Unchanged"), ("Polidocanol", "ambiguous", "New")]

    old = ld.parse(path, sheet="version 1")
    assert [(e.name, e.category, e.comment) for e in old] == [("Atracurium", "no", None)]
    with pytest.raises(ValueError, match="no sheet"):
        ld.resolve_sheet(path, "version 3")


def test_main_dry_run(tmp_path, capsys):
    path = tmp_path / "d.xlsx"
    _two_sheet_workbook(path)
    ld.main(["--file", str(path), "--dry-run"])
    out = capsys.readouterr().out
    assert "Read 2 drugs" in out and "sheet 'version 2'" in out
    assert "1 no, 1 ambiguous" in out


# --- review regressions --------------------------------------------------------

def test_windows_csv_with_title_row_semicolons_and_cp1252(tmp_path):
    path = tmp_path / "dilirank.csv"
    text = ("DILIrank dataset | FDA\r\n"
            "LTKBID;Compound Name;Severity Class;Label Section;vDILIConcern\r\n"
            "LT1;Riboflavin 5'-phosphate;0;No match;vNo-DILI-Concern\r\n"
            "LT2;Café drug;8;Box warning;vMost-DILI-Concern\r\n")
    path.write_bytes(text.encode("cp1252"))
    assert [(e.name, e.category) for e in ld.parse(path)] == [
        ("Riboflavin 5'-phosphate", "no"), ("Café drug", "most")]


def test_tab_delimited_txt(tmp_path):
    path = tmp_path / "dilirank.txt"
    path.write_text("Compound Name\tvDILIConcern\nacetaminophen\tvMost-DILI-Concern\n")
    assert [(e.name, e.category) for e in ld.parse(path)] == [("acetaminophen", "most")]


def test_duplicates_with_the_same_verdict_keep_the_most_serious_category():
    entries = [ld.Entry("Drug A", "less", "vLess-DILI-Concern"),
               ld.Entry("drug a", "most", "vMost-DILI-Concern")]
    merged, warnings = ld.merge_duplicates(entries)
    assert [(e.name, e.category) for e in merged] == [("drug a", "most")] and warnings == []
    # ...but they conflict when Less-DILI-concern is not counted as positive
    merged, warnings = ld.merge_duplicates(entries, "ambiguous")
    assert merged[0].category == "ambiguous" and len(warnings) == 1


def test_two_names_for_one_drug_via_alias(conn, capsys):
    drug_id = conn.execute("INSERT INTO drug (name) VALUES ('Diclofenac') RETURNING drug_id").fetchone()[0]
    conn.execute("INSERT INTO drug_alias (drug_id, alias) VALUES (%s, 'Diclofenac sodium')", (drug_id,))
    conflicting = [ld.Entry("Diclofenac", "most", "vMost-DILI-concern", "LT1"),
                   ld.Entry("Diclofenac sodium", "no", "vNo-DILI-concern", "LT2")]
    stats = ld.load(conn, conflicting)
    assert "different verdicts for one drug" in capsys.readouterr().err
    assert stats["merged: conflicting names for one drug"] == 1
    assert _refs(conn) == [("Diclofenac", "ambiguous", "LT1", True)]

    # Same verdict: merged quietly, keeping the more serious category
    agreeing = [ld.Entry("Diclofenac", "less", "vLess-DILI-concern", "LT1"),
                ld.Entry("Diclofenac sodium", "most", "vMost-DILI-concern", "LT2")]
    stats = ld.load(conn, agreeing)
    assert stats["merged: same drug under two names"] == 1
    assert _refs(conn) == [("Diclofenac", "positive", "LT2", True)]
    assert capsys.readouterr().err == ""


def test_three_names_for_one_drug(conn, capsys):
    drug_id = conn.execute("INSERT INTO drug (name) VALUES ('Paracetamol') RETURNING drug_id").fetchone()[0]
    for alias in ("Acetaminophen", "Paracetamol sodium"):
        conn.execute("INSERT INTO drug_alias (drug_id, alias) VALUES (%s, %s)", (drug_id, alias))
    ld.load(conn, [ld.Entry("Paracetamol", "most", "vMost-DILI-concern", "LT1"),
                   ld.Entry("Acetaminophen", "no", "vNo-DILI-concern", "LT2"),
                   ld.Entry("Paracetamol sodium", "no", "vNo-DILI-concern", "LT3")])
    err = capsys.readouterr().err
    assert err.count("warning") == 1
    assert all(n in err for n in ("'Paracetamol'", "'Acetaminophen'", "'Paracetamol sodium'"))
    finding = conn.execute("SELECT finding FROM reference_outcome").fetchone()[0]
    assert finding.startswith("vMost-DILI-concern / vNo-DILI-concern / vNo-DILI-concern")


def test_loading_a_smaller_list_removes_drugs_not_in_it(conn, tmp_path):
    path = tmp_path / "book.xlsx"
    _two_sheet_workbook(path)
    ld.load(conn, ld.parse(path))                      # version 2: Atracurium, Polidocanol
    stats = ld.load(conn, ld.parse(path, "version 1"))  # version 1: Atracurium only
    assert stats["removed: earlier DILIrank rows for drugs not in this file"] == 1
    assert [r[0] for r in _refs(conn)] == ["Atracurium"]
    assert conn.execute("SELECT count(*) FROM drug WHERE name = 'Polidocanol'").fetchone()[0] == 1


def test_hand_entered_dilirank_rows_are_flagged_and_kept(conn, capsys):
    for name in ("Buspirone", "Paracetamol"):
        conn.execute("INSERT INTO drug (name) VALUES (%s)", (name,))
    conn.execute("""INSERT INTO reference_outcome (drug_id, endpoint, organ, species, source, verdict, method)
                    VALUES (1, 'toxicity', 'liver', 'human', 'DILIrank', 'negative', 'curated'),
                           (2, 'toxicity', 'liver', 'human', 'DILIrank', 'positive', 'database')""")
    ld.load(conn, [ld.Entry("Buspirone", "no", "vNo-DILI-concern"),
                   ld.Entry("acetaminophen", "most", "vMost-DILI-concern")])
    err = capsys.readouterr().err
    assert "sit next to the imported rows" in err          # Buspirone: curated + imported
    assert "drugs this file did not update" in err         # Paracetamol: spelled differently
    # the hand-typed Paracetamol row survives the reload (it was not written by the loader)
    assert conn.execute("""SELECT count(*) FROM reference_outcome r JOIN drug d USING (drug_id)
                           WHERE d.name = 'Paracetamol'""").fetchone()[0] == 1


def test_download_failure_gives_advice(monkeypatch):
    import requests

    def fail(*a, **kw):
        raise requests.ConnectionError("no route to host")
    monkeypatch.setattr(requests, "get", fail)
    with pytest.raises(SystemExit) as e:
        ld.download()
    assert "use --file instead" in str(e.value)


def test_name_whitespace_is_normalised(conn):
    entries = [ld.Entry("Aminosalicylic  acid", "less", "vLess-DILI-concern")]
    ld.load(conn, entries)
    assert conn.execute("SELECT name FROM drug").fetchall() == [("Aminosalicylic acid",)]


def test_mac_line_endings_utf16_and_semicolons_in_names(tmp_path):
    rows = ["LTKBID,Compound Name,vDILIConcern", 'LT1,"A; B; C; D; E",vNo-DILI-Concern',
            "LT2,acetaminophen,vMost-DILI-Concern"]
    cr = tmp_path / "mac.csv"
    cr.write_bytes("\r".join(rows).encode("ascii"))
    utf16 = tmp_path / "unicode.txt"
    utf16.write_bytes("\r\n".join(r.replace(",", "\t").replace('"', "") for r in rows).encode("utf-16"))
    for path in (cr, utf16):
        assert [(e.name, e.category) for e in ld.parse(path)] == [
            ("A; B; C; D; E", "no"), ("acetaminophen", "most")], path.name
