"""Database rules in schema.sql (need OOC_TEST_DSN)."""

import psycopg
import pytest


def rejects(conn, sql, params=()):
    with pytest.raises(psycopg.errors.DatabaseError):
        with conn.transaction():
            conn.execute(sql, params)


@pytest.mark.parametrize("name", ["Aspirin ", " Aspirin", "Aspirin\t", "Aspirin\n", "Aspirin ", ""])
def test_drug_names_without_stray_whitespace(conn, name):
    rejects(conn, "INSERT INTO drug (name) VALUES (%s)", (name,))


def test_names_are_case_insensitively_unique(conn):
    conn.execute("INSERT INTO drug (name) VALUES ('Aspirin')")
    rejects(conn, "INSERT INTO drug (name) VALUES ('ASPIRIN')")
    conn.execute("INSERT INTO readout (name, unit, increase_means) VALUES ('ALT', 'U/L', 'harm')")
    rejects(conn, "INSERT INTO readout (name, unit, increase_means) VALUES ('alt', 'U/L', 'harm')")
    conn.execute("INSERT INTO chip_model (name, organ) VALUES ('Liver chip', 'liver')")
    rejects(conn, "INSERT INTO chip_model (name, organ) VALUES ('LIVER CHIP', 'liver')")


def test_organ_and_species_must_be_lower_case(conn):
    rejects(conn, "INSERT INTO chip_model (name, organ) VALUES ('c', 'Liver')")
    drug_id = conn.execute("INSERT INTO drug (name) VALUES ('d') RETURNING drug_id").fetchone()[0]
    for organ, species in (("Liver", "human"), ("liver", "Human"), ("liver ", "human")):
        rejects(conn, """INSERT INTO reference_outcome (drug_id, endpoint, organ, species, source, verdict)
                         VALUES (%s, 'toxicity', %s, %s, 's', 'positive')""", (drug_id, organ, species))


def test_treated_chip_needs_a_concentration(conn):
    conn.execute("INSERT INTO chip_model (name, organ) VALUES ('c', 'liver')")
    conn.execute("INSERT INTO drug (name) VALUES ('d')")
    rejects(conn, """INSERT INTO experiment (code, chip_model_id, batch_label, is_control, drug_id)
                     VALUES ('E1', 1, 'B', false, 1)""")
    rejects(conn, """INSERT INTO experiment (code, chip_model_id, batch_label, is_control, drug_id,
                     concentration_um) VALUES ('E2', 1, 'B', true, 1, 10)""")


def test_alias_cannot_be_another_drugs_name(conn):
    conn.execute("INSERT INTO drug (name) VALUES ('Acetaminophen'), ('Ibuprofen')")
    rejects(conn, "INSERT INTO drug_alias (drug_id, alias) VALUES (2, 'acetaminophen')")
    conn.execute("INSERT INTO drug_alias (drug_id, alias) VALUES (1, 'Paracetamol')")
    rejects(conn, "INSERT INTO drug (name) VALUES ('PARACETAMOL')")
    rejects(conn, "UPDATE drug SET name = 'Paracetamol' WHERE drug_id = 2")
    conn.execute("INSERT INTO drug_alias (drug_id, alias) VALUES (1, 'ACETAMINOPHEN')")  # own name: fine


def test_control_mean_weights_chips_equally(conn):
    conn.execute("INSERT INTO chip_model (name, organ) VALUES ('c', 'liver')")
    conn.execute("INSERT INTO drug (name, human_cmax_um) VALUES ('d', 1)")
    conn.execute("INSERT INTO readout (name, unit, increase_means) VALUES ('ALT', 'U/L', 'harm')")
    conn.execute("""INSERT INTO experiment (code, chip_model_id, batch_label, is_control, drug_id, concentration_um)
                    VALUES ('C1', 1, 'B', true, NULL, NULL), ('C2', 1, 'B', true, NULL, NULL),
                           ('T1', 1, 'B', false, 1, 10)""")
    # C1 has three replicates at 10, C2 one at 40: chip means 10 and 40 -> 25
    conn.execute("""INSERT INTO measurement (experiment_id, readout_id, timepoint_h, replicate, value)
                    VALUES (1, 1, 24, 1, 10), (1, 1, 24, 2, 10), (1, 1, 24, 3, 10),
                           (2, 1, 24, 1, 40), (3, 1, 24, 1, 50)""")
    mean, n, fold = conn.execute(
        "SELECT control_mean, n_control, fold_change FROM v_measurement_vs_control").fetchone()
    assert (float(mean), n, float(fold)) == (25.0, 2, 2.0)


def _chip_call(conn, drug, verdict="positive"):
    conn.execute("INSERT INTO drug (name) VALUES (%s) ON CONFLICT DO NOTHING", (drug,))
    conn.execute("""INSERT INTO chip_call (drug_id, chip_model_id, endpoint, verdict)
                    SELECT drug_id, 1, 'toxicity', %s FROM drug WHERE name = %s""", (verdict, drug))


def _ref(conn, drug, verdict, source="s", species="human"):
    conn.execute("""INSERT INTO reference_outcome (drug_id, endpoint, organ, species, source, verdict)
                    SELECT drug_id, 'toxicity', 'liver', %s, %s, %s FROM drug WHERE name = %s""",
                 (species, source, verdict, drug))


def test_concordance_keeps_calls_without_a_clear_reference(conn):
    conn.execute("INSERT INTO chip_model (name, organ) VALUES ('c', 'liver')")
    for d in ("known", "ambiguous only", "nothing", "mixed"):
        _chip_call(conn, d)
    _ref(conn, "known", "positive")
    _ref(conn, "ambiguous only", "ambiguous")
    _ref(conn, "mixed", "ambiguous", "s1")
    _ref(conn, "mixed", "negative", "s2")  # ambiguous ignored when a clear answer exists

    outcome = dict(conn.execute("SELECT drug, outcome FROM v_concordance"))
    assert outcome == {"known": "true positive", "ambiguous only": "not scored",
                       "nothing": "no reference", "mixed": "false positive"}
    perf = {r[0]: r[1:] for r in conn.execute(
        "SELECT species, tp, fp, not_scored, no_reference, accuracy_pct FROM v_performance")}
    assert perf["human"][:4] == (1, 1, 1, 1) and float(perf["human"][4]) == 50.0


def test_missing_reference_for_a_species_is_counted_on_that_species(conn):
    conn.execute("INSERT INTO chip_model (name, organ) VALUES ('c', 'liver')")
    _chip_call(conn, "rat only")
    _chip_call(conn, "both")
    _ref(conn, "rat only", "positive", species="rat")
    _ref(conn, "both", "positive", species="rat")
    _ref(conn, "both", "positive", species="human")
    perf = {r[0]: r[1:] for r in conn.execute("SELECT species, tp, no_reference FROM v_performance")}
    assert perf == {"human": (1, 1), "rat": (2, 0)}


def test_consensus_lists_only_the_sources_behind_the_verdict(conn):
    conn.execute("INSERT INTO drug (name) VALUES ('d')")
    _ref(conn, "d", "ambiguous", "DILIrank")
    _ref(conn, "d", "positive", "LiverTox")
    assert conn.execute("SELECT verdict, sources, n_findings FROM v_reference_consensus").fetchone() == (
        "positive", "LiverTox", 1)


def test_schema_works_without_search_path(conn):
    conn.execute("RESET search_path")
    conn.execute("INSERT INTO ooc.drug (name) VALUES ('Aspirin')")
    conn.execute("INSERT INTO ooc.drug_alias (drug_id, alias) VALUES (1, 'ASA')")
    rejects(conn, "INSERT INTO ooc.drug (name) VALUES ('asa')")


def test_interior_whitespace_rejected(conn):
    for name in ("Valproic  acid", "Valproic\u00a0acid", "Valproic\tacid"):
        rejects(conn, "INSERT INTO drug (name) VALUES (%s)", (name,))
    conn.execute("INSERT INTO drug (name) VALUES ('Riboflavin 5''-phosphate')")


def test_conflicting_sources_are_not_scored(conn):
    conn.execute("INSERT INTO chip_model (name, organ) VALUES ('c', 'liver')")
    _chip_call(conn, "d")
    _ref(conn, "d", "positive", "s1")
    _ref(conn, "d", "negative", "s2")
    assert conn.execute("SELECT reference_says, outcome FROM v_concordance").fetchone() == (
        "conflicting", "not scored")


def test_example_data_loads(conn):
    import pathlib
    conn.execute((pathlib.Path(__file__).parents[1] / "example_data.sql").read_text())
    perf = {r[0]: r[1:] for r in conn.execute(
        "SELECT species, sensitivity_pct, specificity_pct FROM v_performance")}
    assert {k: tuple(map(float, v)) for k, v in perf.items()} == {
        "human": (100.0, 100.0), "rat": (100.0, 0.0)}
