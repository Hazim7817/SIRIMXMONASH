"""Database helpers shared by the loaders."""

from __future__ import annotations

import os
import sys

import psycopg

DSN_ENV = "OOC_DATABASE_URL"


def connect(dsn: str | None = None) -> psycopg.Connection:
    """Open a connection with the `ooc` schema on the search path.

    The connection string comes from --dsn or the OOC_DATABASE_URL
    environment variable, e.g. postgresql://postgres:password@localhost/ooc
    """
    dsn = dsn or os.environ.get(DSN_ENV)
    if not dsn:
        sys.exit(
            f"No database given. Pass --dsn or set {DSN_ENV}, e.g.\n"
            "  postgresql://postgres:YOUR_PASSWORD@localhost:5432/ooc"
        )
    conn = psycopg.connect(dsn)
    conn.execute("SET search_path TO ooc")
    return conn


def find_drug(conn: psycopg.Connection, name: str) -> int | None:
    """Drug ID for a name or alias, ignoring case; None if unknown."""
    row = conn.execute(
        """
        SELECT drug_id FROM drug WHERE lower(name) = lower(%(n)s)
        UNION
        SELECT drug_id FROM drug_alias WHERE lower(alias) = lower(%(n)s)
        """,
        {"n": name.strip()},
    ).fetchall()
    if len(row) > 1:
        raise ValueError(f"{name!r} matches more than one drug (check drug_alias)")
    return row[0][0] if row else None


def get_or_create_drug(conn: psycopg.Connection, name: str, notes: str | None = None) -> tuple[int, bool]:
    """Return (drug_id, created)."""
    drug_id = find_drug(conn, name)
    if drug_id is not None:
        return drug_id, False
    drug_id = conn.execute(
        "INSERT INTO drug (name, notes) VALUES (%s, %s) RETURNING drug_id",
        (name.strip(), notes),
    ).fetchone()[0]
    return drug_id, True


def upsert_reference(conn: psycopg.Connection, row: dict) -> None:
    """Insert or update an imported reference_outcome row.

    Imported rows are unique per drug, endpoint, organ, species and source.
    Re-running a loader refreshes the finding and verdict but keeps your
    use_for_scoring choice.
    """
    if row["method"] not in ("database", "statistical_signal"):
        raise ValueError("upsert_reference is only for imported rows")
    conn.execute(
        """
        INSERT INTO reference_outcome
            (drug_id, endpoint, organ, species, evidence_type, source, finding,
             verdict, citation, use_for_scoring, method, source_record_id, retrieved_on)
        VALUES
            (%(drug_id)s, %(endpoint)s, %(organ)s, %(species)s, %(evidence_type)s,
             %(source)s, %(finding)s, %(verdict)s, %(citation)s, %(use_for_scoring)s,
             %(method)s, %(source_record_id)s, %(retrieved_on)s)
        ON CONFLICT (drug_id, endpoint, organ, species, source)
            WHERE method IN ('database', 'statistical_signal')
        DO UPDATE SET
            evidence_type    = EXCLUDED.evidence_type,
            finding          = EXCLUDED.finding,
            verdict          = EXCLUDED.verdict,
            citation         = EXCLUDED.citation,
            method           = EXCLUDED.method,
            source_record_id = EXCLUDED.source_record_id,
            retrieved_on     = EXCLUDED.retrieved_on
        """,
        row,
    )
