"""Database helpers shared by the loaders."""

from __future__ import annotations

import os
import sys

import psycopg

DSN_ENV = "OOC_DATABASE_URL"
SCHEMA_VERSION = "ooc_db schema version 3"   # must match COMMENT ON SCHEMA in schema.sql


def connect(dsn: str | None = None) -> psycopg.Connection:
    """Open a connection with the `ooc` schema on the search path.

    The connection string comes from --dsn or the OOC_DATABASE_URL
    environment variable, e.g. postgresql://postgres:password@localhost/ooc
    """
    dsn = dsn or os.environ.get(DSN_ENV)
    if not dsn:
        sys.exit(
            f"No database given. Pass --dsn or set {DSN_ENV}, e.g.\n"
            "  postgresql://postgres:YOUR_PASSWORD@localhost:5432/ooc\n"
            "or, if the password contains @ : / # ? % or spaces,\n"
            "  host=localhost port=5432 dbname=ooc user=postgres password='YOUR PASSWORD'\n"
            "(inside the quotes, write \\' for a quote and \\\\ for a backslash)"
        )
    conn = psycopg.connect(dsn)
    check_schema(conn)
    conn.execute("SET search_path TO ooc")
    return conn


def check_schema(conn: psycopg.Connection) -> None:
    """Stop with a clear message if schema.sql was not run, or is outdated."""
    version = conn.execute(
        """SELECT obj_description(oid, 'pg_namespace') FROM pg_namespace WHERE nspname = 'ooc'"""
    ).fetchone()
    if version is None:
        sys.exit("This database has no 'ooc' schema yet. Run schema.sql in it first "
                 "(pgAdmin: Query Tool -> open schema.sql -> Run).")
    if version[0] != SCHEMA_VERSION:
        sys.exit(f"This database was created with an older schema.sql ({version[0] or 'version 1'}); "
                 f"these scripts need '{SCHEMA_VERSION}'. See 'Upgrading' in README.md.")


def normalize_name(name: str) -> str:
    """Collapse runs of whitespace (including non-breaking spaces) to one space."""
    return " ".join(name.split())


def find_drug(conn: psycopg.Connection, name: str) -> int | None:
    """Drug ID for a name or alias, ignoring case; None if unknown.

    A drug's own name wins over another drug's alias.
    """
    row = conn.execute(
        """
        SELECT drug_id, 1 AS priority FROM drug WHERE lower(name) = lower(%(n)s)
        UNION ALL
        SELECT drug_id, 2 FROM drug_alias WHERE lower(alias) = lower(%(n)s)
        ORDER BY priority
        LIMIT 1
        """,
        {"n": normalize_name(name)},
    ).fetchone()
    return row[0] if row else None


def get_or_create_drug(conn: psycopg.Connection, name: str, notes: str | None = None) -> tuple[int, bool]:
    """Return (drug_id, created)."""
    drug_id = find_drug(conn, name)
    if drug_id is not None:
        return drug_id, False
    drug_id = conn.execute(
        "INSERT INTO drug (name, notes) VALUES (%s, %s) RETURNING drug_id",
        (normalize_name(name), notes),
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
