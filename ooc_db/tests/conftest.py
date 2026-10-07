import os
import pathlib
import sys

import pytest

OOC_DB = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OOC_DB))

# Integration tests need a throwaway PostgreSQL database, e.g.
#   OOC_TEST_DSN=postgresql://postgres@localhost/ooctest python -m pytest ooc_db/tests
# (PowerShell: $env:OOC_TEST_DSN = "..."; python -m pytest ooc_db/tests)
# The schema is dropped and recreated in it.
TEST_DSN = os.environ.get("OOC_TEST_DSN")


@pytest.fixture
def conn():
    if not TEST_DSN:
        pytest.skip("set OOC_TEST_DSN to run database tests")
    import psycopg

    with psycopg.connect(TEST_DSN, autocommit=True) as setup:
        setup.execute((OOC_DB / "schema.sql").read_text())
    with psycopg.connect(TEST_DSN) as c:
        c.execute("SET search_path TO ooc")
        yield c
