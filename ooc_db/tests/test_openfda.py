import json

import pytest

from fake_openfda import FakeSession
from loaders.openfda import OpenFDA, OpenFDAError, any_of, quote


def client(session, **kw):
    return OpenFDA(session=session, sleep=lambda s: None, min_interval=0, **kw)


def test_search_building():
    assert quote("  hepatic   failure ") == '"hepatic failure"'
    assert any_of("f", ["A", "B C"]) == 'f:("A" OR "B C")'
    with pytest.raises(ValueError):
        quote('bad"name')
    with pytest.raises(ValueError):
        any_of("f", [])


def test_count_and_not_found_is_zero():
    s = FakeSession({None: 1000, "x": 42})
    c = client(s, api_key="KEY")
    assert c.count("x") == 42
    assert c.count("missing") == 0
    assert c.last_updated == "2026-09-30"
    assert all(call["api_key"] == "KEY" and call["limit"] == 1 for call in s.calls)


def test_retries_server_errors_then_succeeds():
    s = FakeSession({None: 10, "x": 5}, failures=[429, 503])
    assert client(s).count("x") == 5


def test_gives_up_after_retries():
    s = FakeSession({None: 10}, failures=[500] * 10)
    with pytest.raises(OpenFDAError, match="after 3 attempts"):
        client(s, max_retries=2).count(None)


def test_client_error_is_not_retried():
    s = FakeSession({None: 10}, failures=[400])
    with pytest.raises(OpenFDAError, match="HTTP 400"):
        client(s).count(None)
    assert len(s.calls) == 1


def test_cache_resumes_same_release(tmp_path):
    cache = tmp_path / "cache.json"
    s1 = FakeSession({None: 100, "x": 7})
    client(s1, cache_path=cache).count("x")

    s2 = FakeSession({None: 100, "x": 999})  # would answer differently if asked
    c2 = client(s2, cache_path=cache)
    assert c2.count("x") == 7
    assert [call.get("search") for call in s2.calls] == [None]  # only the release check


def test_cache_discarded_for_new_release(tmp_path):
    cache = tmp_path / "cache.json"
    client(FakeSession({None: 100, "x": 7}), cache_path=cache).count("x")
    c = client(FakeSession({None: 120, "x": 9}, last_updated="2026-10-07"), cache_path=cache)
    assert c.count("x") == 9
    assert json.loads(cache.read_text())["last_updated"] == "2026-10-07"


def test_release_change_mid_run_is_an_error():
    s = FakeSession({None: 100, "x": 7})
    c = client(s)
    c.count(None)
    s.last_updated = "2026-10-07"
    with pytest.raises(OpenFDAError, match="updated during the run"):
        c.count("x")


# --- review regressions --------------------------------------------------------

def test_corrupt_cache_is_ignored_with_a_warning(tmp_path, capsys):
    cache = tmp_path / "cache.json"
    for content in ("", '{"last_updated": "2026-09-30", "counts": {"x": 7', "[1, 2]"):
        cache.write_text(content)
        c = client(FakeSession({None: 100, "x": 5}), cache_path=cache)
        assert c.count("x") == 5
        assert "ignoring unreadable cache" in capsys.readouterr().err
        assert json.loads(cache.read_text())["counts"]["x"] == 5


def test_cache_write_leaves_no_temporary_file(tmp_path):
    cache = tmp_path / "cache.json"
    client(FakeSession({None: 100, "x": 5}), cache_path=cache).count("x")
    assert [p.name for p in tmp_path.iterdir()] == ["cache.json"]


class Garbled200(FakeSession):
    def get(self, url, params, timeout):
        from fake_openfda import Response
        self.calls.append(dict(params))
        return Response(200, None)  # e.g. an HTML maintenance page


def test_unexpected_200_body_is_retried_then_reported():
    s = Garbled200({})
    with pytest.raises(OpenFDAError, match="unexpected response"):
        client(s, max_retries=2).count(None)
    assert len(s.calls) == 3


def test_api_key_is_redacted_from_errors():
    import requests

    class Down(FakeSession):
        def get(self, url, params, timeout):
            raise requests.ConnectionError(f"Max retries exceeded with url: /drug/event.json?api_key={params['api_key']}")

    with pytest.raises(OpenFDAError) as e:
        client(Down({}), api_key="SECRET/KEY+1", max_retries=0).count(None)
    assert "SECRET" not in str(e.value) and "***" in str(e.value)

    class Page:
        def __init__(self, status, text):
            self.status_code, self.text = status, text

        def json(self):
            raise ValueError("not json")

    class EchoesKey(FakeSession):
        def get(self, url, params, timeout):
            return Page(self.status, f"<html>Blocked: {url}?api_key={params['api_key']}</html>")

    for status in (403, 200):  # an error page, and a 200 'success' page from a proxy
        s = EchoesKey({})
        s.status = status
        with pytest.raises(OpenFDAError) as e:
            client(s, api_key="SECRETKEY", max_retries=0).count(None)
        assert "SECRETKEY" not in str(e.value) and "***" in str(e.value), status


def test_cache_saves_use_a_per_process_temporary_file(tmp_path, monkeypatch):
    import os
    seen = []
    real_replace = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (seen.append(str(a)), real_replace(a, b)))
    client(FakeSession({None: 1}), cache_path=tmp_path / "c.json").count(None)
    assert seen and all(f".{os.getpid()}.tmp" in name for name in seen)


def test_failed_cache_save_is_only_a_warning(tmp_path, capsys):
    blocked = tmp_path / "no_such_dir" / "c.json"
    assert client(FakeSession({None: 3}), cache_path=blocked).count(None) == 3
    assert "could not save cache" in capsys.readouterr().err


def test_interrupted_save_leaves_no_temporary_file(tmp_path, monkeypatch):
    import pathlib as pl
    c = client(FakeSession({None: 1, "x": 2}), cache_path=tmp_path / "c.json")
    c.count(None)
    real_write = pl.Path.write_text

    def interrupted(self, *a, **kw):
        real_write(self, "partial")
        raise KeyboardInterrupt
    monkeypatch.setattr(pl.Path, "write_text", interrupted)
    with pytest.raises(KeyboardInterrupt):
        c.count("x")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["c.json"]
