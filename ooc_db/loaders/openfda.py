"""Minimal client for counting reports in the openFDA drug adverse event API."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import requests

BASE_URL = "https://api.fda.gov/drug/event.json"


class OpenFDAError(RuntimeError):
    pass


def quote(value: str) -> str:
    """A double-quoted phrase for an openFDA search."""
    value = " ".join(value.split())
    if '"' in value:
        raise ValueError(f"Search values cannot contain double quotes: {value!r}")
    return f'"{value}"'


def any_of(field: str, values: list[str]) -> str:
    """Search clause matching reports where `field` matches any of `values`.

    Spaces become '+' in the URL, which openFDA reads as spaces; a literal
    '+' would be sent as %2B and break the query, so none is used here.
    """
    if not values:
        raise ValueError("any_of needs at least one value")
    return f"{field}:({' OR '.join(quote(v) for v in values)})"


class OpenFDA:
    """Counts reports matching a search, with throttling, retries and a cache.

    Without an API key openFDA allows about 240 requests a minute and 1,000 a
    day per IP address; with a free key, 240 a minute and 120,000 a day.
    """

    def __init__(self, api_key: str | None = None, cache_path: pathlib.Path | None = None,
                 min_interval: float = 0.3, max_retries: int = 5, timeout: float = 60,
                 session: requests.Session | None = None, sleep=time.sleep):
        self.api_key = api_key
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = session or requests.Session()
        self.sleep = sleep
        self._last_request = 0.0
        self.requests_made = 0
        self.cache_path = cache_path
        self.counts: dict[str, int] = {}
        self.last_updated: str | None = None   # FAERS release date reported by openFDA

    def count(self, search: str | None) -> int:
        """Number of reports matching `search` (None = all reports).

        All counts in one run come from the same FAERS release: the first
        call checks the dataset's last_updated date, cached counts from an
        older release are discarded, and a release change mid-run is an error.
        """
        if self.last_updated is None:
            self._sync()
        key = search or "<all>"
        if key in self.counts:
            return self.counts[key]
        total = self._live_count(search)
        self.counts[key] = total
        self._save()
        return total

    def _sync(self) -> None:
        total = self._live_count(None)
        cached = {}
        if self.cache_path and self.cache_path.exists():
            try:
                cached = json.loads(self.cache_path.read_text())
                if not isinstance(cached, dict):
                    raise ValueError("not a JSON object")
            except (ValueError, OSError) as e:
                print(f"warning: ignoring unreadable cache {self.cache_path} ({e}); "
                      "counts will be fetched again", file=sys.stderr)
                cached = {}
        if cached.get("last_updated") == self.last_updated:
            self.counts = {**cached.get("counts", {}), "<all>": total}
        else:
            self.counts = {"<all>": total}
        self._save()

    def _save(self) -> None:
        """Write the cache to a temporary file of this run's own, then swap
        it in, so an interruption never leaves a half-written cache and two
        runs sharing a cache do not trip over each other. The cache is only
        an optimisation, so a failed save is a warning, not an error."""
        if not self.cache_path:
            return
        tmp = self.cache_path.with_name(f"{self.cache_path.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(
                {"last_updated": self.last_updated, "counts": self.counts}, indent=0, sort_keys=True))
            os.replace(tmp, self.cache_path)
        except OSError as e:
            print(f"warning: could not save cache {self.cache_path} ({e})", file=sys.stderr)
        finally:
            tmp.unlink(missing_ok=True)  # left over only if the swap did not happen (e.g. Ctrl+C)

    def _live_count(self, search: str | None) -> int:
        params = {"limit": 1}
        if search:
            params["search"] = search
        if self.api_key:
            params["api_key"] = self.api_key
        total, last_updated = self._get_total(params)
        if last_updated:
            if self.last_updated is None:
                self.last_updated = last_updated
            elif last_updated != self.last_updated:
                raise OpenFDAError(
                    f"openFDA data was updated during the run ({self.last_updated} -> "
                    f"{last_updated}); run again to use the new release throughout")
        return total

    def _get_total(self, params: dict) -> tuple[int, str | None]:
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                self.sleep(wait)
            self._last_request = time.monotonic()
            self.requests_made += 1
            try:
                r = self.session.get(BASE_URL, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                error = self._redact(f"network error: {e}")
            else:
                if r.status_code == 200:
                    try:
                        meta = r.json()["meta"]
                        return int(meta["results"]["total"]), meta.get("last_updated")
                    except (ValueError, KeyError, TypeError) as e:
                        error = self._redact(f"unexpected response from openFDA ({e!r}): {r.text[:300]}")
                else:
                    body = _json_or_none(r)
                    err = body.get("error") if isinstance(body, dict) else None
                    code = err.get("code") if isinstance(err, dict) else None
                    if r.status_code == 404 and code == "NOT_FOUND":
                        return 0, None  # openFDA's answer to a search with no matches
                    error = self._redact(f"HTTP {r.status_code}: {err or r.text[:300]}")
                    if r.status_code not in (429, 500, 502, 503, 504):
                        raise OpenFDAError(f"{error} (search: {params.get('search')})")
            if attempt < self.max_retries:
                self.sleep(delay)
                delay *= 2
        raise OpenFDAError(f"{error} after {self.max_retries + 1} attempts (search: {params.get('search')})")

    def _redact(self, text: str) -> str:
        """Hide the API key, which appears in URLs inside error messages."""
        if self.api_key:
            for form in (self.api_key, requests.utils.quote(self.api_key, safe="")):
                text = text.replace(form, "***")
        return text


def _json_or_none(r: requests.Response):
    try:
        return r.json()
    except ValueError:
        return None
