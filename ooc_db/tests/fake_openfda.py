"""A stand-in for the openFDA API that answers count queries from a dict."""

import json


class Response:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = json.dumps(body) if body is not None else "<html>error</html>"

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


class FakeSession:
    def __init__(self, totals, last_updated="2026-09-30", failures=None):
        self.totals = totals              # search string (None = all) -> count
        self.last_updated = last_updated
        self.failures = list(failures or [])  # status codes to return first
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append(dict(params))
        if self.failures:
            return Response(self.failures.pop(0), {"error": {"code": "SERVER_ERROR", "message": "busy"}})
        total = self.totals.get(params.get("search"))
        if not total:
            return Response(404, {"error": {"code": "NOT_FOUND", "message": "No matches found!"}})
        return Response(200, {"meta": {"last_updated": self.last_updated, "results": {"total": total}},
                              "results": [{}]})
