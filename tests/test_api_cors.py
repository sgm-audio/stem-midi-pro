# SPDX-License-Identifier: Apache-2.0
"""CORS preflight tests (T-8.3.11).

Expected behavior (per TODO API-6.2):
- preflight OPTIONS without an Origin header → default (no ACAO header)
- with `Origin: https://app.example` when allowlisted → ACAO echoed
- with a mismatched origin → no Access-Control-Allow-Origin header

Marked xfail-guarded: the current middleware ships a localhost default
allowlist and does not read `CORS_ORIGINS` dynamically.
"""

import pytest

fastapi = pytest.importorskip("fastapi", reason="requires fastapi")
pytest.importorskip("httpx", reason="TestClient requires httpx")


def _client(monkeypatch):
    # CORS middleware is configured at app construction from CORS_ORIGINS,
    # so reload the api module AFTER the env var is set.
    import importlib

    import api

    importlib.reload(api)

    from fastapi.testclient import TestClient

    return TestClient(api.app, raise_server_exceptions=True)


def test_preflight_allowed_origin(monkeypatch):
    """Allowlisted origin receives ACAO echo."""
    monkeypatch.setenv("CORS_ORIGINS", "https://app.example")
    client = _client(monkeypatch)
    resp = client.options(
        "/process",
        headers={
            "Origin": "https://app.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.headers.get("access-control-allow-origin") == "https://app.example"


def test_star_disables_credentials(monkeypatch):
    """CORS_ORIGINS='*' → allow_origins='*' AND allow_credentials=False."""
    monkeypatch.setenv("CORS_ORIGINS", "*")
    client = _client(monkeypatch)
    resp = client.options(
        "/process",
        headers={
            "Origin": "https://anything.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.headers.get("access-control-allow-origin") == "*"
    assert resp.headers.get("access-control-allow-credentials") != "true"
