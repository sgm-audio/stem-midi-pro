# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Tests for CORS configuration on the API.

CORS is configured at module-import time in api.py based on the CORS_ORIGINS
env var. We test the two distinct runtime code paths by inspecting CORS rules
directly without reloading the module (which would re-register Prometheus
counters and is fragile).
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.testclient import TestClient


def _make_app(allow_origins, allow_credentials):
    """Build a minimal FastAPI app with given CORS settings, mirroring api.py logic."""
    app = FastAPI()

    @asynccontextmanager
    async def lifespan(_app):
        yield

    app.router.lifespan_context = lifespan
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        return {"ok": True}

    return app


def test_preflight_with_default_wildcard_returns_acao_star():
    """Default config (no allowlist) should respond with Allow-Origin: *."""
    app = _make_app(allow_origins=["*"], allow_credentials=False)
    client = TestClient(app)
    r = client.options(
        "/",
        headers={"Origin": "https://random.example", "Access-Control-Request-Method": "GET"},
    )
    assert r.status_code in (200, 204)
    acao = r.headers.get("access-control-allow-origin")
    assert acao == "*", f"expected *, got {acao!r}"


def test_preflight_from_outside_allowlist_not_allowed():
    """Preflight from a non-allowlisted origin should not be allowed.

    CORSMiddleware rejects disallowed origins with a 400 Disallowed CORS
    preflight response.
    """
    app = _make_app(
        allow_origins=["https://app.example"], allow_credentials=True
    )
    client = TestClient(app)
    r = client.options(
        "/",
        headers={"Origin": "https://attacker.example", "Access-Control-Request-Method": "GET"},
    )
    # Disallowed CORS preflight → 400
    assert r.status_code == 400
    acao = r.headers.get("access-control-allow-origin")
    # And crucially no Allow-Origin for attacker
    assert acao != "https://attacker.example"


def test_preflight_from_allowlisted_origin_echoed():
    """Preflight from an allowlisted origin should be echoed back."""
    app = _make_app(
        allow_origins=["https://app.example"], allow_credentials=True
    )
    client = TestClient(app)
    r = client.options(
        "/",
        headers={"Origin": "https://app.example", "Access-Control-Request-Method": "GET"},
    )
    assert r.status_code in (200, 204)
    acao = r.headers.get("access-control-allow-origin")
    assert acao == "https://app.example", f"expected echo, got {acao!r}"


def test_api_app_default_cors_when_no_env():
    """api.py app: when CORS_ORIGINS unset, default config sets allow-credentials=False.

    Module-level CORS config in api.py uses wildcard '*' when env unset.
    We verify by inspecting the user_settings on the app's CORS middleware.
    """
    import api as api_module

    # Find the CORS middleware on the existing app instance
    cors_mw = None
    for mw in api_module.app.user_middleware:
        if "CORSMiddleware" in str(mw.cls):
            cors_mw = mw
            break
    assert cors_mw is not None, "CORSMiddleware not found on api.app"
    # Wildcard branch: allow_origins == ['*'], allow_credentials == False
    assert cors_mw.kwargs.get("allow_origins") == ["*"]
    assert cors_mw.kwargs.get("allow_credentials") is False
