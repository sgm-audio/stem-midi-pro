# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Tests for API key auth on /process endpoint."""
from __future__ import annotations

import io
import zipfile
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException
from starlette.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """TestClient with a no-op lifespan so the model isn't loaded."""
    from contextlib import asynccontextmanager
    from api import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    app.router.lifespan_context = noop_lifespan
    return TestClient(app)


def _make_wav_bytes(seconds: float = 1.0, sr: int = 44100) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(sr * seconds), dtype=np.float32), sr, format="WAV")
    return buf.getvalue()


def test_process_without_api_keys_is_open(client):
    """When no keys are configured, /process should not return 401."""
    with patch("api.API_KEYS", set()):
        with patch("api.model", None):  # avoid model use
            resp = client.post(
                "/process",
                files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
            )
    # Expect either 503 (no model) or 200 — NOT 401/403
    assert resp.status_code not in (401, 403)


def test_process_with_api_keys_no_header_is_401(client):
    """When API_KEYS configured but no Authorization header → 401."""
    with patch("api.API_KEYS", {"secret-key-123"}):
        resp = client.post(
            "/process",
            files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
        )
    assert resp.status_code == 401


def test_process_with_invalid_api_key_is_403(client):
    """Valid Authorization header with wrong token → 403."""
    with patch("api.API_KEYS", {"secret-key-123"}):
        resp = client.post(
            "/process",
            files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
            headers={"Authorization": "Bearer wrong-key"},
        )
    assert resp.status_code == 403


def test_process_with_valid_api_key_proceeds_past_auth(client):
    """A valid bearer token should bypass auth (then fail on model load with 503)."""
    with patch("api.API_KEYS", {"secret-key-123"}):
        with patch("api.model", None):
            resp = client.post(
                "/process",
                files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
                headers={"Authorization": "Bearer secret-key-123"},
            )
    # Auth bypassed; model is None → 503
    assert resp.status_code in (503, 413, 400)  # tolerate any further validation
    assert resp.status_code != 401 and resp.status_code != 403


def test_health_endpoints_do_not_require_auth(client):
    """/health, /live, /ready, /metrics, /templates are open even when API_KEYS set."""
    with patch("api.API_KEYS", {"secret-key-123"}):
        for ep in ("/health", "/live", "/ready", "/metrics", "/templates"):
            r = client.get(ep)
            assert r.status_code != 401, f"{ep} returned 401"
            assert r.status_code != 403, f"{ep} returned 403"
