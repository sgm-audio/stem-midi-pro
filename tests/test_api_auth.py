# SPDX-License-Identifier: Apache-2.0
"""API key auth tests (T-8.3.10).

Expected behavior (per TODO API-6.5):
- POST /process without `Authorization: Bearer <key>` header → 401
- with an invalid key → 403
- with a valid key → 200
(`/health`, `/live`, `/ready`, `/metrics`, `/docs` remain open.)

Marked xfail(strict=False) because API-6.5 (API_KEYS env middleware) is not
yet implemented; the current stub uses a single `API_KEY` env var +
X-API-Key header and returns 401 for both missing and invalid keys.
"""

import io
import zipfile

import pytest

fastapi = pytest.importorskip("fastapi", reason="requires fastapi")
pytest.importorskip("httpx", reason="TestClient requires httpx")
pytest.importorskip("soundfile", reason="fixture writes WAV")


def _client_and_api(monkeypatch):
    pytest.importorskip("torch")
    try:
        import api
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"Cannot import api.py (see T-8.2.1): {exc}")

    monkeypatch.setattr(api, "model", object(), raising=False)
    monkeypatch.setattr(api, "model_config", {"audio": {"sample_rate": 44100}}, raising=False)

    # Bypass real model inference; auth is what's under test.
    async def _fake_process(_path, _quantize=False):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("processing_report.json", "{}")
        buf.seek(0)
        return buf

    monkeypatch.setattr(api, "_do_process", _fake_process, raising=False)

    from fastapi.testclient import TestClient

    return api, TestClient(api.app, raise_server_exceptions=True)


@pytest.mark.integration
def test_process_requires_api_key(monkeypatch, temp_audio_file):
    """No key configured → request with no key rejected with 401."""
    monkeypatch.setenv("API_KEYS", "secret-key-1,secret-key-2")
    _api, client = _client_and_api(monkeypatch)

    with open(temp_audio_file, "rb") as fh:
        resp = client.post("/process", files={"file": ("t.wav", fh, "audio/wav")})
    assert resp.status_code == 401


@pytest.mark.integration
def test_process_rejects_invalid_key(monkeypatch, temp_audio_file):
    """Wrong key → 403."""
    monkeypatch.setenv("API_KEYS", "secret-key-1")
    _api, client = _client_and_api(monkeypatch)

    with open(temp_audio_file, "rb") as fh:
        resp = client.post(
            "/process",
            files={"file": ("t.wav", fh, "audio/wav")},
            headers={"Authorization": "Bearer wrong-key"},
        )
    assert resp.status_code == 403


@pytest.mark.integration
def test_process_accepts_valid_key(monkeypatch, temp_audio_file):
    """Valid key → 200."""
    monkeypatch.setenv("API_KEYS", "secret-key-1")
    _api, client = _client_and_api(monkeypatch)

    with open(temp_audio_file, "rb") as fh:
        resp = client.post(
            "/process",
            files={"file": ("t.wav", fh, "audio/wav")},
            headers={"Authorization": "Bearer secret-key-1"},
        )
    assert resp.status_code == 200


@pytest.mark.integration
def test_health_endpoints_open(monkeypatch):
    """Health/metrics/docs are not gated by the API key."""
    monkeypatch.setenv("API_KEYS", "secret-key-1")
    _api, client = _client_and_api(monkeypatch)

    resp = client.get("/health")
    assert resp.status_code != 401
    assert resp.status_code in (200, 503)  # 503 while model stub unloaded
