# SPDX-License-Identifier: Apache-2.0
"""End-to-end API integration test (T-8.3.7).

POST /process with a WAV → 200 + ZIP with 5 expected entries.

`api.py` imports StemMidiModel (→ torch/nemo/mamba_ssm) at module level;
until T-8.2.1 lazy-imports it, this test skips when those are missing.
The model itself is monkeypatched with a lightweight fake so no CUDA/torch
inference is exercised here.
"""

import io
import zipfile
from typing import ClassVar

import pytest

fastapi = pytest.importorskip("fastapi", reason="requires fastapi")
pytest.importorskip("httpx", reason="TestClient requires httpx")
pytest.importorskip("soundfile", reason="fixture writes WAV via soundfile")


def _load_api():
    pytest.importorskip("torch")
    try:
        import api
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"Cannot import api.py (see T-8.2.1): {exc}")
    return api


class _FakeReport:
    si_sdr: float = 22.0
    phase_coherence: float = 0.9
    avg_confidence: float = 0.9
    artifact_flags: ClassVar[list] = ["none"]
    low_confidence_notes: int = 0

    from utils.quality_gates import QualityTier

    quality_tier = QualityTier.STUDIO


class _FakeModel:
    def process_audio_file(self, path, quantize=False):
        import numpy as np

        return {
            "stems": {
                "guitar": np.zeros(4410, dtype="float32"),
                "bass": np.zeros(4410, dtype="float32"),
            },
            "midi": {"midi_events": [], "summary": {}},
            "report": _FakeReport(),
            "routing": {"action": "direct_download"},
            "sample_rate": 44100,
        }


@pytest.mark.integration
def test_process_returns_zip_with_five_entries(temp_audio_file, monkeypatch):
    api = _load_api()

    # Bypass the real (heavy) model — the ZIP contract is what's under test.
    monkeypatch.setattr(api, "model", _FakeModel(), raising=False)
    monkeypatch.setattr(api, "model_config", {"audio": {"sample_rate": 44100}}, raising=False)

    from fastapi.testclient import TestClient

    # Instantiate without the `with` block so the startup hook (which calls
    # the real load_model()) does not run.
    client = TestClient(api.app, raise_server_exceptions=True)

    with open(temp_audio_file, "rb") as fh:
        response = client.post(
            "/process",
            files={"file": ("test.wav", fh, "audio/wav")},
        )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"

    zf = zipfile.ZipFile(io.BytesIO(response.content))
    names = set(zf.namelist())
    assert names == {
        "guitar_stem.wav",
        "bass_stem.wav",
        "guitar.mid",
        "bass.mid",
        "processing_report.json",
    }

    import json

    report = json.loads(zf.read("processing_report.json"))
    assert report["quality_tier"] == "studio"
    assert report["avg_confidence"] == pytest.approx(0.9)
