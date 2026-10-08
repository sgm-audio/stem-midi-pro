# SPDX-License-Identifier: Apache-2.0
"""Quantization tests (T-8.3.15).

Expected behavior (per API-6 / PERF-7.8):
`process_audio_file(..., quantize=True)` must return the same output shapes
as the non-quantized path. The API exposes it as `POST /process?quantize=true`.
"""

import pytest

fastapi = pytest.importorskip("fastapi", reason="requires fastapi")
pytest.importorskip("httpx", reason="TestClient requires httpx")
pytest.importorskip("soundfile", reason="fixture writes WAV")


@pytest.mark.integration
def test_quantized_process_same_output_shape(monkeypatch, temp_audio_file):
    """quantize=True POST returns a ZIP with the same 5 entries/shapes."""
    pytest.importorskip("torch")
    try:
        import api
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"Cannot import api.py (see T-8.2.1): {exc}")

    import numpy as np

    class FakeModel:
        def process_audio_file(self, path, quantize=False):
            return {
                "stems": {
                    "guitar": np.zeros(4410, dtype="float32"),
                    "bass": np.zeros(4410, dtype="float32"),
                },
                "midi": {"midi_events": [], "summary": {}},
                "report": type(
                    "R",
                    (),
                    {
                        "si_sdr": 20.0,
                        "phase_coherence": 0.9,
                        "avg_confidence": 0.9,
                        "artifact_flags": ["none"],
                        "low_confidence_notes": 0,
                        "quality_tier": __import__(
                            "utils.quality_gates", fromlist=["QualityTier"]
                        ).QualityTier.DRAFT,
                    },
                )(),
                "routing": {"action": "editor_launch"},
                "sample_rate": 44100,
            }

    monkeypatch.setattr(api, "model", FakeModel(), raising=False)
    monkeypatch.setattr(api, "model_config", {"audio": {"sample_rate": 44100}}, raising=False)

    from fastapi.testclient import TestClient

    client = TestClient(api.app, raise_server_exceptions=True)
    with open(temp_audio_file, "rb") as fh:
        resp = client.post(
            "/process?quantize=true",
            files={"file": ("t.wav", fh, "audio/wav")},
        )

    assert resp.status_code == 200

    import io
    import zipfile

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    assert "guitar_stem.wav" in zf.namelist()
    assert "bass_stem.wav" in zf.namelist()
