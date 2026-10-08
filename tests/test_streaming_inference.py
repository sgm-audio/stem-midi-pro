# SPDX-License-Identifier: Apache-2.0
"""Streaming inference chunking test (T-8.3.8).

A 6-second file processed with chunk_seconds=2.0 must yield
chunks_processed == 3 when the hop equals the chunk length
(overlap_ratio forced to 0 in the test config).

The forward path itself requires the C-2.5 fix (real streaming via the
separator's state-cache argument); pending that fix the test is written to
the POST-FIX contract.
"""

import pytest

torch = pytest.importorskip("torch", reason="requires torch")
np = pytest.importorskip("numpy", reason="requires numpy")
sf = pytest.importorskip("soundfile", reason="requires soundfile")


def test_streaming_chunks(mock_config, tmp_path):
    try:
        from main import StemMidiModel
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"Cannot import main.py: {exc}")

    # 6 seconds of synthetic audio.
    sr = mock_config["audio"]["sample_rate"]
    t = np.linspace(0, 6, 6 * sr)
    audio = (0.2 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)
    path = tmp_path / "six_seconds.wav"
    sf.write(str(path), audio, sr)

    # Overlap 0 → hop == chunk length → exactly 3 chunks of 2.0s.
    config = {**mock_config, "audio": {**mock_config["audio"], "overlap_ratio": 0.0}}

    model = StemMidiModel(config)
    model.eval()

    # expects fix per TODO C-2.5 (separator forward must accept state_cache)
    try:
        result = model.process_audio_streaming(str(path), chunk_seconds=2.0)
    except TypeError as exc:
        if "state_cache" in str(exc):
            pytest.xfail("streaming requires C-2.5 fix (state_cache arg)")
        raise

    assert result["chunks_processed"] == 3
