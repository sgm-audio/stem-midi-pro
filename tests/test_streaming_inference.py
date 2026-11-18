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


# ---------------------------------------------------------------------------
# CPU-3: cross-chunk SSM state, measured rather than assumed
# ---------------------------------------------------------------------------


def test_state_threading_is_exact_at_the_block_level():
    """The mechanism works: two calls with the state equal one long call.

    This is the property CPU-3 asks for, and it already holds. Without this
    guard, "CPU-3 is closed" would rest on a docstring.
    """
    import torch

    from models._mamba_compat import Mamba

    torch.manual_seed(0)
    blk = Mamba(d_model=64, d_state=16).eval()
    x = torch.randn(1, 128, 64)
    with torch.inference_mode():
        whole = blk(x)
        first, state = blk(x[:, :64], return_state=True)
        second = blk(x[:, 64:], state=state)
    cat = torch.cat([first, second], dim=1)
    rel = float((cat - whole).norm() / whole.norm())
    assert rel < 1e-6, f"block-level state threading drifted: rel err {rel:.3e}"


def test_the_pipeline_gains_nothing_from_carrying_state():
    """Why the default path does not thread state: measured zero benefit.

    If this ever fails in the direction of "dropping state is now much worse",
    the crossfade has stopped doing its job and threading becomes worthwhile.
    """
    import torch
    import yaml

    from models.mamba_separator import MambaSeparator

    with open("configs/model_config.mamba.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    torch.manual_seed(0)
    sep = MambaSeparator(cfg).eval()
    sr = int(cfg["audio"]["sample_rate"])
    t = np.arange(2 * sr) / sr
    x = torch.from_numpy(
        (0.3 * np.sin(2 * np.pi * 110 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    )[None, None]
    cut = (x.shape[-1] // 2 // int(cfg["audio"]["hop_length"])) * int(cfg["audio"]["hop_length"])

    with torch.inference_mode():
        whole = sep(x)[0]
        _first, _b, _r, state, _m = sep(x[:, :, :cut], return_state=True)
        carried = sep(x[:, :, cut:], state=state)[0]
        dropped = sep(x[:, :, cut:])[0]

    def rel(u: "torch.Tensor") -> float:
        return float((u - whole[:, :, cut:]).norm() / whole[:, :, cut:].norm())

    effect = float((carried - dropped).norm() / whole[:, :, cut:].norm())
    improvement = rel(dropped) / max(rel(carried), 1e-12)
    assert effect < 5e-3, (
        f"carrying state moved the output by {effect:.3e}; the STFT edge effect no "
        "longer dominates and the default path should be reconsidered"
    )
    assert improvement < 1.05, (
        f"carrying state now improves chunk-vs-whole accuracy by {improvement:.2f}x; "
        "CPU-3's verdict ('no measurable benefit in the pipeline') no longer holds"
    )
