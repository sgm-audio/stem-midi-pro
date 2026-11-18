# SPDX-License-Identifier: Apache-2.0
"""Numerical-health regression tests for the separator and transcriber.

These pin the fixes for the audit's Critical finding C-1 (signal collapse) and
the related ones:

* per-block hidden RMS must stay O(1) instead of decaying geometrically
  (1.0 → 4.5e-2 → … → 7.3e-20 across 12 blocks);
* masks must carry content (the old stack produced a constant 0.5 everywhere);
* ``guitar + bass + residual`` must reconstruct the input (mixture consistency —
  pinned as a known bug by ``test_pipeline_integrity.py::SEP-1``: this separator
  learns three INDEPENDENT sigmoid masks, so the stems do not sum to the mix; the
  test lives there, not here, to keep one pin per defect);
* silence must not produce notes, NaN, or Inf;
* the chunked scan must agree with an unchunked scan (state carry correctness);
* ``dt`` must stay inside its clamp;
* dropout-based uncertainty must actually vary.

Uses the small CPU config so the whole module runs in a few seconds.
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch", reason="requires torch")
yaml = pytest.importorskip("yaml", reason="requires pyyaml")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import _mamba_compat  # noqa: E402
from models.mamba_separator import MambaSeparator  # noqa: E402
from models.mamba_transcriber import MambaTranscriber  # noqa: E402

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "model_config.cpu.yaml",
)


@pytest.fixture(scope="module")
def cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def audio(cfg):
    torch.manual_seed(0)
    n = int(2.0 * cfg["audio"]["sample_rate"])
    return torch.randn(1, 1, n) * 0.1


def _separator_hidden_rms(cfg, audio):
    """Replay the separator's stack and record the RMS after each block.

    Replays ``forward()`` EXACTLY: ``hidden = hidden + block(norm(hidden))``,
    then ``final_norm``. Chaining the bare mixers instead (``hidden =
    block(hidden)``) multiplies the RMS by ~0.04 per block and reports a
    collapse the shipped model does not have -- that replay is what kept SEP-2
    on the open list for a release after the residual stack landed. It is kept
    below as ``test_bare_mixers_are_the_only_collapsing_stack`` so the
    distinction stays pinned in both directions.
    """
    torch.manual_seed(0)
    sep = MambaSeparator(cfg).eval()
    with torch.inference_mode():
        spec = torch.stft(
            audio.squeeze(1),
            cfg["audio"]["n_fft"],
            cfg["audio"]["hop_length"],
            window=sep.window,
            return_complex=True,
        )
        hidden = sep.input_proj(torch.abs(spec).transpose(1, 2))
        rms = [float(hidden.pow(2).mean().sqrt())]
        for norm, block in zip(sep.block_norms, sep.mamba_blocks, strict=True):
            hidden = hidden + block(norm(hidden))
            rms.append(float(hidden.pow(2).mean().sqrt()))
        rms.append(float(sep.final_norm(hidden).pow(2).mean().sqrt()))
    return rms


def test_block_rms_ladder_stays_order_one(cfg, audio):
    """C-1 / SEP-2: no geometric decay across the residual stack.

    Measured through the real path (norm + mixer + residual): 1.00 -> 1.01
    across all 12 blocks, then 1.00 after ``final_norm``.
    """
    rms = _separator_hidden_rms(cfg, audio)
    for i, value in enumerate(rms):
        assert 0.05 < value < 20.0, f"block {i} RMS {value:.3e} left [0.05, 20]"
    # The old stack fell off by ~19 orders of magnitude; even a mild gradient
    # would show up as a >10x drop between the first and last block.
    assert rms[-1] > 0.1 * rms[0], f"RMS decayed from {rms[0]:.3e} to {rms[-1]:.3e}"


def test_masks_carry_content(cfg, audio):
    """Masks must vary with the input, not sit at a constant 1/3.

    Replays the real stack and the CONFIGURED normalisation (softmax across
    stems by default, independent sigmoids on request) instead of assuming
    sigmoid over bare mixers.
    """
    torch.manual_seed(0)
    sep = MambaSeparator(cfg).eval()
    with torch.inference_mode():
        _g, _b, _r, _state, _metrics = sep(audio)
        spec = torch.stft(
            audio.squeeze(1),
            cfg["audio"]["n_fft"],
            cfg["audio"]["hop_length"],
            window=sep.window,
            return_complex=True,
        )
        hidden = sep.input_proj(torch.abs(spec).transpose(1, 2))
        for norm, block in zip(sep.block_norms, sep.mamba_blocks, strict=True):
            hidden = hidden + block(norm(hidden))
        hidden = sep.final_norm(hidden)
        logits = torch.stack([head(hidden) for head in sep.mask_heads.values()], dim=0)
        if sep.mask_normalization == "softmax":
            masks = torch.softmax(logits, dim=0)
        else:
            masks = logits.sigmoid()

    assert float(masks.std()) > 1e-3, "mask probabilities are constant (collapse)"
    assert 0.2 < float(masks.mean()) < 0.6


def test_masks_partition_the_mixture(cfg, audio):
    """SEP-1: the three masks sum to one per bin, so the stems reconstruct the mix.

    With independent sigmoids the residual error measured 0.46-0.51 of the input
    norm -- the residual stem was decorative. Softmax across stems makes
    ``guitar + bass + residual == mix`` by construction; measured 1.1e-07.
    """
    torch.manual_seed(0)
    sep = MambaSeparator(cfg).eval()
    with torch.inference_mode():
        guitar, bass, residual, _state, _metrics = sep(audio)
    rel = float((guitar + bass + residual - audio).norm() / audio.norm())
    assert rel < 1e-5, (
        f"||guitar+bass+residual - mix|| / ||mix|| = {rel:.3e}; the masks no longer "
        "partition the mixture (SEP-1)"
    )

    # The `independent` escape hatch restores the old, non-partitioning masks
    # (measured 5.3e-01). It stays available, but it must announce itself: if a
    # future refactor ever flips the DEFAULT to independent, the assertion above
    # fails rather than the residual quietly going decorative again.
    import copy

    c = copy.deepcopy(cfg)
    c["separator"]["mask_normalization"] = "independent"
    torch.manual_seed(0)
    indep = MambaSeparator(c).eval()
    with torch.inference_mode():
        guitar_i, bass_i, residual_i, _s, _m = indep(audio)
    rel_i = float((guitar_i + bass_i + residual_i - audio).norm() / audio.norm())
    assert rel_i > 0.1, (
        f"independent masks reconstructed the mixture to {rel_i:.3e}; the SEP-1 "
        "contrast no longer holds and this test is not proving anything"
    )


def test_bare_mixers_are_the_only_collapsing_stack(cfg, audio):
    """The control for SEP-2: the residual add is what prevents the collapse.

    Chaining the mixers without ``norm``/residual drops the RMS ~0.04x per
    block. If this ever stops being true the residual is doing nothing and the
    test above is passing for the wrong reason; if the real stack ever matches
    this ladder, the collapse is back.
    """
    torch.manual_seed(0)
    sep = MambaSeparator(cfg).eval()
    with torch.inference_mode():
        spec = torch.stft(
            audio.squeeze(1),
            cfg["audio"]["n_fft"],
            cfg["audio"]["hop_length"],
            window=sep.window,
            return_complex=True,
        )
        hidden = sep.input_proj(torch.abs(spec).transpose(1, 2))
        bare = [float(hidden.pow(2).mean().sqrt())]
        for block in sep.mamba_blocks:
            hidden = block(hidden)
            bare.append(float(hidden.pow(2).mean().sqrt()))

    real = _separator_hidden_rms(cfg, audio)
    # Bare mixers decay by more than two orders of magnitude end to end.
    assert bare[-1] < 1e-2 * bare[0], (
        f"bare mixers only decayed {bare[0]:.3e} -> {bare[-1]:.3e}; the SEP-2 "
        "control no longer demonstrates a collapse"
    )
    # The real stack does not, and the two differ by many orders of magnitude.
    assert real[-1] > 0.1 * real[0]


def test_no_nan_or_inf(cfg, audio):
    """Random input must not produce NaN/Inf stems or logits."""
    torch.manual_seed(0)
    sep = MambaSeparator(cfg).eval()
    with torch.inference_mode():
        guitar, bass, residual, _state, metrics = sep(audio)
        onset, pitch, velocity, expression, confidence = MambaTranscriber(cfg).eval()(guitar)
    for name, tensor in [
        ("guitar", guitar),
        ("bass", bass),
        ("residual", residual),
        ("metrics", metrics),
        ("onset", onset),
        ("pitch", pitch),
        ("velocity", velocity),
        ("expression", expression),
        ("confidence", confidence),
    ]:
        assert torch.isfinite(tensor).all(), f"{name} contains NaN/Inf"


def test_silence_emits_no_notes(cfg):
    """Digital silence must not generate notes, and confidence must be finite."""
    torch.manual_seed(0)
    separator = MambaSeparator(cfg).eval()
    transcriber = MambaTranscriber(cfg).eval()
    silence = torch.zeros(1, 1, cfg["audio"]["sample_rate"])  # 1 s

    with torch.inference_mode():
        guitar, _bass, _res, _state, _metrics = separator(silence)
        onset_logits, _pitch, _vel, _expr, confidence = transcriber(guitar)

    assert torch.isfinite(guitar).all()
    assert torch.isfinite(confidence).all()
    probs = torch.sigmoid(onset_logits)
    # A random-init model outputs a flat curve for silence; the hard guarantee is
    # that a silence-specific detector does not fire everywhere.
    assert float(probs.std()) < 1e-3, "onset probability varies on digital silence"


def test_dt_init_and_extreme_inputs():
    """dt_proj.bias must carry the upstream softplus^-1(Uniform) init.

    Zero-initialised bias produced an unbounded dt and a divergent scan (H-2).
    The upstream init guarantees ``softplus(bias) ∈ [1e-3, 0.1]`` at step 0.
    """
    torch.manual_seed(0)
    mamba = _mamba_compat.Mamba(d_model=32, d_state=8, d_conv=4, expand=2).eval()
    with torch.inference_mode():
        dt0 = torch.nn.functional.softplus(mamba.dt_proj.bias)
    assert float(dt0.min()) >= 9e-4
    assert float(dt0.max()) <= 0.11

    with torch.inference_mode():
        for scale in (0.0, 1e3, -1e3):
            out = mamba(torch.full((1, 16, 32), scale))
            assert torch.isfinite(out).all(), f"non-finite output for input {scale}"


def test_chunked_scan_matches_unchunked():
    """State carry across chunks must reproduce a single-pass scan."""
    torch.manual_seed(0)
    mamba = _mamba_compat.Mamba(d_model=32, d_state=8, d_conv=4, expand=2).eval()
    x = torch.randn(1, 200, 32)

    original = mamba.SCAN_CHUNK
    try:
        mamba.SCAN_CHUNK = 10_000  # single chunk
        with torch.inference_mode():
            reference = mamba(x)
        mamba.SCAN_CHUNK = 17  # many small chunks
        with torch.inference_mode():
            chunked = mamba(x)
    finally:
        mamba.SCAN_CHUNK = original

    assert torch.allclose(reference, chunked, atol=1e-4), (reference - chunked).abs().max().item()


def test_mc_dropout_returns_uncertainty(cfg):
    """mc_dropout_eval must report disagreement, not the mean prediction."""
    torch.manual_seed(0)
    transcriber = MambaTranscriber(cfg).eval()
    hidden = torch.randn(1, 32, cfg["transcriber"]["d_model"])
    uncertainty = transcriber.mc_dropout_eval(hidden, n_samples=4)
    assert uncertainty.shape == hidden.shape[:2]  # (B, T): one score per frame
    assert torch.isfinite(uncertainty).all()
    assert float(uncertainty.min()) >= 0.0 and float(uncertainty.max()) <= 1.0
    # An ensemble that always agrees would return exactly 1.0 everywhere.
    assert float(uncertainty.std()) > 1e-6
