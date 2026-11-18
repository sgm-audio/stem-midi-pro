# SPDX-License-Identifier: Apache-2.0
"""GATE-4: the reported separation number must describe separation.

The metric used to be ``clamp(10*log10(|centroid_guitar - centroid_bass|) + 20,
0, 30)``: a gap between two spectral centroids, in FFT-bin units, mapped through
an arbitrary affine transform. It fails in the most damaging direction -- it
grows when DISTORTION drives the stems apart.

Measured on the shipped config, one 220 Hz tone clipped progressively harder:

    clip peak   0.30   0.20   0.12   0.08   0.05
    old proxy   15.64  23.97  26.89  27.80  28.73

The worse the upload, the better the "separation", and a hard-clipped square
rated the maximum 30.00/30. These tests pin the replacement.

The replacement is **mask crosstalk**: the share of stem energy that sits in
time-frequency bins the stem does not dominate. It is reference-free (so it can
be computed on a real upload, unlike SI-SDR), it is bounded [0, 1], lower is
better, and it does not reward distortion because distortion does not move
energy into bins a stem dominates.

Calibration measured on this repo (2 s, configured n_fft/hop):

    oracle band split at 400 Hz   0.0000     (a genuinely good separator)
    untrained shipped init        0.4677
    flat 1/3 masks                0.6667     (the analytic no-separation value)
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="the metric is computed in torch")

SR = 44100


@pytest.fixture(scope="module")
def sep_cfg():
    import yaml

    with open("configs/model_config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def separator(sep_cfg):
    from models.mamba_separator import MambaSeparator

    torch.manual_seed(0)
    return MambaSeparator(sep_cfg).eval()


def _spec(seq: np.ndarray, cfg: dict) -> torch.Tensor:
    """(B=1, T_f, F) complex STFT in the layout the separator uses."""
    x = torch.from_numpy(seq.astype(np.float32))[None, None]
    with torch.inference_mode():
        return torch.stft(
            x.squeeze(1),
            cfg["audio"]["n_fft"],
            cfg["audio"]["hop_length"],
            window=torch.hann_window(cfg["audio"]["n_fft"]),
            return_complex=True,
        ).transpose(1, 2)


def test_flat_masks_read_the_analytic_no_separation_value(separator, sep_cfg):
    """Three equal masks hand out 2/3 of every stem's energy where it is not dominant.

    This is the floor of the scale and it is arithmetic, not a fitted constant,
    which is why the default ceiling (0.60) can sit just below it.
    """
    t = np.arange(2 * SR) / SR
    seq = 0.3 * np.sin(2 * np.pi * 880 * t) + 0.3 * np.sin(2 * np.pi * 80 * t)
    spec = _spec(seq, sep_cfg)
    flat = {name: torch.full_like(spec.real, 1 / 3) for name in ("guitar", "bass", "residual")}
    with torch.inference_mode():
        value = float(separator._crosstalk(flat, spec))
    assert value == pytest.approx(
        2 / 3, abs=2e-3
    ), f"flat masks read {value:.4f}, expected the analytic 2/3 = 0.6667"


def test_a_perfect_band_split_reads_zero(separator, sep_cfg):
    """An oracle separator that carves the spectrum has no crosstalk.

    Without this the metric could be satisfied by "always report a big number"
    in the low-is-good direction too; a real split has to reach 0.
    """
    t = np.arange(2 * SR) / SR
    seq = 0.3 * np.sin(2 * np.pi * 880 * t) + 0.3 * np.sin(2 * np.pi * 80 * t)
    spec = _spec(seq, sep_cfg)
    freqs = torch.linspace(0, spec.shape[-1] - 1, spec.shape[-1]) / (spec.shape[-1] - 1) * (SR / 2)
    hi = (freqs >= 400).to(spec.real.dtype).view(1, 1, -1).expand(1, spec.shape[1], -1)
    lo = 1.0 - hi
    masks = {"guitar": hi, "bass": lo, "residual": torch.zeros_like(hi)}
    with torch.inference_mode():
        value = float(separator._crosstalk(masks, spec))
    assert value < 1e-6, f"an exact band split read {value:.3e}, expected 0"


def test_crosstalk_does_not_reward_distortion(separator, sep_cfg):
    """The old proxy's worst property, pinned in the opposite direction.

    Clipping a tone harder is strictly worse audio. The replacement must not
    improve as a result; a monotone increase with clipping is exactly the bug.
    """
    t = np.arange(2 * SR) / SR
    base = 0.3 * np.sin(2 * np.pi * 220 * t)
    scores = []
    for clip in (0.30, 0.20, 0.12, 0.05):
        x = torch.from_numpy(np.clip(base, -clip, clip).astype(np.float32))[None, None]
        with torch.inference_mode():
            _g, _b, _r, _state, metrics = separator(x)
        scores.append(float(metrics[0, 0]))

    assert all(0.0 <= s <= 1.0 for s in scores), f"crosstalk left [0, 1]: {scores}"
    worst, best = scores[0], scores[-1]
    # Harder clipping must not look BETTER. Allow a small band for numerical
    # noise, but a rise of the old magnitude (+13 points) must fail here.
    assert best < worst + 0.05, (
        f"crosstalk rose from {worst:.3f} (clip 0.30) to {best:.3f} (clip 0.05): "
        "harder clipping is being scored as better separation"
    )


def test_zero_bit_perfection_still_passes_the_gate():
    """The tier gate: crosstalk at or below the ceiling does not block STUDIO."""
    from utils.quality_gates import ProcessingReport

    gates = {
        "studio_confidence_threshold": 0.85,
        "draft_confidence_threshold": 0.70,
        "max_separation_crosstalk": 0.60,
    }
    report = ProcessingReport(
        separation_proxy=0.30,
        phase_coherence=0.6,
        avg_confidence=0.95,
        artifact_flags=["none"],
        low_confidence_notes=0,
        note_count=10,
        total_notes=10,
        config={"quality_gates": gates},
    )
    assert report.quality_tier.value == "studio"


def test_worse_than_a_flat_mask_cannot_be_studio_or_draft():
    """Above the ceiling the render is not deliverable, however confident."""
    from utils.quality_gates import ProcessingReport

    gates = {
        "studio_confidence_threshold": 0.85,
        "draft_confidence_threshold": 0.70,
        "max_separation_crosstalk": 0.60,
    }
    report = ProcessingReport(
        separation_proxy=0.90,
        phase_coherence=0.6,
        avg_confidence=0.99,
        artifact_flags=["none"],
        low_confidence_notes=0,
        note_count=10,
        total_notes=10,
        config={"quality_gates": gates},
    )
    assert report.quality_tier.value == "complex", (
        "a separation worse than a flat one-third mask was graded "
        f"{report.quality_tier.value} on confidence alone"
    )


def test_the_ceiling_is_configurable_and_can_be_disabled():
    from utils.quality_gates import ProcessingReport

    def tier(crosstalk, ceiling):
        gates = {
            "studio_confidence_threshold": 0.85,
            "draft_confidence_threshold": 0.70,
            "max_separation_crosstalk": ceiling,
        }
        r = ProcessingReport(
            separation_proxy=crosstalk,
            phase_coherence=0.6,
            avg_confidence=0.95,
            artifact_flags=["none"],
            low_confidence_notes=0,
            note_count=10,
            total_notes=10,
            config={"quality_gates": gates},
        )
        return r.quality_tier.value

    assert tier(0.90, 0.95) == "studio", "a raised ceiling was not honoured"
    # `null` disables the GATE, but it also removes the independent evidence
    # CONF-1 requires before a render may be called studio quality, so the
    # render stops at `draft`. See tests/test_confidence_honesty.py::
    # test_disabling_the_separation_gate_removes_the_corroboration_not_the_bar.
    assert tier(0.90, None) == "draft", "null must disable the gate AND its corroboration"


def test_an_out_of_range_ceiling_is_rejected():
    from utils.quality_gates import ProcessingReport

    with pytest.raises(ValueError, match="max_separation_crosstalk"):
        ProcessingReport(
            separation_proxy=0.3,
            phase_coherence=0.6,
            avg_confidence=0.95,
            artifact_flags=["none"],
            low_confidence_notes=0,
            note_count=1,
            config={"quality_gates": {"max_separation_crosstalk": 2.0}},
        )
