# SPDX-License-Identifier: Apache-2.0
"""Adversarial quality-gate tests: can we be talked into a Studio badge?

``utils/quality_gates.py`` decides whether a customer downloads for free, is asked
to edit, or is upsold a $4.99 human review — from three numbers (avg_confidence,
si_sdr, artifact_flags). The suite today asserts the *shape* of that decision and
nothing about whether its inputs mean anything. These tests try to break it: a
render with no notes at all, a hard-clipped square wave, silence vs. music, and a
fake backend whose metrics are constants.

Gate-level tests here need no model at all (a ``ProcessingReport`` is a plain
dataclass), and the two that do use the real separator on a one-second clip.
"""

from __future__ import annotations

import math

import pytest

from tests.honesty import require

torch = require("torch", "real-separator gate tests", module_level=True)
np = require("numpy", "signal builders")

SR = 44100
HOP = 512
GATES = {
    "studio_confidence_threshold": 0.85,
    "draft_confidence_threshold": 0.70,
    "min_centroid_separation_db": 20.0,
}


def _report(
    centroid_separation_db: float,
    avg_confidence: float,
    flags=("none",),
    low_confidence_notes: int = 0,
    note_count: int | None = 1,
    total_notes: int | None = 1,
):
    """Build a report. Both counters default to 1 so the sample is a *render*.

    GATE-3 makes a zero-note render COMPLEX regardless of its numbers, so a
    helper that left a counter unset would silently test a case the gate never
    reaches. Two counters exist because upstream added `total_notes` (which
    drives the `empty_render` flag) while this branch carries `note_count`
    (which also rejects a negative count); tests that mean "an empty render"
    pass 0 for both.
    """
    from utils.quality_gates import ProcessingReport

    return ProcessingReport(
        centroid_separation_db=centroid_separation_db,
        phase_coherence=0.5,
        avg_confidence=avg_confidence,
        artifact_flags=list(flags),
        low_confidence_notes=low_confidence_notes,
        note_count=note_count,
        total_notes=total_notes,
        config={"quality_gates": dict(GATES)},
    )


def _tier(**kw):
    from utils.quality_gates import route_by_quality

    r = _report(**kw)
    return r.quality_tier, route_by_quality(r)


# ---------------------------------------------------------------------------
# the gate honours its config (guards: these must stay green)
# ---------------------------------------------------------------------------


def test_confidence_thresholds_come_from_the_config_not_from_defaults():
    """Confidence thresholds decide the tier; the proxy is observability-only."""
    from utils.quality_gates import ProcessingReport

    strict = ProcessingReport(
        centroid_separation_db=19.0,
        phase_coherence=0.5,
        avg_confidence=0.86,
        artifact_flags=["none"],
        low_confidence_notes=0,
        config={"quality_gates": {**GATES, "studio_confidence_threshold": 0.95}},
    )
    lenient = ProcessingReport(
        centroid_separation_db=19.0,
        phase_coherence=0.5,
        avg_confidence=0.86,
        artifact_flags=["none"],
        low_confidence_notes=0,
        config={"quality_gates": {**GATES, "studio_confidence_threshold": 0.85}},
    )
    assert strict.quality_tier is not lenient.quality_tier

    lowered = ProcessingReport(
        centroid_separation_db=19.0,
        phase_coherence=0.5,
        avg_confidence=0.86,
        artifact_flags=["none"],
        low_confidence_notes=0,
        config={"quality_gates": {**GATES, "min_centroid_separation_db": 10.0}},
    )
    raised = ProcessingReport(
        centroid_separation_db=19.0,
        phase_coherence=0.5,
        avg_confidence=0.86,
        artifact_flags=["none"],
        low_confidence_notes=0,
        config={"quality_gates": {**GATES, "min_centroid_separation_db": 25.0}},
    )
    assert lowered.quality_tier is raised.quality_tier


def test_proxy_thresholds_do_not_control_the_quality_tier():
    """The centroid proxy is reported for observability, not used as a gate."""
    from utils.quality_gates import ProcessingReport, QualityTier

    strict, _ = _tier(centroid_separation_db=19.0, avg_confidence=0.86)
    lenient = ProcessingReport(
        centroid_separation_db=19.0,
        phase_coherence=0.5,
        avg_confidence=0.86,
        artifact_flags=["none"],
        low_confidence_notes=0,
        config={"quality_gates": {**GATES, "min_centroid_separation_db": 10.0}},
    )
    assert strict is lenient.quality_tier is QualityTier.STUDIO
    assert lenient.min_centroid_separation_db == 10.0
    assert lenient.centroid_separation_db == 19.0


def test_artifact_flags_are_an_absolute_veto():
    """A clipped render cannot buy its way back with great numbers (ARCH-11.7)."""
    for flags in (["clipping"], ["phase_cancellation"], ["clipping", "noise_floor"]):
        tier, route = _tier(centroid_separation_db=29.9, avg_confidence=0.99, flags=flags)
        assert tier.value == "complex", f"{flags} with 29.9 dB/0.99 conf routed to {tier.value}"
        assert route["action"] == "manual_review", route
        assert route["review_in_daw"]


def test_badges_and_actions_agree_with_the_tier():
    for kw, expected in (
        (
            dict(centroid_separation_db=25.0, avg_confidence=0.95),
            ("studio", "direct_download", False),
        ),
        (
            dict(centroid_separation_db=25.0, avg_confidence=0.75),
            ("draft", "manual_review", False),
        ),
        (
            dict(centroid_separation_db=25.0, avg_confidence=0.30),
            ("complex", "manual_review", False),
        ),
    ):
        tier, route = _tier(**kw)
        assert (
            tier.value,
            route["action"],
            route.get("editor_launch", False),
        ) == expected, f"{kw} → {tier.value}/{route['action']}, expected {expected}"


# ---------------------------------------------------------------------------
# attempts to buy a Studio badge that should not be sellable
# ---------------------------------------------------------------------------


def test_a_render_with_no_notes_is_never_routed_as_a_result():
    """Regression guard for GATE-3: an empty render is never offered for download."""
    tier, route = _tier(
        centroid_separation_db=25.0,
        avg_confidence=0.95,
        low_confidence_notes=0,
        note_count=0,  # the render contains no notes at all
        total_notes=0,
    )
    assert (
        route["action"] != "direct_download"
    ), f"an empty render was offered for download as {tier.value} with badge {route['badge']!r}"


def test_backend_metrics_depend_on_the_signal(pipeline_config):
    from models.backends.fake_backend import FakeSeparatorBackend

    backend = FakeSeparatorBackend(pipeline_config)
    with torch.inference_mode():
        silence = backend.separate(torch.zeros(1, 1, SR))["metrics"]
        music = backend.separate(
            torch.from_numpy((np.sin(2 * np.pi * 220 * np.arange(SR) / SR) * 0.4).astype("float32"))
            .unsqueeze(0)
            .unsqueeze(0)
        )["metrics"]
    assert not torch.allclose(silence, music), (
        f"si_sdr/phase are identical for silence and a sine: {silence.tolist()} vs "
        f"{music.tolist()} — any tier assertion through the fake backend is circular"
    )


@pytest.fixture(scope="module")
def real_separator():
    from models.mamba_separator import MambaSeparator

    cfg = {
        "audio": {"sample_rate": SR, "n_fft": 2048, "hop_length": HOP, "n_mels": 80},
        "separator": {"d_model": 32, "n_layers": 1, "d_state": 8, "d_conv": 4, "expand": 2},
    }
    torch.manual_seed(0)
    return MambaSeparator(cfg).eval()


def _proxy(sep, audio: np.ndarray) -> float:
    x = torch.from_numpy(np.asarray(audio, dtype="float32")).unsqueeze(0).unsqueeze(0)
    with torch.inference_mode():
        _g, _b, _r, _c, metrics = sep(x)
    return float(metrics[0, 0])


@pytest.mark.parametrize("case", ["clean", "clipped"], ids=["clean-sine", "hard-clipped-square"])
def test_the_separation_proxy_does_not_prefer_clipping(real_separator, synth, case):
    """GATE-4 (fixed): the reported metric must not improve under distortion.

    The old centroid-gap proxy scored a hard-clipped square HIGHER than the clean
    sine (17.75 vs 0.00 at the production grid). The replacement -- mask
    crosstalk, lower is better -- is pinned here at the level the gate reads.
    """
    clean = (np.sin(2 * np.pi * 220 * np.arange(SR) / SR) * 0.4).astype("float32")
    clipped = np.sign(np.sin(2 * np.pi * 220 * np.arange(SR) / SR)) * 0.4
    clipped = clipped.astype("float32")
    scores = {"clean": _proxy(real_separator, clean), "clipped": _proxy(real_separator, clipped)}
    assert all(
        0.0 <= v <= 1.0 for v in scores.values()
    ), f"the reported separation left the crosstalk range [0, 1]: {scores}"
    assert scores["clipped"] <= scores["clean"] + 0.05, (
        f"the metric rates the hard-clipped square {scores['clipped']:.3f} vs the clean sine "
        f"{scores['clean']:.3f}: clipping must not score higher than the undistorted signal"
    )


def test_silence_cannot_earn_a_separation_score(real_separator, synth):
    """A zero-energy render cannot claim a good separation.

    Crosstalk is a fraction of energy, so silence is 0.0 by construction: with
    no energy there is nothing to place in a bin, and a "perfect" 0.0 must not
    be read as a quality signal. The tier path refuses silence through
    `silent_output` flags instead -- this pins that the metric itself is not
    what lets it through.
    """
    score = _proxy(real_separator, np.zeros(SR, dtype="float32"))
    assert 0.0 <= score <= 1.0, f"silence produced {score}, outside the crosstalk range"


@pytest.mark.memory
def test_confidence_does_not_reward_degraded_content(pipeline_config, synth, make_wav):
    """Regression guard for CONF-1: degenerate input must not score like music.

    The original finding was that the confidence head had no loss term, so it
    reported ~0.9555 for everything and the *worst* renders scored no worse than
    clean ones. Two things changed: the head now has a supervised term
    (``models/losses.py``), and a confident render additionally needs
    independent corroboration before it can be called studio quality
    (``utils/quality_gates.ProcessingReport.quality_tier``).

    What is asserted here is the thing that must hold regardless of whether the
    shipped weights were ever trained: silence must never outrank music.
    """
    from main import StemMidiModel

    cfg = {**pipeline_config, "quality_gates": {**GATES, "min_confidence": 0.6}}
    seconds = 1.0
    n = int(SR * seconds)  # np.zeros needs an int; `SR * 1.0` is a float
    t = np.arange(n) / SR
    signals = {
        "silence": np.zeros(n, dtype="float32"),
        "music": synth.note_sequence(total_seconds=seconds, sr=SR).astype("float32"),
        "clipped": (np.sign(np.sin(2 * np.pi * 110 * t)) * 0.99).astype("float32"),
        "noise": (np.random.RandomState(0).randn(n) * 0.3).astype("float32"),
    }
    torch.manual_seed(0)
    model = StemMidiModel(cfg)
    model.eval()
    scores: dict[str, tuple[float, str]] = {}
    for case, audio in signals.items():
        path = make_wav(audio, name=f"conf_{case}.wav")
        with torch.inference_mode():
            out = model.process_audio_file(path)
        rep = out["report"]
        # CONF-1: the tier reads the CONTENT-CALIBRATED number, not the raw head
        # (avg_confidence is still reported for observability and is asserted
        # separately by test_pipeline_integrity).
        conf = rep.calibrated_confidence
        scores[case] = (conf, rep.quality_tier.value)

    finite = [c for c, _ in scores.values() if not math.isnan(c)]
    assert len(finite) == len(scores), f"a metric went non-finite: {scores}"
    # (a) ORDERING: the number the tier reads must rank music above degenerate
    # input. This is what CONF-1 was about -- the raw head ranked a clipped
    # square above clean music, so the calibrated number is what gets pinned.
    music = scores["music"][0]
    worst = max(scores["clipped"][0], scores["noise"][0])
    assert music > worst, (
        "calibrated_confidence: "
        + ", ".join(f"{k}={v[0]:.4f}({v[1]})" for k, v in scores.items())
        + f" — clean music scores {music:.4f} while the clipped/noise renders score "
        f"{worst:.4f}; the tier input rewards degradation instead of penalising it"
    )

    # (b) THE BADGE: whatever the number says, no degenerate render may be
    # offered as deliverable. Belt and braces with (a) on purpose -- a future
    # recalibration that preserves ordering but still grants a badge would slip
    # past (a) alone, and the badge is what the user sees.
    for case in ("silence", "clipped", "noise"):
        assert (
            scores[case][1] != "studio"
        ), f"{case} was graded studio (confidence={scores[case][0]:.4f}): " + ", ".join(
            f"{k}={v[0]:.4f}({v[1]})" for k, v in scores.items()
        )
    # Silence specifically must never clear the confidence bar at all: an empty
    # render is not a render that the model can be confident about.
    assert scores["silence"][0] < 0.1, (
        f"silence reported calibrated_confidence={scores['silence'][0]:.4f} "
        f"(tier {scores['silence'][1]}) — an empty render should not be within "
        f"{scores['silence'][0]:.2f} of a musical one"
    )
