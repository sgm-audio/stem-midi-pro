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

from tests.honesty import known_bug, require

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
):
    from utils.quality_gates import ProcessingReport

    return ProcessingReport(
        centroid_separation_db=centroid_separation_db,
        phase_coherence=0.5,
        avg_confidence=avg_confidence,
        artifact_flags=list(flags),
        low_confidence_notes=low_confidence_notes,
        config={"quality_gates": dict(GATES)},
    )


def _tier(**kw):
    from utils.quality_gates import route_by_quality

    r = _report(**kw)
    return r.quality_tier, route_by_quality(r)


# ---------------------------------------------------------------------------
# the gate honours its config (guards: these must stay green)
# ---------------------------------------------------------------------------


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


@known_bug(
    "GATE-3",
    "a render containing ZERO notes can be sold as studio quality: the gate only sees "
    "avg_confidence/si_sdr/artifact_flags, so 0 notes + confidence 0.95 + 25 dB → STUDIO → "
    "'direct_download, ✨ Studio Quality'. Nothing in the pipeline feeds note count (or "
    "note-density sanity) into the decision, even though ProcessingReport already carries "
    "low_confidence_notes — require n_notes > 0 and a plausible density (measured: the "
    "untrained transcriber emits 342-343 notes per second even for digital silence).",
)
def test_a_render_with_no_notes_is_never_routed_as_a_result():
    tier, route = _tier(centroid_separation_db=25.0, avg_confidence=0.95, low_confidence_notes=0)
    assert (
        route["action"] != "direct_download"
    ), f"an empty render was offered for download as {tier.value} with badge {route['badge']!r}"


@known_bug(
    "FAKE-1",
    "the Fake backend returns a constant centroid_separation_db=12.0 / phase_coherence=0.5 "
    "for ANY input "
    "(models/backends/fake_backend.py), so a test that runs a quality decision through the fake "
    "path asserts a fixture, not behaviour — and the constant is *below* "
    "min_centroid_separation_db, which makes "
    "the fake path permanently COMPLEX and therefore unable to exercise the studio/draft branches "
    "at all. Return metrics computed from the signal (even cheap ones).",
)
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


@known_bug(
    "GATE-4",
    "the separation number the gate compares against min_centroid_separation_db=20 is not a "
    "separation metric at "
    "all: MambaSeparator._centroid_separation is a ratio of guitar/bass spectral centroids "
    "clamped to [0, 30] and it *rewards* distortion — at the production grid this test measures "
    "17.75 dB for a hard-clipped square wave against 0.00 dB for the same clean sine, and the "
    "corpus case in tests/test_audio_edge_cases.py reaches 24.75 dB, above "
    "min_centroid_separation_db=20, i.e. "
    "an outright Studio badge for a square wave. Silence and a +8 dB overflow both score 0.0, so "
    "the metric neither sees clipping nor verifies that anything was separated. Replace with a "
    "reference-free check (mask sum / residual energy / MR-STFT) or gate on artifact flags first.",
)
@pytest.mark.parametrize("case", ["clean", "clipped"], ids=["clean-sine", "hard-clipped-square"])
def test_the_separation_proxy_does_not_prefer_clipping(real_separator, synth, case):
    clean = (np.sin(2 * np.pi * 220 * np.arange(SR) / SR) * 0.4).astype("float32")
    clipped = np.sign(np.sin(2 * np.pi * 220 * np.arange(SR) / SR)) * 0.4
    clipped = clipped.astype("float32")
    scores = {"clean": _proxy(real_separator, clean), "clipped": _proxy(real_separator, clipped)}
    assert scores["clipped"] < scores["clean"], (
        f"the proxy rates the hard-clipped square {scores['clipped']:.2f} dB vs the clean sine "
        f"{scores['clean']:.2f} dB (min_centroid_separation_db="
        f"{GATES['min_centroid_separation_db']}): clipping scores higher "
        f"than the undistorted signal"
    )


def test_silence_cannot_earn_a_separation_score(real_separator, synth):
    """A zero-energy render must score zero on whatever the gate calls separation."""
    score = _proxy(real_separator, np.zeros(SR, dtype="float32"))
    assert score == pytest.approx(0.0, abs=1e-6), (
        f"silence scored {score:.2f} dB of 'separation': the metric rewards nothing happening, "
        f"so it cannot gate anything"
    )


@known_bug(
    "CONF-1",
    "the number every tier depends on rewards the wrong content: measured end-to-end at "
    "d_model=32/n_layers=1/seed 0 on one-second inputs, avg_confidence is 0.5164 for a "
    "hard-clipped square wave and 0.5101 for white noise, but only 0.2998 for clean music and "
    "0.2833 for silence — the *worst* render scores highest. The head is also uninformative "
    "within a file (0.4892 ± 0.0045 across frames) and mc_dropout_eval() reports the MEAN of "
    "the stochastic passes (0.500 vs 0.420 for one pass), which is why STUDIO's >= 0.85 can "
    "never fire. Add a calibration step and never tier on the raw head.",
)
@pytest.mark.memory
def test_confidence_does_not_reward_degraded_content(pipeline_config, synth, make_wav):
    """Degenerate input must not score like music — one verdict over four signals.

    Kept as a single test on purpose: a parametrised known-bug where one case
    legitimately passes becomes an XPASS(strict) failure that hides the real
    result, which is the opposite of honest reporting.
    """
    from main import StemMidiModel

    cfg = {**pipeline_config, "quality_gates": {**GATES, "min_confidence": 0.6}}
    seconds = 1.0
    n = SR * seconds
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
        conf = rep.avg_confidence
        scores[case] = (conf, rep.quality_tier.value)

    finite = [c for c, _ in scores.values() if not math.isnan(c)]
    assert len(finite) == len(scores), f"a metric went non-finite: {scores}"
    music = scores["music"][0]
    worst = max(scores["clipped"][0], scores["noise"][0])
    assert music > worst, (
        "avg_confidence: "
        + ", ".join(f"{k}={v[0]:.4f}({v[1]})" for k, v in scores.items())
        + f" — clean music scores {music:.4f} while the clipped/noise renders score "
        f"{worst:.4f}; the tier input rewards degradation instead of penalising it"
    )
    assert scores["silence"][0] < 0.1, (
        f"silence reported avg_confidence={scores['silence'][0]:.4f} "
        f"(tier {scores['silence'][1]}) — an empty render should not be within "
        f"{scores['silence'][0]:.2f} of a musical one"
    )
