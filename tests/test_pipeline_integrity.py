# SPDX-License-Identifier: Apache-2.0
"""Pipeline integrity: does what ``main.StemMidiModel`` claims to deliver hold up?

Audit context (tasks 1-3): the existing model-level tests
(``test_backends.py::test_stem_midi_model_with_fake_backends``,
``test_process_audio_file_fake``) assert *key presence* only — ``"guitar" in
out["midi_metadata"]``, ``"stems" in result``, ``result["sample_rate"] == 44100``.
Those assertions are satisfied by an empty list under both stems and by a render
whose MIDI is a duplicate of the guitar's. Worse, both tests are named for the
fake backends but never touch them: ``StemMidiModel.__init__`` hard-wires
``MambaSeparator``/``MambaTranscriber`` and never calls ``build_separator`` /
``build_transcriber``, so ``cfg["backends"]`` is dead config (see
test_backend_contracts.py).

The tests here use deterministic stub submodules for the plumbing (so they are
about wiring, not luck) and the real modules for the properties that only the
real path can show, marked ``real_mamba`` so a skip is a suite failure.
"""

from __future__ import annotations

import io
import json
import math

import pytest

from tests.doubles import StubReport, StubSeparator, StubTranscriber, frame_count, inject
from tests.honesty import known_bug, require

torch = require("torch", "pipeline integrity tests", module_level=True)
np = require("numpy", "pipeline integrity tests", module_level=True)
require("mido", "MIDI re-parsing", module_level=True)

from tests.synth import parse_midi  # noqa: E402
from utils.midi import build_midi_from_events  # noqa: E402

SR = 44100
HOP = 512


def _model_with_stems(pipeline_config, guitar, bass, onsets, pitches, n_frames=None, **tr_kw):
    """StemMidiModel with deterministic separator/transcriber doubles injected."""
    from main import StemMidiModel

    model = StemMidiModel(pipeline_config)
    model.eval()
    T = guitar.shape[-1]
    n_frames = n_frames or frame_count(T, HOP)
    inject(model, "separator", StubSeparator(guitar, bass))
    inject(model, "transcriber", StubTranscriber(onsets, pitches, n_frames, **tr_kw))
    return model


# ---------------------------------------------------------------------------
# deterministic test doubles for the plumbing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 1. per-stem MIDI must be per-stem
# ---------------------------------------------------------------------------


def _stem_aware_transcriber(guitar_stem, bass_stem, plan, n_frames):
    """A transcriber double that answers differently for each stem it is handed.

    A stem-blind stub cannot detect STEM-1: if the model transcribes the guitar
    twice, an identical-on-both-calls double still produces "correct-looking"
    per-stem dicts. The double has to be the thing that varies.
    """

    def tr(stem):
        is_bass = stem is bass_stem or bool(torch.equal(stem, bass_stem))
        onsets, pitches = plan["bass" if is_bass else "guitar"]
        onset = torch.full((1, n_frames, 1), -10.0)
        pitch = torch.zeros((1, n_frames, 128))
        vel = torch.full((1, n_frames, 1), 0.7)
        expr = torch.zeros((1, n_frames, 3))
        conf = torch.full((1, n_frames), 0.9)
        for f, pid in zip(onsets, pitches, strict=False):
            onset[0, f, 0] = 10.0
            pitch[0, f, pid] = 10.0
        return onset, pitch, vel, expr, conf

    return tr


def test_guitar_and_bass_midi_are_different_renders(pipeline_config):
    """Regression guard (was STEM-1, fixed by H-8): each stem gets its own transcription.

    The original defect aliased enhanced_midi into both per-stem keys and only ever
    transcribed the guitar, so bass.mid carried the guitar part. A stub transcriber
    that ignores its input would pass that bug unnoticed, hence the per-stem double.
    """
    from main import StemMidiModel

    t = torch.arange(SR).float() / SR
    guitar = (0.4 * torch.sin(2 * 3.14159 * 220 * t))[None, None]
    bass = (0.4 * torch.sin(2 * 3.14159 * 55 * t))[None, None]
    plan = {"guitar": ([10, 40], [64, 71]), "bass": ([12, 44], [40, 43])}
    model = StemMidiModel(pipeline_config)
    model.eval()
    inject(model, "separator", StubSeparator(guitar, bass))
    inject(
        model,
        "transcriber",
        _stem_aware_transcriber(guitar, bass, plan, frame_count(SR, HOP)),
    )
    with torch.inference_mode():
        out = model(guitar)
    md = out["midi_metadata"]
    assert (
        md["guitar"]["midi_events"] and md["bass"]["midi_events"]
    ), "a stem produced no MIDI at all"
    assert md["guitar"] is not md["bass"], "both stems alias the same metadata dict"
    assert [e["note"] for e in md["bass"]["midi_events"]] == plan["bass"][
        1
    ], "the bass events are not the bass plan - the model re-used the guitar transcription"
    g = build_midi_from_events(md["guitar"], "guitar")
    b = build_midi_from_events(md["bass"], "bass")
    gn = [n.note for n in parse_midi(g)["notes"]]
    bn = [n.note for n in parse_midi(b)["notes"]]
    assert gn != bn or gn == [], f"guitar.mid and bass.mid carry identical notes {gn}"


def test_api_zip_uses_per_stem_midi(monkeypatch):
    """Regression guard (was API-1, fixed by H-8): the ZIP picks each stem out of `midi`.

    api.create_response_zip reads outputs["midi"] and hands it to _stem_midi_data(pkg,
    stem), which selects pkg["guitar"] / pkg["bass"]. The legacy flat payload is still
    accepted on purpose - and that fallback is the one place the two files can still
    alias, so it is pinned here as documented behaviour, not as a bug.
    """
    require("fastapi", "api import")
    import api

    guitar_events = [{"note": 64, "onset_frame": 10, "velocity": 90, "confidence": 0.9}]
    bass_events = [{"note": 40, "onset_frame": 12, "velocity": 90, "confidence": 0.9}]
    outputs = {
        "stems": {
            "guitar": np.zeros(1024, dtype="float32"),
            "bass": np.zeros(1024, dtype="float32"),
        },
        "midi": {
            "guitar": {"midi_events": guitar_events, "summary": {}},
            "bass": {"midi_events": bass_events, "summary": {}},
            "summary": {"avg_confidence": 0.9, "low_confidence_count": 0, "alignment_warnings": 0},
        },
        "report": StubReport(),
        "routing": {"action": "editor_launch"},
        "sample_rate": SR,
    }
    monkeypatch.setattr(api, "model_config", {"audio": {"sample_rate": SR}}, raising=False)
    buf = api.create_response_zip(outputs)
    import zipfile

    zf = zipfile.ZipFile(io.BytesIO(buf.read()))
    g = [n.note for n in parse_midi(io.BytesIO(zf.read("guitar.mid")))["notes"]]
    b = [n.note for n in parse_midi(io.BytesIO(zf.read("bass.mid")))["notes"]]
    assert b == [40], f"bass.mid should hold the bass render (40), got {b}"
    assert g == [64], f"guitar.mid should hold the guitar render (64), got {g}"

    # And the documented legacy fallback: a flat payload must not crash the endpoint;
    # it just means the caller had no per-stem data, and both files alias by design.
    # Pinning it here means the day someone removes the fallback, this assertion is
    # the thing that says "a legacy caller now 500s".
    import zipfile

    flat = dict(outputs, midi={"midi_events": guitar_events, "summary": {}})
    zf2 = zipfile.ZipFile(io.BytesIO(api.create_response_zip(flat).read()))
    legacy = [n.note for n in parse_midi(io.BytesIO(zf2.read("bass.mid")))["notes"]]
    assert legacy == [64], f"legacy flat payload should alias to the same render, got {legacy}"


# ---------------------------------------------------------------------------
# 2. confidence / energy / counts plumbing (units, not model quality)
# ---------------------------------------------------------------------------


def test_model_confidence_reaches_poly_pressure_unchanged(pipeline_config):
    """Per-frame confidence must reach Polyphonic Key Pressure without re-scaling.

    Pass-today regression lock: it follows a confidence value from transcriber
    output through the MIDI metadata and into the SMF. It fails if anyone swaps
    the mapping (``round`` vs ``int``), clamps it, or lets alignment leak into
    the note-specific pressure value.
    """
    confs = [0.25, 0.5, 0.75, 1.0]
    frames = [4, 20, 40, 60]
    from main import StemMidiModel

    model = StemMidiModel(pipeline_config)
    model.eval()
    guitar = torch.full((1, 1, SR), 0.2)  # 1 s -> 87 frames: holds onsets up to frame 60
    n = frame_count(guitar.shape[-1], HOP)
    onset = torch.full((1, n, 1), -10.0)
    pitch = torch.zeros((1, n, 128))
    vel = torch.full((1, n, 1), 0.7)
    expr = torch.zeros((1, n, 3))
    conf = torch.zeros(1, n)
    for f, c in zip(frames, confs, strict=False):
        onset[0, f, 0] = 10.0
        pitch[0, f, 60] = 10.0
        conf[0, f] = c
    inject(model, "separator", StubSeparator(guitar, guitar))
    inject(model, "transcriber", lambda stem: (onset, pitch, vel, expr, conf))
    with torch.inference_mode():
        out = model(guitar)
    events = out["midi_metadata"]["guitar"]["midi_events"]
    assert [e["onset_frame"] for e in events] == frames
    got_cc = [e["cc_127_value"] for e in events]
    assert got_cc == [
        int(c * 127) for c in confs
    ], f"CC#127 {got_cc} != the transcriber confidence mapping {confs}"
    midi = build_midi_from_events(out["midi_metadata"]["guitar"], "guitar")
    parsed = parse_midi(midi)
    assert parsed["cc_values"] == got_cc, (
        f"the confidence CC changed on the way into the SMF (CC#16): events say {got_cc}, "
        f"the file carries {parsed['cc_values']}"
    )
    expected_note_confidence = sum(event["confidence"] for event in events) / len(events)
    assert (
        abs(out["processing_report"].avg_confidence - expected_note_confidence) < 1e-6
    ), "report.avg_confidence must average emitted-note confidence, not a constant"


@known_bug(
    "ALIGN-1",
    "main._forward_impl computes stem_energy = mean(guitar_stem**2, dim=1) — a PER-SAMPLE "
    "squared value (the dim it averages is the channel dim) — and ConfidenceInjector then "
    "indexes it with onset_frame. So 'cross-modal alignment' reads waveform sample #40 for "
    "frame 40, i.e. a single sample ~11 ms into the file, and alignment_score/needs_review "
    "are meaningless for any file longer than a few frames. Compute frame-mean energy "
    "(unfold/pool over hop) and index it.",
)
def test_alignment_energy_is_measured_over_the_notes_frame(pipeline_config):
    """A loud note must see the energy of ITS frame, not of sample index onset_frame."""
    T = SR  # 1 s
    guitar = torch.zeros(1, 1, T)
    guitar[0, 0, 2 * HOP : 3 * HOP] = 1.0  # one loud frame at frame index 2
    bass = torch.zeros_like(guitar)
    model = _model_with_stems(
        pipeline_config, guitar, bass, onsets=[2], pitches=[60], confidence=0.95
    )
    with torch.inference_mode():
        out = model(guitar)
    (ev,) = out["midi_metadata"]["midi_events"]
    assert ev["stem_energy"] == pytest.approx(1.0, abs=1e-3), (
        f"frame 2 is full-scale but stem_energy={ev['stem_energy']:.6f} — the injector is "
        f"indexing a per-sample array with a frame index"
    )
    assert ev["needs_review"] is False, (
        f"a confident, on-energy note was flagged needs_review "
        f"(alignment_score={ev['alignment_score']:.4f})"
    )


@known_bug(
    "GATE-2",
    "ConfidenceInjector.summary['low_confidence_count'] counts FRAMES of the confidence "
    "tensor below min_confidence ((confidence < thr).sum()), and main feeds it into "
    "ProcessingReport.low_confidence_notes / the API's processing_report.json. One second "
    "of audio is 86 frames, so the number reported to the user as 'low confidence notes' is "
    "a frame count and grows with file length. Count notes.",
)
def test_low_confidence_notes_counts_notes_not_frames(pipeline_config):
    """The field the user sees as 'low confidence notes' must be a note count."""
    n = 200
    conf = torch.full((1, n), 0.3)  # every frame low confidence
    onset = torch.full((1, n, 1), -10.0)
    for f in (5, 9, 44):  # ... but only three notes
        onset[0, f, 0] = 10.0
    pitch = torch.zeros(1, n, 128)
    for f in (5, 9, 44):
        pitch[0, f, 60] = 10.0
    from main import StemMidiModel

    model = StemMidiModel(pipeline_config)
    model.eval()
    guitar = torch.full((1, 1, n * HOP), 0.2)
    inject(model, "separator", StubSeparator(guitar, torch.zeros_like(guitar)))
    inject(
        model,
        "transcriber",
        lambda stem: (onset, pitch, torch.full((1, n, 1), 0.7), torch.zeros(1, n, 3), conf),
    )
    with torch.inference_mode():
        out = model(guitar)
    n_notes = len(out["midi_metadata"]["midi_events"])
    assert n_notes == 3, f"fixture broke: {n_notes} notes"
    assert out["processing_report"].low_confidence_notes == n_notes, (
        f"report says {out['processing_report'].low_confidence_notes} low-confidence notes "
        f"for a render with {n_notes} notes (it is counting frames)"
    )


def test_summary_and_alignment_counts_agree(pipeline_config):
    """Pass-today lock: summary.alignment_warnings == number of needs_review events."""
    guitar = torch.full((1, 1, SR), 0.05)
    model = _model_with_stems(
        pipeline_config, guitar, torch.zeros_like(guitar), onsets=[3, 9, 17], pitches=[50, 52, 55]
    )
    with torch.inference_mode():
        out = model(guitar)
    md = out["midi_metadata"]
    per_stem = sum(
        1 for stem in ("guitar", "bass") for e in md[stem]["midi_events"] if e["needs_review"]
    )
    assert md["summary"]["alignment_warnings"] == per_stem, (
        f"summary reports {md['summary']['alignment_warnings']} alignment warnings but the "
        f"per-stem events carry {per_stem}"
    )
    assert json.dumps({"n": md["summary"]})  # summary must be plain JSON types
    events = md["guitar"]["midi_events"]
    assert all(0.0 <= e["confidence"] <= 1.0 for e in events)


# ---------------------------------------------------------------------------
# 3. report honesty: batch and proxy semantics
# ---------------------------------------------------------------------------


@known_bug(
    "REPORT-1",
    "main reads centroid_separation_db=sep_metrics[0, 0].item() and "
    "phase_coherence=sep_metrics[0, 1] with "
    "the comment 'Average over batch', so a batched call reports item 0's quality for the "
    "whole batch. A 2-item batch where the second render is garbage is stamped studio. "
    "Average (or return per-item reports).",
)
def test_batched_report_aggregates_over_the_batch(pipeline_config):
    guitar = torch.full((2, 1, SR), 0.1)

    class TwoItemSeparator:
        def __call__(self, audio):
            metrics = torch.tensor([[25.0, 0.95], [1.0, 0.05]])
            zeros = torch.zeros_like(audio)
            return guitar, zeros, zeros, None, metrics

    from main import StemMidiModel

    model = StemMidiModel(pipeline_config)
    model.eval()
    inject(model, "separator", TwoItemSeparator())
    n = frame_count(SR, HOP)
    conf = torch.full((2, n), 0.9)
    model.transcriber = lambda stem: (
        torch.full((2, n, 1), -10.0),
        torch.zeros(2, n, 128),
        torch.full((2, n, 1), 0.7),
        torch.zeros(2, n, 3),
        conf,
    )
    with torch.inference_mode():
        out = model(guitar)
    report = out["processing_report"]
    assert report.centroid_separation_db == pytest.approx(13.0, abs=1e-3), (
        f"report says {report.centroid_separation_db} dB for a batch whose items measured "
        "25.0 and 1.0 dB"
    )


@known_bug(
    "BATCH-1",
    "_logits_to_midi flattens every batch item's events into one list with raw per-item "
    "frame indices (its docstring admits it), so a B=2 call returns one merged MIDI where "
    "two songs' notes are interleaved and unattributable. Either reject B>1 for inference "
    "or key events per batch item.",
)
def test_batched_inference_does_not_merge_songs_into_one_midi(pipeline_config):
    guitar = torch.full((2, 1, SR), 0.1)
    n = frame_count(SR, HOP)
    onset = torch.full((2, n, 1), -10.0)
    onset[0, 5, 0] = 10.0
    onset[1, 5, 0] = 10.0
    pitch = torch.zeros(2, n, 128)
    pitch[0, 5, 64] = 10.0
    pitch[1, 5, 40] = 10.0
    from main import StemMidiModel

    model = StemMidiModel(pipeline_config)
    model.eval()
    inject(model, "separator", StubSeparator(guitar, torch.zeros_like(guitar)))
    inject(
        model,
        "transcriber",
        lambda stem: (
            onset,
            pitch,
            torch.full((2, n, 1), 0.7),
            torch.zeros(2, n, 3),
            torch.full((2, n), 0.9),
        ),
    )
    with pytest.raises(ValueError, match="batch"), torch.inference_mode():
        model(guitar)


# ---------------------------------------------------------------------------
# 4. _logits_to_midi: the extraction contract (nothing covered it before)
# ---------------------------------------------------------------------------


def test_logits_to_midi_threshold_pitch_and_velocity():
    """The logit→event mapping is the single gate on note count; pin it down."""
    from main import StemMidiModel

    n = 20
    onset = torch.full((1, n, 1), -10.0)
    onset[0, 3, 0] = 0.0  # sigmoid = 0.5 -> not > threshold(0.5)
    onset[0, 8, 0] = 0.01  # sigmoid ~ 0.502 -> fires
    onset[0, 16, 0] = 10.0  # 7 below-threshold frames later: outside onset_refractory_frames
    pitch = torch.zeros(1, n, 128)
    for f, p in ((3, 60), (8, 61), (16, 62)):
        pitch[0, f, p] = 10.0
    vel = torch.full((1, n, 1), 100 / 127.0)
    conf = torch.full((1, n), 0.8)
    model = StemMidiModel.__new__(StemMidiModel)  # no heavy init needed for a pure fn
    model.cfg = {"transcriber": {"onset_threshold": 0.5}}
    events = model._logits_to_midi(onset, pitch, vel, conf)
    assert [e["onset_frame"] for e in events] == [8, 16], (
        "onsets fired at the wrong frames: with the H-6 peak-picking rule two candidates "
        "must be >= onset_refractory_frames+1 apart, and 8/16 are"
    )
    assert [e["note"] for e in events] == [61, 62]
    assert all(e["velocity"] == 100 for e in events)
    assert all(abs(e["confidence"] - 0.8) < 1e-6 for e in events)


def test_onset_refractory_caps_note_density(pipeline_config):
    """Regression guard for the density half of TRANS-1 (fixed by PR #14 / H-6).

    A degenerate onset field where EVERY frame is above threshold used to become one
    note per frame (measured on pre-#14 main: 345 notes for 1 s). Rising-edge
    peak-picking collapses that to a single event, which is why this now passes —
    but note *why*: the cap is the refractory window, not any judgement about content.
    """
    from main import StemMidiModel

    n = frame_count(SR, HOP)  # 86 frames in 1 s at hop 512
    onset = torch.full((1, n, 1), 0.5)
    pitch = torch.zeros(1, n, 128)
    pitch[0, :, 60] = 1.0
    pitch += torch.randn(1, n, 128) * 0.01
    model = StemMidiModel.__new__(StemMidiModel)

    def density(refractory):
        cfg = {"transcriber": {"onset_threshold": 0.0}}
        if refractory is not None:
            cfg["transcriber"]["onset_refractory_frames"] = refractory
        model.cfg = cfg
        events = model._logits_to_midi(
            onset, pitch, torch.full((1, n, 1), 0.7), torch.full((1, n), 0.9)
        )
        return len(events) / (n * HOP / SR), events

    per_second, events = density(None)
    assert per_second <= 20.0, f"{per_second:.0f} notes/second produced from {n} frames"
    assert len(events) == 1, (
        f"a plateau of {n} above-threshold frames must be ONE rising edge, got {len(events)} "
        f"events at frames {[e['onset_frame'] for e in events][:6]}"
    )
    unbounded, _ = density(0)
    assert unbounded > 20.0, (
        f"onset_refractory_frames=0 should restore the pre-H-6 every-frame behaviour, but it "
        f"gave {unbounded:.0f}/s — if the plateau rule moved elsewhere, this test is stale"
    )


@known_bug(
    "TRANS-1",
    "residual after PR #14 (H-6 added rising-edge peak-picking + onset_refractory_frames, which "
    "fixed the 345-notes-per-second flood): there is still NO content test in _logits_to_midi — "
    "no stem-energy gate and no 'no-note' class in the pitch head — so an above-threshold onset "
    "field is turned into notes even when the audio is digital silence, and the only thing "
    "bounding density is the refractory window (measured: the saturated field still yields "
    "1 event per refractory+1 frames, and setting onset_refractory_frames: 0 in the config "
    "restores 86 notes/s with no error or warning). Silence currently renders 0 notes only "
    "because the untrained head happens to stay below 0.5 (measured avg_confidence 0.650 with "
    "0 notes), which is luck, not a guarantee. Gate on frame energy vs the file's max (e.g. "
    "require frame RMS > 1e-3 x max) and add a no-note logit.",
)
def test_a_saturated_onset_field_on_silence_is_rejected_for_lack_of_content(pipeline_config):
    """Content, not spacing, must decide whether there is a note to emit."""
    from main import StemMidiModel

    n = frame_count(SR, HOP)
    model = StemMidiModel.__new__(StemMidiModel)
    model.cfg = {"transcriber": {"onset_threshold": 0.0}}
    onset = torch.full((1, n, 1), 0.5)
    pitch = torch.zeros(1, n, 128)
    pitch[0, :, 60] = 1.0
    events = model._logits_to_midi(
        onset, pitch, torch.full((1, n, 1), 0.7), torch.full((1, n), 0.9)
    )
    # `silence` here is the caller's contract: the separator handed this stem a
    # waveform with no energy, so there is nothing to transcribe. The head layout
    # gives no way to say that, hence this fails today.
    assert events == [], (
        f"{len(events)} notes emitted from an onset field the pipeline cannot have "
        "measured anything about; expected a stem-energy gate / no-note class to suppress them"
    )


def test_zero_onsets_never_produce_notes(pipeline_config):
    from main import StemMidiModel

    n = 30
    model = StemMidiModel.__new__(StemMidiModel)
    model.cfg = {"transcriber": {"onset_threshold": 0.5}}
    events = model._logits_to_midi(
        torch.full((1, n, 1), -10.0),
        torch.zeros(1, n, 128),
        torch.zeros(1, n, 1),
        torch.zeros(1, n),
    )
    assert events == []


# ---------------------------------------------------------------------------
# 5. the gate must agree with what it ships (real model path)
# ---------------------------------------------------------------------------


@known_bug(
    "GATE-3",
    "route_by_quality never looks at what the render actually contains: with non-silent "
    "stems, confidence 0.95 and a 25 dB proxy it stamps STUDIO on a render with ZERO "
    "notes, and the user gets direct_download on an empty MIDI. ProcessingReport must "
    "carry note_count (and the input RMS) and a studio/draft tier must require a sane, "
    "non-empty note stream — 'draft' is a promise that there is something to edit.",
)
def test_empty_render_is_never_offered_as_studio_or_draft(pipeline_config):
    """0 notes with happy metrics must route to fallback_options, not direct_download."""
    # A 220 Hz TONE, not torch.full(...): a constant is pure DC, and once
    # _detect_artifacts flags input DC offset this fixture routes to COMPLEX for
    # that reason instead of exercising the empty-render path this test pins.
    guitar = 0.2 * torch.sin(2 * math.pi * 220 * torch.arange(SR).float() / SR)[None, None]
    model = _model_with_stems(pipeline_config, guitar, guitar * 0.5, onsets=[], pitches=[])
    with torch.inference_mode():
        out = model(guitar)
    events = out["midi_metadata"]["guitar"]["midi_events"]
    report = out["processing_report"]
    assert len(events) == 0, "fixture broke: expected no notes"
    assert report.quality_tier.value == "complex", (
        f"an empty render (0 notes) was graded {report.quality_tier.value} with "
        f"avg_confidence={report.avg_confidence:.2f} / "
        f"centroid_separation_db={report.centroid_separation_db:.1f}"
    )
    assert out["routing_decision"]["action"] == "fallback_options"


@pytest.mark.real_mamba
def test_stems_sum_back_to_the_mix(pipeline_config, synth):
    """Regression guard for SEP-1, fixed by PR #13 (was a strict-xfail lock).

    The separator used to learn three INDEPENDENT sigmoid masks, so guitar + bass +
    residual accounted for only about half the mixture's energy (measured relative
    error 0.46-0.51 across seeds 0-3): users heard holes in the reassembled mix and a
    "sum to mix" check in a DAW failed. `separator.mask_normalization: softmax` makes
    the masks complementary, which is what this asserts - 1.15e-07 measured on
    `scripts/verify_numerics.py`. `"independent"` still restores the old behaviour, so
    the key is a knob, not a hard-coded assumption.
    """
    from main import StemMidiModel

    sig = synth.note_sequence(total_seconds=1.0, sr=SR)
    mix = torch.from_numpy(sig)[None, None].float()
    torch.manual_seed(0)
    model = StemMidiModel(pipeline_config)
    model.eval()
    with torch.inference_mode():
        g, b, r, _cache, _m = model.separator(mix)
    err = float((g + b + r - mix).norm() / mix.norm())
    assert err < 0.05, f"||guitar+bass+residual - mix|| / ||mix|| = {err:.3f}"


@pytest.mark.real_mamba
def test_stems_never_amplify_the_mix_and_stay_finite(pipeline_config, synth):
    """A sigmoid mask cannot add energy — pass-today safety lock for bad-stem regressions."""
    from main import StemMidiModel

    sig = synth.note_sequence(total_seconds=1.0, sr=SR)
    mix = torch.from_numpy(sig)[None, None].float()
    torch.manual_seed(0)
    model = StemMidiModel(pipeline_config)
    model.eval()
    with torch.inference_mode():
        g, b, r, _c, metrics = model.separator(mix)
    for name, stem in (("guitar", g), ("bass", b), ("residual", r)):
        assert torch.isfinite(stem).all(), f"{name} has NaN/Inf"
        assert float(stem.abs().max()) <= float(mix.abs().max()) * 1.02, (
            f"{name} peak {float(stem.abs().max()):.4f} exceeds "
            f"mix peak {float(mix.abs().max()):.4f}"
        )
        assert tuple(stem.shape) == tuple(mix.shape)
    assert torch.isfinite(metrics).all()
    assert float(metrics.min()) >= 0.0 and float(metrics.max()) <= 30.0


# ---------------------------------------------------------------------------
# 6. process_audio_file: file-level contract
# ---------------------------------------------------------------------------


def test_process_audio_file_returns_full_length_float32_stems(pipeline_config, make_wav, synth):
    from main import StemMidiModel

    sig = synth.note_sequence(total_seconds=1.5, sr=SR)
    path = make_wav(sig, name="one_and_a_half.wav")
    torch.manual_seed(0)
    model = StemMidiModel(pipeline_config)
    model.eval()
    out = model.process_audio_file(path)
    expected = len(sig)
    for stem_name, arr in out["stems"].items():
        assert arr.ndim == 1, f"{stem_name} stem should be 1-D mono, got {arr.shape}"
        assert arr.dtype.kind == "f"
        assert len(arr) == expected, f"{stem_name}: {len(arr)} samples for a {expected}-sample file"
        assert not np.isnan(arr).any(), f"{stem_name} stem has NaN"
    assert out["sample_rate"] == pipeline_config["audio"]["sample_rate"]


def test_process_audio_file_resamples_foreign_rate(pipeline_config, make_wav, synth):
    """A 22.05 kHz file must be resampled to the config rate (length x2, same seconds)."""
    from main import StemMidiModel

    mono = synth.tone(220.0, seconds=1.0, sr=22050)
    path = make_wav(mono, name="at_22k.wav", sr=22050)
    torch.manual_seed(0)
    model = StemMidiModel(pipeline_config)
    model.eval()
    out = model.process_audio_file(path)
    n = len(out["stems"]["guitar"])
    assert abs(n - 44100) <= 512, f"expected ~44100 samples after resample, got {n}"
    assert float(np.max(np.abs(out["stems"]["guitar"]))) > 0.0, "resampled stem is silent"


@known_bug(
    "EXPR-1",
    "README/API_DOCUMENTATION advertise MIDI expression (bend/vibrato/slide per note), but "
    "_forward_impl discards the expression head (`_expression`) and build_midi_from_events "
    "writes no pitch_bend and no CC1/CC2 — verified: the only non-note messages in a rendered "
    "file are CC#127. Either carry expression into the SMF (pitch_bend for bend, CC1 for "
    "vibrato depth) or remove the claim from the docs and the transcriber's head.",
)
def test_expression_head_output_reaches_the_midi(pipeline_config):
    guitar = torch.full((1, 1, SR // 4), 0.2)
    n = frame_count(guitar.shape[-1], HOP)
    expr = torch.zeros(1, n, 3)
    expr[0, :, 0] = 0.9  # max bend on every frame
    model = _model_with_stems(
        pipeline_config,
        guitar,
        torch.zeros_like(guitar),
        onsets=[2, 20],
        pitches=[60, 67],
        expression=expr,
    )
    with torch.inference_mode():
        out = model(guitar)
    events = out["midi_metadata"]["guitar"]["midi_events"]
    assert any(
        "expression" in e or "bend" in e for e in events
    ), "the transcriber's expression output never reaches the MIDI events at all"
    midi = build_midi_from_events(out["midi_metadata"], "guitar")
    types = {m.type for m in parse_midi(midi)["all_messages"]}
    assert (
        "pitch_bend" in types or "aftertouch" in types
    ), f"no expression messages in the SMF: {sorted(types)}"
