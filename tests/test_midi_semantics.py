# SPDX-License-Identifier: Apache-2.0
"""MIDI *semantics*: does the SMF encode the music we were given?

Why this file exists (audit finding, task 2): the two pre-existing MIDI files
(``test_midi_generation.py``, ``test_midi_roundtrip.py``) assert **message
counts** and **relative ordering** — ``len(note_ons) == n``, ``times ==
sorted(times)``, ``note_offs[1] == 0 or note_offs[1] > note_offs[0]``. Those
predicates are satisfied by a file whose notes are all in the wrong place, and
one of them literally accepts a zero-length note. On this repository they hide
three real defects, each verified by running the code:

* ``utils/midi.py`` computes ``ticks_per_second = ticks_per_beat * (60 /
  (tempo_us/1e6))`` = 57_600 ticks/s for 120 BPM, but 480 ticks per 0.5 s beat
  is **960** ticks/s. Every note (onset *and* duration) is written 60x too far
  apart: 1 s of audio becomes a 60 s MIDI timeline. Verified: a 1.00 s render
  with 345 notes produces ``MidiFile.length == 239.8 s``.
* Events are written as ``note_on, note_off, cc`` sequences against one running
  cursor, and negative deltas are clamped with ``max(0, delta)``. A short note
  that starts *during* a longer previous note therefore gets pushed to the end
  of that previous note instead of its real onset — silently, in every render,
  because ``_logits_to_midi`` emits one note per frame with the default 0.25-beat
  duration (overlapping by construction).
* The confidence CC#127 for note *i* is written after note *i*'s note_off, i.e.
  it lands in the DAW as the value for note *i+1* and leaves note 0 unlabelled.

Everything here is unit-level (no torch, no model): the writer/reader contract is
where MIDI usability is won or lost, and it is cheap to pin down.
"""

from __future__ import annotations

import io
import math

import pytest

from tests.honesty import known_bug, require

mido = require("mido", "MIDI semantics require mido", module_level=True)

from tests.synth import parse_midi  # noqa: E402
from utils.midi import (  # noqa: E402
    DEFAULT_TEMPO_US,
    TICKS_PER_BEAT,
    build_midi_from_events,
)

SR = 44100
HOP = 512  # utils.midi.HOP_LENGTH; the grid the production configs use


def _ev(note, onset_frame, velocity=90, confidence=0.9, duration_frames=None):
    e = {"note": note, "onset_frame": onset_frame, "velocity": velocity, "confidence": confidence}
    if duration_frames is not None:
        e["duration_frames"] = duration_frames
    return e


def _events(specs):
    return [_ev(*s) for s in specs]


def _tick(
    frame: int, tpb: int = 480, hop: int = 512, sr: int = 44100, tempo_us: int = 500_000
) -> int:
    """The writer's own frame->tick law, for the property test's multiset check."""
    return round(frame * hop / sr * (tpb * 1_000_000.0 / tempo_us))


def _build(events, **kw):
    return build_midi_from_events({"midi_events": events}, "guitar", **kw)


def _sec_per_frame():
    return HOP / SR


# ---------------------------------------------------------------------------
# 1. wall-clock correctness (the 60x tempo scaling)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frames", [[0, 10, 20], [0, 431, 862, 1724], [5, 6, 7, 8]])
def test_onset_wall_clock_matches_the_audio_timeline(frames):
    """note_on seconds must equal onset_frame * hop / sample_rate (±1 frame)."""
    midi = _build([_ev(60, f) for f in frames])
    parsed = parse_midi(midi)
    sec_per_frame = _sec_per_frame()
    assert len(parsed["notes"]) == len(frames)
    for note, frame in zip(parsed["notes"], frames, strict=False):
        expected = frame * sec_per_frame
        assert abs(note.on_sec - expected) <= sec_per_frame, (
            f"onset frame {frame} should be {expected:.4f} s, MIDI says {note.on_sec:.4f} s "
            f"(ratio {note.on_sec / max(expected, 1e-9):.1f}x)"
        )


def test_note_duration_equals_duration_frames_in_seconds():
    """duration_frames=32 (~0.37 s at hop 512) must survive as ~0.37 s of sustain."""
    dur = 32
    midi = _build([_ev(60, 0, duration_frames=dur)])
    (note,) = parse_midi(midi)["notes"]
    expected = dur * _sec_per_frame()
    assert (
        abs(note.duration_sec - expected) <= 2 * _sec_per_frame()
    ), f"{dur} frames = {expected:.4f} s, MIDI sustain = {note.duration_sec:.4f} s"


def test_midi_length_tracks_render_length():
    """30 s of audio at hop 512 -> 2580 frames; the SMF must end near 30 s, not 30 minutes."""
    seconds = 30.0
    n_frames = int(seconds * SR / HOP)
    events = [_ev(40 + (i % 24), i * 10) for i in range(n_frames // 10)]
    midi = _build(events)
    parsed = parse_midi(midi)
    assert parsed["length_sec"] <= seconds + 1.0, (
        f"{seconds:.0f} s of audio produced a {parsed['length_sec']:.0f} s MIDI "
        f"({parsed['length_sec'] / seconds:.0f}x) — a DAW importing this lands every "
        f"note in the next measure-bar epoch"
    )


# ---------------------------------------------------------------------------
# 2. overlapping events / delta clamping
# ---------------------------------------------------------------------------

_LONG = 400  # frames
_SHORT_AT = 10  # second note starts inside the first one's sustain


def test_overlapping_notes_keep_their_own_onset_time():
    """A note starting during another note's sustain must still start on time."""
    events = [
        _ev(60, 0, duration_frames=_LONG),
        _ev(61, _SHORT_AT, duration_frames=5),
    ]
    parsed = parse_midi(_build(events))
    # select by pitch: the writer resolves re-articulation as release-then-strike,
    # so parsed notes come back in closing order, not striking order
    second = next(n for n in parsed["notes"] if n.note == 61)
    expected = _SHORT_AT * _sec_per_frame()
    assert abs(second.on_sec - expected) <= 2 * _sec_per_frame(), (
        f"note 61 starts at {second.on_sec:.3f} s but its onset frame says {expected:.3f} s "
        f"(drifted by {second.on_sec - expected:+.3f} s — the previous note's tail)"
    )


def test_unsorted_events_are_reordered_or_rejected_not_clamped():
    """Out-of-order events must either be sorted or raise — never collapse onto one tick."""
    frames = [90, 0, 45, 20]
    events = [_ev(60 + i, f) for i, f in enumerate(frames)]
    try:
        parsed = parse_midi(_build(events))
    except ValueError as exc:  # an explicit rejection is acceptable
        pytest.fail(f"unexpected ValueError: {exc}")
    ons = [n.on_tick for n in parsed["notes"]]
    assert len(set(ons)) == len(
        ons
    ), f"notes collapsed onto duplicate ticks: {ons} (input frames {frames})"
    assert sorted(ons) == ons, "writer must emit a tick-sorted track"
    # and every note must land within one frame of its own onset, in sorted order
    for note, frame in zip(
        sorted(parsed["notes"], key=lambda n: n.on_tick), sorted(frames), strict=False
    ):
        assert abs(note.on_sec - frame * _sec_per_frame()) <= _sec_per_frame()


def test_no_stuck_notes_at_end_of_track():
    """Every note_on needs a note_off: a dangling note hangs a synth in the DAW."""
    events = [_ev(50 + i, i * 30, duration_frames=17) for i in range(12)]
    parsed = parse_midi(_build(events))
    assert (
        parsed["dangling_notes"] == {}
    ), f"notes still sounding at end_of_track: {parsed['dangling_notes']}"


def test_repeated_pitch_emits_balanced_voices():
    """Same pitch twice = two voices, each with its own on/off pair (no voice stealing)."""
    events = [_ev(60, 0, duration_frames=20), _ev(60, 25, duration_frames=20)]
    parsed = parse_midi(_build(events))
    ons = [n for n in parsed["notes"] if n.note == 60]
    assert len(ons) == 2
    for n in ons:
        assert n.off_tick > n.on_tick, "zero/late note_off kills the second voice"
    offs = [m for m in parsed["all_messages"] if m.type == "note_off"]
    assert len(offs) == 2 and all(m.velocity == 0 for m in offs)


# ---------------------------------------------------------------------------
# 3. confidence CC placement and range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("confidences", [[0.95, 0.20, 0.70], [0.10, 0.99]])
def test_confidence_cc_describes_the_note_it_sits_on(confidences):  # regression guard (was MIDI-3)
    """The confidence CC in force at note i's onset must be note i's own confidence."""
    events = [_ev(60 + i, i * 20, confidence=c) for i, c in enumerate(confidences)]
    parsed = parse_midi(_build(events))
    got = [n.cc_at_onset for n in parsed["notes"]]
    want = [int(c * 127) for c in confidences]
    assert got == want, f"notes carry CC {got}, expected {want} (off-by-one placement)"


@known_bug(
    "MIDI-4",
    "confidence is truncated twice: models/confidence_injector.py:72 and utils/midi.py:145 "
    "both use int(conf*127), so conf=0.999 -> 126 and 0.5 -> 63. A DAW that treats the top of "
    "the CC range as 'fully confident' never sees it. Fix BOTH call sites — rounding only the "
    "writer changes nothing for a model-produced event, which is the trap here.",
)
def test_top_end_of_confidence_range_reaches_cc_127():
    """Confidence ~1.0 must map to the top of the CC range, not one below it."""
    midi = _build([_ev(60, 0, confidence=0.999)])
    (value,) = parse_midi(midi)["cc_values"]
    assert value == round(0.999 * 127) == 127, f"got {value}"


def test_confidence_controller_is_gp1_not_a_channel_mode_message():
    """Regression guard for main's fix: CC#127 is 'All Sound Off', not a data lane.

    Confidence rides CC#16 (General Purpose 1) by default; #127 only appears if an
    operator opts back in through MIDI_CONFIDENCE_CC, because spec-compliant gear
    treats 120-127 as Channel Mode messages (a 0.9-confidence render used to send
    All Sound Off / Panic to every device on the channel).
    """
    from utils.midi import DEFAULT_CONFIDENCE_CC, build_midi_from_events, resolve_confidence_cc

    assert DEFAULT_CONFIDENCE_CC == 16
    assert resolve_confidence_cc() == 16
    events = [_ev(60, 0, confidence=0.9), _ev(61, 20, confidence=0.4)]
    parsed = parse_midi(build_midi_from_events({"midi_events": events}, "guitar"), cc=16)
    assert parsed["cc_values"] == [int(0.9 * 127), int(0.4 * 127)]
    legacy = parse_midi(build_midi_from_events({"midi_events": events}, "guitar"), cc=127)
    assert legacy["cc_values"] == [], "CC#127 must not be emitted unless explicitly requested"

    # the env escape hatch the docs promise
    import os

    os.environ["MIDI_CONFIDENCE_CC"] = "127"
    try:
        assert resolve_confidence_cc() == 127
        with_mode = parse_midi(build_midi_from_events({"midi_events": events}, "guitar"), cc=127)
        assert (
            len(with_mode["cc_values"]) == 2
        ), "MIDI_CONFIDENCE_CC=127 must restore the legacy mapping"
    finally:
        del os.environ["MIDI_CONFIDENCE_CC"]
    assert resolve_confidence_cc(127) == 127  # explicit value warns but is allowed

    with pytest.raises(ValueError, match="0-127"):
        resolve_confidence_cc(999)


def test_confidence_ordering_survives_into_cc_values():
    """Monotone confidence in -> monotone CC out (the DAW highlights by this)."""
    confidences = [i / 127.0 for i in range(0, 128, 7)]
    events = [_ev(60, i * 10, confidence=c) for i, c in enumerate(confidences)]
    vals = parse_midi(_build(events))["cc_values"]
    assert vals == sorted(vals), "CC ordering must be monotone in confidence"
    assert len(vals) == len(confidences)
    assert all(0 <= v <= 127 for v in vals)


# ---------------------------------------------------------------------------
# 4. velocity / pitch / header invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [(-5, 1), (0, 1), (127, 127), (999, 127)])
def test_velocity_never_zero(raw, expected):
    """velocity 0 on a note_on IS a note-off in MIDI: a silent, untriggerable note."""
    midi = _build([_ev(60, 0, velocity=raw)])
    parsed = parse_midi(midi)
    assert len(parsed["notes"]) == 1, "note vanished — a 0-velocity note_on is a note_off"
    assert parsed["notes"][0].velocity == expected


def test_out_of_range_pitch_is_rejected_with_a_domain_error():
    """A bad pitch must not become a corrupt file; a clear error is acceptable."""
    with pytest.raises(ValueError, match=r"range|127|pitch|note"):
        _build([_ev(200, 0)])


def test_header_is_daw_friendly():
    """Format 0/1, 480 ticks/beat, and an explicit tempo, or DAWs quantise wrongly."""
    parsed = parse_midi(_build([_ev(60, 0)]))
    mid = mido.MidiFile(file=io.BytesIO(_build([_ev(60, 0)])))
    assert mid.type in (0, 1)
    assert parsed["ticks_per_beat"] == TICKS_PER_BEAT == 480
    assert parsed["tempo_us"] == DEFAULT_TEMPO_US
    assert any(m.type == "set_tempo" for m in parsed["all_messages"])


def test_default_duration_beats_argument_is_honoured():
    """API-6.12: the fallback note length is configurable, and the ratio must show it."""
    ev = [_ev(60, 0)]  # no duration_frames -> fallback path
    short = parse_midi(_build(ev, default_duration_beats=0.25))["notes"][0]
    long_ = parse_midi(_build(ev, default_duration_beats=1.0))["notes"][0]
    assert math.isclose(long_.duration_sec / short.duration_sec, 4.0, rel_tol=0.02)
    assert short.duration_sec > 0.05, "fallback note length of 0 makes unusable MIDI"


def test_ticks_per_beat_argument_propagates():
    """A non-default PPQ must rescale consistently — same musical time for every PPQ.

    (Absolute timing is MIDI-1's job; this locks that the argument is actually
    threaded through onsets AND durations instead of being half-applied.)
    """
    events = [_ev(60, f * 4, duration_frames=20) for f in (0, 5, 9)]
    rows = []
    for tpb in (96, 192, 384, 480):
        parsed = parse_midi(_build(events, ticks_per_beat=tpb))
        assert parsed["ticks_per_beat"] == tpb
        notes = parsed["notes"]
        assert len(notes) == len(events)
        ons = [n.on_tick for n in notes]
        assert ons == sorted(ons), f"tpb={tpb}: ticks not monotone"
        rows.append((notes[1].on_sec - notes[0].on_sec, notes[0].duration_sec))
    onsets = [r[0] for r in rows]
    durations = [r[1] for r in rows]
    # int() tick truncation costs up to one tick of resolution (5 ms at 96 PPQ),
    # so compare relatively: a half-applied PPQ argument would differ by ~5x.
    # Rounding to an integer tick costs at most one tick at the coarsest PPQ
    # (96 PPQ @120 BPM = 5.2 ms), so compare with tick-aware slack, not 1e-3.
    tick_at_96 = 0.5 / 96
    assert (
        max(onsets) - min(onsets) <= 2 * tick_at_96
    ), f"PPQ changes onset seconds by more than 2 ticks at 96 PPQ: {onsets}"
    assert (
        max(durations) - min(durations) <= 2 * tick_at_96
    ), f"PPQ changes note length by more than 2 ticks at 96 PPQ: {durations}"


# ---------------------------------------------------------------------------
# 5. property-style: random streams must stay well-formed
# ---------------------------------------------------------------------------


@pytest.mark.property
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_property_random_event_streams_are_wellformed(seed):
    """For arbitrary event streams: parseable, balanced, in-range, no stuck notes.

    Deliberately does NOT assert absolute timing (that is MIDI-1/MIDI-2's job) — it
    locks the invariants that a corrupt writer breaks first: dropped notes, negative
    deltas, out-of-range bytes, dangling voices.
    """
    rng = __import__("numpy").random.default_rng(seed)
    n = int(rng.integers(1, 40))
    frames = sorted(int(x) for x in rng.integers(0, 20000, size=n))
    events = [
        {
            "note": int(rng.integers(28, 97)),
            "onset_frame": f,
            "velocity": int(rng.integers(1, 128)),
            "confidence": float(rng.random()),
            "duration_frames": int(rng.integers(1, 500)),
        }
        for f in frames
    ]
    midi = _build(events)
    assert midi[:4] == b"MThd"
    parsed = parse_midi(midi)
    assert len(parsed["notes"]) == n
    assert parsed["dangling_notes"] == {}
    # The writer resolves same-pitch re-articulation as release-then-strike at one
    # tick, so the *closing* order may differ from the strike order. Compare as a
    # multiset: every requested onset must appear exactly once.
    ons = sorted(x.on_tick for x in parsed["notes"])
    assert ons == sorted(e["onset_frame"] * 96 for e in events) or ons == sorted(
        _tick(e["onset_frame"]) for e in events
    ), "the rendered onset ticks do not match the requested frames"
    for x in parsed["notes"]:
        assert 0 <= x.note <= 127
        assert 1 <= x.velocity <= 127
        assert x.off_tick >= x.on_tick
        assert x.duration_sec > 0, "zero-length notes are invisible in a DAW"
    assert len(parsed["cc_values"]) == n, "one confidence CC per note, on the configured controller"
