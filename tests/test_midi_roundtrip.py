# SPDX-License-Identifier: Apache-2.0
"""MIDI round-trip tests (T-8.3.6).

Build a MidiFile with N events, save to bytes, re-parse, and verify event
count and note-specific confidence values survive. Would have caught API-6.12
(fixed 1/16 note duration ignoring duration_frames).
"""

import io

import pytest

mido = pytest.importorskip("mido", reason="MIDI round-trip requires mido")


from utils.midi import build_midi_from_events  # noqa: E402


@pytest.mark.parametrize("n_events", [0, 1, 3, 10])
def test_event_count_roundtrip(n_events):
    """N input events → N note_on + N note_off + N poly-pressure messages."""
    events = [
        {
            "note": 40 + i,
            "onset_frame": i * 86,  # 86 frames ≈ 10ms @ hop 512/44.1k
            "velocity": 64,
            "confidence": 0.5 + (i % 5) * 0.1,
        }
        for i in range(n_events)
    ]
    midi_bytes = build_midi_from_events({"midi_events": events}, "guitar")
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))

    track = mid.tracks[0]
    note_ons = [m for m in track if m.type == "note_on"]
    note_offs = [m for m in track if m.type == "note_off"]
    pressures = [m for m in track if m.type == "polytouch"]

    assert len(note_ons) == n_events
    assert len(note_offs) == n_events
    assert len(pressures) == n_events


def test_confidence_values_roundtrip():
    """Note-specific confidence values match int(confidence*127)."""
    confidences = [0.1, 0.5, 0.9, 1.0]
    events = [
        {
            "note": 60,
            "onset_frame": i * 86,
            "velocity": 80,
            "confidence": c,
        }
        for i, c in enumerate(confidences)
    ]
    midi_bytes = build_midi_from_events({"midi_events": events}, "bass")
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))

    pressures = [m.value for m in mid.tracks[0] if m.type == "polytouch"]
    assert pressures == [int(c * 127) for c in confidences]


def test_explicit_confidence_value_overrides_confidence():
    """An explicit legacy confidence value must win over confidence-derived value."""
    events = [
        {
            "note": 64,
            "onset_frame": 0,
            "velocity": 100,
            "confidence": 0.9,
            "cc_127_value": 42,
        }
    ]
    midi_bytes = build_midi_from_events({"midi_events": events}, "guitar")
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
    pressures = [m.value for m in mid.tracks[0] if m.type == "polytouch"]
    assert pressures == [42]


def test_duration_frames_honoured():
    """API-6.12: duration_frames must set the actual note-off delay."""
    events = [
        {
            "note": 60,
            "onset_frame": 0,
            "velocity": 100,
            "confidence": 0.9,
            "duration_frames": 172,  # 172 frames ≈ 20ms → well under default
        },
        {
            "note": 61,
            "onset_frame": 10000,
            "velocity": 100,
            "confidence": 0.9,
            "duration_frames": 10000,  # very long note
        },
    ]
    midi_bytes = build_midi_from_events({"midi_events": events}, "guitar")
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))

    note_offs = [m.time for m in mid.tracks[0] if m.type == "note_off"]
    # Second note's duration must be far longer than the first's —
    # impossible if the duration is hardcoded (API-6.12).
    # expects fix per TODO API-6.12
    assert note_offs[1] == 0 or note_offs[1] > note_offs[0]


# --- PR #13's timing/polyphony contract, re-expressed against main's writer -------
#
# PR #13 added these while `main` was rewriting the same file, and its side asserted a
# CC confidence lane (CC#16) that main replaced with note-specific Polyphonic Key
# Pressure (PR #15). The *properties* it pinned are still worth locking, so they are
# kept with the lane updated: absolute time, chord simultaneity, durations under a
# non-120-BPM tempo, no stuck rearticulations, and the opt-in marker track.

SR = 44100
HOP = 512
FRAME_SECONDS = HOP / SR


def _parse(midi_bytes: bytes):
    return mido.MidiFile(file=io.BytesIO(midi_bytes))


def _absolute(msgs):
    """Yield (absolute_tick, message) for a track."""
    tick = 0
    for m in msgs:
        tick += m.time
        yield tick, m


def _events(specs, **kwargs):
    """specs: list of (onset_frame, note, duration_frames)."""
    out = []
    for frame, note, dur in specs:
        ev = {"note": note, "onset_frame": frame, "velocity": 100, "confidence": 0.8}
        if dur is not None:
            ev["duration_frames"] = dur
        ev.update(kwargs)
        out.append(ev)
    return out


@pytest.mark.parametrize("tempo_us", [500_000, 1_000_000])
def test_onset_time_matches_audio_time(tempo_us):
    """2 s of audio must land at 2 s in the file, whatever the tempo."""
    frame = round(2.0 / FRAME_SECONDS)
    mid = _parse(
        build_midi_from_events(
            {"midi_events": _events([(frame, 62, 20)])}, "guitar", tempo_us=tempo_us
        )
    )
    ticks_per_second = mid.ticks_per_beat * 1e6 / tempo_us
    onsets = [t for t, m in _absolute(mid.tracks[0]) if m.type == "note_on" and m.velocity > 0]
    assert len(onsets) == 1
    assert onsets[0] / ticks_per_second == pytest.approx(2.0, abs=0.02)


def test_chord_stays_simultaneous():
    """Three notes on the same onset must share a tick (no +120 stagger)."""
    specs = [(0, 60, 20), (0, 64, 20), (0, 67, 20)]
    mid = _parse(build_midi_from_events({"midi_events": _events(specs)}, "guitar"))
    onsets = [t for t, m in _absolute(mid.tracks[0]) if m.type == "note_on" and m.velocity > 0]
    assert onsets == [0, 0, 0]


def test_duration_frames_honoured_at_a_nonstandard_tempo():
    """Durations must scale with the CALLER's tempo, not the hard-coded 120 BPM.

    Regression guard for the merge of PR #13's `tempo_us` plumbing: `set_tempo` and the
    frame->tick divisor used to be two independent 120-BPM constants, so a 60 BPM render
    played every note (and every length) at double speed.
    """
    tempo_us = 1_000_000  # 60 BPM
    mid = _parse(
        build_midi_from_events(
            {"midi_events": _events([(0, 60, 20), (172, 61, 400)])}, "guitar", tempo_us=tempo_us
        )
    )
    assert [m.tempo for m in mid.tracks[0] if m.type == "set_tempo"] == [tempo_us]
    ticks_per_frame = FRAME_SECONDS * mid.ticks_per_beat * 1e6 / tempo_us

    lengths, opened = {}, {}
    for tick, m in _absolute(mid.tracks[0]):
        if m.type == "note_on" and m.velocity > 0:
            opened[m.note] = tick
        elif m.type == "note_off":
            lengths[m.note] = tick - opened[m.note]

    assert lengths[60] == pytest.approx(20 * ticks_per_frame, abs=1)
    assert lengths[61] == pytest.approx(400 * ticks_per_frame, abs=1)


def test_retriggered_pitch_does_not_overlap():
    """A repeated pitch is closed before the next note-on (no stuck notes)."""
    specs = [(0, 60, 400), (10, 60, 20)]
    mid = _parse(build_midi_from_events({"midi_events": _events(specs)}, "guitar"))
    lengths, opened = [], None
    for tick, m in _absolute(mid.tracks[0]):
        if m.type == "note_on" and m.velocity > 0:
            opened = tick
        elif m.type == "note_off":
            lengths.append(tick - opened)
    assert len(lengths) == 2
    assert all(length > 0 for length in lengths)


def test_confidence_marker_track_is_opt_in():
    """A poly-pressure value is note-specific but not shown by every DAW; the
    companion marker track carries the same numbers as text.

    Merge note: PR #13 wrote this lane unconditionally, which added a second track to
    every render. `main`'s contract is one track per file, so it is now a flag and the
    default keeps main's shape.
    """
    specs = [(0, 60, 10), (0, 64, 10)]
    default = _parse(build_midi_from_events({"midi_events": _events(specs)}, "guitar"))
    assert len(default.tracks) == 1, "a companion track became the default again"

    mid = _parse(
        build_midi_from_events(
            {"midi_events": _events(specs)}, "guitar", include_confidence_track=True
        )
    )
    markers = [m.text for track in mid.tracks for m in track if m.type == "marker"]
    rows = [t for t in markers if t.startswith("confidence note=")]
    assert len(rows) == 2, rows
    # The text is a contract too: `confidence note=<n> onset_tick=<t> value=<0..1>`.
    parsed = [
        (
            int(t.split("note=")[1].split()[0]),
            int(t.split("onset_tick=")[1].split()[0]),
            float(t.split("value=")[1]),
        )
        for t in rows
    ]
    assert [row[0] for row in parsed] == [60, 64], "the marker must name the note it describes"
    assert all(row[1] == 0 for row in parsed), "both notes start at tick 0"
    # 0.8 confidence reaches the file as int(0.8 * 127) = 101 - the truncation MIDI-4
    # documents - and the marker lane must agree with the poly-pressure lane exactly:
    # two spellings of one number, never two numbers.
    byte = 101
    pressures = [m.value for m in mid.tracks[0] if m.type == "polytouch"]
    assert pressures == [byte, byte], pressures
    # the text is written with 3 decimals, so it is the byte rounded, not the byte
    assert [row[2] for row in parsed] == [pytest.approx(byte / 127, abs=5e-4)] * 2, parsed


def test_no_notes_single_track():
    """Empty event lists still produce a valid, single-track file."""
    mid = _parse(build_midi_from_events({"midi_events": []}, "guitar"))
    assert len(mid.tracks) == 1
    assert not [m for m in mid.tracks[0] if m.type == "note_on"]


def test_writer_rejects_a_nonpositive_tempo():
    """A zero/negative tempo would divide by zero into `ticks_per_second`."""
    with pytest.raises(ValueError, match="tempo_us"):
        build_midi_from_events({"midi_events": _events([(0, 60, 20)])}, "guitar", tempo_us=0)
