# SPDX-License-Identifier: Apache-2.0
"""MIDI round-trip tests.

Build a MidiFile with N events, save to bytes, re-parse, and verify timing,
polyphony, durations, the confidence controller and the confidence marker track.

These pin the behaviour fixed after the audit (AUDIT_REPORT.md C-3/API-6.12):

* tick positions come from ``ticks_per_beat * 1e6 / tempo_us`` — the old code
  used ``ticks_per_beat * 60 / seconds_per_beat``, i.e. 60x too late;
* simultaneous onsets stay simultaneous (no arpeggiation);
* confidence goes to a general-purpose controller (default CC#16); CC#120-127
  are Channel Mode messages, so CC#127 is only reachable by explicit request;
* ``duration_frames`` sets the real note length instead of a fixed 1/16 note.
"""

import io

import pytest

mido = pytest.importorskip("mido", reason="MIDI round-trip requires mido")

from utils.midi import build_midi_from_events  # noqa: E402

SR = 44100
HOP = 512
FRAME_SECONDS = HOP / SR
DEFAULT_CC = 16


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
        ev = {
            "note": note,
            "onset_frame": frame,
            "velocity": 100,
            "confidence": 0.8,
        }
        if dur is not None:
            ev["duration_frames"] = dur
        ev.update(kwargs)
        out.append(ev)
    return out


@pytest.mark.parametrize("n_events", [0, 1, 3, 10])
def test_event_count_roundtrip(n_events):
    """N events → N note_on + N note_off + N confidence CC messages."""
    specs = [(i * 86, 40 + i, 20) for i in range(n_events)]
    mid = _parse(build_midi_from_events({"midi_events": _events(specs)}, "guitar"))
    track = mid.tracks[0]

    note_ons = [m for m in track if m.type == "note_on" and m.velocity > 0]
    note_offs = [m for m in track if m.type == "note_off"]
    ccs = [m for m in track if m.type == "control_change"]

    assert len(note_ons) == n_events
    assert len(note_offs) == n_events
    assert len(ccs) == n_events
    assert {m.control for m in ccs} <= {DEFAULT_CC}


def test_confidence_values_roundtrip():
    """Controller values must match the per-event confidence (scaled to 0-127)."""
    confidences = [0.1, 0.5, 0.9, 1.0]
    specs = [(i * 86, 60, 10) for i in range(len(confidences))]
    events = [
        {"note": n, "onset_frame": f, "velocity": 80, "confidence": c}
        for (f, n, _d), c in zip(specs, confidences, strict=True)
    ]
    mid = _parse(build_midi_from_events({"midi_events": events}, "bass"))
    ccs = [m.value for m in mid.tracks[0] if m.type == "control_change"]
    assert ccs == [int(c * 127) for c in confidences]


def test_legacy_cc127_is_opt_in():
    """CC#127 (Channel Mode) is only reachable by explicit request."""
    mid = _parse(
        build_midi_from_events({"midi_events": _events([(0, 60, 10)])}, "guitar", confidence_cc=127)
    )
    controls = {m.control for m in mid.tracks[0] if m.type == "control_change"}
    assert controls == {127}

    default_mid = _parse(build_midi_from_events({"midi_events": _events([(0, 60, 10)])}, "guitar"))
    default_controls = {m.control for m in default_mid.tracks[0] if m.type == "control_change"}
    assert default_controls == {DEFAULT_CC}
    assert not (default_controls & set(range(120, 128)))


def test_explicit_confidence_value_overrides():
    """An explicit confidence_cc_value (or legacy cc_127_value) wins."""
    events = [
        {
            "note": 64,
            "onset_frame": 0,
            "velocity": 100,
            "confidence": 0.9,
            "cc_127_value": 42,
        }
    ]
    mid = _parse(build_midi_from_events({"midi_events": events}, "guitar"))
    ccs = [m.value for m in mid.tracks[0] if m.type == "control_change"]
    assert ccs == [42]


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


def test_duration_frames_honoured():
    """duration_frames must set the real note length, not a fixed 1/16."""
    tempo_us = 500_000
    mid = _parse(
        build_midi_from_events(
            {"midi_events": _events([(0, 60, 20), (172, 61, 400)])}, "guitar", tempo_us=tempo_us
        )
    )
    ticks_per_frame = FRAME_SECONDS * mid.ticks_per_beat * 1e6 / tempo_us

    lengths = {}
    opened = {}
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
    lengths = []
    opened = None
    for tick, m in _absolute(mid.tracks[0]):
        if m.type == "note_on" and m.velocity > 0:
            opened = tick
        elif m.type == "note_off":
            lengths.append(tick - opened)
    assert len(lengths) == 2
    assert lengths[1] > 0


def test_confidence_marker_track():
    """Per-note confidence is also written as marker text (one channel CC cannot
    distinguish simultaneous notes)."""
    mid = _parse(
        build_midi_from_events({"midi_events": _events([(0, 60, 10), (0, 64, 10)])}, "guitar")
    )
    markers = [m.text for track in mid.tracks for m in track if m.type == "marker"]
    assert sum(1 for t in markers if t.startswith("confidence note=")) == 2


def test_no_notes_single_track():
    """Empty event lists still produce a valid, single-track file."""
    mid = _parse(build_midi_from_events({"midi_events": []}, "guitar"))
    assert len(mid.tracks) == 1
    assert not [m for m in mid.tracks[0] if m.type == "note_on"]
