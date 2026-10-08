# SPDX-License-Identifier: Apache-2.0
"""Tests for MIDI generation from metadata (T-8.2.4 — no skip markers).

Covers: empty events, multi-event ordering, CC#127 values, round-trip parse.
Uses utils/midi.py:build_midi_from_events (post RF-3.1.3 rename; the old
api.create_placeholder_midi is identical behavior pre-refactor).
"""

import io

import pytest

mido = pytest.importorskip("mido", reason="MIDI tests require mido")


from utils.midi import build_midi_from_events  # noqa: E402


def _events(n, start_frame=0, frame_step=86):
    return [
        {
            "note": 60 + i,
            "onset_frame": start_frame + i * frame_step,
            "velocity": min(127, 60 + i),
            "confidence": 0.9 - i * 0.05,
            "cc_127_value": int((0.9 - i * 0.05) * 127),
        }
        for i in range(n)
    ]


def _parse(midi_bytes):
    return mido.MidiFile(file=io.BytesIO(midi_bytes))


def test_empty_events_produces_valid_midi():
    """No events should still yield a parseable MIDI file with header only."""
    midi_bytes = build_midi_from_events({"midi_events": [], "summary": {}}, "guitar")
    mid = _parse(midi_bytes)
    assert len(mid.tracks) == 1
    note_ons = [m for m in mid.tracks[0] if m.type == "note_on"]
    assert note_ons == []


def test_multi_event_ordering():
    """Events should appear in onset order in the track."""
    data = {"midi_events": _events(4), "summary": {}}
    mid = _parse(build_midi_from_events(data, "guitar"))
    track = mid.tracks[0]

    note_ons = [m for m in track if m.type == "note_on"]
    assert [m.note for m in note_ons] == [60, 61, 62, 63]
    # Absolute ticks must be non-decreasing.
    times = []
    t = 0
    for m in track:
        t += m.time
        if m.type == "note_on":
            times.append(t)
    assert times == sorted(times)


def test_cc127_confidence_values():
    """Each note event must carry a CC#127 message with the confidence value."""
    events = _events(3)
    data = {"midi_events": events, "summary": {}}
    mid = _parse(build_midi_from_events(data, "bass"))

    ccs = [m for m in mid.tracks[0] if m.type == "control_change" and m.control == 127]
    assert len(ccs) == 3
    assert [m.value for m in ccs] == [e["cc_127_value"] for e in events]


def test_midi_roundtrip_parse():
    """Saved bytes must re-parse with the same event count."""
    data = {"midi_events": _events(5), "summary": {}}
    midi_bytes = build_midi_from_events(data, "guitar")

    assert midi_bytes[:4] == b"MThd"  # SMF header magic

    reparsed = _parse(midi_bytes)
    note_ons = [m for m in reparsed.tracks[0] if m.type == "note_on"]
    note_offs = [m for m in reparsed.tracks[0] if m.type == "note_off"]
    assert len(note_ons) == 5
    assert len(note_offs) == 5


def test_velocity_clamped():
    """Velocities outside 1-127 are clamped into MIDI range."""
    events = [
        {"note": 60, "onset_frame": 0, "velocity": 250, "confidence": 0.9},
        {"note": 61, "onset_frame": 86, "velocity": 0, "confidence": 0.9},
    ]
    mid = _parse(build_midi_from_events({"midi_events": events}, "guitar"))
    velocities = [m.velocity for m in mid.tracks[0] if m.type == "note_on"]
    assert velocities[0] == 127
    assert velocities[1] == 1
