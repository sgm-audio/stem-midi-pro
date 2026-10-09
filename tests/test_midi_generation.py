"""Tests for MIDI generation from metadata."""
import io

import mido

from api import create_placeholder_midi


def _sample_midi_data():
    return {
        "midi_events": [
            {
                "note": 60,
                "onset_frame": 0,
                "velocity": 100,
                "confidence": 0.9,
                "cc_127_value": 114,
            },
            {
                "note": 64,
                "onset_frame": 86,
                "velocity": 80,
                "confidence": 0.8,
                "cc_127_value": 102,
            },
        ],
        "summary": {
            "avg_confidence": 0.85,
            "low_confidence_count": 0,
            "alignment_warnings": 0,
        },
    }


def test_create_midi_from_events():
    """The output should be a valid MIDI file with note and confidence events."""
    midi_bytes = create_placeholder_midi(_sample_midi_data(), "guitar")
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))

    assert len(mid.tracks) == 1
    messages = mid.tracks[0]
    note_ons = [message for message in messages if message.type == "note_on"]
    assert [message.note for message in note_ons] == [60, 64]
    assert [message.velocity for message in note_ons] == [100, 80]

    confidence_values = [
        message.value
        for message in messages
        if message.type == "control_change" and message.control == 127
    ]
    assert confidence_values == [114, 102]


def test_create_midi_empty_events():
    """Empty input should still produce a valid track without note events."""
    midi_bytes = create_placeholder_midi({"midi_events": []}, "bass")
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))

    assert len(mid.tracks) == 1
    assert not any(message.type == "note_on" for message in mid.tracks[0])


def test_midi_onset_frames_use_absolute_ticks():
    """Frame positions should map to elapsed time, not accumulate as deltas."""
    mid = mido.MidiFile(file=io.BytesIO(
        create_placeholder_midi(_sample_midi_data(), "guitar")
    ))

    elapsed_ticks = 0
    note_on_ticks = []
    for message in mid.tracks[0]:
        elapsed_ticks += message.time
        if message.type == "note_on":
            note_on_ticks.append(elapsed_ticks)

    expected_second_onset = round(86 * 512 / 44100 * 960)
    assert note_on_ticks == [0, expected_second_onset]
