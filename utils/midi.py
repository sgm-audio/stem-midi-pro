# SPDX-License-Identifier: Apache-2.0
"""
MIDI building utilities for Stem+MIDI Pro.

Converts model MIDI-event metadata into Standard MIDI Files.  Confidence is
encoded as MIDI Polyphonic Key Pressure (a note-specific, 0-127 value), not
CC#127: CC#127 is a channel-mode message and cannot carry per-note metadata.

Event dict shape (as produced by ConfidenceInjector / main.py):
    {
        "onset_frame": int,        # audio frame index (hop_length grid)
        "note": int,               # MIDI pitch 0-127
        "velocity": int,           # 1-127
        "confidence": float,       # 0-1
        "confidence_value": int,   # optional; defaults to confidence*127
        "cc_127_value": int,       # deprecated input alias; never emitted as CC#127
        "duration_frames": int,    # optional; falls back to default duration
        "channel": int,            # optional MIDI channel, defaults to 0
        "expression": {             # optional normalized values in [0, 1]
            "bend": float,         # .5 = center; maps to MIDI pitch-wheel
            "vibrato": float,      # maps to CC#1 (modulation)
            "slide": float,        # maps to CC#74 (brightness proxy)
        },
    }
"""

import io
import os
from typing import Any

from mido import Message, MetaMessage, MidiFile, MidiTrack

TICKS_PER_BEAT = 480
DEFAULT_TEMPO_US = 500_000  # 120 BPM
HOP_LENGTH = 512
DEFAULT_SAMPLE_RATE = 44100
# Configurable fallback note duration, in beats, when the event carries no
# duration_frames (API-6.12). Was a hardcoded 0.25 (1/16 note).
DEFAULT_NOTE_DURATION_BEATS = float(os.getenv("MIDI_DEFAULT_NOTE_DURATION_BEATS", "0.25"))


def _clamp_midi(value: Any, low: int = 0, high: int = 127) -> int:
    return min(high, max(low, int(value)))


def _expression_messages(event: dict[str, Any], channel: int) -> list[Message]:
    """Translate normalized expression fields to standard MIDI messages.

    Pitch bend and CC expression are channel-wide in ordinary MIDI.  The event
    channel is therefore explicit so callers can provide MPE-style channels if
    they need independent simultaneous bends.
    """
    expression = event.get("expression") or {}
    messages: list[Message] = []

    if "bend" in expression:
        # The model head emits [0, 1], where .5 is the neutral pitch-wheel
        # position. Mido uses the signed 14-bit pitch-wheel range.
        bend = max(0.0, min(1.0, float(expression["bend"])))
        pitch = _clamp_midi(round((bend - 0.5) * 16384), -8192, 8191)
        messages.append(Message("pitchwheel", channel=channel, pitch=pitch, time=0))
    if "vibrato" in expression:
        messages.append(
            Message(
                "control_change",
                channel=channel,
                control=1,
                value=_clamp_midi(round(float(expression["vibrato"]) * 127)),
                time=0,
            )
        )
    if "slide" in expression:
        messages.append(
            Message(
                "control_change",
                channel=channel,
                control=74,
                value=_clamp_midi(round(float(expression["slide"]) * 127)),
                time=0,
            )
        )
    return messages


def build_midi_from_events(
    midi_data: dict[str, Any],
    stem_type: str,
    ticks_per_beat: int = TICKS_PER_BEAT,
    default_duration_beats: float | None = None,
    hop_length: int = HOP_LENGTH,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    include_expression: bool = False,
) -> bytes:
    """Convert MIDI event metadata to SMF bytes.

    Events are scheduled on an absolute tick timeline before being written as
    Mido delta times. This preserves chords and overlapping different-pitch
    notes, while truncating duplicate same-channel/pitch retriggers so ordinary
    MIDI receivers do not see ambiguous overlapping note-ons. Audio-frame timing
    uses ``ticks_per_beat * 1_000_000 / tempo_us``; at 120 BPM and 480 PPQ,
    that is 960 ticks/sec.

    ``duration_frames`` is honored when available; otherwise the function
    falls back to ``default_duration_beats`` (environment variable
    ``MIDI_DEFAULT_NOTE_DURATION_BEATS``, default 0.25 beats).
    """
    if default_duration_beats is None:
        default_duration_beats = DEFAULT_NOTE_DURATION_BEATS
    if default_duration_beats <= 0:
        raise ValueError("default_duration_beats must be positive")
    if hop_length <= 0 or sample_rate <= 0 or ticks_per_beat <= 0:
        raise ValueError("hop_length, sample_rate, and ticks_per_beat must be positive")

    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(MetaMessage("set_tempo", tempo=DEFAULT_TEMPO_US))
    track.append(MetaMessage("time_signature", numerator=4, denominator=4))
    track.append(MetaMessage("track_name", name=f"{stem_type} transcription"))
    instrument = "Electric Guitar" if stem_type == "guitar" else "Electric Bass"
    track.append(MetaMessage("instrument_name", name=instrument))
    track.append(MetaMessage("marker", text=f"{stem_type.capitalize()} - Stem+MIDI Pro"))

    # tempo is microseconds per beat; no extra factor of 60 is needed.
    ticks_per_second = ticks_per_beat * 1_000_000.0 / DEFAULT_TEMPO_US

    # Scheduled entries are (absolute_tick, priority, sequence, message).
    # Note-offs precede note-ons at the same tick; note-specific metadata and
    # expression follow the note-on at that tick.
    scheduled: list[tuple[int, int, int, Message]] = []
    sequence = 0
    active_until: dict[tuple[int, int], int] = {}
    last_onset: dict[tuple[int, int], int] = {}
    note_off_entry: dict[tuple[int, int], int] = {}
    events = sorted(midi_data.get("midi_events", []), key=lambda e: int(e["onset_frame"]))
    for ev in events:
        note = int(ev["note"])
        if not 0 <= note <= 127:
            raise ValueError(f"MIDI note outside MIDI range [0, 127]: {note}")
        channel = int(ev.get("channel", 0))
        if not 0 <= channel <= 15:
            raise ValueError(f"MIDI channel outside MIDI range [0, 15]: {channel}")

        onset_frame = int(ev["onset_frame"])
        if onset_frame < 0:
            raise ValueError(f"onset_frame must be non-negative, got {onset_frame}")
        onset_tick = max(0, round(onset_frame * hop_length / sample_rate * ticks_per_second))
        key = (channel, note)
        # Drop exact duplicate retriggers and truncate a prior same-pitch note
        # before opening the next one. Different pitches remain fully
        # overlapping, preserving chords and polyphonic durations.
        if last_onset.get(key) == onset_tick:
            continue
        if key in active_until and onset_tick < active_until[key]:
            old_index = note_off_entry[key]
            _old_tick, old_priority, old_sequence, old_message = scheduled[old_index]
            scheduled[old_index] = (onset_tick, old_priority, old_sequence, old_message)

        duration_frames = ev.get("duration_frames")
        if duration_frames is not None:
            duration_frames = int(duration_frames)
            if duration_frames <= 0:
                raise ValueError(f"duration_frames must be positive, got {duration_frames}")
            duration_ticks = max(
                1, round(duration_frames * hop_length / sample_rate * ticks_per_second)
            )
        else:
            duration_ticks = max(1, round(default_duration_beats * ticks_per_beat))
        off_tick = onset_tick + duration_ticks

        sequence += 1
        scheduled.append(
            (
                onset_tick,
                1,
                sequence,
                Message(
                    "note_on",
                    channel=channel,
                    note=note,
                    velocity=_clamp_midi(ev.get("velocity", 1), 1, 127),
                    time=0,
                ),
            )
        )

        # Polyphonic key pressure is note-specific and is the confidence lane.
        # Keep the old cc_127_value key as an input alias for callers, but do
        # not emit CC#127 because it is a channel-mode command.
        confidence_value = ev.get(
            "confidence_value",
            ev.get("cc_127_value", float(ev.get("confidence", 0.0)) * 127),
        )
        sequence += 1
        scheduled.append(
            (
                onset_tick,
                2,
                sequence,
                Message(
                    "polytouch",
                    channel=channel,
                    note=note,
                    value=_clamp_midi(confidence_value),
                    time=0,
                ),
            )
        )

        # Expression is off by default: the head is untrained (ARCH-11.4.4), so
        # writing it into every export would fabricate phrasing the model never
        # learned. Callers that want it ask for it.
        if include_expression:
            for expression_message in _expression_messages(ev, channel):
                sequence += 1
                scheduled.append((onset_tick, 3, sequence, expression_message))

        sequence += 1
        note_off_entry[key] = len(scheduled)
        scheduled.append(
            (
                off_tick,
                0,
                sequence,
                Message("note_off", channel=channel, note=note, velocity=0, time=0),
            )
        )
        active_until[key] = off_tick
        last_onset[key] = onset_tick

    scheduled.sort(key=lambda item: (item[0], item[1], item[2]))
    current_tick = 0
    for tick, _priority, _sequence, message in scheduled:
        message.time = max(0, tick - current_tick)
        track.append(message)
        current_tick = tick

    track.append(MetaMessage("end_of_track"))

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


# Back-compat alias for any external code importing the old name (API-6.11).
def create_placeholder_midi(midi_data: dict[str, Any], stem_type: str) -> bytes:
    """Deprecated alias for build_midi_from_events."""
    return build_midi_from_events(midi_data, stem_type)
