# SPDX-License-Identifier: Apache-2.0
"""
MIDI building utilities for Stem+MIDI Pro.

Owns conversion of model MIDI-event metadata into Standard MIDI Files.
(Replaces the old `api.py:create_placeholder_midi`, which actually emitted
real events — API-6.11 / RF-3.1.4 / RF-3.1.3.)

Event dict shape (as produced by ConfidenceInjector / main.py):
    {
        "onset_frame": int,        # audio frame index (hop_length grid)
        "note": int,               # MIDI pitch 0-127
        "velocity": int,           # 1-127
        "confidence": float,       # 0-1
        "cc_127_value": int,       # optional; defaults to confidence*127
        "duration_frames": int,    # optional; falls back to default duration
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


def build_midi_from_events(
    midi_data: dict[str, Any],
    stem_type: str,
    ticks_per_beat: int = TICKS_PER_BEAT,
    default_duration_beats: float | None = None,
    hop_length: int = HOP_LENGTH,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> bytes:
    """
    Convert MIDI event metadata to SMF bytes.

    Uses `duration_frames` on each event when available (API-6.12);
    otherwise falls back to `default_duration_beats`
    (env `MIDI_DEFAULT_NOTE_DURATION_BEATS`, default 0.25 beats).
    """
    if default_duration_beats is None:
        default_duration_beats = DEFAULT_NOTE_DURATION_BEATS

    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(MetaMessage("set_tempo", tempo=DEFAULT_TEMPO_US))
    track.append(MetaMessage("time_signature", numerator=4, denominator=4))
    track.append(MetaMessage("track_name", name=f"{stem_type} transcription"))
    instrument = "Electric Guitar" if stem_type == "guitar" else "Electric Bass"
    track.append(MetaMessage("instrument_name", name=instrument))
    track.append(MetaMessage("marker", text=f"{stem_type.capitalize()} - Stem+MIDI Pro"))

    # Tempo: DEFAULT_TEMPO_US is microseconds per beat, i.e. 2 beats/sec at
    # 120 BPM -> 480 * 2 = 960 ticks/second. (Was 480 * (60/0.5) = 57600, which
    # placed every note 60x too late in the bar; see BUG-MIDI-1.)
    bpm = 60_000_000.0 / DEFAULT_TEMPO_US
    ticks_per_second = ticks_per_beat * bpm / 60.0

    events = midi_data.get("midi_events", [])

    # Mido `time` is delta-ticks, not absolute, so build the absolute-tick
    # schedule first and convert to deltas in a single monotone pass. Scheduling
    # note_on/note_off independently keeps *simultaneous* notes simultaneous:
    # advancing a single cursor past each note_off (the previous approach)
    # serialised chords into arpeggios (BUG-MIDI-2).
    #
    # Sort key: tick, then kind (note_off before note_on before control_change
    # so a re-triggered pitch is released before it is struck), then insertion
    # order for stability.
    _ORDER_OFF, _ORDER_ON, _ORDER_CC = 0, 1, 2
    schedule: list[tuple[int, int, int, Message]] = []

    for seq, ev in enumerate(events):
        tick = max(0, round(ev["onset_frame"] * hop_length / sample_rate * ticks_per_second))
        note = max(0, min(127, int(ev["note"])))
        velocity = min(127, max(1, int(ev["velocity"])))

        duration_frames = ev.get("duration_frames")
        if duration_frames:
            dur_ticks = max(1, round(duration_frames * hop_length / sample_rate * ticks_per_second))
        else:
            dur_ticks = max(1, int(default_duration_beats * ticks_per_beat))

        schedule.append(
            (tick + dur_ticks, _ORDER_OFF, seq, Message("note_off", note=note, velocity=0, time=0))
        )
        schedule.append(
            (
                tick,
                _ORDER_ON,
                seq,
                Message("note_on", note=note, velocity=velocity, time=0),
            )
        )

        cc_val = ev.get("cc_127_value", int(ev.get("confidence", 0.0) * 127))
        cc_clamped = min(127, max(0, int(cc_val)))
        schedule.append(
            (tick, _ORDER_CC, seq, Message("control_change", control=127, value=cc_clamped, time=0))
        )

    schedule.sort(key=lambda item: (item[0], item[1], item[2]))

    current_tick = 0
    for tick, _kind, _seq, msg in schedule:
        msg.time = max(0, tick - current_tick)
        current_tick = max(current_tick, tick)
        track.append(msg)

    track.append(MetaMessage("end_of_track"))

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


# Back-compat alias for any external code importing the old name (API-6.11).
def create_placeholder_midi(midi_data: dict[str, Any], stem_type: str) -> bytes:
    """Deprecated alias for build_midi_from_events."""
    return build_midi_from_events(midi_data, stem_type)
