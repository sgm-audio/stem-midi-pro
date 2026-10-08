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

    # 120 BPM -> 2 beats/sec -> ticks_per_beat * 2 ticks per second
    ticks_per_second = ticks_per_beat * (60.0 / (DEFAULT_TEMPO_US / 1_000_000.0))

    events = midi_data.get("midi_events", [])
    for ev in events:
        tick = int(ev["onset_frame"] * hop_length / sample_rate * ticks_per_second)
        note = int(ev["note"])
        velocity = min(127, max(1, int(ev["velocity"])))
        track.append(Message("note_on", note=note, velocity=velocity, time=tick))

        duration_frames = ev.get("duration_frames")
        if duration_frames:
            dur_ticks = max(1, int(duration_frames * hop_length / sample_rate * ticks_per_second))
        else:
            dur_ticks = max(1, int(default_duration_beats * ticks_per_beat))
        track.append(Message("note_off", note=note, velocity=0, time=dur_ticks))

        cc_val = ev.get("cc_127_value", int(ev.get("confidence", 0.0) * 127))
        cc_clamped = min(127, max(0, int(cc_val)))
        track.append(Message("control_change", control=127, value=cc_clamped, time=0))

    track.append(MetaMessage("end_of_track"))

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


# Back-compat alias for any external code importing the old name (API-6.11).
def create_placeholder_midi(midi_data: dict[str, Any], stem_type: str) -> bytes:
    """Deprecated alias for build_midi_from_events."""
    return build_midi_from_events(midi_data, stem_type)
