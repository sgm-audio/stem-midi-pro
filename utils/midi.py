# SPDX-License-Identifier: Apache-2.0
"""
MIDI building utilities for Stem+MIDI Pro.

Owns conversion of model MIDI-event metadata into Standard MIDI Files.
(Replaces the old `api.py:create_placeholder_midi`, which actually emitted
real events - API-6.11 / RF-3.1.4 / RF-3.1.3.)

Event dict shape (as produced by ConfidenceInjector / main.py):
    {
        "onset_frame": int,        # audio frame index (hop_length grid)
        "note": int,               # MIDI pitch 0-127 (validated)
        "velocity": int,           # 1-127 (clamped)
        "confidence": float,       # 0-1
        "confidence_cc_value": int,  # optional; defaults to confidence*127
        "cc_127_value": int,       # legacy alias for confidence_cc_value
        "duration_frames": int,    # optional; falls back to default duration
    }

Confidence CC channel (H-5): the MIDI spec reserves controllers 120-127 as
Channel Mode messages (CC#127 = "Poly Mode On", which also implies All Notes
Off on spec-compliant gear), so the default confidence controller is CC#16
(General Purpose 1). Set ``MIDI_CONFIDENCE_CC=127`` to restore the legacy
behavior explicitly.

The writer is two-pass: events are sorted by onset tick, same-pitch overlaps
are resolved deterministically (a later same-pitch note truncates the earlier
one at ``onset - 1``; identical-onset duplicates are collapsed), and all
messages are emitted as sorted absolute ticks converted to deltas. This
guarantees a valid SMF: non-negative deltas, exactly one note_off per note_on,
and no overlapping same-pitch notes (H-6).
"""

import io
import os
from typing import Any

from mido import Message, MetaMessage, MidiFile, MidiTrack

TICKS_PER_BEAT = 480
DEFAULT_TEMPO_US = 500_000  # 120 BPM
HOP_LENGTH = 512
DEFAULT_SAMPLE_RATE = 44100
# Default confidence controller: CC#16 (General Purpose 1). CC#120-127 are
# Channel Mode messages per the MIDI spec - CC#127 would switch synth mode
# and imply All Notes Off (H-5).
DEFAULT_CONFIDENCE_CC = 16
# Configurable fallback note duration, in beats, when the event carries no
# duration_frames (API-6.12). Was a hardcoded 0.25 (1/16 note).
DEFAULT_NOTE_DURATION_BEATS = float(os.getenv("MIDI_DEFAULT_NOTE_DURATION_BEATS", "0.25"))


def resolve_confidence_cc(explicit: int | None = None) -> int:
    """Resolve the confidence CC controller number.

    Priority: explicit argument > ``MIDI_CONFIDENCE_CC`` env > default (16).
    Raises ValueError outside 0-127 and warns when a Channel Mode message
    (120-127) is requested.
    """
    import logging

    if explicit is None:
        raw = os.getenv("MIDI_CONFIDENCE_CC", str(DEFAULT_CONFIDENCE_CC))
    else:
        raw = explicit
    try:
        cc = int(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Confidence CC controller must be an int 0-127, got {raw!r}") from e
    if not 0 <= cc <= 127:
        raise ValueError(f"Confidence CC controller must be in 0-127, got {cc}")
    if cc >= 120:
        logging.getLogger(__name__).warning(
            "Confidence CC set to %d, a MIDI Channel Mode message (120-127); "
            "spec-compliant gear may treat it as a mode switch, not data.",
            cc,
        )
    return cc


def build_midi_from_events(
    midi_data: dict[str, Any],
    stem_type: str,
    ticks_per_beat: int = TICKS_PER_BEAT,
    default_duration_beats: float | None = None,
    hop_length: int = HOP_LENGTH,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    confidence_cc: int | None = None,
) -> bytes:
    """
    Convert MIDI event metadata to SMF bytes.

    Uses `duration_frames` on each event when available (API-6.12);
    otherwise falls back to `default_duration_beats`
    (env `MIDI_DEFAULT_NOTE_DURATION_BEATS`, default 0.25 beats).
    """
    if default_duration_beats is None:
        default_duration_beats = DEFAULT_NOTE_DURATION_BEATS
    cc_controller = resolve_confidence_cc(confidence_cc)

    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(MetaMessage("set_tempo", tempo=DEFAULT_TEMPO_US))
    track.append(MetaMessage("time_signature", numerator=4, denominator=4))
    track.append(MetaMessage("track_name", name=f"{stem_type} transcription"))
    instrument = "Electric Guitar" if stem_type == "guitar" else "Electric Bass"
    track.append(MetaMessage("instrument_name", name=instrument))
    track.append(MetaMessage("marker", text=f"{stem_type.capitalize()} - Stem+MIDI Pro"))

    # 120 BPM (500000 µs/beat) → 2 beats/sec → ticks_per_beat * 2 ticks per
    # second. NOTE: this must be ticks_per_beat / seconds_per_beat; the old
    # formula (x 60 / seconds_per_beat) computed 57600 ticks/sec - every file
    # the repo ever wrote played 60x too fast, and no test checked absolute
    # time until the 2026-11 review regression suite.
    ticks_per_second = ticks_per_beat * 1_000_000.0 / DEFAULT_TEMPO_US

    events = midi_data.get("midi_events", []) or []

    # Pass 1: validate and compute absolute ticks.
    notes: list[list[int]] = []  # [onset_tick, off_tick, note, velocity, cc]
    for i, ev in enumerate(events):
        try:
            note = int(ev["note"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"MIDI event {i}: invalid or missing 'note': {ev!r}") from e
        if not 0 <= note <= 127:
            raise ValueError(
                f"MIDI event {i}: note {note} outside MIDI range 0-127; refusing to "
                "write an invalid file. Check the pitch head / source backend."
            )
        onset_frame = int(ev.get("onset_frame", 0))
        tick = round(onset_frame * hop_length / sample_rate * ticks_per_second)
        tick = max(0, tick)
        velocity = min(127, max(1, int(ev.get("velocity", 64))))

        duration_frames = ev.get("duration_frames")
        if duration_frames:
            dur_ticks = max(1, int(duration_frames * hop_length / sample_rate * ticks_per_second))
        else:
            dur_ticks = max(1, int(default_duration_beats * ticks_per_beat))

        cc_val = ev.get(
            # MIDI-4: round, not truncate -- see confidence_injector.py.
            "confidence_cc_value",
            ev.get("cc_127_value", round(ev.get("confidence", 0.0) * 127)),
        )
        cc_clamped = min(127, max(0, int(cc_val)))

        notes.append([tick, tick + dur_ticks, note, velocity, cc_clamped])

    # Pass 2: resolve same-pitch overlaps (H-6). Events are sorted by onset;
    # a later same-pitch note truncates the earlier one at onset-1; notes with
    # an identical onset tick are duplicates - keep the first, drop the rest.
    notes.sort(key=lambda n: (n[0], n[2]))
    kept: list[list[int]] = []
    last_for_pitch: dict[int, list[int]] = {}
    for note_rec in notes:
        onset, _off, pitch = note_rec[0], note_rec[1], note_rec[2]
        prev = last_for_pitch.get(pitch)
        if prev is not None and onset < prev[1]:
            if onset <= prev[0]:
                continue  # duplicate onset for the same pitch - drop
            # truncate the earlier note so it ends before this onset
            prev[1] = max(prev[0] + 1, onset - 1)
        kept.append(note_rec)
        last_for_pitch[pitch] = note_rec
    notes = kept

    # Pass 3: emit sorted messages with delta times. At the same tick:
    # note_off (release) before note_on (strike) before CC (H-5: CC goes
    # with the note-on, not after the note-off).
    messages: list[tuple[int, int, Message]] = []
    for onset, off, note, velocity, cc_val in notes:
        messages.append((onset, 1, Message("note_on", note=note, velocity=velocity)))
        messages.append((off, 0, Message("note_off", note=note, velocity=0)))
        messages.append((onset, 2, Message("control_change", control=cc_controller, value=cc_val)))
    messages.sort(key=lambda m: (m[0], m[1]))

    current_tick = 0
    for abs_tick, _kind, msg in messages:
        delta = max(0, abs_tick - current_tick)
        msg.time = delta
        track.append(msg)
        current_tick = abs_tick

    track.append(MetaMessage("end_of_track"))

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


# Back-compat alias for any external code importing the old name (API-6.11).
def create_placeholder_midi(midi_data: dict[str, Any], stem_type: str, **kwargs: Any) -> bytes:
    """Deprecated alias for build_midi_from_events."""
    return build_midi_from_events(midi_data, stem_type, **kwargs)
