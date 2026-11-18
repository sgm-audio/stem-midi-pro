#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the MIDI claims in FIXES.md (C-3, M-3).

Run: ``python3 scripts/verify_midi.py``

Checks the writer and the target builder against the audit's reproductions:

* 3000 frames (~34.8 s at 44.1 kHz / hop 512) land at 34.8 s of MIDI time, not 60x that,
* simultaneous onsets stay simultaneous,
* confidence rides a per-note Polyphonic Key Pressure event (CC#120-127 are Channel
  Mode, so they can neither carry per-note data nor appear at all), and
  ``include_confidence_track=True`` adds a meta-only Marker track for DAWs that do not
  show poly-pressure; no channel CC carries confidence,
* ``duration_frames`` sets note length,
* ``tempo_us`` steers the writer (its ``set_tempo`` meta and its frame->tick divisor are
  one promise: absolute seconds survive a tempo change),
* the reader's tempo map puts beat 2 at frame 86 @120 BPM and 172 @60 BPM.
"""

from __future__ import annotations

import io
import sys
import tempfile

import mido
import torch

sys.path.insert(0, ".")

from utils.midi import build_midi_from_events
from utils.midi_targets import build_tempo_map, midi_to_targets, tick_to_seconds

HOP = 512
SR = 44100


def _parse(data: bytes) -> mido.MidiFile:
    return mido.MidiFile(file=io.BytesIO(data))


def _absolute(track):
    tick = 0
    for msg in track:
        tick += msg.time
        yield tick, msg


def check_clock(left: int, right: int, channel: str = "guitar") -> None:
    mid = _parse(
        build_midi_from_events(
            {
                "midi_events": [
                    {
                        "note": 57,
                        "onset_frame": left,
                        "duration_frames": 40,
                        "velocity": 90,
                        "confidence": 0.8,
                    },
                    {
                        "note": 59,
                        "onset_frame": right,
                        "duration_frames": 40,
                        "velocity": 90,
                        "confidence": 0.8,
                    },
                ]
            },
            channel,
            hop_length=HOP,
            sample_rate=SR,
        )
    )
    audio_seconds = right * HOP / SR
    channel_controls = sorted({m.control for m in mid.tracks[0] if m.type == "control_change"})
    print(
        f"clock       : {right} frames = {audio_seconds:.2f}s audio -> MIDI {mid.length:.2f}s "
        f"(ratio {mid.length / audio_seconds:.2f}x, controls {channel_controls})"
    )


def check_polyphony() -> None:
    events = [
        {"note": n, "onset_frame": 0, "duration_frames": 30, "velocity": 90, "confidence": 0.8}
        for n in (60, 64, 67)
    ]
    mid = _parse(build_midi_from_events({"midi_events": events}, "guitar"))
    onsets = [t for t, m in _absolute(mid.tracks[0]) if m.type == "note_on" and m.velocity > 0]
    print(f"polyphony   : 3 notes on one frame -> onsets {onsets}")


def check_confidence() -> None:
    """Confidence must be per note, and must not squat on a Channel Mode CC."""
    events = [
        {"note": 60, "onset_frame": 0, "duration_frames": 20, "velocity": 90, "confidence": 0.9},
        {"note": 64, "onset_frame": 0, "duration_frames": 20, "velocity": 90, "confidence": 0.4},
    ]
    mid = _parse(build_midi_from_events({"midi_events": events}, "guitar"))
    track = mid.tracks[0]
    polytouch = {m.note: m.value for m in track if m.type == "polytouch"}
    controls = sorted({m.control for m in track if m.type == "control_change"})
    # The writer's lane truncates (`int(conf * 127)`, the surviving half of MIDI-4);
    # `round()` would give 114 / 51. Both spellings are printed so the difference is a
    # documented divergence rather than an unexplained diff, and the assert pins the one
    # the writer actually guarantees. Note the expression lanes above DO round
    # (`round((bend - 0.5) * 16384)`, `round(vibrato * 127)`) - one file, two mappings.
    truncated = {60: int(0.9 * 127), 64: int(0.4 * 127)}
    rounded = {60: round(0.9 * 127), 64: round(0.4 * 127)}
    assert polytouch == truncated, f"polytouch {polytouch} != truncated {truncated}"
    assert all(abs(polytouch[n] - rounded[n]) <= 1 for n in rounded), polytouch
    print(f"confidence  : polytouch per note {polytouch} (int() mapping; round() -> {rounded})")
    print(f"              CC lanes used {controls} (CC#120-127 are Channel Mode; none emitted)")


def check_expression() -> None:
    """The untrained expression head is exported only when asked for, and labelled."""
    event = {
        "note": 62,
        "onset_frame": 0,
        "duration_frames": 40,
        "velocity": 90,
        "confidence": 0.8,
        "expression": {
            "bend": 0.75,
            "vibrato": 1.0,
            "slide": 0.0,
            "source": "expression_head (untrained)",
        },
    }
    default_types = {
        m.type for m in _parse(build_midi_from_events({"midi_events": [event]}, "guitar")).tracks[0]
    }
    print(f"expression  : default render message types {sorted(default_types)} (no pitchwheel)")

    mid = _parse(
        build_midi_from_events({"midi_events": [event]}, "guitar", include_expression=True)
    )
    bend = next(m for m in mid.tracks[0] if m.type == "pitchwheel")
    ccs = {m.control: m.value for m in mid.tracks[0] if m.type == "control_change"}
    print(f"              opt-in -> pitchwheel {bend.pitch}, CC1 {ccs.get(1)}, CC74 {ccs.get(74)}")


def check_durations() -> None:
    mid = _parse(
        build_midi_from_events(
            {
                "midi_events": [
                    {
                        "note": 60,
                        "onset_frame": 0,
                        "duration_frames": 20,
                        "velocity": 90,
                        "confidence": 0.8,
                    },
                    {
                        "note": 62,
                        "onset_frame": 400,
                        "duration_frames": 400,
                        "velocity": 90,
                        "confidence": 0.8,
                    },
                ]
            },
            "guitar",
            hop_length=HOP,
            sample_rate=SR,
        )
    )
    ticks_per_frame = (HOP / SR) * mid.ticks_per_beat * 1e6 / 500_000
    lengths, opened = {}, {}
    for tick, msg in _absolute(mid.tracks[0]):
        if msg.type == "note_on" and msg.velocity > 0:
            opened[msg.note] = tick
        elif msg.type == "note_off":
            lengths[msg.note] = tick - opened[msg.note]
    for note, frames in ((60, 20), (62, 400)):
        print(
            f"duration    : note {note}, {frames} frames -> {lengths[note]} ticks "
            f"(expected {frames * ticks_per_frame:.0f} ±1 from integer rounding)"
        )


def check_writer_params() -> None:
    """Cover the two writer parameters PR #13 grafts onto main's lane.

    ``tempo_us`` used to be hard-coded to 120 BPM in both the ``set_tempo`` meta and the
    frame->tick divisor, so a 60 BPM render contradicted its own header by 2x. The guard
    below is the round trip: the file must report the tempo it was given, and the note's
    length in SECONDS must not move when the tempo does.
    """
    frames_per_second = SR / HOP
    events = [
        {
            "note": 60,
            "onset_frame": 0,
            "duration_frames": round(2.0 * frames_per_second),
            "velocity": 90,
            "confidence": 0.9,
        }
    ]
    for tempo_us in (500_000, 1_000_000):
        raw = build_midi_from_events(
            {"midi_events": events}, "guitar", hop_length=HOP, sample_rate=SR, tempo_us=tempo_us
        )
        mid = _parse(raw)
        metas = [m for m in mid.tracks[0] if m.type == "set_tempo"]
        ticks_per_second = mid.ticks_per_beat * 1_000_000 / tempo_us
        onsets, offs = {}, {}
        for tick, msg in _absolute(mid.tracks[0]):
            if msg.type == "note_on" and msg.velocity > 0:
                onsets[msg.note] = tick
            elif msg.type == "note_off":
                offs[msg.note] = tick
        seconds = (offs[60] - onsets[60]) / ticks_per_second
        assert len(metas) == 1 and metas[0].tempo == tempo_us, f"set_tempo = {metas}"
        assert abs(seconds - 2.0) < 0.01, f"{tempo_us} us/beat -> note length {seconds:.3f}s"
        print(
            f"writer tempo  : tempo_us={tempo_us} -> set_tempo {metas[0].tempo}, "
            f"{ticks_per_second:.0f} ticks/s, note = {seconds:.3f}s of MIDI time"
        )

    marked = _parse(
        build_midi_from_events(
            {"midi_events": events},
            "guitar",
            hop_length=HOP,
            sample_rate=SR,
            include_confidence_track=True,
        )
    )
    rows = [m.text for m in marked.tracks[1] if m.type == "marker"]
    plain = _parse(
        build_midi_from_events({"midi_events": events}, "guitar", hop_length=HOP, sample_rate=SR)
    )
    assert len(marked.tracks) == 2 and len(plain.tracks) == 1, "marker track is not opt-in"
    print(f"marker track  : {rows}")


def check_tempo() -> None:
    for tempo_us, note_tick in ((500_000, 960), (1_000_000, 960)):
        mid = mido.MidiFile(ticks_per_beat=480)
        meta = mido.MidiTrack()
        mid.tracks.append(meta)
        meta.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
        meta.append(mido.MetaMessage("end_of_track", time=480 * 8))
        notes = mido.MidiTrack()
        mid.tracks.append(notes)
        notes.append(mido.Message("note_on", note=57, velocity=80, time=note_tick))
        notes.append(mido.Message("note_off", note=57, velocity=0, time=480))

        changes = build_tempo_map(mid)
        seconds = tick_to_seconds(note_tick, changes, mid.ticks_per_beat)
        bpm = 60_000_000 / tempo_us

        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as fh:
            path = fh.name
        mid.save(path)
        onset, pitch, _velocity = midi_to_targets(
            path, n_frames=600, hop_length=HOP, sample_rate=SR
        )
        frames = torch.nonzero(onset[0, :, 0]).flatten().tolist()
        expected = seconds * SR / HOP
        print(
            f"tempo map   : tick {note_tick} @{bpm:.0f} BPM = {seconds:.2f}s "
            f"-> frame {expected:.0f}; targets put the onset at {frames} "
            f"(pitch sum {float(pitch.sum()):.0f})"
        )


if __name__ == "__main__":
    check_clock(86, 3000)
    check_polyphony()
    check_confidence()
    check_expression()
    check_durations()
    check_writer_params()
    check_tempo()
