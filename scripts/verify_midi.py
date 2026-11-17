#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the MIDI claims in FIXES.md (C-3, M-3).

Run: ``python3 scripts/verify_midi.py``

Checks the writer and the target builder against the audit's reproductions:

* 3000 frames (~34.8 s at 44.1 kHz / hop 512) land at 34.8 s of MIDI time, not 60x that,
* simultaneous onsets stay simultaneous,
* the default confidence controller is CC#16 (CC#120-127 are Channel Mode),
  CC#127 only via ``confidence_cc=127``, per-note values in the marker track,
* ``duration_frames`` sets note length,
* tempo maps put beat 2 at frame 86 @120 BPM and 172 @60 BPM.
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


def check_controller() -> None:
    events = [
        {"note": 60, "onset_frame": 0, "duration_frames": 20, "velocity": 90, "confidence": 0.9},
        {"note": 64, "onset_frame": 0, "duration_frames": 20, "velocity": 90, "confidence": 0.4},
    ]
    default_mid = _parse(build_midi_from_events({"midi_events": events}, "guitar"))
    controls = sorted({m.control for m in default_mid.tracks[0] if m.type == "control_change"})
    markers = [m.text for track in default_mid.tracks for m in track if m.type == "marker"]
    print(f"confidence  : default controls {controls}, marker track {markers}")

    legacy = _parse(build_midi_from_events({"midi_events": events}, "guitar", confidence_cc=127))
    legacy_controls = sorted({m.control for m in legacy.tracks[0] if m.type == "control_change"})
    print(f"legacy CC   : confidence_cc=127 -> controls {legacy_controls}")

    try:
        build_midi_from_events({"midi_events": events}, "guitar", confidence_cc=121)
    except ValueError as exc:
        print(f"channel mode: CC#121 rejected -> {exc}")


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
            tempo_us=500_000,
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
    check_controller()
    check_durations()
    check_tempo()
