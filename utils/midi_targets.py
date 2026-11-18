# SPDX-License-Identifier: Apache-2.0
"""MIDI → frame-aligned training targets.

Converts MIDI files (e.g. the per-stem MIDI shipped with Slakh2100) into
onset/pitch/velocity target tensors aligned to the model's STFT hop grid.

This is the real-label path for training (C-4): the dataset loaders call it
with whatever MIDI the track directory provides. Tempo is honored — a tempo
map is built from all ``set_tempo`` events and absolute seconds are integrated
per tick (the previous version hard-assumed 120 BPM and ignored tempo
changes, misaligning every target when a track carried a tempo event).
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch

# Default tempo (microseconds per beat) when a MIDI file has no set_tempo event.
DEFAULT_TEMPO_US = 500000  # 120 BPM


def build_tempo_map(mid) -> list[tuple[int, int]]:
    """Collect (abs_tick, tempo_us) tempo changes from all tracks, sorted."""
    changes: list[tuple[int, int]] = []
    for track in mid.tracks:
        t = 0
        for msg in track:
            t += msg.time
            if msg.type == "set_tempo":
                changes.append((t, msg.tempo))
    changes.sort(key=lambda x: x[0])
    if not changes or changes[0][0] > 0:
        changes.insert(0, (0, DEFAULT_TEMPO_US))
    return changes


def tick_to_seconds(tick: int, changes: Sequence[tuple[int, int]], ticks_per_beat: int) -> float:
    """Integrate the tempo map up to ``tick`` (tempo applies from its tick on)."""
    seconds = 0.0
    prev_tick, prev_tempo = changes[0]
    for t, tempo in changes[1:]:
        if tick <= t:
            break
        seconds += (t - prev_tick) * prev_tempo / 1e6 / ticks_per_beat
        prev_tick, prev_tempo = t, tempo
    seconds += (tick - prev_tick) * prev_tempo / 1e6 / ticks_per_beat
    return seconds


def midi_to_targets(
    midi_paths: str | Sequence[str],
    n_frames: int,
    hop_length: int = 512,
    sample_rate: int = 44100,
    pitch_vocab: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Read MIDI file(s) and produce frame-aligned target tensors.

    Args:
        midi_paths: one MIDI path or a list (all instruments merged — e.g. the
            per-stem ``S##.mid`` files of a Slakh track) into a polyphonic
            target grid.
        n_frames: number of frames to produce.
        hop_length / sample_rate: the model's frame grid.

    Returns:
        onset: (1, n_frames, 1) — 1.0 where a note_on starts.
        pitch: (1, n_frames, pitch_vocab) — **multi-hot**: every note starting on
            a frame is marked (simultaneous notes no longer overwrite each
            other; H-5). The model's pitch head is still single-label, so a
            polyphonic head must be added before this target can be exploited.
        velocity: (1, n_frames, 1) — normalized [0, 1] velocity at onset frames.
    """
    import mido

    if isinstance(midi_paths, os.PathLike):
        midi_paths = [os.fspath(midi_paths)]
    elif isinstance(midi_paths, str):
        midi_paths = [midi_paths]

    onset = torch.zeros(1, n_frames, 1)
    pitch = torch.zeros(1, n_frames, pitch_vocab)
    velocity = torch.zeros(1, n_frames, 1)

    for path in midi_paths:
        mid = mido.MidiFile(path)
        ticks_per_beat = mid.ticks_per_beat or 480
        changes = build_tempo_map(mid)

        for track in mid.tracks:
            abs_tick = 0
            for msg in track:
                abs_tick += msg.time
                if msg.type == "note_on" and msg.velocity > 0:
                    abs_time_s = tick_to_seconds(abs_tick, changes, ticks_per_beat)
                    frame = round(abs_time_s * sample_rate / hop_length)
                    if 0 <= frame < n_frames:
                        onset[0, frame, 0] = 1.0
                        if 0 <= msg.note < pitch_vocab:
                            pitch[0, frame, msg.note] = 1.0
                        # Keep the loudest note on a shared frame.
                        velocity[0, frame, 0] = max(
                            float(velocity[0, frame, 0]), msg.velocity / 127.0
                        )

    return onset, pitch, velocity


# Back-compat aliases: earlier code (and some call sites) used the private
# names with the same signatures.
# Back-compat aliases for external callers that used the private spellings while
# main's version had them (no in-tree caller does; the public names above are the
# definitions, so these must stay *after* them).
_build_tempo_map = build_tempo_map
_tick_to_seconds = tick_to_seconds
