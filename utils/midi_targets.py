# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""MIDI → frame-aligned training targets.

Converts a MIDI file (e.g. Slakh2100 stems/MIDI) into onset/pitch/velocity
target tensors aligned to the model's STFT hop grid.
"""
from __future__ import annotations

from typing import Tuple

import torch

# Default tempo (microseconds per beat) when a MIDI file has no set_tempo event.
DEFAULT_TEMPO_US = 500000  # 120 BPM


def midi_to_targets(
    midi_path: str,
    n_frames: int,
    hop_length: int = 512,
    sample_rate: int = 44100,
    pitch_vocab: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Read a MIDI file and produce frame-aligned target tensors.

    Returns:
        onset: (1, n_frames, 1) — 1.0 where a note_on starts.
        pitch: (1, n_frames, pitch_vocab) — one-hot per onset frame.
        velocity: (1, n_frames, 1) — normalized [0, 1] velocity at onset frames.
    """
    import mido

    mid = mido.MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat or 480
    seconds_per_tick = DEFAULT_TEMPO_US / 1e6 / ticks_per_beat

    onset = torch.zeros(1, n_frames, 1)
    pitch = torch.zeros(1, n_frames, pitch_vocab)
    velocity = torch.zeros(1, n_frames, 1)

    for track in mid.tracks:
        abs_time_s = 0.0
        for msg in track:
            abs_time_s += msg.time * seconds_per_tick
            if msg.type == "note_on" and msg.velocity > 0:
                frame = int(round(abs_time_s * sample_rate / hop_length))
                if 0 <= frame < n_frames:
                    onset[0, frame, 0] = 1.0
                    if 0 <= msg.note < pitch_vocab:
                        pitch[0, frame, msg.note] = 1.0
                    velocity[0, frame, 0] = msg.velocity / 127.0
            elif msg.type == "set_tempo":
                # Recompute timing from this point (rare; not tracked per-tick in v1)
                pass

    return onset, pitch, velocity
