# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Tests for MIDI → frame-aligned training targets (real label pipeline)."""
from __future__ import annotations

import mido
import pytest
import torch

from utils.midi_targets import midi_to_targets


def _make_midi(path, note=60, velocity=100, ticks_per_beat=480, note_ticks=480):
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000))  # 120 BPM
    track.append(mido.Message("note_on", note=note, velocity=velocity, time=note_ticks))
    track.append(mido.Message("note_off", note=note, velocity=0, time=480))
    mid.save(path)
    return path


def test_midi_to_targets_marks_onset_pitch_velocity(tmp_path):
    """A single note_on should produce one onset + correct pitch + velocity."""
    p = _make_midi(tmp_path / "one_note.mid", note=60, velocity=100, note_ticks=480)
    hop = 512
    sr = 44100
    n_frames = 200

    onset, pitch, velocity = midi_to_targets(p, n_frames, hop_length=hop, sample_rate=sr)

    assert onset.shape == (1, n_frames, 1)
    assert pitch.shape == (1, n_frames, 128)
    assert velocity.shape == (1, n_frames, 1)

    # exactly one onset frame
    assert int(onset.sum().item()) == 1
    frame = int(onset.squeeze(-1).squeeze(0).argmax().item())
    # 480 ticks @ 120 BPM = 0.5 s → frame = round(0.5 * 44100 / 512)
    assert frame == round(0.5 * sr / hop)

    assert pitch[0, frame, 60].item() == 1.0
    assert abs(velocity[0, frame, 0].item() - 100 / 127.0) < 1e-5


def test_midi_to_targets_ignores_note_off(tmp_path):
    """note_off (velocity 0) must not create an onset."""
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000))
    track.append(mido.Message("note_off", note=60, velocity=0, time=0))
    p = tmp_path / "off_only.mid"
    mid.save(p)

    onset, _, _ = midi_to_targets(p, 100)
    assert int(onset.sum().item()) == 0


def test_midi_to_targets_respects_pitch_vocab(tmp_path):
    """Notes >= pitch_vocab are skipped for pitch, but onset still recorded."""
    p = _make_midi(tmp_path / "clip.mid", note=110, velocity=90, note_ticks=0)
    onset, pitch, velocity = midi_to_targets(p, 50, pitch_vocab=100)
    # onset at frame 0 still recorded
    assert onset[0, 0, 0].item() == 1.0
    # note 110 >= vocab 100 → no pitch bin set
    assert int(pitch.sum().item()) == 0
    assert abs(velocity[0, 0, 0].item() - 90 / 127.0) < 1e-5
