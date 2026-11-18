# SPDX-License-Identifier: Apache-2.0
"""MIDI → training-target alignment tests.

Regression coverage for M-3 (audit): targets were aligned at a hard-coded
120 BPM regardless of the file's ``set_tempo``, and simultaneous notes
overwrote each other because the pitch target was a single index.
"""

import pytest

mido = pytest.importorskip("mido", reason="requires mido")

from utils.midi_targets import build_tempo_map, midi_to_targets, tick_to_seconds  # noqa: E402

SR = 44100
HOP = 512


def _write(tmp_path, tempo_us, notes, duration_ticks=10):
    """notes: list of (absolute_tick, pitch, velocity)."""
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))

    # Absolute events, sorted by tick, with note_off before note_on at equal ticks.
    events = []
    for tick, pitch, velocity in notes:
        events.append((tick, 1, "note_on", pitch, velocity))
        events.append((tick + duration_ticks, 0, "note_off", pitch, 0))
    events.sort()
    last = 0
    for tick, _order, kind, pitch, velocity in events:
        track.append(mido.Message(kind, note=pitch, velocity=velocity, time=tick - last))
        last = tick
    path = tmp_path / "clip.mid"
    mid.save(str(path))
    return str(path)


@pytest.mark.parametrize(
    "tempo_us,expected_frame",
    [(500_000, 86), (1_000_000, 172)],  # beat 2 at 120 BPM / 60 BPM
)
def test_tempo_is_honoured(tmp_path, tempo_us, expected_frame):
    path = _write(tmp_path, tempo_us, [(960, 60, 100)])  # tick 960 = beat 2
    onset, pitch, velocity = midi_to_targets(path, n_frames=400, hop_length=HOP, sample_rate=SR)

    frames = onset[0, :, 0].nonzero().flatten().tolist()
    assert frames == [expected_frame]
    assert pitch[0, expected_frame, 60] == 1.0
    assert velocity[0, expected_frame, 0] == pytest.approx(100 / 127)


def test_chord_is_multi_hot(tmp_path):
    path = _write(tmp_path, 500_000, [(480, 60, 90), (480, 64, 70)])
    onset, pitch, _velocity = midi_to_targets(path, n_frames=400, hop_length=HOP, sample_rate=SR)

    frame = int(onset[0, :, 0].nonzero()[0])
    assert pitch[0, frame, 60] == 1.0
    assert pitch[0, frame, 64] == 1.0
    assert float(pitch[0, frame].sum()) == 2.0


def test_tempo_map_defaults_to_120_bpm(tmp_path):
    path = _write(tmp_path, 500_000, [(0, 60, 100)])
    mid = mido.MidiFile(path)
    tempo_map = build_tempo_map(mid)
    assert tempo_map[0][0] == 0
    assert tick_to_seconds(480, tempo_map, 480) == pytest.approx(0.5)


def test_tempo_change_mid_file(tmp_path):
    """A tempo change must shift later events (old code assumed one tempo)."""
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    track.append(mido.Message("note_on", note=60, velocity=100, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    track.append(mido.MetaMessage("set_tempo", tempo=1_000_000, time=0))  # 60 BPM
    track.append(mido.Message("note_on", note=62, velocity=100, time=480))
    path = tmp_path / "tempo_change.mid"
    mid.save(str(path))

    onset, _pitch, _velocity = midi_to_targets(
        str(path), n_frames=400, hop_length=HOP, sample_rate=SR
    )
    # First note at 0.5 s, second one beat later at the slower tempo:
    # 0.5 s + 1.0 s = 1.5 s.
    frames = onset[0, :, 0].nonzero().flatten().tolist()
    assert frames[0] == 0  # note at tick 0
    assert frames[1] == round(1.5 * SR / HOP)  # 0.5 s at 120 BPM + 1.0 s at 60 BPM
