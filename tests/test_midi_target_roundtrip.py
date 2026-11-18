# SPDX-License-Identifier: Apache-2.0
"""Writer ↔ reader round trip.

``utils.midi.build_midi_from_events`` → ``utils.midi_targets.midi_to_targets``.

This is the test the audit asks for first (task 1: "MIDI round-trip
correctness"), because the suite tests the writer and the reader **separately,
each against its own re-derived expectation**:

* ``test_midi_roundtrip.py`` / ``test_midi_generation.py`` check message counts
  and CC values against ``int(confidence * 127)`` — never wall-clock seconds.
* ``test_midi_targets.py`` builds its MIDI with mido directly (correct timing)
  and checks only the reader.

A shared unit error therefore cannot be seen: the writer multiplies ticks by
``ticks_per_beat * 60 / seconds_per_beat`` (60x too big) while the reader divides
by the correct ``ticks_per_beat / seconds_per_beat``, and ``test_midi_targets.py``
never touches the writer, so both halves look right. Feeding one into the other
is what fails — and it is also the only place the ``hop_length`` coupling between
the model's frame grid and the MIDI writer gets exercised, plus the reader's
discarded ``set_tempo`` (every Slakh MIDI that is not 120 BPM produces targets on
the wrong frames, i.e. the model is *trained* onto shifted onsets).
"""

from __future__ import annotations

import io

import pytest

from tests.honesty import known_bug, require

mido = require("mido", "round-trip tests require mido", module_level=True)

from tests.synth import parse_midi  # noqa: E402
from utils.midi import DEFAULT_TEMPO_US, HOP_LENGTH, build_midi_from_events  # noqa: E402
from utils.midi_targets import DEFAULT_TEMPO_US as READER_DEFAULT_TEMPO_US  # noqa: E402
from utils.midi_targets import midi_to_targets  # noqa: E402

SR = 44100
HOP = 512


def _ev(note, onset_frame, velocity=100, confidence=0.9, duration_frames=None):
    e = {"note": note, "onset_frame": onset_frame, "velocity": velocity, "confidence": confidence}
    if duration_frames is not None:
        e["duration_frames"] = duration_frames
    return e


def _roundtrip(events, tmp_path, hop=HOP, sr=SR, n_frames=4000, pitch_vocab=128):
    """Write SMF bytes with ``hop`` as the frame grid, then read targets with ``hop``."""
    midi = build_midi_from_events({"midi_events": events}, "guitar", hop_length=hop, sample_rate=sr)
    path = tmp_path / "roundtrip.mid"
    path.write_bytes(midi)
    onset, pitch, velocity = midi_to_targets(
        str(path), n_frames, hop_length=hop, sample_rate=sr, pitch_vocab=pitch_vocab
    )
    return midi, onset, pitch, velocity


# ---------------------------------------------------------------------------
# writer -> reader: the full circle on frames, pitches, velocities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hop", [512, 128, 256])
def test_roundtrip_recovers_onsets_and_pitches(tmp_path, hop):
    """N events in -> the same N (frame, pitch, velocity) triples out, ±1 frame."""
    specs = [(40, 12), (47, 60), (55, 130), (60, 250), (67, 512)]
    events = [_ev(n, f, duration_frames=20) for n, f in specs]
    _midi, onset, pitch, velocity = _roundtrip(events, tmp_path, hop=hop)

    got = [(int(i), int(pitch[0, i].argmax())) for i in range(onset.shape[1]) if onset[0, i, 0] > 0]
    want = [(f, n) for n, f in specs]
    assert len(got) == len(want), f"expected onsets {want}, got {got}"
    for (gframe, gpitch), (wframe, wnote) in zip(got, want, strict=False):
        assert abs(gframe - wframe) <= 1, f"onset frame {wframe} came back as {gframe} (hop {hop})"
        assert gpitch == wnote, f"pitch at frame {gframe}: {gpitch} != {wnote}"
        assert abs(velocity[0, gframe, 0].item() - 100 / 127.0) < 1e-5


def test_roundtrip_pitch_rows_are_one_hot(tmp_path):
    # frames 315/350/399 -> n_frames must cover them (MIDI-5 keeps the writer on
    # its own 512/44100 defaults, which is what _roundtrip asks the reader for)
    events = [_ev(n, n * 7, duration_frames=9) for n in (45, 50, 57)]
    _midi, onset, pitch, _v = _roundtrip(events, tmp_path, n_frames=500)
    rows = [i for i in range(onset.shape[1]) if onset[0, i, 0] > 0]
    assert rows, "no onsets recovered at all"
    for i in rows:
        assert float(pitch[0, i].sum()) == 1.0, f"frame {i} pitch mass {float(pitch[0, i].sum())}"


def test_roundtrip_duration_matches_requested_frames(tmp_path):
    for dur in (5, 40, 200):
        events = [_ev(60, 0, duration_frames=dur)]
        midi, *_ = _roundtrip(events, tmp_path)
        (note,) = parse_midi(midi)["notes"]
        expected = dur * HOP / SR
        assert abs(note.duration_sec - expected) <= 2 * (
            HOP / SR
        ), f"duration_frames={dur} -> {note.duration_sec:.4f} s, expected {expected:.4f} s"


# ---------------------------------------------------------------------------
# unit coupling: hop / sample rate / tempo must be shared, not defaulted
# ---------------------------------------------------------------------------


def test_non_default_hop_changes_note_time_one_to_one():
    """Documents the coupling: the writer's hop_length scales note seconds 1:1.

    Useful as a canary: if someone "fixes" the writer by ignoring hop_length, this
    fails and the pipeline test (MIDI-5) explains why ignoring it is wrong.
    """
    ev = [_ev(60, 100)]
    a = parse_midi(build_midi_from_events({"midi_events": ev}, "guitar", hop_length=512))["notes"][
        0
    ]
    b = parse_midi(build_midi_from_events({"midi_events": ev}, "guitar", hop_length=128))["notes"][
        0
    ]
    assert abs(a.on_sec / b.on_sec - 4.0) < 0.01, (a.on_sec, b.on_sec)


def test_api_midi_writer_is_given_the_models_frame_grid():
    """The API must pass the model's hop/sample-rate into the MIDI writer."""
    import inspect

    require("fastapi", "api.py import")
    require("soundfile", "api.py import")
    import api

    src = inspect.getsource(api.create_response_zip)
    assert "hop_length" in src and "sample_rate" in src, (
        "create_response_zip builds MIDI with utils.midi's default 512/44100 grid "
        "instead of the loaded model's audio config"
    )


def test_writer_and_reader_agree_on_default_tempo():
    """Both halves hard-code 120 BPM; if they ever diverge, timing is garbage."""
    assert DEFAULT_TEMPO_US == READER_DEFAULT_TEMPO_US == 500_000
    assert HOP_LENGTH == 512


# ---------------------------------------------------------------------------
# reader: tempo-map handling (independent of the writer's scaling bug)
# ---------------------------------------------------------------------------


def _write_midi(path, tempo_us, notes, tpb=480, extra_tempo_events=()):
    mid = mido.MidiFile(ticks_per_beat=tpb)
    trk = mido.MidiTrack()
    mid.tracks.append(trk)
    trk.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
    for kwargs in extra_tempo_events:
        trk.append(mido.MetaMessage("set_tempo", **kwargs))
    last = 0
    for tick, note, vel in notes:
        trk.append(mido.Message("note_on", note=note, velocity=vel, time=tick - last))
        last = tick
        trk.append(mido.Message("note_off", note=note, velocity=0, time=240))
    mid.save(str(path))
    return str(path)


def test_reader_handles_120bpm_exactly(tmp_path):
    """Baseline for the reader at the one tempo it assumes."""
    p = _write_midi(tmp_path / "a.mid", 500_000, [(480, 60, 100)])  # 1 beat = 0.5 s
    onset, pitch, _v = midi_to_targets(p, 100, hop_length=HOP, sample_rate=SR)
    f = int(onset[0].argmax())
    assert onset[0, f, 0] == 1.0
    assert f == round(0.5 * SR / HOP)
    assert int(pitch[0, f].argmax()) == 60


@pytest.mark.parametrize(
    "tempo_us,beats",
    [(600_000, 1), (428_571, 1), (750_000, 2)],
    ids=["100bpm", "140bpm", "80bpm-2beats"],  # 120 BPM is the passing baseline above
)
def test_reader_honours_the_files_tempo(tmp_path, tempo_us, beats):
    """Regression guard for MIDI-6 (fixed by PR #11): the parsed set_tempo is applied.

    `utils/midi_targets.py` used to read `set_tempo` and then convert ticks with the
    120-BPM default, stamping every non-120-BPM source MIDI (i.e. all of Slakh) onto
    the wrong frames — silent training-label corruption that no count-based test could
    see. A tempo *change* mid-file is still ignored; that is MIDI-7 and stays an xfail.
    """
    seconds = beats * (tempo_us / 1e6)  # beats x microseconds-per-beat
    p = _write_midi(tmp_path / f"t{tempo_us}_{beats}.mid", tempo_us, [(480 * beats, 61, 100)])
    onset, _pitch, _v = midi_to_targets(p, 200, hop_length=HOP, sample_rate=SR)
    expected_frame = round(seconds * SR / HOP)
    got_frames = [i for i in range(200) if onset[0, i, 0] > 0]
    assert got_frames, f"tempo {tempo_us}: note dropped entirely"
    assert abs(got_frames[0] - expected_frame) <= 1, (
        f"{tempo_us} us/beat, {beats} beat(s) = {seconds:.3f} s -> frame {expected_frame}; "
        f"reader produced frame {got_frames[0]} (it assumed 120 BPM)"
    )


@known_bug(
    "MIDI-7",
    "utils/midi_targets.py reads ONE tempo (DEFAULT_TEMPO_US / the file's first set_tempo, "
    "whichever it picks) and converts every message's tick offset with it, so a tempo MAP in "
    "the middle of a file is ignored: measured on a 480-PPQ file with 120 BPM at tick 0, "
    "60 BPM from tick 480 and a note at tick 1440, midi_to_targets reports frame 129 (=1.500 s, "
    "the 120 BPM answer) where the correct onset is 2.500 s (frame 215) — a 1-second drift on a "
    "60 BPM ballad movement. Anything exported from a DAW with a tempo change lands off-grid. "
    "Accumulate time per track while applying each set_tempo at its own tick.",
)
def test_reader_handles_a_tempo_change_mid_file(tmp_path):
    p = _write_midi(
        tmp_path / "map.mid",
        500_000,
        [(240, 60, 100), (1440, 62, 100)],
        extra_tempo_events=[{"time": 480, "tempo": 1_000_000}],  # 60 BPM from beat 1
    )
    onset, _p, _v = midi_to_targets(p, 300, hop_length=HOP, sample_rate=SR)
    frames = [i for i in range(300) if onset[0, i, 0] > 0]
    assert len(frames) == 2, frames
    # note 1: half a beat at 120 BPM = 0.25 s (frame 22). note 2: 0.25 s + 2 beats
    # at 60 BPM = 2.500 s (frame 215) — the reader must apply the tempo map per tick.
    assert abs(frames[0] - round(0.25 * SR / HOP)) <= 1, f"first note at frame {frames[0]}"
    assert abs(frames[1] - round(2.50 * SR / HOP)) <= 1, (
        f"second note at frame {frames[1]}; 2.500 s is frame {round(2.50 * SR / HOP)} — "
        f"frame {frames[1]} is where it lands if the 60 BPM change is ignored"
    )


def test_reader_marks_velocity_only_on_onset_frames(tmp_path):
    """Non-onset frames must carry zero velocity, or dynamics training is noise."""
    p = _write_midi(tmp_path / "vel.mid", 500_000, [(480, 58, 111)])
    onset, _pitch, velocity = midi_to_targets(p, 60, hop_length=HOP, sample_rate=SR)
    f = int(onset[0].argmax())
    assert abs(velocity[0, f, 0].item() - 111 / 127.0) < 1e-5
    leak = velocity.clone()
    leak[0, f, 0] = 0
    assert float(leak.abs().sum()) == 0.0, "velocity leaked to non-onset frames"


def test_reader_ignores_note_off_and_zero_velocity_notes(tmp_path):
    """Velocity-0 'note_on' IS a note-off in MIDI: it must not create an onset."""
    mid = mido.MidiFile(ticks_per_beat=480)
    trk = mido.MidiTrack()
    mid.tracks.append(trk)
    trk.append(mido.MetaMessage("set_tempo", tempo=500_000))
    trk.append(mido.Message("note_on", note=60, velocity=0, time=240))
    trk.append(mido.Message("note_off", note=61, velocity=0, time=240))
    p = tmp_path / "offs.mid"
    mid.save(str(p))
    onset, _pitch, _v = midi_to_targets(str(p), 40, hop_length=HOP, sample_rate=SR)
    assert float(onset.sum()) == 0.0


def test_reader_collects_notes_from_all_tracks(tmp_path):
    """Multi-track MIDI (guitar + bass + drums) contributes onsets from every track."""
    mid = mido.MidiFile(ticks_per_beat=480)
    for i, note in enumerate((40, 52, 64)):
        trk = mido.MidiTrack()
        mid.tracks.append(trk)
        trk.append(mido.MetaMessage("set_tempo", tempo=500_000))
        trk.append(mido.Message("note_on", note=note, velocity=90 + i, time=480 * (i + 1)))
        trk.append(mido.Message("note_off", note=note, velocity=0, time=240))
    p = tmp_path / "multi.mid"
    mid.save(str(p))
    onset, _pitch, _v = midi_to_targets(str(p), 100, hop_length=HOP, sample_rate=SR)
    assert float(onset.sum()) >= 2, "multi-track MIDI lost whole tracks"


def test_reader_skips_out_of_vocab_pitches_but_keeps_the_onset(tmp_path):
    """A note above pitch_vocab must not be silently written into a valid bin."""
    p = _write_midi(tmp_path / "hi.mid", 500_000, [(0, 120, 100)])
    onset, pitch, _v = midi_to_targets(p, 10, hop_length=HOP, sample_rate=SR, pitch_vocab=100)
    assert float(onset.sum()) == 1.0
    assert float(pitch.sum()) == 0.0, "out-of-vocab pitch leaked into a valid bin"


def test_reader_drops_notes_beyond_the_window(tmp_path):
    """Notes past n_frames must be dropped, not wrapped by negative indexing."""
    p = _write_midi(tmp_path / "late.mid", 500_000, [(480 * 200, 60, 100)])
    onset, _pitch, _v = midi_to_targets(p, 5, hop_length=HOP, sample_rate=SR)
    assert float(onset.sum()) == 0.0


def test_writer_bytes_are_parseable_by_a_third_party_reader(tmp_path):
    """mido must read the file from disk, not just the buffer it wrote."""
    ev = [_ev(n, n * 13, duration_frames=25) for n in (45, 50, 57)]
    midi = build_midi_from_events({"midi_events": ev}, "bass")
    p = tmp_path / "f.mid"
    p.write_bytes(midi)
    mid = mido.MidiFile(str(p))
    assert len(mid.tracks) >= 1
    parsed = parse_midi(io.BytesIO(p.read_bytes()))
    assert len(parsed["notes"]) == 3
    assert parsed["dangling_notes"] == {}
