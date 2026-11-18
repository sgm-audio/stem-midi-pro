# SPDX-License-Identifier: Apache-2.0
"""MIDI *timing* tests — the gap the existing round-trip suite leaves open.

``tests/test_midi_roundtrip.py`` and ``tests/test_midi_generation.py`` assert
event *counts*, CC#127 values, velocity clamping, and that absolute ticks are
"non-decreasing".  None of them assert that a tick lands where the audio
actually is, so a systematic tempo error or a collapsed chord passes silently.

Everything here is pure ``mido`` + numpy-free arithmetic: no torch, no model,
runs in well under a second.  These are the cheapest tests in the suite and
they guard the artifact the product actually sells (a .mid a DAW can import).

Reference geometry (must match utils/midi.py):
    sample_rate   = 44100 Hz
    hop_length    = 512 samples  -> 86.13 frames/sec
    tempo         = 120 BPM      -> 2 beats/sec
    ticks_per_beat= 480          -> 960 ticks/sec
So: ticks(onset_frame) = onset_frame * 512 / 44100 * 960

These tests were written as ``xfail(strict=True)`` while the timing defects
were open (BUG-MIDI-1/2/3/4). The defects are now fixed in ``utils/midi.py`` and
``utils/midi_targets.py``, so the markers are gone and these are ordinary
regression tests: if the 60x tempo error or the chord serialisation returns,
they fail.
"""

import io
import itertools

import pytest

mido = pytest.importorskip("mido", reason="MIDI timing tests require mido")

from utils.midi import build_midi_from_events  # noqa: E402
from utils.midi_targets import midi_to_targets  # noqa: E402

SAMPLE_RATE = 44100
HOP = 512
TICKS_PER_BEAT = 480
TEMPO_US = 500_000
# Derived independently of utils/midi.py so a units bug in production cannot be
# mirrored here (a helper that copies the buggy formula makes the test vacuous —
# this exact mistake was caught by strict xfail on first run).
BPM = 60_000_000 / TEMPO_US  # 120
TICKS_PER_SECOND = TICKS_PER_BEAT * (BPM / 60)  # 480 * 2 = 960
FRAMES_PER_SECOND = SAMPLE_RATE / HOP


def ticks_for_frame(frame: int) -> int:
    """Tick position a note at ``frame`` must occupy in a 120 BPM SMF."""
    return round(frame * HOP / SAMPLE_RATE * TICKS_PER_SECOND)


def frames_for_seconds(seconds: float) -> int:
    return round(seconds * FRAMES_PER_SECOND)


def load(midi_bytes: bytes) -> "mido.MidiFile":  # type: ignore[name-defined]
    return mido.MidiFile(file=io.BytesIO(midi_bytes))


def note_onsets(midi_bytes: bytes) -> list[tuple[int, int]]:
    """[(absolute_tick, note), ...] for every sounding note_on."""
    out: list[tuple[int, int]] = []
    tick = 0
    for msg in load(midi_bytes).tracks[0]:
        tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            out.append((tick, msg.note))
    return out


def note_lengths(midi_bytes: bytes) -> list[int]:
    """Ticks between each note_on and its matching note_off."""
    lengths: list[int] = []
    tick = 0
    on_tick: dict[int, int] = {}
    for msg in load(midi_bytes).tracks[0]:
        tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            on_tick[msg.note] = tick
        elif (msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0)) and (
            msg.note in on_tick
        ):
            lengths.append(tick - on_tick.pop(msg.note))
    return lengths


def events(*frames_and_notes: tuple[int, int], **extra) -> dict:
    return {
        "midi_events": [
            {
                "onset_frame": frame,
                "note": note,
                "velocity": 100,
                "confidence": 0.9,
                **extra,
            }
            for frame, note in frames_and_notes
        ]
    }


# ---------------------------------------------------------------------------
# Absolute timing: does a tick land where the audio is?
# ---------------------------------------------------------------------------


def test_onset_tick_matches_audio_time():
    """A note at frame 100 (~1.16 s) must sit at tick 1115, not 66873.

    This is the single most user-visible MIDI defect: at 60x, a two-second
    phrase is written two *minutes* into the file, past the end of every DAW
    session region the audio came from.
    """
    onsets = note_onsets(build_midi_from_events(events((100, 60)), "guitar"))
    assert onsets == [(ticks_for_frame(100), 60)]


@pytest.mark.parametrize("seconds", [1.0, 2.0, 5.0])
def test_midi_length_matches_audio_length(seconds):
    """A transcription of ``seconds`` of audio must not be longer than that.

    Asserted on the *last* note of a phrase spanning the whole buffer, which
    is the quantity a DAW timeline cares about.
    """
    last_frame = frames_for_seconds(seconds) - 1
    onsets = note_onsets(build_midi_from_events(events((0, 60), (last_frame, 62)), "guitar"))
    assert onsets[-1][0] <= round(seconds * TICKS_PER_SECOND), (
        f"last note at tick {onsets[-1][0]} exceeds the "
        f"{int(seconds * TICKS_PER_SECOND)}-tick span of {seconds}s of audio"
    )


def test_sixteenth_note_spacing_is_one_sixteenth():
    """A 16th-note run at 120 BPM is 120 ticks per step; spacing must be uniform.

    Deliberately independent of BUG-MIDI-1's absolute offset: this fails on the
    *ratio* too, so fixing the constant alone is not enough if the cursor
    arithmetic changes.
    """
    step = 11  # 11 frames ~= 0.128 s ~= a 16th note at 120 BPM
    run = events(*[(i * step, 60 + i) for i in range(8)])
    ticks = [t for t, _ in note_onsets(build_midi_from_events(run, "guitar"))]
    gaps = [b - a for a, b in itertools.pairwise(ticks)]
    expected = ticks_for_frame(step)
    # Ticks are integers, so a 122.6-tick step round-trips as 122 or 123 —
    # allow that quantisation but nothing more (the 60x bug gave 7 356).
    assert all(abs(gap - expected) <= 1 for gap in gaps), f"gaps {gaps} vs expected {expected}"


# ---------------------------------------------------------------------------
# Polyphony: a chord must be a chord
# ---------------------------------------------------------------------------


def test_chord_notes_share_one_tick():
    """Four pitches at one onset_frame must all be written at the same tick.

    Guitar and bass are polyphonic instruments. The current writer advances its
    tick cursor by each note's duration *before* computing the next delta, so a
    4-note chord becomes four sequential 1/16 notes spread over 360 ticks —
    heard as a fast strum, unreadable as harmony.
    """
    chord = events((50, 52), (50, 55), (50, 59), (50, 64))
    onsets = note_onsets(build_midi_from_events(chord, "guitar"))
    assert len(onsets) == 4
    assert {t for t, _ in onsets} == {ticks_for_frame(50)}


def test_chord_after_short_note_keeps_its_position():
    """A chord on the next beat must not be dragged later by the previous note.

    Depending on note order this either clamps to zero (notes collapse onto the
    earlier beat) or drifts (notes land after where they belong) — both are
    wrong; the assertion is simply "one tick per frame of the audio".
    """
    data = {
        "midi_events": [
            {"onset_frame": 0, "note": 64, "velocity": 90, "confidence": 0.9},
            {"onset_frame": 11, "note": 52, "velocity": 90, "confidence": 0.9},
            {"onset_frame": 11, "note": 55, "velocity": 90, "confidence": 0.9},
        ]
    }
    onsets = note_onsets(build_midi_from_events(data, "guitar"))
    ticks = [t for t, _ in onsets]
    assert ticks[1] == ticks[2] == ticks_for_frame(11)


# ---------------------------------------------------------------------------
# Round-trip asymmetry: the project's own writer into its own reader
# ---------------------------------------------------------------------------


def test_own_writer_reads_back_at_the_same_frames(tmp_path):
    """build -> save -> midi_to_targets must recover the onset frames.

    Neither existing test file crosses this boundary: the round-trip tests stop
    at SMF bytes, the target tests start from hand-written mido files. Anything
    trained on targets derived from our own writer is therefore silently
    misaligned with what inference produces.
    """
    source_frames = [0, 43, 86, 172]
    data = events(*[(f, 57) for f in source_frames])
    path = tmp_path / "phrase.mid"
    path.write_bytes(build_midi_from_events(data, "bass"))

    onset, _pitch, _vel = midi_to_targets(
        str(path), n_frames=512, hop_length=HOP, sample_rate=SAMPLE_RATE
    )
    recovered = [int(f) for f in onset[0, :, 0].nonzero().flatten()]
    assert recovered == source_frames


def test_midi_to_targets_honours_file_tempo(tmp_path):
    """A 60 BPM file's tick 480 is 1.0 s -> frame 86, not frame 43.

    Real corpus MIDI (Slakh/MUSDB) is not all at 120 BPM; assuming it is shifts
    every training target by a tempo-dependent ratio, which is undetectable by
    eye in a loss curve.
    """
    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=1_000_000))  # 60 BPM
    track.append(mido.Message("note_on", note=40, velocity=90, time=TICKS_PER_BEAT))  # +1 beat
    track.append(mido.Message("note_off", note=40, velocity=0, time=TICKS_PER_BEAT))
    path = tmp_path / "slow.mid"
    mid.save(str(path))

    onset, _pitch, _vel = midi_to_targets(
        str(path), n_frames=256, hop_length=HOP, sample_rate=SAMPLE_RATE
    )
    hits = onset[0, :, 0].nonzero().flatten().tolist()
    assert hits == [frames_for_seconds(1.0)], f"expected frame 86 for a 1 s offset, got {hits}"


def test_midi_to_targets_keeps_every_chord_pitch(tmp_path):
    """Three simultaneous pitches must all be present in the pitch target.

    Polyphonic transcription trains against these targets. This one already
    holds (pitch is written per-note, not argmaxed) and is kept as a guard:
    it is the *writer* that cannot express a chord, not the target builder.
    """
    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=TEMPO_US))
    for note in (52, 55, 59):
        track.append(mido.Message("note_on", note=note, velocity=90, time=0))
    for note in (52, 55, 59):
        track.append(mido.Message("note_off", note=note, velocity=0, time=TICKS_PER_BEAT))
    path = tmp_path / "chord.mid"
    mid.save(str(path))

    _onset, pitch, _vel = midi_to_targets(
        str(path), n_frames=256, hop_length=HOP, sample_rate=SAMPLE_RATE
    )
    frame = frames_for_seconds(0.0)
    active = pitch[0, frame, :].nonzero().flatten().tolist()
    assert sorted(active) == [52, 55, 59]


# ---------------------------------------------------------------------------
# Regressions that already hold — keep them cheap and keep them here
# ---------------------------------------------------------------------------


def test_empty_event_list_still_parses():
    mid = load(build_midi_from_events({"midi_events": []}, "guitar"))
    assert mid.ticks_per_beat == TICKS_PER_BEAT
    assert [m for m in mid.tracks[0] if m.type == "note_on"] == []


def test_note_count_matches_input_events():
    data = events(*[(i * 43, 60 + (i % 12)) for i in range(10)])
    onsets = note_onsets(build_midi_from_events(data, "guitar"))
    assert len(onsets) == 10


def test_no_zero_velocity_sounding_notes():
    """note_on with velocity 0 is a note_off in MIDI; never emit one as a note."""
    data = events((0, 60), (86, 62), **{"velocity": 0})
    mid = load(build_midi_from_events(data, "guitar"))
    assert all(m.velocity != 0 for m in mid.tracks[0] if m.type == "note_on")


def test_explicit_duration_frames_sets_note_length():
    """duration_frames must scale the written note length, not hit the fallback.

    Asserted as a ratio (86 frames vs 172 frames -> 1:2) plus "not the 120-tick
    default", because the absolute tick value is currently wrong for the same
    reason as BUG-MIDI-1; a 1:2 ratio still proves the event field is honoured.
    """

    def length_for(frames: int) -> int:
        data = {
            "midi_events": [
                {
                    "onset_frame": 0,
                    "note": 60,
                    "velocity": 90,
                    "confidence": 0.9,
                    "duration_frames": frames,
                }
            ]
        }
        (length,) = note_lengths(build_midi_from_events(data, "guitar"))
        return length

    short, long_ = length_for(86), length_for(172)
    assert short != round(0.25 * TICKS_PER_BEAT), "fell back to the 1/16 default"
    assert 1.9 <= long_ / short <= 2.1, f"duration_frames not proportional ({short} vs {long_})"


def test_events_are_emitted_in_the_order_given():
    data = events(*[(i * 43, 60 + i) for i in range(6)])
    notes = [n for _, n in note_onsets(build_midi_from_events(data, "guitar"))]
    assert notes == [60, 61, 62, 63, 64, 65]
