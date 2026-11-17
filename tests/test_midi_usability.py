# SPDX-License-Identifier: Apache-2.0
"""What the user actually downloads: the guitar.mid / bass.mid pair.

The existing MIDI tests stop at ``build_midi_from_events`` with hand-written
event dicts. They therefore never see the two artefacts the product ships per
job, and never ask whether those two files are different, whether the bass file
contains bass notes, or whether the note density is playable.

Every test here pins the transcriber to an explicit logit pattern instead of
relying on random initialisation, so the assertions are about the *pipeline*.
"""

import io

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires torch")
mido = pytest.importorskip("mido", reason="requires mido")
pytest.importorskip("soundfile")

import soundfile as sf  # noqa: E402

from utils.midi import build_midi_from_events  # noqa: E402

SR = 44100
GUITAR_HIGH_NOTE = 84  # C6 — well above any bass part
BASS_NOTE = 40  # E2 — inside the bass register
BASS_RANGE = range(24, 68)  # E1..G4; a bass stem should live here


class _RegisterSeparator(torch.nn.Module):
    """Emits a fixed 220 Hz "guitar" stem and a 55 Hz "bass" stem.

    Known registers make the bass assertions deterministic: a register-aware
    transcriber can only produce a bass pitch if it was handed the *bass* stem,
    which is exactly what "bass.mid was transcribed from the bass stem" means.
    """

    def __init__(self, sr: int = SR):
        super().__init__()
        self.sr = sr

    def forward(self, audio):
        b, _c, t = audio.shape
        tt = torch.arange(t, dtype=torch.float32) / self.sr
        guitar = (0.3 * torch.sin(2 * torch.pi * 220.0 * tt)).expand(b, 1, t)
        bass = (0.3 * torch.sin(2 * torch.pi * 55.0 * tt)).expand(b, 1, t)
        metrics = torch.tensor([[21.0, 0.9]])
        return guitar, bass, torch.zeros_like(guitar), None, metrics


class _RegisterTranscriber(torch.nn.Module):
    """Eager onsets on every frame; pitch chosen by the stem's zero-crossing rate.

    220 Hz -> zcr ~0.010, 55 Hz -> zcr ~0.0025 (zcr = 2f/sr), so the pitch
    follows the register of whatever stem this is called on.
    """

    def __init__(self, eager: bool = True):
        super().__init__()
        self.eager = eager

    def forward(self, stem):
        b, _c, t = stem.shape
        frames = max(1, t // 512)
        zcr = float(((stem[:, 0, 1:] * stem[:, 0, :-1]) < 0).float().mean()) if t > 1 else 0.0
        pitch = GUITAR_HIGH_NOTE if zcr > 0.005 else BASS_NOTE
        onset = torch.full((b, frames, 1), 4.0 if self.eager else -4.0)
        logits = torch.full((b, frames, 128), -10.0)
        logits[..., pitch] = 10.0
        return (
            onset,
            logits,
            torch.ones(b, frames, 1),
            torch.zeros(b, frames, 3),
            torch.full((b, frames), 0.99),
        )


def _make_model(mock_config, eager: bool = True):
    """StemMidiModel with stub components whose output is fully determined."""
    from main import StemMidiModel

    model = StemMidiModel(mock_config).eval()
    model.separator = _RegisterSeparator()
    model.transcriber = _RegisterTranscriber(eager=eager)
    return model


@pytest.fixture
def model_with_guitar_notes(mock_config):
    return _make_model(mock_config)


def _audio(tmp_path, seconds: float = 1.0) -> str:
    t = np.arange(int(seconds * SR)) / SR
    path = tmp_path / f"tone_{seconds}.wav"
    sf.write(str(path), (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), SR)
    return str(path)


def _onsets(midi_bytes: bytes) -> list[tuple[int, int]]:
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
    out, tick = [], 0
    for msg in mid.tracks[0]:
        tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            out.append((tick, msg.note))
    return out


# ---------------------------------------------------------------------------
# The guitar/bass pair
# ---------------------------------------------------------------------------


def test_guitar_and_bass_midi_are_different_files(model_with_guitar_notes, tmp_path):
    """Two stems must produce two different *note sequences*.

    Note the trap this test avoids: comparing the SMF *bytes* is not enough,
    because ``track_name``/``instrument_name`` differ per stem even when the
    note content is a copy. The comparison is therefore on (note, onset) pairs.
    """
    result = model_with_guitar_notes.process_audio_file(_audio(tmp_path))
    guitar = _onsets(build_midi_from_events(result["midi_guitar"], "guitar"))
    bass = _onsets(build_midi_from_events(result["midi_bass"], "bass"))
    assert guitar != bass, (
        f"bass.mid repeats the guitar transcription exactly "
        f"({len(guitar)} identical note events)"
    )


def test_bass_midi_contains_no_guitar_range_notes(model_with_guitar_notes, tmp_path):
    """A bass part must not contain C6 (MIDI 84).

    The transcriber is pinned to emit pitch 84. If the bass file contains 84,
    the "bass" MIDI is provably a copy of the guitar one.
    """
    result = model_with_guitar_notes.process_audio_file(_audio(tmp_path))
    bass_notes = {note for _t, note in _onsets(build_midi_from_events(result["midi_bass"], "bass"))}
    assert bass_notes, "no bass notes at all"
    assert bass_notes <= set(BASS_RANGE), f"bass.mid contains non-bass pitches {sorted(bass_notes)}"


# ---------------------------------------------------------------------------
# Playability
# ---------------------------------------------------------------------------


def test_note_density_is_musical(model_with_guitar_notes, tmp_path):
    """One second of monophonic audio must not yield dozens of notes.

    87 notes/second (one per 11.6 ms frame) is not playable, not editable, and
    not what the audio contains. A real transcriber needs an onset refractory
    window; nothing in ``_logits_to_midi`` provides one.
    """
    result = model_with_guitar_notes.process_audio_file(_audio(tmp_path, 1.0))
    events = result["midi"]["midi_events"]
    assert len(events) / 1.0 <= 24, f"{len(events)} note events in 1 s of monophonic audio"


def test_no_duplicate_notes_on_the_same_frame(model_with_guitar_notes, tmp_path):
    """A frame may contribute at most one note per pitch (already holds)."""
    result = model_with_guitar_notes.process_audio_file(_audio(tmp_path, 1.0))
    keys = [(e["onset_frame"], e["note"]) for e in result["midi"]["midi_events"]]
    assert len(keys) == len(set(keys))


def test_every_event_is_a_playable_note(model_with_guitar_notes, tmp_path):
    """Velocity 1..127, pitch 0..127, confidence 0..1, ordered onsets."""
    result = model_with_guitar_notes.process_audio_file(_audio(tmp_path, 1.0))
    events = result["midi"]["midi_events"]
    assert events, "no events produced for a steady 220 Hz tone"
    for ev in events:
        assert 0 <= ev["note"] <= 127
        assert 1 <= ev["velocity"] <= 127, f"velocity {ev['velocity']} would sound as a note-off"
        assert 0.0 <= ev["confidence"] <= 1.0
        assert "needs_review" in ev
    frames = [e["onset_frame"] for e in events]
    assert frames == sorted(frames)


def test_summary_counts_agree_with_events(model_with_guitar_notes, tmp_path):
    """``summary`` must not contradict the per-event flags it summarises.

    ``needs_review`` is derived from ``alignment_score`` while
    ``low_confidence_count`` is derived from raw confidence, so the report can
    say "0 notes need review" next to a MIDI where every note is flagged.
    """
    result = model_with_guitar_notes.process_audio_file(_audio(tmp_path, 1.0))
    events, summary = result["midi"]["midi_events"], result["midi"]["summary"]
    flagged = sum(1 for e in events if e.get("needs_review"))
    assert summary["low_confidence_count"] == flagged, (
        f"summary says {summary['low_confidence_count']} low-confidence notes, "
        f"but {flagged}/{len(events)} events carry needs_review=True"
    )


def test_summary_exposes_total_note_count(model_with_guitar_notes, tmp_path):
    """The report a client renders needs the note count; it is absent."""
    result = model_with_guitar_notes.process_audio_file(_audio(tmp_path, 1.0))
    assert result["midi"]["summary"]["total_notes"] == len(result["midi"]["midi_events"])


# ---------------------------------------------------------------------------
# Streaming path
# ---------------------------------------------------------------------------


def test_streaming_notes_are_not_all_pitch_zero(mock_config, tmp_path):
    """Streamed MIDI must contain real pitches, not 128 bins of silence.

    ``process_audio_streaming`` discards the transcriber's pitch output and
    substitutes ``torch.zeros(...)``; argmax over zeros is pitch 0, velocity 0
    (clamped to 1). Every note in a streamed transcription is literally the
    lowest MIDI note.
    """
    model = _make_model(mock_config)
    result = model.process_audio_streaming(_audio(tmp_path, 2.0), chunk_seconds=1.0)
    notes = {e["note"] for e in result["midi"]["midi_events"]}
    assert notes, "streaming produced no notes"
    assert notes != {0}, "every streamed note has pitch 0 (C-1)"


def test_streaming_reports_chunk_count_and_sample_rate(mock_config, tmp_path):
    model = _make_model(mock_config, eager=False)
    result = model.process_audio_streaming(_audio(tmp_path, 2.0), chunk_seconds=1.0)
    assert result["sample_rate"] == SR
    assert result["chunks_processed"] >= 1
    assert set(result["stems"]) == {"guitar", "bass"}


def test_streaming_report_uses_real_chunk_metrics(mock_config, tmp_path):
    """The streaming report must carry the separator's metrics, not zeros.

    ``process_audio_streaming`` used to hard-code ``final_metrics = [0.0, 0.0]``,
    so a streamed job could never be routed on its actual quality.
    """
    model = _make_model(mock_config)
    result = model.process_audio_streaming(_audio(tmp_path, 1.0), chunk_seconds=1.0)
    # _RegisterSeparator reports si_sdr=21.0 / phase=0.9 for every chunk.
    assert result["report"].si_sdr == pytest.approx(21.0)
    assert result["report"].phase_coherence == pytest.approx(0.9)
