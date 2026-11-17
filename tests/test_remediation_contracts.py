# SPDX-License-Identifier: Apache-2.0
"""Focused regressions for the verification-matrix remediation.

These tests intentionally cover contracts rather than model quality. A real
trained checkpoint and held-out audio are still required before making quality
or production-performance claims.
"""

from __future__ import annotations

import io

import pytest

mido = pytest.importorskip("mido")
torch = pytest.importorskip("torch")

from models.confidence_injector import ConfidenceInjector  # noqa: E402
from utils.midi import build_midi_from_events  # noqa: E402
from utils.quality_gates import ProcessingReport, QualityTier  # noqa: E402


def _absolute_messages(mid):
    tick = 0
    rows = []
    for message in mid.tracks[0]:
        tick += message.time
        rows.append((tick, message))
    return rows


def test_midi_uses_audio_frame_timing_not_minute_scaling():
    """One second of audio is approximately 960 ticks at 120 BPM/480 PPQ."""
    events = [
        {
            "note": 60,
            "onset_frame": 86,  # approximately one second at 44.1 kHz / 512 hop
            "duration_frames": 86,
            "velocity": 100,
            "confidence": 0.9,
        }
    ]
    mid = mido.MidiFile(file=io.BytesIO(build_midi_from_events({"midi_events": events}, "guitar")))
    rows = _absolute_messages(mid)
    note_on_tick = next(t for t, msg in rows if msg.type == "note_on")
    note_off_tick = next(t for t, msg in rows if msg.type == "note_off")

    assert 900 <= note_on_tick <= 1_020
    assert 900 <= note_off_tick - note_on_tick <= 1_020
    assert note_on_tick < 2_000  # catches the former 60x timing error


def test_midi_preserves_same_onset_chords_and_overlaps():
    """All note-ons share their onset and a later note starts before an earlier off."""
    events = [
        {"note": 60, "onset_frame": 0, "duration_frames": 100, "velocity": 90},
        {"note": 64, "onset_frame": 0, "duration_frames": 100, "velocity": 90},
        {"note": 67, "onset_frame": 50, "duration_frames": 100, "velocity": 90},
    ]
    mid = mido.MidiFile(file=io.BytesIO(build_midi_from_events({"midi_events": events}, "guitar")))
    rows = _absolute_messages(mid)
    note_ons = [(tick, msg.note) for tick, msg in rows if msg.type == "note_on"]
    note_offs = [(tick, msg.note) for tick, msg in rows if msg.type == "note_off"]

    assert [note for tick, note in note_ons[:2]] == [60, 64]
    assert note_ons[0][0] == note_ons[1][0] == 0
    assert next(t for t, note in note_ons if note == 67) < next(
        t for t, note in note_offs if note == 60
    )


def test_expression_is_mapped_to_standard_midi_messages():
    """Bend, vibrato, and slide fields are emitted as pitch-wheel/CC messages."""
    event = {
        "note": 60,
        "onset_frame": 0,
        "duration_frames": 20,
        "velocity": 100,
        "confidence": 0.9,
        "expression": {"bend": 0.75, "vibrato": 1.0, "slide": 0.0},
    }
    mid = mido.MidiFile(file=io.BytesIO(build_midi_from_events({"midi_events": [event]}, "guitar")))
    messages = list(mid.tracks[0])

    bend = next(msg for msg in messages if msg.type == "pitchwheel")
    ccs = {msg.control: msg.value for msg in messages if msg.type == "control_change"}
    assert bend.pitch == 4096
    assert ccs[1] == 127
    assert ccs[74] == 0


def test_model_expression_fields_reach_midi_serializer(mock_config):
    """The model decoder and SMF writer form one mapped expression path."""
    from main import StemMidiModel

    model = StemMidiModel(mock_config)
    events = model._logits_to_midi(
        torch.tensor([[[10.0]]]),
        torch.nn.functional.one_hot(torch.tensor([[60]]), num_classes=128).float(),
        torch.tensor([[[0.8]]]),
        torch.tensor([[0.9]]),
        torch.tensor([[[0.75, 1.0, 0.0]]]),
    )
    assert events[0]["expression"] == {"bend": 0.75, "vibrato": 1.0, "slide": 0.0}
    mid = mido.MidiFile(
        file=io.BytesIO(build_midi_from_events({"midi_events": events}, "guitar"))
    )
    assert any(message.type == "pitchwheel" for message in mid.tracks[0])
    assert {message.control for message in mid.tracks[0] if message.type == "control_change"} == {
        1,
        74,
    }


def test_confidence_count_is_limited_to_emitted_notes():
    """Frame confidence arrays must not inflate the note count in the report."""
    events = [{"note": 60, "onset_frame": 2}, {"note": 64, "onset_frame": 80}]
    confidence = torch.full((100,), 0.1)
    confidence[2] = 0.9
    confidence[80] = 0.7
    energy = torch.ones(100)
    result = ConfidenceInjector(min_confidence=0.6).inject_midi_metadata(
        events, confidence, energy
    )

    assert len(result["midi_events"]) == 2
    assert result["summary"]["low_confidence_count"] == 0
    assert result["summary"]["confidence_encoding"] == "polyphonic_key_pressure"


def test_checkpoint_loader_accepts_model_state_dict_and_marks_weights(tmp_path, mock_config):
    """Training-loop checkpoints become eligible only after state_dict load."""
    import yaml

    from main import StemMidiModel, load_from_checkpoint

    config = dict(mock_config)
    config.pop("backends", None)
    config.pop("allow_untrained_inference", None)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    source = StemMidiModel(config)
    checkpoint_path = tmp_path / "weights.pt"
    torch.save({"model_state_dict": source.state_dict(), "epoch": 1}, checkpoint_path)

    loaded = load_from_checkpoint(str(checkpoint_path), str(config_path))
    assert loaded.weights_loaded is True
    assert not loaded._allow_untrained_inference


def test_untrained_model_report_is_not_studio_certified(mock_config):
    """A direct test forward labels random weights and forces manual review."""
    from main import StemMidiModel

    model = StemMidiModel(mock_config)
    with torch.inference_mode():
        output = model(torch.randn(1, 1, 8192))
    report = output["processing_report"]

    assert report.model_weights_loaded is False
    assert "untrained_weights" in report.artifact_flags
    assert report.quality_tier == QualityTier.COMPLEX
    assert output["routing_decision"]["action"] == "manual_review"


def test_untrained_file_inference_requires_explicit_opt_in(mock_config, temp_audio_file):
    """File inference refuses random weights unless a test/dev opt-in is set."""
    from main import StemMidiModel

    config = dict(mock_config)
    config.pop("allow_untrained_inference", None)
    model = StemMidiModel(config)
    with pytest.raises(RuntimeError, match="trained checkpoint"):
        model.process_audio_file(temp_audio_file)


def test_configured_fake_backends_keep_guitar_and_bass_events_independent(mock_config):
    """The orchestrator invokes transcription separately for each stem."""
    from main import StemMidiModel

    model = StemMidiModel(mock_config)
    with torch.inference_mode():
        output = model(torch.randn(1, 1, 8192))

    guitar = output["midi_metadata"]["guitar"]["midi_events"]
    bass = output["midi_metadata"]["bass"]["midi_events"]
    assert guitar and bass
    assert guitar[0]["note"] != bass[0]["note"]


def test_api_zip_uses_independent_midi_and_configured_sample_rate(monkeypatch):
    """The package builder must not duplicate guitar events into bass.mid."""
    import json
    import zipfile

    import numpy as np

    import api

    monkeypatch.setattr(
        api,
        "model_config",
        {"audio": {"sample_rate": 48000, "hop_length": 512}},
        raising=False,
    )
    report = ProcessingReport(
        separation_proxy=22.0,
        phase_coherence=0.8,
        avg_confidence=0.9,
        artifact_flags=["none"],
        model_weights_loaded=True,
    )
    outputs = {
        "sample_rate": 48000,
        "stems": {
            "guitar": np.zeros(4800, dtype=np.float32),
            "bass": np.zeros(4800, dtype=np.float32),
        },
        "midi": {
            "guitar": {
                "midi_events": [
                    {"note": 72, "onset_frame": 94, "duration_frames": 20, "confidence": 0.9}
                ]
            },
            "bass": {
                "midi_events": [
                    {"note": 36, "onset_frame": 94, "duration_frames": 20, "confidence": 0.8}
                ]
            },
        },
        "report": report,
    }

    package = api.create_response_zip(outputs)
    with zipfile.ZipFile(package) as archive:
        guitar_mid = mido.MidiFile(file=io.BytesIO(archive.read("guitar.mid")))
        bass_mid = mido.MidiFile(file=io.BytesIO(archive.read("bass.mid")))
        report_json = json.loads(archive.read("processing_report.json"))

    assert [m.note for m in guitar_mid.tracks[0] if m.type == "note_on"] == [72]
    assert [m.note for m in bass_mid.tracks[0] if m.type == "note_on"] == [36]
    assert report_json["separation_proxy"] == pytest.approx(22.0)
    assert report_json["metric_semantics"] == "centroid_separation_proxy_not_si_sdr"
    assert report_json["model_weights_loaded"] is True


def test_api_refuses_to_load_without_checkpoint(monkeypatch):
    """Startup must become not-ready rather than constructing random weights."""
    import api

    monkeypatch.setenv("MODEL_CONFIG_PATH", "configs/model_config.yaml")
    monkeypatch.delenv("MODEL_CHECKPOINT_PATH", raising=False)
    with pytest.raises(RuntimeError, match="trained checkpoint"):
        api.load_model()
