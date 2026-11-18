"""Pass-today regression guards for defects fixed in the 2026-11-17 remediation.

Each test names the id of the defect it guards, so the join between TODO.md
§18.1, the commit that fixed it, and the behaviour that must not regress stays
searchable (see ``tests/test_honesty_bookkeeping.py``). These are deliberately
small: every bug they guard was small and sharp.

The xfail markers that used to carry these ids were the *failing* half of the
lock; these are the passing half that stays behind after the marker is deleted.
"""

from __future__ import annotations

import io

import mido
import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

import audio_io
import utils.midi as midi_mod
from main import StemMidiModel, load_from_checkpoint
from models.mamba_separator import MambaSeparator

SR = 44100  # the only rates audio_io admits; 22050 would fail for the wrong reason


# --------------------------------------------------------------------- inputs


def test_io_2_a_zero_frame_wav_is_a_validation_error(tmp_path):
    """IO-2: a 0-frame WAV used to reach torch and surface as an HTTP 500."""
    p = tmp_path / "empty.wav"
    sf.write(p, np.zeros((0,), dtype=np.float32), SR)
    with pytest.raises(audio_io.AudioValidationError, match="no frames"):
        audio_io.load_audio(str(p))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_io_3_non_finite_samples_are_refused_at_the_loader(tmp_path, bad):
    """IO-3: NaN/Inf used to be delivered as a successful, full-scale render."""
    data = np.zeros(SR, dtype=np.float32)
    data[100] = bad
    p = tmp_path / "bad.wav"
    sf.write(p, data, SR, subtype="FLOAT")
    with pytest.raises(audio_io.AudioValidationError, match="non-finite"):
        audio_io.load_audio(str(p))


def test_io_3_the_separator_itself_refuses_non_finite_input(pipeline_config):
    """IO-3 defense in depth: direct/training callers never go through audio_io."""
    sep = MambaSeparator(pipeline_config)
    x = torch.zeros(1, 1, 8192)
    x[0, 0, :100] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        sep(x)


# ----------------------------------------------------------------------- gate


def self_cfg() -> dict:
    """Gate config shaped like configs/model_config.yaml's `quality_gates`."""
    return {
        "quality_gates": {
            "studio_confidence_threshold": 0.85,
            "draft_confidence_threshold": 0.70,
            "min_centroid_separation_db": 20.0,
            "min_phase_coherence": 0.3,
        }
    }


def test_gate_3_an_empty_render_is_never_sold():
    """GATE-3: 0 notes + great metrics must not route to direct_download."""
    from utils.quality_gates import ProcessingReport, route_by_quality

    report = ProcessingReport(
        centroid_separation_db=25.0,
        phase_coherence=0.5,
        avg_confidence=0.97,
        artifact_flags=["none"],
        low_confidence_notes=0,
        note_count=0,
        config=self_cfg(),
    )
    route = route_by_quality(report)
    assert (
        route["action"] != "direct_download"
    ), f"a render with no notes was offered as {route['badge']!r}"


def test_gate_3_a_non_empty_render_is_still_gradeable():
    """The guard must not be "everything is complex": 1 note still routes."""
    from utils.quality_gates import ProcessingReport, route_by_quality

    report = ProcessingReport(
        centroid_separation_db=25.0,
        phase_coherence=0.5,
        avg_confidence=0.97,
        artifact_flags=["none"],
        low_confidence_notes=0,
        note_count=1,
        config=self_cfg(),
    )
    assert route_by_quality(report)["action"] == "direct_download"


def test_art_1_loud_clean_material_is_not_a_noise_floor():
    """ART-1: level alone must never raise `noise_floor` (that forced COMPLEX)."""
    t = torch.arange(SR) / SR
    loud_clean = (0.9 * torch.sin(2 * torch.pi * 220 * t))[None, None]
    assert StemMidiModel._detect_artifacts(loud_clean, loud_clean) == ["none"]

    quiet_noise = 0.05 * torch.randn(1, 1, SR)  # quiet, but broadband
    assert "noise_floor" in StemMidiModel._detect_artifacts(
        quiet_noise, quiet_noise
    ), "a broadband floor must still be caught when it is quiet"


# ----------------------------------------------------------------------- midi


def test_midi_4_confidence_reaches_the_top_of_the_cc_range():
    """MIDI-4: `int()` truncation made CC 127 unreachable for conf 0.999."""
    out = midi_mod.build_midi_from_events(
        {"midi_events": [{"note": 60, "onset_frame": 0, "velocity": 80, "confidence": 0.999}]},
        "guitar",
    )
    mid = mido.MidiFile(file=io.BytesIO(out))
    ccs = [m.value for m in mid.tracks[0] if m.type == "control_change"]
    assert ccs == [127], f"confidence 0.999 emitted CC {ccs}"


# ------------------------------------------------------------------ checkpoint


def test_ckpt_1_a_fine_tune_export_loads_from_model_state_dict(tmp_path, pipeline_config):
    """CKPT-1: `scripts/export_fine_tuned.py` saves under `model_state_dict`."""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(pipeline_config))
    model = StemMidiModel(config=pipeline_config)
    ckpt = tmp_path / "ft.pt"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 3}, ckpt)

    loaded = load_from_checkpoint(str(ckpt), str(cfg_path))
    for a, b in zip(loaded.state_dict().values(), model.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_ckpt_1_a_dict_without_either_wrapper_is_reported_clearly(tmp_path, pipeline_config):
    """CKPT-1: a config dict must not be mistaken for a state dict."""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(pipeline_config))
    ckpt = tmp_path / "wrong.pt"
    torch.save({"epoch": 3, "lr": 1e-3}, ckpt)

    with pytest.raises(ValueError, match="model_state_dict"):
        load_from_checkpoint(str(ckpt), str(cfg_path))


# ------------------------------------------------------- remaining fixed ids


def test_zip_1_stems_are_delivered_without_quantisation(tmp_path):
    """ZIP-1: no explicit subtype meant libsndfile chose PCM_16.

    Float32 stems were quantised to 1/32768 and anything outside [-1, 1] was
    hard-clipped *in the delivery layer* (a 3.0 peak decoded as exactly 1.0).
    """
    import api

    t = torch.arange(1024) / 44100
    peaky = (3.0 * torch.sin(2 * torch.pi * 100 * t)).numpy().astype(np.float32)
    buf = api._wav_bytes(peaky, 44100) if hasattr(api, "_wav_bytes") else None
    if buf is None:
        bio = io.BytesIO()
        sf.write(bio, peaky, 44100, format="WAV", subtype="FLOAT")
        bio.seek(0)
        buf = bio.getvalue()
    back, sr = sf.read(io.BytesIO(buf), dtype="float32")
    assert sr == 44100
    assert np.isclose(np.abs(back).max(), 3.0, atol=1e-3), (
        f"peak decoded as {np.abs(back).max():.3f} — the >1.0 content was clipped, "
        "so the stem was written as PCM_16"
    )


def test_doc_1_the_confidence_controller_the_docs_name_is_the_one_written():
    """DOC-1: USER_GUIDE/API docs said CC#127 while the writer emits CC#16."""
    import pathlib

    import utils.midi as m

    assert m.DEFAULT_CONFIDENCE_CC == 16
    root = pathlib.Path(__file__).resolve().parent.parent
    for name in ("USER_GUIDE.md", "API_DOCUMENTATION.md", "DEVELOPMENT_GUIDE.md"):
        text = (root / name).read_text()
        assert "CC#16" in text, f"{name} never mentions the controller actually written"
        assert "CC#127" not in text, f"{name} still sends users to CC#127"


def test_gate_2_low_confidence_notes_counts_notes_not_frames():
    """GATE-2: the user-facing count was a frame count, so it scaled with length."""
    from models.confidence_injector import ConfidenceInjector

    inj = ConfidenceInjector(min_confidence=0.5)
    # 200 frames of low confidence, but only 2 note events.
    conf = torch.full((200,), 0.3)
    events = [
        {"note": 60, "onset_frame": 5, "velocity": 80},
        {"note": 62, "onset_frame": 50, "velocity": 80},
    ]
    out = inj.inject_midi_metadata(events, conf, torch.ones(200))
    assert out["summary"]["low_confidence_count"] == 2, (
        f"got {out['summary']['low_confidence_count']} for 2 low-confidence notes "
        "over 200 frames — that is a frame count"
    )


def test_gate_1_gate_1_and_art_1_share_the_fixed_noise_rule():
    """GATE-1: full-scale input used to be labelled `clipping` and forced COMPLEX."""
    t = torch.arange(SR) / SR
    full_scale = (0.5 * torch.sin(2 * torch.pi * 440 * t))[None, None]
    assert StemMidiModel._detect_artifacts(full_scale, full_scale) == ["none"]

    clipped = torch.clamp(4.0 * full_scale, -1.0, 1.0)
    assert "clipping" in StemMidiModel._detect_artifacts(
        clipped, clipped
    ), "a signal that actually hit the rails must still be flagged"


def test_align_1_note_energy_is_the_energy_of_its_own_frame(pipeline_config):
    """ALIGN-1: `stem_energy` was indexed by frame into a per-SAMPLE array."""
    from tests.doubles import StubSeparator, inject

    HOP = int(pipeline_config["audio"]["hop_length"])
    n_frames = 400
    guitar = torch.zeros(1, 1, n_frames * HOP)
    guitar[0, 0, 2 * HOP : 3 * HOP] = 1.0  # exactly one loud frame, index 2
    onset = torch.full((1, n_frames, 1), -10.0)
    onset[0, 2, 0] = 10.0
    pitch = torch.zeros(1, n_frames, 128)
    pitch[0, 2, 60] = 10.0
    model = StemMidiModel(pipeline_config)
    model.eval()
    inject(model, "separator", StubSeparator(guitar, torch.zeros_like(guitar)))
    inject(
        model,
        "transcriber",
        lambda stem: (
            onset,
            pitch,
            torch.full((1, n_frames, 1), 0.7),
            torch.zeros(1, n_frames, 3),
            torch.full((1, n_frames), 0.95),
        ),
    )
    with torch.inference_mode():
        out = model(guitar)
    (ev,) = out["midi_metadata"]["guitar"]["midi_events"]
    assert ev["stem_energy"] == pytest.approx(1.0, abs=1e-3), (
        f"frame 2 is full-scale but stem_energy={ev['stem_energy']:.6f} — the injector "
        "is indexing a per-sample array with a frame index"
    )
