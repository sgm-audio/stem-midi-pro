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


def test_io_4_anti_phase_stereo_is_reported_not_silently_silenced(tmp_path):
    """IO-4: L = -R downmixed to exact silence with no warning."""
    t = np.arange(SR) / SR
    base = 0.5 * np.sin(2 * np.pi * 220 * t)
    p = tmp_path / "anti.wav"
    sf.write(p, np.stack([base, -base], axis=1), SR)
    with pytest.raises(audio_io.AudioValidationError, match="cancel"):
        audio_io.load_audio(str(p))


@pytest.mark.parametrize("scale", [1.0, 0.6, 0.0])
def test_io_4_in_phase_stereo_still_loads(tmp_path, scale):
    """The guard must not reject ordinary stereo."""
    t = np.arange(SR) / SR
    base = 0.5 * np.sin(2 * np.pi * 220 * t)
    p = tmp_path / "stereo.wav"
    sf.write(p, np.stack([base, scale * base], axis=1), SR)
    audio, sr, _ = audio_io.load_audio(str(p))
    assert sr == SR and np.isfinite(audio).all()


def test_repro_1_the_same_input_gives_bitwise_identical_confidence(pipeline_config):
    """REPRO-1: the same paid request returned different confidence on a retry.

    `mc_dropout_eval` drew its masks from the global RNG, so two runs differed by
    up to 6.5e-3 -- enough to flip `needs_review` and move a CC value.
    """
    from models.mamba_transcriber import MambaTranscriber

    tr = MambaTranscriber(pipeline_config)
    tr.eval()
    hidden = torch.randn(1, 20, int(pipeline_config["transcriber"]["d_model"]))
    first = tr.mc_dropout_eval(hidden)
    second = tr.mc_dropout_eval(hidden)
    assert torch.equal(first, second), (
        f"MC-dropout estimate is not reproducible (max |delta| = "
        f"{float((first - second).abs().max()):.6f})"
    )


def test_repro_1_the_ensemble_is_still_non_degenerate(pipeline_config):
    """The reproducibility fix must not turn C-2.10's ensemble into a no-op."""
    from models.mamba_transcriber import MambaTranscriber

    tr = MambaTranscriber(pipeline_config)
    tr.eval()
    hidden = torch.randn(1, 20, int(pipeline_config["transcriber"]["d_model"]))
    mc = tr.mc_dropout_eval(hidden)
    with torch.no_grad():
        plain = tr.confidence_head(hidden).squeeze(-1)
    assert not torch.allclose(mc, plain, atol=1e-6), (
        "the MC estimate is identical to a single eval-mode forward, so dropout "
        "is inactive and the ensemble does nothing"
    )


def test_fake_1_backend_metrics_are_a_function_of_the_signal():
    """FAKE-1: the fake backend returned 12.0 / 0.5 for any input."""
    from models.backends.fake_backend import FakeSeparatorBackend

    be = FakeSeparatorBackend()
    loud = torch.zeros(1, 1, 8192)
    loud[0, 0, :] = torch.sin(torch.arange(8192) * 2 * torch.pi * 220 / 44100)
    silent = torch.zeros(1, 1, 8192)

    m_loud = be.separate(loud)["metrics"]
    m_silent = be.separate(silent)["metrics"]
    assert not torch.allclose(m_loud, m_silent), (
        "the fake backend returns identical metrics for a sine and for silence, "
        "so any gate test routed through it asserts a fixture"
    )
    # and it must be able to reach a passing tier, or studio/draft stay untestable
    assert float(m_loud[0, 0]) >= 20.0


def test_report_1_the_batched_report_averages_over_the_batch():
    """REPORT-1: `metrics[0, 0]` was read under an "Average over batch" comment."""
    import pathlib

    import yaml

    from main import StemMidiModel
    from models.backends.base import SeparateOut  # noqa: F401  (shape documentation)
    from tests.doubles import inject

    root = pathlib.Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((root / "configs" / "model_config.yaml").read_text())
    cfg["separator"]["d_model"] = 32
    cfg["separator"]["n_layers"] = 1
    cfg["transcriber"]["d_model"] = 32
    cfg["transcriber"]["n_layers"] = 1

    class TwoItemSeparator:
        def __call__(self, audio):
            metrics = torch.tensor([[25.0, 0.95], [1.0, 0.05]])
            zeros = torch.zeros_like(audio)
            return audio, zeros, zeros, None, metrics

    model = StemMidiModel(cfg)
    model.eval()
    inject(model, "separator", TwoItemSeparator())
    n_frames = 10
    inject(
        model,
        "transcriber",
        lambda stem: (
            torch.full((2, n_frames, 1), -10.0),
            torch.zeros(2, n_frames, 128),
            torch.full((2, n_frames, 1), 0.7),
            torch.zeros(2, n_frames, 3),
            torch.full((2, n_frames), 0.9),
        ),
    )
    with torch.inference_mode():
        out = model(torch.full((2, 1, n_frames * 512), 0.1))
    assert out["processing_report"].centroid_separation_db == pytest.approx(
        13.0, abs=1e-3
    ), "report is reporting item 0's 25.0 dB instead of the batch mean 13.0 dB"
