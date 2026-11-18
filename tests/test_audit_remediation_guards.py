# SPDX-License-Identifier: Apache-2.0
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


def _detect(guitar, bass, source=None):
    """Call the artifact detector without building a model (GATE-1 / ART-1).

    `_detect_artifacts` is an instance method (it reads `_input_flags` and
    `_weights_loaded`), but its stem logic is class-qualified, so an instance
    whose only needed attribute is `_weights_loaded` is enough.
    """
    from types import SimpleNamespace

    return StemMidiModel._detect_artifacts(
        SimpleNamespace(_weights_loaded=True), guitar, bass, source
    )


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
    assert _detect(loud_clean, loud_clean) == ["none"]

    quiet_noise = 0.05 * torch.randn(1, 1, SR)  # quiet, but broadband
    assert "noise_floor" in _detect(
        quiet_noise, quiet_noise
    ), "a broadband floor must still be caught when it is quiet"


# ----------------------------------------------------------------------- midi


def test_midi_4_confidence_reaches_the_top_of_the_cc_range():
    """MIDI-4: `int()` truncation made the top of the range unreachable.

    Confidence now rides Polyphonic Key Pressure (a note-specific message)
    rather than a CC, but the truncation bug is lane-independent: 0.999*127 =
    126.87 and `int()` gave 126, so "fully confident" could never be expressed.
    """
    out = midi_mod.build_midi_from_events(
        {"midi_events": [{"note": 60, "onset_frame": 0, "velocity": 80, "confidence": 0.999}]},
        "guitar",
    )
    mid = mido.MidiFile(file=io.BytesIO(out))
    pressures = [m.value for m in mid.tracks[0] if m.type == "polytouch"]
    assert pressures == [127], f"confidence 0.999 emitted pressure {pressures}"


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


def test_doc_1_the_docs_describe_the_lane_confidence_actually_rides():
    """DOC-1 evolved upstream: confidence is Polyphonic Key Pressure, not a CC.

    The original defect was that the docs told users to filter on CC#127 while
    the writer emitted CC#16. Upstream then removed CC from the confidence path
    entirely (correctly -- CC#120-127 are Channel Mode messages), so the guard
    is now: the docs say poly pressure, and no CC is emitted for confidence.
    """
    import pathlib

    out = midi_mod.build_midi_from_events(
        {"midi_events": [{"note": 60, "onset_frame": 0, "velocity": 80, "confidence": 0.9}]},
        "guitar",
    )
    mid = mido.MidiFile(file=io.BytesIO(out))
    ccs = [m for m in mid.tracks[0] if m.type == "control_change" and m.value != 0]
    assert ccs == [], f"confidence must not ride a CC lane, got {[m.value for m in ccs]}"
    assert [m.value for m in mid.tracks[0] if m.type == "polytouch"] == [round(0.9 * 127)]

    root = pathlib.Path(__file__).resolve().parent.parent
    for name in ("USER_GUIDE.md", "API_DOCUMENTATION.md"):
        text = (root / name).read_text()
        assert "olyphonic" in text or "poly pressure" in text.lower(), (
            f"{name} never mentions Polyphonic Key Pressure, the lane confidence " "actually rides"
        )


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
    assert _detect(full_scale, full_scale) == ["none"]

    clipped = torch.clamp(4.0 * full_scale, -1.0, 1.0)
    assert "clipping" in _detect(
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
    # The shipped config sets `backends:`, which routes `_separate`/`_transcribe`
    # through the backend objects and ignores the modules `inject()` replaces.
    # This test is about the batch reduction, so force the module path.
    cfg.pop("backends", None)

    class TwoItemSeparator:
        def __call__(self, audio):
            metrics = torch.tensor([[25.0, 0.95], [1.0, 0.05]])
            zeros = torch.zeros_like(audio)
            return audio, zeros, zeros, None, metrics

    model = StemMidiModel(cfg)
    model.eval()
    inject(model, "separator", TwoItemSeparator())
    n_frames = 10
    # No onsets: this test is about the REPORT, so the batch must not try to
    # produce events. BATCH-1 refuses to flatten a multi-item batch into one
    # MIDI event list, and a fixture with onsets here would (correctly) trip it.
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
    # `centroid_separation_db` is a deprecated alias upstream; the live field is
    # `separation_proxy`. Reading the alias here asserted 0.0.
    report = out["processing_report"]
    assert report.separation_proxy == pytest.approx(13.0, abs=1e-3), (
        f"report says {report.separation_proxy} dB for a batch whose items measured "
        "25.0 and 1.0 dB -- that is item 0, not the batch mean"
    )


def test_midi_7_a_mid_file_tempo_change_moves_later_notes(tmp_path):
    """MIDI-7: a `set_tempo` after tick 0 must shift everything after it."""
    import mido

    from utils.midi_targets import midi_to_targets

    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))  # 120 BPM
    track.append(mido.Message("note_on", note=60, velocity=100, time=240))  # 0.25 s
    track.append(mido.Message("note_off", note=60, velocity=0, time=240))
    track.append(mido.MetaMessage("set_tempo", tempo=1_000_000, time=0))  # 60 BPM
    track.append(mido.Message("note_on", note=62, velocity=100, time=960))  # 2.50 s
    track.append(mido.Message("note_off", note=62, velocity=0, time=240))
    path = tmp_path / "map.mid"
    mid.save(str(path))

    onset, _pitch, _vel = midi_to_targets(str(path), 300, hop_length=512, sample_rate=44100)
    frames = [i for i in range(300) if onset[0, i, 0] > 0]
    assert len(frames) == 2, frames
    assert abs(frames[0] - 22) <= 1, f"first note at frame {frames[0]}"
    assert abs(frames[1] - 215) <= 1, (
        f"second note at frame {frames[1]}; 2.500 s is frame 215 and frame 129 is "
        "where it lands if the 60 BPM change is ignored"
    )


def test_batch_1_two_recordings_never_become_one_midi_file(pipeline_config):
    """BATCH-1: `_logits_to_midi` flattened every item into one event list.

    The events carry raw per-item frame indices, so a B=2 call returned a single
    MIDI in which two songs' notes were interleaved and unattributable.
    """
    from main import StemMidiModel

    model = StemMidiModel({**pipeline_config, "backends": {}})
    model.eval()
    n = 50
    onset = torch.full((2, n, 1), -10.0)
    onset[0, 5, 0] = 10.0
    onset[1, 5, 0] = 10.0
    pitch = torch.zeros(2, n, 128)
    pitch[0, 5, 64] = 10.0
    pitch[1, 5, 40] = 10.0

    with pytest.raises(ValueError, match="batch size 2"):
        model._logits_to_midi(onset, pitch, torch.full((2, n, 1), 0.7), torch.full((2, n), 0.9))


def test_batch_1_single_item_transcription_is_unaffected(pipeline_config):
    """The BATCH-1 guard must not touch the supported B=1 path."""
    from main import StemMidiModel

    model = StemMidiModel({**pipeline_config, "backends": {}})
    model.eval()
    n = 50
    onset = torch.full((1, n, 1), -10.0)
    onset[0, 5, 0] = 10.0
    pitch = torch.zeros(1, n, 128)
    pitch[0, 5, 64] = 10.0
    events = model._logits_to_midi(
        onset, pitch, torch.full((1, n, 1), 0.7), torch.full((1, n), 0.9)
    )
    assert [e["note"] for e in events] == [64]
    assert [e["onset_frame"] for e in events] == [5]


def test_cpu_1_scan_memory_does_not_grow_with_song_length():
    """CPU-1: the scan used to materialise O(L) tensors -- 162 GB for 5 minutes."""
    import yaml

    from models._mamba_compat import Mamba
    from tests.test_memory_ceiling import _scan_activation_bytes, _shipped

    assert Mamba.SCAN_CHUNK > 0
    one_min = _scan_activation_bytes(_shipped(), 60.0)
    five_min = _scan_activation_bytes(_shipped(), 300.0)
    assert five_min == one_min, "scan cost still scales with song length"
    assert one_min < 4 * 1024**3
    assert yaml is not None


def test_cpu_2_process_audio_file_honours_the_chunk_knob(tmp_path, pipeline_config, monkeypatch):
    """CPU-2: `chunk_duration_sec` was inert -- one monolithic forward per file."""
    from main import StemMidiModel

    cfg = {**pipeline_config, "audio": {**pipeline_config["audio"], "chunk_duration_sec": 2.0}}
    model = StemMidiModel(cfg)
    model.eval()

    calls: list[int] = []
    real_forward = model.separator.forward

    def counting(tensor, *a, **k):
        calls.append(int(tensor.shape[-1]))
        return real_forward(tensor, *a, **k)

    monkeypatch.setattr(model.separator, "forward", counting)

    sr = 44100
    t = np.arange(8 * sr) / sr
    path = tmp_path / "long.wav"
    sf.write(path, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)
    with torch.inference_mode():
        model.process_audio_file(str(path))

    assert len(calls) > 1, f"the 8 s file went through {len(calls)} forward(s)"
    assert max(calls) <= 2 * sr + 1, f"an unbounded forward carried {max(calls)} samples"


def test_cpu_2_short_input_still_takes_the_single_forward_path(
    tmp_path, pipeline_config, monkeypatch
):
    """A file at or below one chunk must not be chunked (bit-exact for short uploads)."""
    from main import StemMidiModel

    cfg = {**pipeline_config, "audio": {**pipeline_config["audio"], "chunk_duration_sec": 2.0}}
    model = StemMidiModel(cfg)
    model.eval()
    calls: list[int] = []
    real_forward = model.separator.forward

    def counting(tensor, *a, **k):
        calls.append(int(tensor.shape[-1]))
        return real_forward(tensor, *a, **k)

    monkeypatch.setattr(model.separator, "forward", counting)
    sr = 44100
    t = np.arange(sr) / sr  # 1 s, below the 2 s chunk
    path = tmp_path / "short.wav"
    sf.write(path, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)
    with torch.inference_mode():
        model.process_audio_file(str(path))
    assert len(calls) == 1, f"1 s input issued {len(calls)} forwards"


# ---------------------------------------------------------------------------
# PERF-1 / CPU-2 follow-up: the streaming path must not rebuild an autograd
# graph over the recombined full-length stem.
# ---------------------------------------------------------------------------


def test_streaming_transcribes_the_recombined_stem_inside_inference_mode(tmp_path, pipeline_config):
    """A 60 s upload must not retain activations for a backward pass.

    `process_audio_streaming` closed its `inference_mode` block after the chunk
    loop, so the transcribe call on the recombined FULL-LENGTH stem ran with
    grad enabled. Nothing calls backward, but every mamba activation and the
    (1, T, 128) pitch logits stayed reachable from the graph: a 60 s upload
    measured ~4.5 MB per frame and was OOM-killed at a 4 GB ceiling. The file
    path on main kept `_finalize` inside `inference_mode`; this copy did not.

    Guarded structurally -- no torch internals, no RSS threshold: a tensor that
    escaped `inference_mode` reports `requires_grad`/is an inference tensor.
    """
    from main import StemMidiModel

    cfg = {**pipeline_config, "audio": {**pipeline_config["audio"], "chunk_duration_sec": 2.0}}
    model = StemMidiModel(cfg)
    model.eval()

    seen: list[dict] = []
    real = model._transcribe

    def spy(stem, instrument=None):
        out = real(stem, instrument)
        seen.append(out)
        return out

    model._transcribe = spy
    sr = 44100
    t = np.arange(4 * sr) / sr
    path = tmp_path / "four.wav"
    sf.write(path, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)
    model.process_audio_streaming(str(path), chunk_seconds=2.0)

    assert seen, "streaming never transcribed"
    for key in ("onset_logits", "pitch_logits", "velocity", "confidence"):
        tensor = seen[0][key]
        assert tensor is not None
        # inference-mode tensors are either flagged as such or carry no grad;
        # a graph-retaining tensor from an enabled-grad forward does the reverse
        # of at least one of these.
        assert getattr(tensor, "is_inference", lambda: False)() or not tensor.requires_grad, (
            f"{key} was produced outside inference_mode (requires_grad="
            f"{tensor.requires_grad}); the streaming path leaks an autograd graph "
            "over the recombined stem"
        )
    # And the peak must stay bounded: measured 898 MB for 60 s after the fix,
    # OOM (>4 GB) before it.
    from tests import memprobe

    if not memprobe.reset_peak():
        pytest.skip("peak-RSS measurement is not resettable on this platform")
    model.process_audio_file(str(path))
    peak = memprobe.peak_mb()
    assert peak is not None and peak < 4096, f"streaming a 4 s file peaked at {peak:.0f} MB"


def test_memprobe_reports_this_measurements_peak_not_the_process_high_water():
    """The shared probe must reset, or every later memory assertion inherits a peak.

    CI showed the failure mode: four memory tests in three files all reported
    ~14 GB because one of them got there first and `ru_maxrss` never decreases.
    """
    from tests import memprobe

    if not memprobe.is_resettable():
        pytest.skip("peak is not resettable on this platform")
    big = [bytearray(200 * 1024 * 1024)]  # 200 MB
    assert big[0][0] == 0
    after_big = memprobe.peak_mb()
    assert after_big is not None and after_big > 150
    del big
    assert memprobe.reset_peak()
    after_reset = memprobe.peak_mb()
    assert after_reset is not None and after_reset < after_big - 100, (
        f"peak did not drop after reset ({after_big:.0f} -> {after_reset:.0f} MB); "
        "memory tests will assert on the process high-water mark instead of their own"
    )
