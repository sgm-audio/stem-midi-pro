# SPDX-License-Identifier: Apache-2.0
"""Round-2 audit fixes: tempo, streaming state, checkpoint migration, threads, dt.

Each test states the finding it covers (see AUDIT_REPORT.md / FIXES.md):

* M-4  — the export used a fixed 120 BPM grid (no tempo detection)
* C-2.5 — no SSM state crossed a chunk boundary
* —    — the pre-norm stack broke old checkpoints with no migration path
* M-8  — ``torch.set_num_threads(os.cpu_count())`` oversubscribed containers
* —    — ``dt`` was only bounded at initialisation, not at runtime
* H-4  — expression values could never reach the MIDI file
"""

import io
import os

import numpy as np
import pytest

from tests.honesty import require


def _clicks(bpm: float, sr: int = 44100, seconds: float = 6.0, seed: int = 0) -> np.ndarray:
    """Click track at ``bpm``: a decaying noise burst on every beat."""
    n = int(sr * seconds)
    out = np.zeros(n, dtype=np.float32)
    period = round(sr * 60.0 / bpm)
    rng = np.random.default_rng(seed)
    burst = np.exp(-np.arange(200) / 40.0) * rng.standard_normal(200) * 0.5
    for start in range(0, n - 200, period):
        out[start : start + 200] += burst.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# M-4 — tempo detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bpm", [60, 90, 120, 140, 200])
def test_tempo_estimate_is_within_one_bpm(bpm):
    from utils.tempo import estimate_tempo_bpm

    estimate, confidence = estimate_tempo_bpm(_clicks(bpm), 44100, 512)
    assert estimate is not None, f"no tempo found in a {bpm} BPM click track"
    assert abs(estimate - bpm) < 1.0, f"{bpm} BPM estimated as {estimate:.1f}"
    assert confidence > 0.3


@pytest.mark.parametrize(
    "signal",
    [
        np.zeros(44100, dtype=np.float32),  # silence
        np.random.default_rng(1).standard_normal(44100).astype(np.float32) * 0.1,  # noise
        (0.3 * np.sin(2 * np.pi * 220 * np.arange(44100) / 44100)).astype(np.float32),  # drone
    ],
)
def test_no_pulse_means_no_tempo(signal):
    """Silence/noise/a drone have no beat: the fixed grid is the honest answer."""
    from utils.tempo import estimate_tempo_bpm

    estimate, _confidence = estimate_tempo_bpm(signal, 44100, 512)
    assert estimate is None


def test_writer_tempo_moves_the_grid_not_the_notes():
    """Changing ``tempo_us`` changes ticks, never absolute note times."""
    from mido import MidiFile

    from utils.midi import build_midi_from_events

    events = [{"onset_frame": 400, "note": 60, "velocity": 100, "confidence": 0.9}]
    seconds = []
    for tempo_us in (500_000, 428_571):
        mid = MidiFile(
            file=io.BytesIO(
                build_midi_from_events({"midi_events": events}, "guitar", tempo_us=tempo_us)
            )
        )
        written = next(m for m in mid.tracks[0] if m.type == "set_tempo").tempo
        tick = 0
        for message in mid.tracks[0]:
            tick += message.time
            if message.type == "note_on" and message.velocity > 0:
                break
        seconds.append(tick * written / 1e6 / mid.ticks_per_beat)
        assert written == tempo_us
    # 400 frames * 512 / 44100 = 4.644 s, within one tick of rounding
    assert seconds[0] == pytest.approx(4.644, abs=2e-3)
    assert seconds[1] == pytest.approx(seconds[0], abs=2e-3)


def test_writer_rejects_impossible_tempo():
    from utils.midi import build_midi_from_events

    with pytest.raises(ValueError, match="tempo_us"):
        build_midi_from_events({"midi_events": []}, "guitar", tempo_us=0)


def test_pipeline_attaches_detected_tempo(mock_config):
    """The orchestrator reports where the beat grid came from."""
    require("torch")
    import torch

    from main import StemMidiModel

    cfg = {**mock_config}
    cfg.pop("backends", None)
    cfg["midi"] = {"tempo": "auto"}
    model = StemMidiModel(cfg)
    model.eval()

    audio = torch.from_numpy(_clicks(120)).float().view(1, 1, -1)
    with torch.inference_mode():
        outputs = model(audio)
    metadata = outputs["midi_metadata"]
    timing = metadata["summary"]["timing"]
    assert timing["source"] in {"detected", "fixed"}
    assert timing["tempo_us"] > 0
    for stem in ("guitar", "bass"):
        assert metadata[stem]["tempo_us"] == timing["tempo_us"]
        assert metadata[stem]["tempo_source"] == timing["source"]


def test_pipeline_honours_a_configured_bpm(mock_config):
    require("torch")
    import torch

    from main import StemMidiModel

    cfg = {**mock_config}
    cfg.pop("backends", None)
    cfg["midi"] = {"tempo": 100}
    model = StemMidiModel(cfg)
    model.eval()
    with torch.inference_mode():
        outputs = model(torch.randn(1, 1, 8192))
    timing = outputs["midi_metadata"]["summary"]["timing"]
    assert timing["source"] == "configured"
    assert timing["bpm"] == pytest.approx(100.0)
    assert timing["tempo_us"] == 600_000


# ---------------------------------------------------------------------------
# C-2.5 — SSM state across calls
# ---------------------------------------------------------------------------


def test_block_state_carrying_matches_one_shot():
    """Streaming three pieces must equal processing the whole sequence."""
    require("torch")
    import torch

    from models._mamba_compat import MAMBA_BACKEND, Mamba

    if MAMBA_BACKEND != "pure-torch":
        pytest.skip("state cache is a property of the pure-torch fallback")

    torch.manual_seed(0)
    block = Mamba(d_model=32, d_state=8, d_conv=4, expand=2)
    block.eval()
    x = torch.randn(1, 96, 32)
    with torch.inference_mode():
        reference = block(x)
        state = block.initial_state(x)
        pieces = []
        start = 0
        for length in (7, 13, 40, 36):
            piece, state = block(x[:, start : start + length], state=state, return_state=True)
            pieces.append(piece)
            start += length
        streamed = torch.cat(pieces, dim=1)
    assert streamed.shape == reference.shape
    assert torch.allclose(
        streamed, reference, atol=1e-6
    ), f"state carrying diverges from one pass: {(streamed - reference).abs().max():.2e}"


def test_state_resets_boundaries_when_not_carried():
    """Sanity check on the test above: without state the seams do differ."""
    require("torch")
    import torch

    from models._mamba_compat import MAMBA_BACKEND, Mamba

    if MAMBA_BACKEND != "pure-torch":
        pytest.skip("state cache is a property of the pure-torch fallback")

    torch.manual_seed(0)
    block = Mamba(d_model=32, d_state=8, d_conv=4, expand=2)
    block.eval()
    x = torch.randn(1, 96, 32)
    with torch.inference_mode():
        reference = block(x)
        parts = torch.cat([block(x[:, :40]), block(x[:, 40:])], dim=1)
    assert not torch.allclose(parts, reference, atol=1e-6)


def test_separator_returns_state_only_when_asked(mock_config):
    require("torch")
    import torch

    from models.mamba_separator import MambaSeparator

    cfg = {**mock_config}
    cfg["audio"] = dict(cfg["audio"])
    separator = MambaSeparator(cfg)
    separator.eval()
    audio = torch.randn(1, 1, 8192)
    with torch.inference_mode():
        _g, _b, _r, no_state, _m = separator(audio)
        _g2, _b2, _r2, state, _m2 = separator(audio, return_state=True)
    assert no_state is None, "the default path must stay stateless"
    assert isinstance(state, list) and len(state) == len(separator.mamba_blocks)
    assert set(state[0]) == {"h", "conv"}


# ---------------------------------------------------------------------------
# Checkpoint compatibility
# ---------------------------------------------------------------------------


def test_legacy_checkpoint_loads_with_identity_norms(mock_config):
    require("torch")
    from main import StemMidiModel
    from models.checkpoint_compat import is_legacy_state_dict, load_state_dict_compat

    cfg = {**mock_config}
    cfg.pop("backends", None)
    model = StemMidiModel(cfg)
    state = model.state_dict()
    legacy = {
        key: value
        for key, value in state.items()
        if "block_norms." not in key and "final_norm." not in key
    }
    assert is_legacy_state_dict(legacy, model)
    report = load_state_dict_compat(model, legacy)
    assert report["legacy"] and report["filled"]
    assert not report["missing"]


def test_legacy_checkpoint_refused_when_not_allowed(mock_config):
    require("torch")

    from main import StemMidiModel
    from models.checkpoint_compat import load_state_dict_compat

    cfg = {**mock_config}
    cfg.pop("backends", None)
    model = StemMidiModel(cfg)
    legacy = {
        key: value
        for key, value in model.state_dict().items()
        if "block_norms." not in key and "final_norm." not in key
    }
    with pytest.raises(RuntimeError, match="predates the pre-norm"):
        load_state_dict_compat(model, legacy, allow_legacy=False)


def test_a_genuinely_broken_checkpoint_still_fails_loudly(mock_config):
    require("torch")

    from main import StemMidiModel
    from models.checkpoint_compat import load_state_dict_compat

    cfg = {**mock_config}
    cfg.pop("backends", None)
    model = StemMidiModel(cfg)
    broken = {key: value for key, value in model.state_dict().items() if "input_proj" not in key}
    with pytest.raises(RuntimeError, match="does not match the model"):
        load_state_dict_compat(model, broken)


# ---------------------------------------------------------------------------
# M-8 — thread count
# ---------------------------------------------------------------------------


def test_thread_count_respects_env_and_caps_at_four():
    import main

    assert main._resolve_thread_count("3") == 3
    assert main._resolve_thread_count("99") == 99  # explicit wins, even past the cap
    assert main._resolve_thread_count("0") <= 4
    assert main._resolve_thread_count("nonsense") <= 4
    assert main._resolve_thread_count(None) >= 1


def test_import_does_not_oversubscribe():
    import main

    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        available = os.cpu_count() or 1
    expected_cap = 4 if not os.environ.get("OMP_NUM_THREADS") else None
    if expected_cap is not None:
        assert max(1, min(4, available)) >= main.TORCH_THREADS


# ---------------------------------------------------------------------------
# dt clamp
# ---------------------------------------------------------------------------


def test_dt_is_clamped_at_runtime():
    """A huge dt input must not turn the SSM into an unbounded integrator."""
    require("torch")
    import torch

    from models._mamba_compat import MAMBA_BACKEND, Mamba

    if MAMBA_BACKEND != "pure-torch":
        pytest.skip("clamp lives in the pure-torch fallback")

    torch.manual_seed(0)
    block = Mamba(d_model=16, d_state=8, d_conv=4, expand=2, dt_max=0.05)
    block.eval()
    with torch.no_grad():
        block.dt_proj.bias.fill_(50.0)  # softplus(50) would be ~50, i.e. dt >> 1
    x = torch.ones(1, 32, 16)
    with torch.inference_mode():
        out = block(x)
    assert torch.isfinite(out).all()
    # With dt clamped to 0.05 the state cannot integrate to a huge constant.
    assert out.abs().max() < 50, f"output saturates at {out.abs().max():.1f} despite dt_max"


def test_dt_clamp_argument_is_validated():
    require("torch")
    from models._mamba_compat import MAMBA_BACKEND, Mamba

    if MAMBA_BACKEND != "pure-torch":
        pytest.skip("clamp lives in the pure-torch fallback")
    with pytest.raises(ValueError, match="dt_max"):
        Mamba(d_model=8, d_state=4, dt_max=0.0)


# ---------------------------------------------------------------------------
# H-4 — expression reaches the MIDI file when asked
# ---------------------------------------------------------------------------


def test_expression_is_exported_only_on_request():
    from mido import MidiFile

    from utils.midi import build_midi_from_events

    event = {
        "onset_frame": 10,
        "note": 60,
        "velocity": 90,
        "confidence": 0.8,
        "expression": {"bend": 0.75, "vibrato": 0.4, "slide": 0.6, "source": "untrained_head"},
    }
    plain = MidiFile(file=io.BytesIO(build_midi_from_events({"midi_events": [event]}, "guitar")))
    plain_types = {m.type for m in plain.tracks[0]}
    assert "pitchwheel" not in plain_types and "control_change" not in plain_types

    expressive = MidiFile(
        file=io.BytesIO(
            build_midi_from_events({"midi_events": [event]}, "guitar", include_expression=True)
        )
    )
    types = [m.type for m in expressive.tracks[0]]
    assert "pitchwheel" in types
    assert "control_change" in types
