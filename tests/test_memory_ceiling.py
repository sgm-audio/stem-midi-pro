# SPDX-License-Identifier: Apache-2.0
"""CPU practicality: does a 4-8 GB i5 actually survive a real song?

The suite's only memory test (``tests/test_perf_smoke.py``) builds the tiny
``mock_config`` model, forwards **one second** of audio and compares a single RSS
number against 2 500 MB. Nothing in the repo connects the shipped ``d_model: 768
/ n_layers: 12`` architecture, the 44.1 kHz / hop-512 frame grid and the
"process the whole file in one forward" implementation, which is what a user's
machine meets. These tests do that arithmetic — mostly without running the big
model at all, so they are fast and deterministic — and they pin the three claims
the README makes: chunked inference, gradient checkpointing, and "CPU-only".
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

from tests.honesty import known_bug, require

np = require("numpy", "memory ceiling tests")
yaml = require("yaml", "config-vs-code checks")

ROOT = Path(__file__).resolve().parents[1]
SR = 44100
HOP = 512
CPU_TARGET_BYTES = 4 * 1024**3  # the low end of the advertised 4-8 GB budget


def _shipped() -> dict:
    with open(ROOT / "configs" / "model_config.yaml") as fh:
        return yaml.safe_load(fh)


def _frames(seconds: float, sr: int = SR, hop: int = HOP) -> int:
    return 1 + int(sr * seconds // hop)


def _scan_activation_bytes(cfg: dict, seconds: float) -> int:
    """dA and dB*x in the pure-torch scan: (B, L, d_inner, d_state) fp32 each.

    ``models/_mamba_compat.py`` materialises both per layer (see lines 90-91), and
    ``MambaSeparator``/``MambaTranscriber`` keep every layer's activations alive for
    the whole forward (inference_mode, no checkpointing), so this is a lower bound on
    the transient cost of a forward of a ``seconds``-long song.
    """
    total = 0
    for section in ("separator", "transcriber"):
        arch = cfg[section]
        d_inner = int(arch["d_model"]) * int(arch.get("expand", 2))
        per_step = int(arch["n_layers"]) * d_inner * int(arch["d_state"]) * 4  # fp32
        total += (
            2
            * per_step
            * _frames(seconds, int(cfg["audio"]["sample_rate"]), int(cfg["audio"]["hop_length"]))
        )
    return total


# ---------------------------------------------------------------------------
# the arithmetic, at the shipped config
# ---------------------------------------------------------------------------


@pytest.mark.memory
def test_scan_activations_for_two_seconds_fit_the_cpu_budget():
    """Documents the constant the rest of the file reasons about (a real number, not a claim)."""
    cfg = _shipped()
    need = _scan_activation_bytes(cfg, 2.0)
    per_sec = need / 2.0
    assert need < CPU_TARGET_BYTES, f"even 2 s needs {need / 1e9:.2f} GB"
    assert per_sec > 1e8, (
        f"{per_sec / 1e6:.0f} MB per audio-second at the shipped shape — if this ever drops "
        f"below 100 MB/s the ceiling analysis below is obsolete and CPU-1 can be closed"
    )


@pytest.mark.memory
@pytest.mark.parametrize("seconds", [30.0, 60.0, 300.0], ids=["30s", "1min", "5min"])
@known_bug(
    "CPU-1",
    "the shipped architecture cannot hold a song: models/_mamba_compat.py materialises dA and "
    "dB*x as (B, L, d_inner, d_state) fp32 per layer, which at configs/model_config.yaml "
    "(768x2 d_inner, 12 layers + 512x2, 8 layers, d_state 16/12 on the 44.1k/512 grid) costs "
    "271 MB per second of audio — 0.54 GB for 2 s, 8.1 GB for a 30 s clip, 16.3 GB for a minute, "
    "162 GB for five — so anything past ~15 s blows the 4-8 GB target before the STFT masks, the "
    "waveform copies and the autograd graph are counted at all. Chunked/streaming inference is "
    "the only fix that keeps the CPU promise, and process_audio_file does one monolithic forward "
    "today (CPU-2).",
)
def test_scan_activations_for_a_real_song_fit_the_cpu_budget(seconds):
    cfg = _shipped()
    need = _scan_activation_bytes(cfg, seconds)
    assert need < CPU_TARGET_BYTES, (
        f"{seconds:g}s of audio needs {need / 1e9:.2f} GB for the two scan tensors alone "
        f"({need / _frames(seconds) / 1e6:.0f} MB per audio-second); budget is "
        f"{CPU_TARGET_BYTES / 1e9:.0f} GB"
    )


@pytest.mark.memory
def test_scan_activation_cost_is_linear_in_song_length():
    """Guard for the shape of the problem (a sub-quadratic law users can plan around)."""
    cfg = _shipped()
    small = _scan_activation_bytes(cfg, 2.0)
    large = _scan_activation_bytes(cfg, 8.0)
    assert large / small == pytest.approx(_frames(8.0) / _frames(2.0), rel=0.02), (
        f"2s→8s cost ratio {large / small:.2f} is not the frame ratio "
        f"{_frames(8.0) / _frames(2.0):.2f}: the scaling law changed"
    )


def test_gradient_checkpointing_call_resolves_in_a_fresh_interpreter():
    """Regression guard for CFG-3 (wired in PR #11, import fixed on this branch): the
    mitigation must actually be callable.

    main's fix for CPU-1 added a `torch.utils.checkpoint.checkpoint(...)` branch to the
    pure-Torch scan, but `import torch` does not bind `torch.utils.checkpoint` — so until
    `models/_mamba_compat.py` imports the submodule, enabling
    `training.gradient_checkpointing` raised AttributeError at the first training step.
    Checked in a SUBPROCESS on purpose: inside a shared session `torch.utils.checkpoint`
    becomes importable as soon as any other test imports it, so an in-process probe passes
    depending on test order and cannot pin the bug.
    """
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    cfg = _shipped()
    assert cfg["training"]["gradient_checkpointing"] is True, "config no longer claims it"
    import models._mamba_compat as compat

    src = Path(compat.__file__).read_text(encoding="utf-8")
    if "torch.utils.checkpoint.checkpoint" in src:
        assert (
            "import torch.utils.checkpoint" in src
            or "from torch.utils.checkpoint" in src
            or "from torch.utils import checkpoint" in src
        ), f"{compat.__file__} calls torch.utils.checkpoint.checkpoint(...) without importing it"

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            f"sys.path.insert(0, {str(root)!r})\n"
            "import models._mamba_compat as m\n"
            "assert hasattr(m.torch.utils, 'checkpoint'), 'torch.utils unbound'\n"
            "assert callable(m.torch.utils.checkpoint.checkpoint)\n",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert probe.returncode == 0, (
        "importing models._mamba_compat in a clean interpreter leaves torch.utils.checkpoint "
        "unbound, so gradient_checkpointing=True raises AttributeError: "
        f"{probe.stderr.strip()[-300:]}"
    )


# ---------------------------------------------------------------------------
# chunking: promised in the config, absent from the code path
# ---------------------------------------------------------------------------


def _chunked_forwards(pipeline_config, make_wav, synth, monkeypatch, seconds):
    """Run process_audio_file and return the sample count of every separator forward."""
    import torch

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
    path = make_wav(synth.note_sequence(total_seconds=seconds, sr=SR), name=f"chunk{seconds}.wav")
    with torch.inference_mode():
        model.process_audio_file(path)

    return calls


def test_short_input_needs_exactly_one_forward(pipeline_config, make_wav, synth, monkeypatch):
    """Control for the chunking test: one chunk in, one forward out."""
    seconds = 2.0
    calls = _chunked_forwards(pipeline_config, make_wav, synth, monkeypatch, seconds)
    assert (
        len(calls) == 1
    ), f"a {seconds}s clip (chunk_duration_sec=2.0) issued {len(calls)} forwards"


@known_bug(
    "CPU-2",
    "configs/model_config.yaml sets audio.chunk_duration_sec: 2.0 and the README sells chunked "
    "inference for RAM-bounded hosts, but main.StemMidiModel.process_audio_file loads the whole "
    "file and runs exactly one forward (352 800 samples for an 8 s song), so the chunk knob is "
    "inert and peak memory grows with song length. Only process_audio_streaming chunks. Route "
    "process_audio_file through it (or chunk whenever duration > chunk_duration_sec).",
)
def test_process_audio_file_chunks_long_input(pipeline_config, make_wav, synth, monkeypatch):
    seconds = 8.0
    chunk = 2.0
    calls = _chunked_forwards(pipeline_config, make_wav, synth, monkeypatch, seconds)
    expected = max(1, math.ceil(seconds / chunk))
    assert len(calls) == expected, (
        f"{seconds:g}s at chunk_duration_sec={chunk:g}s issued {len(calls)} separator "
        f"forward(s) covering {sum(calls)} samples; the whole song went through ONE forward "
        f"of {max(calls)} samples instead of {expected} bounded chunks of {int(chunk * SR)}"
    )
    assert max(calls) <= int(chunk * SR) + 1, f"a single forward carried {max(calls)} samples"


@pytest.mark.memory
def test_peak_rss_probe_runs_at_all():
    """Non-xfail smoke check for the plumbing below.

    An xfail(strict) test is 'confirmed' by ANY failure, including a broken
    harness — so every measurement used by a known-bug test needs a guard that is
    not allowed to fail quietly. If this trips, CPU-3's verdict is meaningless.
    """
    peak = _child_peak_rss_mb(2.0)
    assert 120 < peak < 4000, f"child peak RSS {peak:.0f} MB is not a plausible pipeline footprint"
    assert _child_peak_rss_mb(2.0, "stream") > 120, "streaming probe broken"


CHILD_SCRIPT = """
import resource, sys
sys.path.insert(0, {root!r})
import numpy as np, soundfile as sf, torch, yaml
from main import StemMidiModel
with open(sys.argv[2]) as fh:
    cfg = yaml.safe_load(fh)
torch.manual_seed(0)
m = StemMidiModel(cfg); m.eval()
sr = 44100
secs = float(sys.argv[3])
n = int(sr * secs)
t = np.arange(n) / sr
sig = (np.sin(2 * np.pi * 220 * t) * 0.3).astype("float32")
sf.write("/tmp/_mem_child.wav", sig, sr, format="WAV", subtype="FLOAT")
mode = sys.argv[4]
with torch.inference_mode():
    if mode == "mono":
        out = m.process_audio_file("/tmp/_mem_child.wav")
    else:
        out = m.process_audio_streaming("/tmp/_mem_child.wav", chunk_seconds=2.0)
assert out["stems"]["guitar"].shape[0] > 0
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
"""


def _child_peak_rss_mb(seconds: float, mode: str = "mono") -> float:
    """Peak RSS (VmHWM) of a fresh process that runs one full pipeline; no psutil needed."""
    cfg = {
        "name": "mem-probe",
        "audio": {"sample_rate": SR, "n_fft": 2048, "hop_length": HOP, "n_mels": 80},
        "separator": {"d_model": 64, "n_layers": 2, "d_state": 8, "d_conv": 4, "expand": 2},
        "transcriber": {
            "d_model": 64,
            "n_layers": 2,
            "d_state": 8,
            "onset_head_dim": 8,
            "pitch_vocab_size": 128,
            "velocity_bins": 128,
            "expression_heads": ["bend"],
            "onset_threshold": 0.5,
        },
        "quality_gates": {
            "studio_confidence_threshold": 0.85,
            "draft_confidence_threshold": 0.70,
            "min_confidence": 0.6,
            "min_si_sdr": 20.0,
        },
        "training": {"precision": "fp32", "gradient_checkpointing": False},
    }
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh)
        cfg_path = fh.name
    code = CHILD_SCRIPT.replace("{root!r}", repr(str(ROOT)))
    proc = subprocess.run(
        [sys.executable, "-c", code, "unused", cfg_path, str(seconds), mode],
        capture_output=True,
        text=True,
        timeout=900,
        cwd=str(ROOT),
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": ""},
    )
    Path(cfg_path).unlink(missing_ok=True)
    assert proc.returncode == 0, f"child failed ({seconds}s): {proc.stderr[-400:]}"
    return float(proc.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# failing loudly beats returning a short render
# ---------------------------------------------------------------------------


def test_memoryerror_from_the_model_is_a_503_with_retry_after(tmp_path):
    """OOM must be a retryable service error, not a 500 and not a silent partial result."""
    import api

    class OomModel:
        def process_audio_file(self, path, quantize=False):
            raise MemoryError("CUDA/CPU out of memory")

    import numpy as np
    import soundfile as sf

    src = tmp_path / "in.wav"
    sf.write(
        src, (np.sin(2 * np.pi * 220 * np.arange(SR)) * 0.3).astype("float32"), SR, format="WAV"
    )
    old = (api.model, api.model_config, api._processing_sem, api._model_ready)
    api.model = OomModel()
    api.model_config = {"audio": {"sample_rate": SR, "hop_length": HOP}}
    api._processing_sem = None
    api._model_ready = True
    try:
        from fastapi.testclient import TestClient

        with src.open("rb") as fh:
            r = TestClient(api.app).post("/process", files={"file": ("in.wav", fh, "audio/wav")})
    finally:
        api.model, api.model_config, api._processing_sem, api._model_ready = old
    assert r.status_code == 503, f"OOM surfaced as {r.status_code}: {r.text[:160]}"
    assert (
        r.headers.get("retry-after") == "1"
    ), f"no Retry-After on the OOM response: {dict(r.headers)}"


def test_streaming_failure_never_returns_a_truncated_render(
    pipeline_config, make_wav, synth, monkeypatch
):
    """If chunk 3 of 6 dies, the caller must not get a 3-chunk 'success'."""
    import torch

    from main import StemMidiModel

    model = StemMidiModel(pipeline_config)
    model.eval()
    n_calls = {"i": 0}
    real = model.separator.forward

    def flaky(tensor, *a, **k):
        n_calls["i"] += 1
        if n_calls["i"] == 3:
            raise MemoryError("simulated OOM on chunk 3")
        return real(tensor, *a, **k)

    monkeypatch.setattr(model.separator, "forward", flaky)
    path = make_wav(synth.note_sequence(total_seconds=6.0, sr=SR), name="stream6.wav")
    with pytest.raises(MemoryError), torch.inference_mode():
        model.process_audio_streaming(path, chunk_seconds=2.0)


def test_streaming_render_length_matches_the_input(pipeline_config, make_wav, synth):
    """Overlap-add must hand back exactly the samples that went in, or MIDI/audio drift."""
    import torch

    from main import StemMidiModel

    model = StemMidiModel(pipeline_config)
    model.eval()
    seconds = 5.0
    audio = synth.note_sequence(total_seconds=seconds, sr=SR)
    path = make_wav(audio, name="len.wav")
    with torch.inference_mode():
        out = model.process_audio_streaming(path, chunk_seconds=2.0)
    got = out["stems"]["guitar"].shape[0]
    want = len(audio)
    assert abs(got - want) <= 2 * HOP, (
        f"streaming returned {got} samples for {want} samples of input "
        f"({(got - want) / SR:+.3f} s): the render and the MIDI frames no longer line up"
    )


@known_bug(
    "PERF-1",
    "tests/test_perf_smoke.py is the only memory/latency gate and it proves nothing about the "
    "CPU target: it builds `mock_config` (d_model 64, 2 layers — 12x fewer parameters per layer "
    "than the shipped 768x12), forwards 1 s (44 100 samples, ~1/180 of a song), asserts "
    "rss < 2500 MB and elapsed < 30 s, and its psutil branch reads *current* RSS after the "
    "forward while its `resource` fallback reads the process peak — so the same test measures "
    "two different quantities depending on whether psutil happens to be installed. Neither "
    "branch can see the O(L·d_inner·d_state) scan cost, and the pure-torch fallback's "
    "`for t in range(L)` loop (models/_mamba_compat.py:97) is never timed at realistic L.",
)
@pytest.mark.slow
def test_perf_smoke_uses_the_shipped_shape_and_measures_a_peak():
    src = (ROOT / "tests" / "test_perf_smoke.py").read_text(encoding="utf-8")
    cfg = _shipped()
    assert "mock_config" not in src, (
        "the perf gate builds the tiny mock model; use configs/model_config.yaml's shape (or a "
        "documented fraction of it) so the RSS/latency budget means something"
    )
    assert "peak" in src.lower(), "RSS is sampled once after the forward; measure the peak"
    assert "352800" in src or "8 * " in src, (
        "the perf gate forwards 44100 samples (1 s); a 4-8 GB CPU target must be checked at "
        "least at 8 s, where the monolithic forward is already 8x larger"
    )
    assert cfg["separator"]["d_model"] >= 64, "sanity: shipped model is the big one"
