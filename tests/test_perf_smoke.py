# SPDX-License-Identifier: Apache-2.0
"""Performance smoke test (T-8.3.13) — the CPU memory/latency gate.

PERF-1: this test used to prove nothing about the CPU target. It built the tiny
test fixture (d_model 64, 2 layers -- 12x smaller per layer than the shipped
768x12), forwarded 1 s of audio (about 1/180 of a song), and its psutil branch
read *current* RSS after the forward while its `resource` fallback read the
process *peak* -- so it measured two different quantities depending on which
module happened to be installed.

What it does now:
  * runs the SHIPPED architecture (configs/model_config.yaml), not the toy one;
  * forwards 8 s (`PERF_SECONDS`), which is longer than one
    `audio.chunk_duration_sec` so the chunked path is exercised too;
  * measures the true peak, via `ru_maxrss` (peak, Linux) or psutil's
    `memory_info().peak_wset` where available, and never silently substitutes a
    current-RSS reading for a peak;
  * reports MB-per-audio-second so a regression is legible in the failure.

At 8 s the shipped model is slow enough that this cannot run in the fast suite,
so it stays `slow` and the correctness gate excludes it via `-m "not slow"`.
"""

import time

import pytest


def _peak_mb():
    """Best available PEAK resident set, in MB. Never returns current RSS.

    Uses ``tests.memprobe`` so the peak can be RESET at the start of each test.
    Without that, a full-suite run makes these gates assert on whatever the
    largest earlier test allocated: on CI four different memory tests all
    reported ~14 GB for this reason (see tests/memprobe.py).
    """
    from tests import memprobe

    return memprobe.peak_mb()


PERF_SECONDS = 8
# Budget rationale (measured in this sandbox, shipped shape, 2 threads):
#   interpreter + torch import   ~410 MB
#   model build (768x12 + 512x8) ~663 MB
#   chunked 6 s pipeline         ~2260 MB   (+1.6 GB for stems, transients, MIDI)
# The advertised CPU target is 4-8 GB, and the streaming path holds one
# 2 s chunk's activations at a time, so the headroom above 2.5 GB is where a
# regression would show up. `ru_maxrss` is process-wide, so in a full-suite run
# this also includes whatever earlier tests left behind -- the number is a
# high-water mark for the process, which is the honest thing to gate on.
PERF_BUDGET_MB = 4096
PERF_BUDGET_SECONDS = 300


@pytest.mark.slow
def test_forward_smoke_at_the_shipped_shape():
    pytest.importorskip("torch")

    import torch
    import yaml

    from main import StemMidiModel
    from tests import memprobe

    baseline = memprobe.current_mb()
    if not memprobe.reset_peak():
        pytest.skip("peak-RSS measurement is not resettable on this platform")

    root = __import__("pathlib").Path(__file__).resolve().parent.parent
    with open(root / "configs" / "model_config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    model = StemMidiModel(cfg)
    model.eval()

    sr = int(cfg["audio"]["sample_rate"])
    audio = torch.randn(1, 1, PERF_SECONDS * sr)
    with torch.inference_mode():
        start = time.perf_counter()
        outputs = model(audio)
        elapsed = time.perf_counter() - start

    assert "guitar_stem" in outputs

    peak = _peak_mb()
    assert peak is not None and baseline is not None
    assert peak < PERF_BUDGET_MB, (
        f"peak RSS {peak:.0f} MB (baseline {baseline:.0f} MB) exceeds the "
        f"{PERF_BUDGET_MB} MB CPU budget after a {PERF_SECONDS}s forward"
    )
    assert elapsed < PERF_BUDGET_SECONDS, (
        f"{PERF_SECONDS}s forward took {elapsed:.1f}s "
        f"(budget {PERF_BUDGET_SECONDS}s; {elapsed / PERF_SECONDS:.1f} s per audio-second)"
    )


@pytest.mark.slow
def test_chunked_processing_stays_inside_the_budget():
    """The chunked path is what the README sells; gate it too."""
    pytest.importorskip("torch")

    import numpy as np
    import soundfile as sf
    import yaml

    from main import StemMidiModel
    from tests import memprobe

    if not memprobe.reset_peak():
        pytest.skip("peak-RSS measurement is not resettable on this platform")

    root = __import__("pathlib").Path(__file__).resolve().parent.parent
    with open(root / "configs" / "model_config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    cfg["audio"]["chunk_duration_sec"] = 2.0
    # Upstream refuses to run `process_audio_file` on random weights unless this
    # is set. That guard is correct for production; this is a memory gate, so it
    # opts in explicitly rather than the check being weakened.
    cfg["allow_untrained_inference"] = True

    sr = int(cfg["audio"]["sample_rate"])
    seconds = 6
    t = np.arange(seconds * sr) / sr
    audio = np.stack([0.3 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1)
    path = root / ".perf_probe.wav"
    sf.write(path, audio.astype(np.float32), sr)
    try:
        model = StemMidiModel(cfg)
        model.eval()
        start = time.perf_counter()
        out = model.process_audio_file(str(path))
        elapsed = time.perf_counter() - start
    finally:
        path.unlink(missing_ok=True)

    assert "report" in out
    # The peak was reset just above, so this is the high-water mark reached
    # DURING the chunked run, not the suite's.
    peak = _peak_mb()
    assert peak is not None
    assert peak < PERF_BUDGET_MB, (
        f"peak RSS {peak:.0f} MB exceeds the {PERF_BUDGET_MB} MB budget "
        f"after chunked processing of a {seconds}s file"
    )
    assert elapsed < PERF_BUDGET_SECONDS
