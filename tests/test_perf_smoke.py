# SPDX-License-Identifier: Apache-2.0
"""Performance smoke test (T-8.3.13).

Build the downsized-CPU model, run one forward on (1, 1, 44100) and assert:
- RSS stays under ~2.5 GB (torch import alone is ~0.9-1 GB; original 500 MB was pre-torch reality)
- wall time stays under 30 s (i5 reference)

Marked `slow` so it is excluded from fast runs via `-m "not slow"`.
RSS measurement prefers psutil; falls back to `resource` (POSIX); skips if
neither is available.
"""

import time

import pytest


@pytest.mark.slow
def test_forward_smoke(mock_config):
    pytest.importorskip("torch")

    import torch

    rss_probe = None
    try:
        import psutil

        def rss_mb():
            return psutil.Process().memory_info().rss / 1e6

        rss_probe = rss_mb
    except ImportError:
        try:
            import resource

            def rss_mb():
                # ru_maxrss is KiB on Linux, bytes on macOS — POSIX test envs
                # are Linux in CI.
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000

            rss_probe = rss_mb
        except ImportError:
            pytest.skip("neither psutil nor resource available for RSS")

    from main import StemMidiModel

    model = StemMidiModel(mock_config)
    model.eval()

    # expects fix per TODO C-2.2 (transcriber self.cfg) for full forward
    audio = torch.randn(1, 1, 44100)
    with torch.inference_mode():
        start = time.perf_counter()
        outputs = model(audio)
        elapsed = time.perf_counter() - start

    assert "guitar_stem" in outputs
    rss = rss_probe()
    assert rss < 2500, f"RSS {rss:.0f} MB exceeds 2500 MB budget"
    assert elapsed < 30, f"forward took {elapsed:.1f}s, budget is 30s"
