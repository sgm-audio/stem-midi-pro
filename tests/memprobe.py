# SPDX-License-Identifier: Apache-2.0
"""Peak-RSS measurement that survives a full-suite run.

Why this exists
---------------
``resource.getrusage(RUSAGE_SELF).ru_maxrss`` is a **process-wide high-water
mark that never decreases**.  It is the right quantity for a standalone probe
and the wrong one for a test that runs after ~400 other tests: once any earlier
test has touched a large allocation, every later peak-RSS assertion inherits
that number and fails no matter how well-behaved the code under test is.

That is exactly what happened on CI: four memory tests in different files all
reported ~14 GB, because one of them reached that peak first and the rest read
the same monotonic value.  The failure message ("60 s upload grew peak RSS by
12434 MB") pointed at the upload path when it was really reporting an
accumulator.

On Linux the kernel exposes the same high-water mark per process as ``VmHWM``
in ``/proc/self/status``, and it **can be reset** by writing the current RSS
into ``/proc/self/clear_refs``.  Resetting at the start of a measurement gives
the peak *during that measurement*, which is what a memory test actually means.
``VmRSS`` is the current value and is returned for the baseline.

Non-Linux falls back to ``ru_maxrss``, where a single-measurement-per-process
run still gets an honest answer; the function says so via ``is_resettable()``
so a caller can skip the assertion rather than assert on a monotonic value.
"""

from __future__ import annotations

import sys


def _read_status_kb(field: str) -> float | None:
    """Read a ``VmXxx:`` field from /proc/self/status, in kB, or None."""
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return float(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def is_resettable() -> bool:
    """True when the peak can be reset (Linux ``clear_refs`` is available)."""
    return sys.platform.startswith("linux") and _read_status_kb("VmHWM") is not None


def peak_mb() -> float | None:
    """Peak resident set since the last :func:`reset_peak`, in MB."""
    hwm = _read_status_kb("VmHWM") if sys.platform.startswith("linux") else None
    if hwm is not None:
        return hwm / 1024
    try:
        import resource

        # KiB on Linux, bytes on macOS. CI and every measurement we report is
        # Linux; this branch exists so the helper never returns a wrong unit
        # silently.
        if sys.platform == "darwin":
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except (ImportError, AttributeError):
        return None


def current_mb() -> float | None:
    """Current resident set, in MB."""
    rss = _read_status_kb("VmRSS")
    if rss is not None:
        return rss / 1024
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except (ImportError, AttributeError):
        return None


def reset_peak() -> bool:
    """Reset the process peak to the current RSS. Returns True on success.

    Writing to ``/proc/self/clear_refs`` needs no privileges for one's own
    process. ``5`` is the documented "reset the peak RSS" value (Linux 4.0+).
    """
    if not sys.platform.startswith("linux"):
        return False
    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as f:
            f.write("5")
        return True
    except OSError:
        return False


def measure(fn):
    """Run ``fn`` and return ``(result, peak_growth_mb)``.

    Resets the peak first when possible. Growth is measured against the current
    RSS at entry, so allocator state inherited from earlier tests does not count
    as this test's fault, while everything the call allocates does.
    """
    baseline = current_mb()
    reset_peak()
    result = fn()
    peak = peak_mb()
    if peak is None or baseline is None:
        return result, None
    return result, peak - baseline
