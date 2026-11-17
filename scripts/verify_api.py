#!/usr/bin/env python3
"""Reproduce the API claims in FIXES.md (H-6, API-6.7, API-6.9).

Run: ``python3 scripts/verify_api.py``

* ``load_model()`` must refuse to serve a randomly initialised network unless
  ``ALLOW_UNTRAINED_MODEL=1`` / ``inference.allow_untrained_model`` is set,
* a CPU allocator OOM must be recognised (the handler maps it to 503),
* NaN/Inf must not survive into ``processing_report.json``.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, ".")

import api


def check_untrained_gate() -> None:
    os.environ.pop("MODEL_CHECKPOINT_PATH", None)
    os.environ.pop("ALLOW_UNTRAINED_MODEL", None)
    try:
        api.load_model()
    except RuntimeError as exc:
        print("untrained gate : refused ->", str(exc).split(":")[0])
    else:
        print("untrained gate : FAILED — a random-weight model was loaded")

    os.environ["ALLOW_UNTRAINED_MODEL"] = "1"
    try:
        model = api.load_model()
        print(
            "override       : loaded with ALLOW_UNTRAINED_MODEL=1,",
            f"trained_weights={api._model_trained}, model={type(model).__name__}",
        )
    finally:
        os.environ.pop("ALLOW_UNTRAINED_MODEL", None)


def check_oom_mapping() -> None:
    cases = [
        RuntimeError("DefaultCPUAllocator: can't allocate memory: you tried to allocate 8 bytes"),
        RuntimeError("CUDA out of memory. Tried to allocate 2.00 MiB"),
        RuntimeError("model exploded"),
    ]
    for exc in cases:
        print(f"oom mapping    : {api._is_oom_error(exc)!s:<5} <- {str(exc)[:48]}")


def check_json_guard() -> None:
    nan = float("nan")
    inf = float("-inf")
    payload = {
        "centroid_separation_db": api._finite(nan),
        "phase_coherence": api._finite(inf),
        "avg_confidence": api._finite(0.87),
    }
    text = json.dumps(payload, allow_nan=False)
    print("json guard     :", text)


if __name__ == "__main__":
    check_untrained_gate()
    check_oom_mapping()
    check_json_guard()
