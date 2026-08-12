#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Warm the Demucs + Basic Pitch weight caches."""
from __future__ import annotations

import sys


def main() -> int:
    print("Loading Demucs htdemucs (downloads on first use)...")
    try:
        from demucs.api import Separator

        sep = Separator(model="htdemucs", device="cpu", progress=True)
        print(f"  OK samplerate={sep.samplerate} sources={sep.model.sources}")
    except ImportError:
        print("  SKIP: demucs not installed (pip install demucs)", file=sys.stderr)
    except Exception as e:
        print(f"  FAIL demucs: {e}", file=sys.stderr)
        return 1

    print("Loading Basic Pitch model...")
    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH

        print(f"  OK model path={ICASSP_2022_MODEL_PATH}")
    except ImportError:
        print("  SKIP: basic-pitch not installed (pip install basic-pitch)", file=sys.stderr)
    except Exception as e:
        print(f"  FAIL basic-pitch: {e}", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
