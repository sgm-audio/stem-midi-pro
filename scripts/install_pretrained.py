#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""
Install the production pretrained backends (Demucs + Basic Pitch).

Why this script exists: basic-pitch 0.4.0 declares a hard dependency on
`tensorflow<2.15.1,>=2.4.1` when platform != Darwin and python >= 3.11.
TensorFlow has NO CPython 3.13 wheels, so `pip install basic-pitch` fails with
ResolutionImpossible. The ONNX runtime path works fine on 3.13 without TF, so
we install basic-pitch with --no-deps and pin its real runtime deps explicitly.
"""
from __future__ import annotations

import subprocess
import sys


def _run(*args: str) -> None:
    cmd = [sys.executable, "-m", "pip", "install", *args]
    print(">", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> int:
    # 1. Matched torch + torchaudio (first CPython 3.13 CPU pair)
    _run("torch>=2.11.0", "torchaudio>=2.11.0")

    # 2. Audio stack (librosa 0.11+ dropped numba; needs numpy>=2.0)
    _run("numpy>=2.0,<2.3", "librosa>=0.11.0", "scipy", "soundfile>=0.12.0")

    # 3. Demucs separator
    _run("demucs>=4.0.0")

    # 4. Basic Pitch via ONNX (no TensorFlow)
    _run("basic-pitch==0.4.0", "--no-deps")
    _run(
        "onnxruntime",
        "pretty-midi>=0.2.9",
        "resampy>=0.2.2,<0.4.3",
        "mir-eval>=0.6",
        "scikit-learn",
        "typing-extensions",
        # setuptools<81 keeps pkg_resources (resampy imports it)
        "setuptools>=69,<81",
    )

    print("\nPretrained backends installed. Verify with:")
    print("  python scripts/download_pretrained.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
