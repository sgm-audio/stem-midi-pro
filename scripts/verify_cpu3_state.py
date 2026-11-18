#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-3: what cross-chunk SSM state continuity is actually worth.

Run: ``python scripts/verify_cpu3_state.py [configs/model_config.mamba.yaml]``

CPU-3 asks for the SSM state to be carried across chunks so a streamed file is
processed "as one sequence". The capability already exists
(``MambaSeparator.forward(..., state=, return_state=)``); what was never
established is whether the pipeline should USE it. This script separates the
three levels so the decision is not a matter of opinion:

1. **block**     one ``Mamba`` block, two calls with the state threaded versus
                 one long call. This is what the mechanism promises.
2. **separator** two chunks through the whole separator, with and without the
                 state, against a single long pass. Each chunk gets its own
                 STFT, so this is where the transform's edge padding enters.
3. **pipeline**  the real chunked path (50 % overlap + complementary crossfade,
                 H-4), with and without the state, against a single pass.

Measured on this repo with ``configs/model_config.mamba.yaml``:

    block     2-call vs 1-call, state threaded     0.0e+00
    separator chunk-vs-whole, state threaded       1.7e-02
    separator chunk-vs-whole, state dropped        1.7e-02
    separator carried-vs-dropped difference        1.0e-03
    pipeline  chunk-vs-whole, state threaded       4.9e-03
    pipeline  chunk-vs-whole, state dropped        4.9e-03
    pipeline  carried-vs-dropped difference        1.5e-04   (1.000x accuracy)

The mechanism is exact where it applies and the pipeline cannot benefit from it:
the crossfade already absorbs the discontinuity. That is why the default path
does not thread state, and why CPU-3 is closed as measured rather than assumed.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, ".")

from models._mamba_compat import Mamba
from models.mamba_separator import MambaSeparator


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp(min=1e-12))


def _signal(cfg: dict, seconds: float) -> torch.Tensor:
    sr = int(cfg["audio"]["sample_rate"])
    t = np.arange(int(seconds * sr)) / sr
    y = 0.3 * np.sin(2 * np.pi * 110 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)
    return torch.from_numpy(y.astype(np.float32))[None, None]


def block_level() -> float:
    torch.manual_seed(0)
    blk = Mamba(d_model=64, d_state=16).eval()
    x = torch.randn(1, 128, 64)
    with torch.inference_mode():
        whole = blk(x)
        first, state = blk(x[:, :64], return_state=True)
        second = blk(x[:, 64:], state=state)
    return _rel(torch.cat([first, second], dim=1), whole)


def separator_level(sep: MambaSeparator, x: torch.Tensor, hop: int) -> tuple[float, float, float]:
    cut = ((x.shape[-1] // 2) // hop) * hop  # align to the STFT grid
    with torch.inference_mode():
        whole = sep(x)[0]
        a, _b, _r, state, _m = sep(x[:, :, :cut], return_state=True)
        with_state = sep(x[:, :, cut:], state=state)[0]
        without = sep(x[:, :, cut:])[0]
    return (
        _rel(torch.cat([a, with_state], -1), whole),
        _rel(torch.cat([a, without], -1), whole),
        _rel(with_state, without),
    )


def pipeline_level(
    sep: MambaSeparator, x: torch.Tensor, sr: int, thread_state: bool
) -> torch.Tensor:
    chunk = 2 * sr
    overlap = int(chunk * 0.5)
    step = chunk - overlap
    total = x.shape[-1]
    out = torch.zeros(1, 1, total)
    weight = torch.zeros_like(out)
    state = None
    pos = 0
    with torch.inference_mode():
        while pos < total:
            end = min(pos + chunk, total)
            seg = x[:, :, pos:end]
            valid = end - pos
            if valid < chunk:
                seg = torch.nn.functional.pad(seg, (0, chunk - valid))
            if thread_state:
                g, _b, _r, state, _m = sep(seg, state=state, return_state=True)
            else:
                g = sep(seg)[0]
            w = torch.ones(1, 1, valid)
            if pos > 0:
                n = min(overlap, valid)
                w[:, :, :n] *= torch.arange(1, n + 1, dtype=torch.float32) / overlap
            if end < total:
                n = min(overlap, valid)
                w[:, :, valid - n :] *= torch.arange(n - 1, -1, -1, dtype=torch.float32) / overlap
            out[:, :, pos:end] += g[:, :, :valid] * w
            weight[:, :, pos:end] += w
            pos += step
    return out / weight.clamp(min=1e-8)


def main(config_path: str = "configs/model_config.mamba.yaml") -> None:
    torch.set_num_threads(2)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    torch.manual_seed(0)
    sep = MambaSeparator(cfg).eval()
    sr = int(cfg["audio"]["sample_rate"])
    hop = int(cfg["audio"]["hop_length"])

    print(f"block      2-call vs 1-call, state threaded   : {block_level():.1e}")

    x = _signal(cfg, 4.0)
    with_state, without, delta = separator_level(sep, x, hop)
    print(f"separator  chunk-vs-whole, state threaded     : {with_state:.1e}")
    print(f"separator  chunk-vs-whole, state dropped      : {without:.1e}")
    print(f"separator  carried-vs-dropped difference      : {delta:.1e}")

    x = _signal(cfg, 6.0)
    with torch.inference_mode():
        whole = sep(x)[0]
    threaded = pipeline_level(sep, x, sr, True)
    dropped = pipeline_level(sep, x, sr, False)
    print(f"pipeline   chunk-vs-whole, state threaded     : {_rel(threaded, whole):.1e}")
    print(f"pipeline   chunk-vs-whole, state dropped      : {_rel(dropped, whole):.1e}")
    print(f"pipeline   carried-vs-dropped difference      : {_rel(threaded, dropped):.1e}")
    gain = _rel(dropped, whole) / max(_rel(threaded, whole), 1e-12)
    print(f"pipeline   accuracy improvement               : {gain:.3f}x")
    print()
    print("Verdict: the mechanism is exact at the block level, and the pipeline")
    print("cannot benefit from it -- the H-4 crossfade already absorbs the")
    print("boundary discontinuity. Threading state in the default path would")
    print("change every render by the measured-nonzero difference above for no")
    print("measurable accuracy gain, which is why it is not done.")


if __name__ == "__main__":
    main(*sys.argv[1:])
