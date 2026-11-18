#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the numeric claims in FIXES.md (C-1, C-4, dt init).

Run: ``python3 scripts/verify_numerics.py [configs/model_config.yaml]``

Prints, for the separator and the transcriber:

* the hidden-state RMS after each Mamba block (a stack that collapses shows a
  ladder like 1.0 -> 1e-10; the healthy stack stays flat). Measured on this repo
  (small CPU config): 1 -> 4.9e-2 -> 1.2e-3 -> 2.9e-5 -> 7.4e-7 -> 1.7e-8 ->
  4.0e-10 — the collapse is REAL here and pinned as C-1 in
  ``tests/test_numerics.py``,
* per-stem mask mean/std (a collapsing stack pins every mask at the same value),
* mixture consistency ||guitar+bass+residual - mix|| / ||mix|| (independent
  sigmoid masks do not partition the mix: ~0.5 on this repo, pinned as SEP-1),
* ``dt_proj.bias`` range and chunked-vs-unchunked scan agreement.
"""

from __future__ import annotations

import sys

import torch
import yaml

sys.path.insert(0, ".")

from models import _mamba_compat
from models.mamba_separator import MambaSeparator
from models.mamba_transcriber import MambaTranscriber


def _rms(t: torch.Tensor) -> float:
    return float(t.pow(2).mean().sqrt())


def main(config_path: str = "configs/model_config.yaml") -> None:
    torch.set_num_threads(2)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    torch.manual_seed(0)
    sep = MambaSeparator(cfg).eval()
    tr = MambaTranscriber(cfg).eval()

    x = torch.randn(1, 1, 3 * 44100) * 0.1
    with torch.inference_mode():
        spec = torch.stft(
            x.squeeze(1),
            cfg["audio"]["n_fft"],
            cfg["audio"]["hop_length"],
            window=sep.window,
            return_complex=True,
        )
        h = sep.input_proj(torch.abs(spec).transpose(1, 2))
        ladder = [_rms(h)]
        for block in sep.mamba_blocks:  # no per-block norm / residual add here
            h = block(h)
            ladder.append(_rms(h))
        masks = torch.stack([head(h).sigmoid() for head in sep.mask_heads.values()], 0)

        guitar, bass, residual, _, metrics = sep(x)
        mel = tr._mel_spectrogram(guitar.squeeze(1)).transpose(1, 2)
        ht = tr.input_proj(mel)
        tr_ladder = [_rms(ht)]
        for block in tr.mamba:
            ht = block(ht)
            tr_ladder.append(_rms(ht))
        _, _, _, _, confidence = tr(guitar)

    print("separator block RMS :", " ".join(f"{v:.2e}" for v in ladder))
    print("transcriber block RMS:", " ".join(f"{v:.2e}" for v in tr_ladder))
    for name, mask in zip(sep.mask_heads, masks, strict=True):
        print(f"mask {name:<9}: mean {float(mask.mean()):.3f}  std {float(mask.std()):.3f}")
    rel = float((guitar + bass + residual - x).norm() / x.norm())
    print(f"mixture consistency : {rel:.2e}")
    print("proxy / phase       :", [round(float(v), 2) for v in metrics[0]])
    print(f"confidence mean/std : {float(confidence.mean()):.3f} / {float(confidence.std()):.3f}")

    torch.manual_seed(0)
    mamba = _mamba_compat.Mamba(d_model=32, d_state=8, d_conv=4, expand=2).eval()
    with torch.inference_mode():
        dt0 = torch.nn.functional.softplus(mamba.dt_proj.bias)
        probe = torch.randn(1, 200, 32)
        mamba.SCAN_CHUNK = 10_000
        single = mamba(probe)
        mamba.SCAN_CHUNK = 17
        chunked = mamba(probe)
    print(f"softplus(dt_bias)   : {float(dt0.min()):.5f} … {float(dt0.max()):.5f}")
    print(f"chunked vs 1-pass   : max|Δ| = {float((single - chunked).abs().max()):.2e}")

    finite = all(torch.isfinite(t).all() for t in (guitar, bass, residual, confidence))
    print("all outputs finite  :", finite)


if __name__ == "__main__":
    main(*sys.argv[1:])
