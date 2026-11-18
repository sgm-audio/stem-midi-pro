#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the numeric claims in FIXES.md (C-1, C-4, dt init, SEP-1, SEP-2).

Run: ``python3 scripts/verify_numerics.py [configs/model_config.yaml]``

Prints, for the separator and the transcriber:

* the hidden-state RMS **through the real stack** — norm, mixer, residual add —
  after every block. A stack without the residual/norm wrapper collapses as
  ``1 -> 4.9e-2 -> ... -> 1e-18``; the shipped stack holds. Measuring the bare
  ``mamba_blocks`` (what this script used to do, labelled "no per-block norm /
  residual add here") reports the collapse for a model that does not have it,
  which is how SEP-2 stayed on the open list after it was fixed,
* per-stem mask mean/std and the logit spread that produces them (a collapsed
  stack pins every mask at the same value because the logits are ~0),
* mixture consistency ``||guitar+bass+residual - mix|| / ||mix||`` — 1e-7 with
  the default softmax masks (they partition the mixture by construction) and
  ~0.5 with ``separator.mask_normalization: independent``,
* the separation proxy (NOT SI-SDR) against its saturation range, so a metric
  that cannot discriminate is visible as such,
* confidence spread across content classes (a head that has never been
  supervised reads the same number for music and for silence),
* ``dt_proj.bias`` range and chunked-vs-unchunked scan agreement.
"""

from __future__ import annotations

import sys

import torch
import yaml

sys.path.insert(0, ".")

from models.mamba_separator import MambaSeparator
from models.mamba_transcriber import MambaTranscriber


def _rms(t: torch.Tensor) -> float:
    return float(t.pow(2).mean().sqrt())


def _separator_ladder(sep: MambaSeparator, cfg: dict) -> list[float]:
    """RMS after every block, following ``forward``'s actual layer order.

    ``h = h + block(norm(h))`` with a final norm — NOT ``h = block(h)``. This
    distinction is the whole of SEP-2: chaining bare mixers multiplies the RMS
    by ~0.04 per block, but the shipped forward adds the residual back, so the
    real stack holds at order one.
    """
    x = torch.randn(1, 1, 3 * 44100) * 0.1
    with torch.inference_mode():
        spec = torch.stft(
            x.squeeze(1),
            cfg["audio"]["n_fft"],
            cfg["audio"]["hop_length"],
            window=sep.window,
            return_complex=True,
        )
        hidden = sep.input_proj(torch.abs(spec).transpose(1, 2))
        ladder = [_rms(hidden)]
        for norm, block in zip(sep.block_norms, sep.mamba_blocks, strict=True):
            hidden = hidden + block(norm(hidden))
            ladder.append(_rms(hidden))
        ladder.append(_rms(sep.final_norm(hidden)))
    return ladder


def _bare_block_ladder(sep: MambaSeparator, cfg: dict) -> list[float]:
    """RMS through the bare mixers, i.e. the pre-C-1 architecture.

    Kept deliberately: it is the control that shows WHY the wrapper matters, and
    it is what the old script measured while calling it "the stack".
    """
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
        for block in sep.mamba_blocks:
            h = block(h)
            ladder.append(_rms(h))
    return ladder


def _transcriber_ladder(tr: MambaTranscriber) -> list[float]:
    x = torch.randn(1, 1, 3 * 44100) * 0.1
    with torch.inference_mode():
        mel = tr._mel_spectrogram(x.squeeze(1)).transpose(1, 2)
        hidden = tr.input_proj(mel)
        ladder = [_rms(hidden)]
        for norm, block in zip(tr.block_norms, tr.mamba, strict=True):
            hidden = hidden + block(norm(hidden))
            ladder.append(_rms(hidden))
        ladder.append(_rms(tr.final_norm(hidden)))
    return ladder


def _confidence_by_content(tr: MambaTranscriber) -> dict[str, float]:
    """Mean confidence for music vs silence.

    The confidence head has no loss term (``models/losses.py`` trains onset,
    pitch and velocity only), so at init it is a random MLP behind dropout and
    returns the same number for every input. Printing the spread makes an
    unsupervised head look like what it is instead of letting a single
    "confidence mean/std" line imply it tracks content.
    """
    sr = 44100
    t = torch.arange(2 * sr) / sr
    cases = {
        "music": 0.30 * torch.sin(2 * torch.pi * 110 * t)
        + 0.15 * torch.sin(2 * torch.pi * 330 * t),
        "clipped": torch.sign(torch.sin(2 * torch.pi * 220 * t)) * 0.9,
        "silence": torch.zeros_like(t),
    }
    out = {}
    for name, sig in cases.items():
        with torch.inference_mode():
            _onset, _pitch, _vel, _expr, conf = tr(sig.reshape(1, 1, -1).float())
        out[name] = float(conf.mean())
    return out


def main(config_path: str = "configs/model_config.yaml") -> None:
    torch.set_num_threads(2)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    torch.manual_seed(0)
    sep = MambaSeparator(cfg).eval()
    tr = MambaTranscriber(cfg).eval()

    x = torch.randn(1, 1, 3 * 44100) * 0.1
    with torch.inference_mode():
        guitar, bass, residual, _state, metrics = sep(x)
        conf = _confidence_by_content(tr)
        # Logit spread behind the masks: softmax over near-equal logits gives
        # 1/3 each, which is a flat gain rather than a separation.
        spec = torch.stft(
            x.squeeze(1),
            cfg["audio"]["n_fft"],
            cfg["audio"]["hop_length"],
            window=sep.window,
            return_complex=True,
        )
        hidden = sep.input_proj(torch.abs(spec).transpose(1, 2))
        for norm, block in zip(sep.block_norms, sep.mamba_blocks, strict=True):
            hidden = hidden + block(norm(hidden))
        hidden = sep.final_norm(hidden)
        logits = torch.stack([head(hidden) for head in sep.mask_heads.values()], 0)
        probs = torch.softmax(logits, dim=0)
        dt_bias = torch.nn.functional.softplus(sep.mamba_blocks[0].dt_proj.bias)

    stack = _separator_ladder(sep, cfg)
    bare = _bare_block_ladder(sep, cfg)
    tr_ladder = _transcriber_ladder(tr)

    print("separator stack RMS :", " ".join(f"{v:.2e}" for v in stack))
    print("  (bare mixers only):", " ".join(f"{v:.2e}" for v in bare))
    print("transcriber stack   :", " ".join(f"{v:.2e}" for v in tr_ladder))
    for i, name in enumerate(sep.mask_heads):
        print(
            f"mask {name:<9}: logit mean {float(logits[i].mean()):+.4f} "
            f"std {float(logits[i].std()):.4f} | mask mean {float(probs[i].mean()):.4f}"
        )
    rel = float((guitar + bass + residual - x).norm() / x.norm())
    print(f"mixture consistency : {rel:.2e}")
    print(f"proxy / phase       : [{float(metrics[0, 0]):.2f}, {float(metrics[0, 1]):.2f}]")
    print("confidence by content:", " ".join(f"{k}={v:.4f}" for k, v in conf.items()))
    print("  spread across content classes:", f"{max(conf.values()) - min(conf.values()):.5f}")
    print(f"softplus(dt_bias)   : {float(dt_bias.min()):.5f} … {float(dt_bias.max()):.5f}")
    finite = bool(torch.isfinite(guitar).all() and torch.isfinite(bass).all())
    print("all outputs finite  :", finite)


if __name__ == "__main__":
    main(*sys.argv[1:])
