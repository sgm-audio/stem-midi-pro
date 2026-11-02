# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Demucs / HTDemucs separator backend (pretrained)."""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from models.backends.base import SeparateOut, device_from_cfg, empty_metrics

log = logging.getLogger(__name__)

# HTDemucs sources are typically: drums, bass, other, vocals
# Guitar lives in "other" for standard 4-stem models.
_GUITAR_SOURCE_CANDIDATES = ("guitar", "other")
_BASS_SOURCE_CANDIDATES = ("bass",)


class DemucsSeparatorBackend:
    """
    Pretrained Demucs separator via ``demucs.api.Separator``.

    Maps:
      - guitar ← first of guitar/other present
      - bass   ← bass
      - residual ← mix − guitar − bass (time-aligned)
    """

    def __init__(self, cfg: dict):
        try:
            from demucs.api import Separator
        except ImportError as e:
            raise ImportError(
                "demucs is required for separator backend 'demucs'. "
                "Install with: pip install 'stem-midi-pro[pretrained]' "
                "or pip install demucs"
            ) from e

        backends = cfg.get("backends") or {}
        model_name = backends.get("demucs_model", "htdemucs")
        self.sample_rate = int(cfg.get("audio", {}).get("sample_rate", 44100))
        self.device = device_from_cfg(cfg)
        self._separator = Separator(
            model=model_name,
            device=str(self.device),
            shifts=int(backends.get("demucs_shifts", 1)),
            split=bool(backends.get("demucs_split", True)),
            progress=bool(backends.get("demucs_progress", False)),
        )
        log.info(
            "DemucsSeparatorBackend ready model=%s device=%s sr_model=%s",
            model_name,
            self.device,
            self._separator.samplerate,
        )

    def separate(self, audio: torch.Tensor) -> SeparateOut:
        """
        Args:
            audio: (B, 1, T) mono float tensor at cfg sample rate.
        """
        if audio.dim() != 3 or audio.shape[1] != 1:
            raise ValueError(f"Expected audio (B, 1, T), got {tuple(audio.shape)}")

        B, _, T = audio.shape
        guitars = []
        basses = []
        residuals = []
        metrics_rows = []

        for b in range(B):
            mono = audio[b, 0].detach().cpu()  # (T,)
            # Demucs expects (C, T); duplicate mono → stereo
            wav = mono.unsqueeze(0).repeat(2, 1).contiguous()
            _orig, stems = self._separator.separate_tensor(wav, sr=self.sample_rate)

            guitar = self._pick_stem(stems, _GUITAR_SOURCE_CANDIDATES)
            bass = self._pick_stem(stems, _BASS_SOURCE_CANDIDATES)
            # Mono downmix
            guitar_m = guitar.mean(dim=0)
            bass_m = bass.mean(dim=0)
            # Align length to input T
            guitar_m = self._match_len(guitar_m, T)
            bass_m = self._match_len(bass_m, T)
            residual_m = mono[:T] - guitar_m - bass_m

            guitars.append(guitar_m)
            basses.append(bass_m)
            residuals.append(residual_m)
            # Proxy metric: energy ratio guitar vs bass (not true SI-SDR)
            g_e = float((guitar_m**2).mean().clamp_min(1e-8))
            b_e = float((bass_m**2).mean().clamp_min(1e-8))
            proxy = min(30.0, max(0.0, 10.0 * torch.log10(torch.tensor(g_e / b_e + 1e-8)).item() + 15.0))
            metrics_rows.append([proxy, 0.5])

        device = audio.device
        guitar_t = torch.stack(guitars, dim=0).unsqueeze(1).to(device)
        bass_t = torch.stack(basses, dim=0).unsqueeze(1).to(device)
        residual_t = torch.stack(residuals, dim=0).unsqueeze(1).to(device)
        metrics = torch.tensor(metrics_rows, dtype=torch.float32, device=device)
        if metrics.numel() == 0:
            metrics = empty_metrics(B, device)

        return {
            "guitar": guitar_t,
            "bass": bass_t,
            "residual": residual_t,
            "metrics": metrics,
        }

    @staticmethod
    def _pick_stem(stems: dict, candidates: tuple) -> torch.Tensor:
        for name in candidates:
            if name in stems:
                return stems[name]
        available = ", ".join(sorted(stems.keys()))
        raise KeyError(
            f"None of {candidates} found in Demucs stems. Available: {available}"
        )

    @staticmethod
    def _match_len(wav: torch.Tensor, length: int) -> torch.Tensor:
        if wav.shape[-1] == length:
            return wav
        if wav.shape[-1] > length:
            return wav[..., :length]
        return F.pad(wav, (0, length - wav.shape[-1]))
