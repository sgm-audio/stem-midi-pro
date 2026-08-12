"""Mamba-3 Per-Track Adaptive Filter Bank model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


@dataclass
class TrackProcessorConfig:
    sample_rate: int = 44100
    hop_length: int = 512
    n_fft: int = 2048
    n_mels: int = 80
    d_model: int = 256
    d_state: int = 32
    d_conv: int = 4
    expand: int = 2
    n_layers: int = 6
    crepe_dim: int = 360
    use_fast_path: bool = True


class STFTEncoder(nn.Module):
    """Fixed STFT with learnable post-projection."""

    def __init__(self, cfg: TrackProcessorConfig):
        super().__init__()
        self.n_fft = cfg.n_fft
        self.hop_length = cfg.hop_length
        self.register_buffer("window", torch.hann_window(self.n_fft))

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spec = torch.stft(
            wav, n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, return_complex=True, pad_mode='reflect'
        )
        mag = spec.abs()
        mag_sq = mag ** 2
        phase = spec.angle()
        return mag_sq, phase

    def invert(self, mag: torch.Tensor, phase: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        spec = mag * torch.exp(1j * phase)
        return torch.istft(
            spec, n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, length=length
        )


class MelEncoder(nn.Module):
    """Mel filterbank from magnitude spectrogram."""

    def __init__(self, cfg: TrackProcessorConfig):
        super().__init__()
        self.n_mels = cfg.n_mels
        self.n_fft = cfg.n_fft
        self.sr = cfg.sample_rate
        self.register_buffer("mel_basis", self._create_mel_basis())

    def _create_mel_basis(self) -> torch.Tensor:
        fmin, fmax = 0, self.sr // 2
        mel_fmin = 2595 * torch.log10(torch.tensor(1 + fmin / 700))
        mel_fmax = 2595 * torch.log10(torch.tensor(1 + fmax / 700))
        mel_points = torch.linspace(mel_fmin, mel_fmax, self.n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_idx = torch.floor((self.n_fft + 1) * hz_points / self.sr).long()
        fbank = torch.zeros((self.n_mels, self.n_fft // 2 + 1))
        for j in range(1, self.n_mels + 1):
            left, center, right = int(bin_idx[j - 1]), int(bin_idx[j]), int(bin_idx[j + 1])
            for i in range(left, center):
                if center - left > 0:
                    fbank[j - 1, i] = (i - left) / (center - left)
            for i in range(center, right):
                if right - center > 0:
                    fbank[j - 1, i] = (right - i) / (right - center)
        return fbank

    def forward(self, mag_sq: torch.Tensor) -> torch.Tensor:
        mel = torch.matmul(self.mel_basis, mag_sq)
        return torch.log1p(mel)


class R8Encoder(nn.Module):
    """8-band raw-feature encoder (spectral centroid, bandwidth, etc.)."""

    def __init__(self, cfg: TrackProcessorConfig):
        super().__init__()
        self.n_fft = cfg.n_fft
        self.hop_length = cfg.hop_length
        self.proj = nn.Linear(8, cfg.d_model // 4)

    def forward(self, wav: torch.Tensor, mag_sq: torch.Tensor) -> torch.Tensor:
        B, F, T = mag_sq.shape
        freqs = torch.linspace(0, self.n_fft // 2, F, device=mag_sq.device)

        centroid = (mag_sq * freqs.unsqueeze(0).unsqueeze(-1)).sum(dim=1) / (mag_sq.sum(dim=1) + 1e-8)
        bandwidth = ((mag_sq * (freqs.unsqueeze(0).unsqueeze(-1) - centroid.unsqueeze(1)) ** 2).sum(dim=1) / (mag_sq.sum(dim=1) + 1e-8)).sqrt()
        crest = mag_sq.max(dim=1).values / (mag_sq.mean(dim=1) + 1e-8)
        flatness = mag_sq.exp().mean(dim=1).log() / (mag_sq.mean(dim=1) + 1e-8).log()

        rms = (wav ** 2).mean(dim=-1, keepdim=True).sqrt()
        zcr = ((wav[:, :-1] * wav[:, 1:]) < 0).float().mean(dim=-1, keepdim=True)

        r8 = torch.stack([
            centroid, bandwidth, crest, flatness,
            rms.squeeze(-1), zcr.squeeze(-1),
            centroid / (self.n_fft // 2 + 1),
            flatness / (crest + 1e-8),
        ], dim=-1)
        return self.proj(r8).transpose(1, 2)


class SemanticEncoder(nn.Module):
    """Transformer encoder for semantic embedding (contrastive pretraining)."""

    def __init__(self, cfg: TrackProcessorConfig):
        super().__init__()
        d = cfg.d_model
        self.proj = nn.Linear(d + cfg.d_model // 4, d)
        self.pos_enc = nn.Parameter(torch.randn(1, 500, d) * 0.02)
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, num_heads=4, batch_first=True)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, mel: torch.Tensor, r8: torch.Tensor, return_sequence: bool = True) -> torch.Tensor:
        x = torch.cat([mel.transpose(1, 2), r8.transpose(1, 2)], dim=-1)
        x = self.proj(x)
        x = x + self.pos_enc[:, :x.shape[1], :]
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        if not return_sequence:
            x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        return x


class PerTrackProcessor(nn.Module):
    """Mamba-3 per-track adaptive filter bank with semantic encoder."""

    def __init__(self, cfg: TrackProcessorConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.stft_enc = STFTEncoder(cfg)
        self.mel_enc = MelEncoder(cfg)
        self.r8_enc = R8Encoder(cfg)
        self.semantic_enc = SemanticEncoder(cfg)

        self.crepe_proj = nn.Linear(cfg.crepe_dim, d)

        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d, d_state=cfg.d_state, d_conv=cfg.d_conv,
                expand=cfg.expand, use_fast_path=cfg.use_fast_path,
            ) for _ in range(cfg.n_layers)
        ])

        self.output_proj = nn.Linear(d, cfg.n_fft // 2 + 1)
        self.track_emb_proj = nn.Linear(d, d)

    def _compute_stft(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.stft_enc(wav)

    def _compute_mel(self, mag_sq: torch.Tensor) -> torch.Tensor:
        return self.mel_enc(mag_sq)

    def _extract_r8_raw(self, wav: torch.Tensor, mag_sq: torch.Tensor) -> torch.Tensor:
        return self.r8_enc(wav, mag_sq)

    def forward(
        self,
        wav: torch.Tensor,
        crepe: Optional[torch.Tensor] = None,
        other_embs: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        mag_sq, phase = self._compute_stft(wav)
        mel = self._compute_mel(mag_sq)
        r8 = self._extract_r8_raw(wav, mag_sq)
        sem = self.semantic_enc(mel, r8, return_sequence=True)

        x = sem.transpose(1, 2)

        if crepe is not None:
            crepe_feat = self.crepe_proj(crepe)
            x = x + crepe_feat.transpose(1, 2)

        for layer in self.mamba_layers:
            x = layer(x)

        output_stft = self.output_proj(x).transpose(1, 2)
        track_emb = self.track_emb_proj(x.mean(dim=1))

        if other_embs is not None:
            cross = (track_emb.unsqueeze(1) * other_embs).sum(dim=-1)
            track_emb = track_emb + 0.1 * cross

        return {"output_stft": output_stft, "track_emb": track_emb, "phase": phase}

    def waveform_from_stft(self, pred_stft: torch.Tensor, phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        if phase is None:
            phase = torch.zeros_like(pred_stft)
        mag = pred_stft.abs()
        return self.stft_enc.invert(mag, phase if phase is not None else torch.zeros_like(pred_stft))
