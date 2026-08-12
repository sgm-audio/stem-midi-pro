"""Loss functions for Mamba-3 Per-Track Adaptive Filter Bank training."""
from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchaudio

try:
    import auraloss.freq as alf
    HAS_AURALOSS = True
except ImportError:
    HAS_AURALOSS = False


class TrackProcessorLoss(nn.Module):
    """Composite loss: MR-STFT + Mel-L1 + SI-SDR (ramped)."""

    def __init__(self, cfg, sample_rate: int = 44100):
        super().__init__()
        self.cfg = cfg
        self.sr = sample_rate

        if HAS_AURALOSS:
            self.mr_stft = alf.MultiResolutionSTFTLoss(
                fft_sizes=[512, 1024, 2048, 4096],
                hop_sizes=[128, 256, 512, 1024],
                win_lengths=[512, 1024, 2048, 4096],
                w_sc=1.0, w_log_mag=1.0, sample_rate=sample_rate,
            )
        else:
            self.mr_stft = None

        self.mel_transforms = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_fft=n,
                hop_length=n // 4, n_mels=n_mels,
            ) for n, n_mels in [(512, 64), (1024, 80), (2048, 128)]
        ])

        self.global_step = 0

    def _si_sdr(self, pred: Tensor, target: Tensor) -> Tensor:
        eps = 1e-8
        target_energy = (target ** 2).sum(dim=-1, keepdim=True)
        dot = (pred * target).sum(dim=-1, keepdim=True)
        proj = dot / (target_energy + eps) * target
        noise = pred - proj
        si_sdr = 10 * torch.log10(
            (proj ** 2).sum(-1) / ((noise ** 2).sum(-1) + eps) + eps
        )
        return -si_sdr.mean()

    def _mel_l1(self, pred_wave: Tensor, target_wave: Tensor) -> Tensor:
        loss = 0.0
        for mel_tf in self.mel_transforms:
            mel_tf = mel_tf.to(pred_wave.device)
            p = torch.log1p(mel_tf(pred_wave))
            t = torch.log1p(mel_tf(target_wave))
            loss = loss + F.l1_loss(p, t)
        return loss / len(self.mel_transforms)

    def _si_sdr_weight(self) -> float:
        if self.global_step < 5000:
            return 0.0
        elif self.global_step > 15000:
            return 0.1
        else:
            return 0.1 * (self.global_step - 5000) / 10000

    def forward(
        self, pred_stft: Tensor, target_wave: Tensor, model: nn.Module,
    ) -> dict[str, Tensor]:
        pred_wave = model.waveform_from_stft(pred_stft)
        L = min(pred_wave.shape[-1], target_wave.shape[-1])
        pred_wave = pred_wave[..., :L]
        target_wave = target_wave[..., :L]

        losses = {}

        if self.mr_stft is not None:
            losses['mr_stft'] = self.mr_stft(
                pred_wave.unsqueeze(1), target_wave.unsqueeze(1)
            )
        else:
            window = torch.hann_window(2048, device=pred_wave.device)
            p_stft = torch.stft(pred_wave, 2048, 512, window=window, return_complex=True)
            t_stft = torch.stft(target_wave, 2048, 512, window=window, return_complex=True)
            losses['mr_stft'] = F.l1_loss(p_stft.abs(), t_stft.abs())

        losses['mel_l1'] = self._mel_l1(pred_wave, target_wave)

        si_sdr_w = self._si_sdr_weight()
        if si_sdr_w > 0:
            losses['si_sdr'] = self._si_sdr(pred_wave, target_wave) * si_sdr_w
        else:
            losses['si_sdr'] = torch.tensor(0.0, device=pred_wave.device)

        losses['total'] = (
            1.0 * losses['mr_stft'] + 0.5 * losses['mel_l1'] + losses['si_sdr']
        )
        self.global_step += 1
        return losses


class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross-entropy loss for contrastive learning."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.T = temperature

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        B = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.T) / self.T
        labels = torch.arange(B, device=z.device)
        labels = torch.cat([labels + B, labels])
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float('-inf'))
        return F.cross_entropy(sim, labels)


def augment_for_contrastive(wav: Tensor) -> Tensor:
    """Apply augmentation for contrastive learning pairs."""
    gain_db = random.uniform(-4, 4)
    wav = wav * (10 ** (gain_db / 20))
    if random.random() < 0.5:
        wav = -wav
    shift = random.randint(0, wav.shape[-1] // 4)
    wav = torch.roll(wav, shift, dims=-1)
    noise_level = random.uniform(0, 0.005)
    wav = wav + noise_level * torch.randn_like(wav)
    return wav.clamp(-1.0, 1.0)


def encode_for_contrastive(
    model: nn.Module, wav: Tensor, cfg,
) -> Tensor:
    """Encode waveform to embedding for contrastive loss."""
    mag_sq, _ = model._compute_stft(wav)
    mel = model._compute_mel(mag_sq)
    r8 = model._extract_r8_raw(wav, mag_sq)
    return model.semantic_enc(mel, r8, return_sequence=False)
