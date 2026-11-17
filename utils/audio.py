# SPDX-License-Identifier: Apache-2.0
"""Shared STFT/mel audio utilities for MambaSeparator and MambaTranscriber."""

from __future__ import annotations

import torch
import torch.nn as nn


def build_hann_window(n_fft: int) -> torch.Tensor:
    """Create a Hann window."""
    return torch.hann_window(n_fft)


def compute_stft(
    audio: torch.Tensor,
    n_fft: int,
    hop_length: int,
    window: torch.Tensor,
    *,
    pad_mode: str = "reflect",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute STFT magnitude and phase. Returns (mag, phase) tuple."""
    spec = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
        pad_mode=pad_mode,
    )
    return torch.abs(spec), torch.angle(spec)


def polar_to_complex(mag: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Convert magnitude + phase back to complex tensor."""
    return mag * torch.exp(1j * phase)


def compute_istft(
    spec: torch.Tensor,
    n_fft: int,
    hop_length: int,
    window: torch.Tensor,
    length: int | None = None,
) -> torch.Tensor:
    """Inverse STFT from complex spectrogram."""
    return torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        length=length,
    )


def build_mel_basis(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> torch.Tensor:
    """Build mel filter bank matrix. Returns (n_mels, n_fft//2+1)."""
    if fmax is None:
        fmax = sr // 2

    mel_fmin = 2595.0 * torch.log10(torch.tensor(1.0 + fmin / 700.0))
    mel_fmax = 2595.0 * torch.log10(torch.tensor(1.0 + fmax / 700.0))

    mel_points = torch.linspace(mel_fmin, mel_fmax, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_indices = torch.floor((n_fft + 1) * hz_points / sr)

    fbank = torch.zeros((n_mels, n_fft // 2 + 1))
    for j in range(1, n_mels + 1):
        left = int(bin_indices[j - 1])
        center = int(bin_indices[j])
        right = int(bin_indices[j + 1])
        for i in range(left, center):
            if center - left > 0:
                fbank[j - 1, i] = (i - left) / (center - left)
        for i in range(center, right):
            if right - center > 0:
                fbank[j - 1, i] = (right - i) / (right - center)
    return fbank


def mel_spectrogram(
    audio: torch.Tensor,
    n_fft: int,
    hop_length: int,
    window: torch.Tensor,
    mel_basis: torch.Tensor,
) -> torch.Tensor:
    """Compute log-mel spectrogram from raw audio."""
    spec = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
        pad_mode="reflect",
    )
    mag = torch.abs(spec)
    mel_spec = torch.matmul(mel_basis.to(audio.device), mag)
    return torch.log(torch.clamp(mel_spec, min=1e-8))


class STFTModule(nn.Module):
    """Module wrapper around STFT utilities (for use in nn.Module subclasses)."""

    def __init__(self, n_fft: int, hop_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return compute_stft(audio, self.n_fft, self.hop_length, self.window)

    def inverse(self, spec: torch.Tensor, length: int | None = None) -> torch.Tensor:
        return compute_istft(spec, self.n_fft, self.hop_length, self.window, length)


def resample_audio(audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Band-limited resampling for (B, C, T) tensors or 1-D input.

    ``F.interpolate(mode="linear")`` is a triangle filter with no anti-imaging
    or anti-aliasing, so an 8 kHz upload acquired a -4 dB spectral image at
    4.5 kHz (BUG-AUDIO-6) and a 48 kHz source could fold content above the new
    Nyquist back into the audible band. Both corrupt the mel input the
    transcriber sees, so the sample-rate conversion must be band-limited.

    Uses ``librosa.resample`` (polyphase / soxr) when available, which is a
    project dependency; falls back to linear interpolation only if librosa is
    missing, so an install without it degrades rather than crashes.
    """
    if orig_sr == target_sr:
        return audio

    squeeze = audio.dim() == 1
    x = audio.unsqueeze(0).unsqueeze(0) if squeeze else audio
    if x.dim() != 3:
        raise ValueError(f"expected (B, C, T) or (T,) audio, got shape {tuple(x.shape)}")

    n_out = max(1, round(x.shape[-1] * target_sr / orig_sr))
    flat = x.reshape(-1, x.shape[-1]).cpu().numpy()

    try:
        import librosa

        out = librosa.resample(flat, orig_sr=orig_sr, target_sr=target_sr, axis=-1)
        resampled = torch.from_numpy(out).to(dtype=x.dtype).reshape(x.shape[0], x.shape[1], -1)
    except ImportError:  # pragma: no cover - librosa is a hard dependency
        resampled = torch.nn.functional.interpolate(
            x, size=n_out, mode="linear", align_corners=False
        )

    resampled = resampled.to(device=x.device).reshape(*x.shape[:-1], -1)
    return resampled.squeeze(0).squeeze(0) if squeeze else resampled
