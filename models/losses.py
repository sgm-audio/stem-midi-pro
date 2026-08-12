# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionSTFTLoss(nn.Module):
    """MR-STFT loss for perceptual stem fidelity (3 resolutions)."""

    def __init__(self, resolutions=[(2048, 512), (1024, 256), (512, 128)]):
        super().__init__()
        self.resolutions = resolutions
        self.register_buffer("win_2048", torch.hann_window(2048))
        self.register_buffer("win_1024", torch.hann_window(1024))
        self.register_buffer("win_512", torch.hann_window(512))

    def _window_for(self, n_fft: int) -> torch.Tensor:
        if n_fft == 2048:
            return self.win_2048
        if n_fft == 1024:
            return self.win_1024
        if n_fft == 512:
            return self.win_512
        return torch.hann_window(n_fft, device=self.win_2048.device)

    def forward(self, x, y):
        # Flatten (B, C, T) -> (B*C, T) so torch.stft accepts the shape
        x_flat = x.reshape(-1, x.shape[-1])
        y_flat = y.reshape(-1, y.shape[-1])

        loss = 0
        for n_fft, hop in self.resolutions:
            win = self._window_for(n_fft)
            X = torch.log(torch.stft(x_flat, n_fft, hop, window=win, return_complex=True).abs() + 1e-8)
            Y = torch.log(torch.stft(y_flat, n_fft, hop, window=win, return_complex=True).abs() + 1e-8)
            loss += F.l1_loss(X, Y)

            X_mag = torch.stft(x_flat, n_fft, hop, window=win, return_complex=True).abs()
            Y_mag = torch.stft(y_flat, n_fft, hop, window=win, return_complex=True).abs()
            loss += F.mse_loss(X_mag, Y_mag)
        return loss / len(self.resolutions)


def _make_windows(resolutions):
    """Build a list of Hann windows buffer for each resolution (unused; kept for compat)."""
    return [torch.hann_window(n_fft) for n_fft, _ in resolutions]


class PerceptualAudioLoss(nn.Module):
    """Composite loss: MR-STFT + flatness + crest + onset + pitch + velocity + alignment."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.mr_stft = MultiResolutionSTFTLoss()
        self.cfg = cfg

    def forward(
        self,
        pred_stems,
        target_stems,
        pred_onsets=None,
        target_onsets=None,
        pred_pitch=None,
        target_pitch=None,
        pred_velocity=None,
        target_velocity=None,
    ):
        """
        Args:
            pred_stems / target_stems: lists of [guitar, bass] (B, 1, T) tensors.
            pred_onsets / target_onsets: (B, T_frames, 1).
            pred_pitch: (B, T_frames, V) logits.
            target_pitch: (B, T_frames, V) one-hot (or soft).
            pred_velocity / target_velocity: (B, T_frames, 1) in [0, 1].
        """
        device = pred_stems[0].device

        # Primary: MR-STFT for spectral fidelity
        spectral_loss = 0
        for pred, target in zip(pred_stems, target_stems):
            spectral_loss += self.mr_stft(pred, target)

        # Regularization: preserve dynamic range
        flatness_loss = 0
        crest_loss = 0
        for pred, target in zip(pred_stems, target_stems):
            flatness_loss += self._spectral_flatness_loss(pred, target)
            crest_loss += self._crest_factor_loss(pred, target)
        flatness_loss /= len(pred_stems)
        crest_loss /= len(pred_stems)

        # Transcription losses (skipped when targets absent)
        onset_loss = torch.zeros((), device=device)
        if pred_onsets is not None and target_onsets is not None:
            onset_loss = self._onset_loss(pred_onsets, target_onsets)

        pitch_loss = torch.zeros((), device=device)
        if pred_pitch is not None and target_pitch is not None:
            pitch_loss = self._pitch_ce_loss(pred_pitch, target_pitch)

        velocity_loss = torch.zeros((), device=device)
        if pred_velocity is not None and target_velocity is not None:
            velocity_loss = F.l1_loss(pred_velocity, target_velocity)

        # Cross-modal: align MIDI onsets with stem energy peaks
        alignment_loss = torch.zeros((), device=device)
        if pred_onsets is not None and target_onsets is not None:
            alignment_loss = self._onset_alignment_loss(
                pred_onsets, target_onsets, pred_stems[0]
            )

        total = (
            self.cfg["loss"]["mr_stft_weight"] * spectral_loss
            + self.cfg["loss"]["spectral_flatness_weight"] * flatness_loss
            + self.cfg["loss"]["crest_factor_weight"] * crest_loss
            + self.cfg["loss"]["onset_f1_weight"] * onset_loss
            + self.cfg["loss"]["pitch_ce_weight"] * pitch_loss
            + self.cfg["loss"]["velocity_mae_weight"] * velocity_loss
            + self.cfg["loss"]["cross_modal_alignment_weight"] * alignment_loss
        )
        return total, {
            "spectral": float(spectral_loss),
            "flatness": float(flatness_loss),
            "crest": float(crest_loss),
            "onset": float(onset_loss),
            "pitch": float(pitch_loss),
            "velocity": float(velocity_loss),
            "alignment": float(alignment_loss),
        }

    @staticmethod
    def _onset_loss(pred_onsets: torch.Tensor, target_onsets: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy (with logits) for onset detection."""
        return F.binary_cross_entropy_with_logits(pred_onsets, target_onsets)

    @staticmethod
    def _pitch_ce_loss(pred_pitch: torch.Tensor, target_pitch: torch.Tensor) -> torch.Tensor:
        """Cross-entropy over V-way pitch logits against one-hot/soft targets."""
        log_probs = F.log_softmax(pred_pitch, dim=-1)
        return -(target_pitch * log_probs).sum(dim=-1).mean()

    def _spectral_flatness_loss(self, x, y):
        """Penalize unnatural spectral smoothing (preserves transient detail)."""
        x_flat = x.reshape(-1, x.shape[-1])
        y_flat = y.reshape(-1, y.shape[-1])
        win = torch.hann_window(2048, device=x.device)

        def flatness(spec):
            geom = torch.exp(torch.mean(torch.log(spec + 1e-8), dim=-1, keepdim=True))
            arith = torch.mean(spec, dim=-1, keepdim=True)
            return geom / (arith + 1e-8)

        x_spec = torch.stft(x_flat, 2048, 512, window=win, return_complex=True).abs()
        y_spec = torch.stft(y_flat, 2048, 512, window=win, return_complex=True).abs()
        return F.mse_loss(flatness(x_spec), flatness(y_spec))

    def _crest_factor_loss(self, x, y):
        """Preserve peak-to-RMS ratio (critical for transients)."""
        def crest(audio):
            peak = audio.abs().max(dim=-1).values
            rms = torch.sqrt(torch.mean(audio**2, dim=-1))
            return peak / (rms + 1e-8)
        return F.mse_loss(crest(x), crest(y))

    def _onset_alignment_loss(self, pred_onsets, target_onsets, stem):
        """Ensure MIDI onsets align with stem energy peaks."""
        energy = torch.mean(stem**2, dim=1, keepdim=True)  # (B, 1, T)
        energy_smooth = F.avg_pool1d(energy, kernel_size=101, padding=50, stride=1)

        pred_energy = (pred_onsets * energy_smooth).sum()
        target_energy = (target_onsets * energy_smooth).sum()
        return F.mse_loss(pred_energy, target_energy)
