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
    """Composite loss: MR-STFT + spectral flatness + crest factor + onset alignment."""
    
    def __init__(self, cfg: dict):
        super().__init__()
        self.mr_stft = MultiResolutionSTFTLoss()
        self.cfg = cfg
        # TODO: Implement onset_f1, pitch_ce, and velocity_mae losses
        # (config keys: onset_f1_weight, pitch_ce_weight, velocity_mae_weight)
        
    def forward(self, pred_stems, target_stems, pred_onsets=None, target_onsets=None):
        # We expect pred_stems and target_stems to be lists of [guitar, bass] tensors
        # Primary: MR-STFT for spectral fidelity (applied to each stem and summed)
        spectral_loss = 0
        for pred, target in zip(pred_stems, target_stems):
            spectral_loss += self.mr_stft(pred, target)
        
        # Regularization: preserve dynamic range (applied to each stem and averaged)
        flatness_loss = 0
        crest_loss = 0
        for pred, target in zip(pred_stems, target_stems):
            flatness_loss += self._spectral_flatness_loss(pred, target)
            crest_loss += self._crest_factor_loss(pred, target)
        flatness_loss /= len(pred_stems)
        crest_loss /= len(pred_stems)
        
        # Cross-modal: align MIDI onsets with stem energy peaks
        alignment_loss = 0
        if pred_onsets is not None and target_onsets is not None:
            # We assume we are aligning to the guitar stem for transcription
            alignment_loss = self._onset_alignment_loss(pred_onsets, target_onsets, pred_stems[0])
        
        # Weighted sum
        total = (
            self.cfg['loss']['mr_stft_weight'] * spectral_loss +
            self.cfg['loss']['spectral_flatness_weight'] * flatness_loss +
            self.cfg['loss']['crest_factor_weight'] * crest_loss +
            self.cfg['loss']['cross_modal_alignment_weight'] * alignment_loss
        )
        return total, {
            'spectral': float(spectral_loss),
            'flatness': float(flatness_loss),
            'crest': float(crest_loss),
            'alignment': float(alignment_loss),
        }
    
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
        # Compute stem energy envelope
        energy = torch.mean(stem**2, dim=1, keepdim=True)  # (B, 1, T)
        energy_smooth = F.avg_pool1d(energy, kernel_size=101, padding=50, stride=1)
        
        # Onset-triggered energy attention
        pred_energy = (pred_onsets * energy_smooth).sum()
        target_energy = (target_onsets * energy_smooth).sum()
        return F.mse_loss(pred_energy, target_energy)
