import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionSTFTLoss(nn.Module):
    """MR-STFT loss for perceptual stem fidelity (6 resolutions)."""

    def __init__(self, resolutions=None):
        super().__init__()
        if resolutions is None:
            resolutions = [(2048, 512), (1024, 256), (512, 128)]
        self.resolutions = resolutions

    def forward(self, x, y):
        loss = 0
        for n_fft, hop in self.resolutions:
            # Log-magnitude loss
            X = torch.log(torch.stft(x, n_fft, hop, return_complex=True).abs() + 1e-8)
            Y = torch.log(torch.stft(y, n_fft, hop, return_complex=True).abs() + 1e-8)
            loss += F.l1_loss(X, Y)

            # Spectral convergence (L2 on magnitude)
            X_mag = torch.stft(x, n_fft, hop, return_complex=True).abs()
            Y_mag = torch.stft(y, n_fft, hop, return_complex=True).abs()
            loss += F.mse_loss(X_mag, Y_mag)
        return loss / len(self.resolutions)


class PerceptualAudioLoss(nn.Module):
    """Composite loss: MR-STFT + spectral flatness + crest factor + onset alignment."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.mr_stft = MultiResolutionSTFTLoss()
        self.cfg = cfg

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
            alignment_loss = self._onset_alignment_loss(
                pred_onsets, target_onsets, pred_stems[0]
            )

        # Weighted sum
        total = (
            self.cfg["loss"]["mr_stft_weight"] * spectral_loss
            + self.cfg["loss"]["spectral_flatness_weight"] * flatness_loss
            + self.cfg["loss"]["crest_factor_weight"] * crest_loss
            + self.cfg["loss"]["cross_modal_alignment_weight"] * alignment_loss
        )
        return total, {
            "spectral": spectral_loss.item(),
            "flatness": flatness_loss.item(),
            "crest": crest_loss.item(),
            "alignment": alignment_loss.item(),
        }

    def _spectral_flatness_loss(self, x, y):
        """Penalize unnatural spectral smoothing (preserves transient detail)."""

        def flatness(spec):
            geom = torch.exp(torch.mean(torch.log(spec + 1e-8), dim=-1))
            arith = torch.mean(spec, dim=-1)
            return geom / (arith + 1e-8)

        x_spec = torch.stft(x, 2048, 512, return_complex=True).abs()
        y_spec = torch.stft(y, 2048, 512, return_complex=True).abs()
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
