# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionSTFTLoss(nn.Module):
    """MR-STFT loss for perceptual stem fidelity (6 resolutions)."""

    def __init__(self, resolutions=None):
        super().__init__()
        self.resolutions = resolutions or [(2048, 512), (1024, 256), (512, 128)]

    def forward(self, x, y):
        # torch.stft accepts only 1D/2D input; flatten (B, C, T) -> (B, T)
        if x.dim() == 3:
            x = x.squeeze(1)
        if y.dim() == 3:
            y = y.squeeze(1)
        # pad so the largest resolution window fits
        max_fft = max(r[0] for r in self.resolutions)
        if x.shape[-1] < max_fft:
            pad = max_fft - x.shape[-1]
            x = F.pad(x, (0, pad))
            y = F.pad(y, (0, pad))
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

    def forward(
        self,
        pred_stems,
        target_stems,
        pred_onsets=None,
        target_onsets=None,
        pred_pitch_logits=None,
        target_pitches=None,
        pred_velocities=None,
        target_velocities=None,
        pred_durations=None,
        target_durations=None,
    ):
        # We expect pred_stems and target_stems to be lists of [guitar, bass] tensors
        # Primary: MR-STFT for spectral fidelity (applied to each stem and summed)
        spectral_loss = 0
        for pred, target in zip(pred_stems, target_stems, strict=False):
            spectral_loss += self.mr_stft(pred, target)

        # Regularization: preserve dynamic range (applied to each stem and averaged)
        flatness_loss = 0
        crest_loss = 0
        for pred, target in zip(pred_stems, target_stems, strict=False):
            flatness_loss += self._spectral_flatness_loss(pred, target)
            crest_loss += self._crest_factor_loss(pred, target)
        flatness_loss /= len(pred_stems)
        crest_loss /= len(pred_stems)

        # Onset F1 loss (1 - F1 of binarized onset predictions vs targets)
        onset_f1_loss = 0
        if pred_onsets is not None and target_onsets is not None:
            onset_f1_loss = onset_f1_loss_fn(pred_onsets, target_onsets)

        # Pitch cross-entropy (128-way MIDI pitch)
        pitch_ce_loss = 0
        if pred_pitch_logits is not None and target_pitches is not None:
            pitch_ce_loss = pitch_ce_loss_fn(pred_pitch_logits, target_pitches)

        # Velocity MAE
        velocity_mae_loss = 0
        if pred_velocities is not None and target_velocities is not None:
            velocity_mae_loss = velocity_mae_loss_fn(pred_velocities, target_velocities)

        # Duration IoU (1 - mean IoU of predicted vs target note durations)
        duration_iou_loss = 0
        if pred_durations is not None and target_durations is not None:
            duration_iou_loss = duration_iou_loss_fn(pred_durations, target_durations)

        # Cross-modal: align MIDI onsets with stem energy peaks
        alignment_loss = 0
        if pred_onsets is not None and target_onsets is not None:
            # We assume we are aligning to the guitar stem for transcription
            alignment_loss = self._onset_alignment_loss(pred_onsets, target_onsets, pred_stems[0])

        loss_cfg = self.cfg["loss"]
        # Weighted sum
        total = (
            loss_cfg["mr_stft_weight"] * spectral_loss
            + loss_cfg["spectral_flatness_weight"] * flatness_loss
            + loss_cfg["crest_factor_weight"] * crest_loss
            + loss_cfg.get("onset_f1_weight", 1.0) * onset_f1_loss
            + loss_cfg.get("pitch_ce_weight", 0.8) * pitch_ce_loss
            + loss_cfg.get("velocity_mae_weight", 0.3) * velocity_mae_loss
            + loss_cfg.get("duration_iou_weight", 0.5) * duration_iou_loss
            + loss_cfg["cross_modal_alignment_weight"] * alignment_loss
        )

        def _scalar(x):
            """Float value of a loss component that may be a Tensor or number."""
            return float(x.item()) if torch.is_tensor(x) else x

        return total, {
            "spectral": spectral_loss.item(),
            "flatness": flatness_loss.item(),
            "crest": crest_loss.item(),
            "onset_f1": _scalar(onset_f1_loss),
            "pitch_ce": _scalar(pitch_ce_loss),
            "velocity_mae": _scalar(velocity_mae_loss),
            "duration_iou": _scalar(duration_iou_loss),
            "alignment": _scalar(alignment_loss),
        }

    def _spectral_flatness_loss(self, x, y):
        """Penalize unnatural spectral smoothing (preserves transient detail)."""
        # torch.stft accepts only 1D/2D input; flatten (B, C, T) -> (B, T)
        if x.dim() == 3:
            x = x.squeeze(1)
        if y.dim() == 3:
            y = y.squeeze(1)
        if x.shape[-1] < 2048:
            pad = 2048 - x.shape[-1]
            x = F.pad(x, (0, pad))
            y = F.pad(y, (0, pad))

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
        """Ensure MIDI onsets align with stem energy peaks.

        Compares the energy envelope at predicted onset frames against the
        energy at target onset frames. If target_onsets is missing or
        degenerate, falls back to zero alignment loss.
        """
        if target_onsets is None:
            return pred_onsets.new_zeros(())
        # Normalize shapes for broadcasting against (B, 1, T) energy
        if pred_onsets.dim() == 2:
            pred_onsets = pred_onsets.unsqueeze(1)
        if target_onsets.dim() == 2:
            target_onsets = target_onsets.unsqueeze(1)
        # Compute stem energy envelope
        energy = torch.mean(stem**2, dim=1, keepdim=True)  # (B, 1, T)
        energy_smooth = F.avg_pool1d(energy, kernel_size=101, padding=50, stride=1)

        tgt_len = min(pred_onsets.shape[-1], target_onsets.shape[-1], energy_smooth.shape[-1])
        if tgt_len == 0:
            return pred_onsets.new_zeros(())
        pred_onsets = pred_onsets[..., :tgt_len]
        target_onsets = target_onsets[..., :tgt_len]
        energy_smooth = energy_smooth[..., :tgt_len]

        # Onset-triggered energy attention, normalized per batch
        pred_energy = (torch.sigmoid(pred_onsets) * energy_smooth).mean()
        target_energy = (target_onsets.clamp(0, 1).float() * energy_smooth).mean()
        return F.mse_loss(pred_energy, target_energy)


def onset_f1_loss_fn(pred_onsets, target_onsets, threshold: float = 0.5, eps: float = 1e-8):
    """Onset F1 loss: 1 - F1 of binarized predictions vs binary targets.

    Shape-robust: inputs may be (B, T, 1), (B, T), or (B, 1, T, 1) — they are
    flattened and paired elementwise (padding/truncation to the predicted
    frame grid is expected to happen at the caller for unequal lengths).

    Args:
        pred_onsets: probabilities/logits, any shape
        target_onsets: binary targets, same number of elements
    Returns:
        Scalar loss in [0, 1].
    """
    is_logits = pred_onsets.min() < 0 or pred_onsets.max() > 1
    probs = torch.sigmoid(pred_onsets) if is_logits else pred_onsets

    pred_flat = probs.reshape(-1)
    target_flat = target_onsets.reshape(-1).float()
    n = min(pred_flat.numel(), target_flat.numel())

    pred_bin = (pred_flat[:n] >= threshold).float()
    target_bin = (target_flat[:n] >= threshold).float().clamp(0, 1)

    tp = (pred_bin * target_bin).sum()
    fp = (pred_bin * (1 - target_bin)).sum()
    fn = ((1 - pred_bin) * target_bin).sum()

    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return 1.0 - f1


def pitch_ce_loss_fn(pred_pitch_logits, target_pitches, ignore_index: int = -100):
    """128-way pitch cross-entropy loss.

    Args:
        pred_pitch_logits: (B, T, 128) or (B, 128, T) pitch logits
        target_pitches: (B, T) longs in [0, 127]
    Returns:
        Scalar cross-entropy loss.
    """
    logits = pred_pitch_logits
    targets = target_pitches.long()
    if logits.dim() == 3:
        # Bring classes to dim=-1
        if logits.shape[-1] != 128 and logits.shape[1] == 128:
            logits = logits.transpose(1, 2)
        logits = logits.reshape(-1, logits.shape[-1])
        targets = targets.reshape(-1)
    else:
        logits = logits.reshape(-1, logits.shape[-1])
        targets = targets.reshape(-1)
    return F.cross_entropy(
        logits, targets.clamp(0, logits.shape[-1] - 1), ignore_index=ignore_index
    )


def velocity_mae_loss_fn(pred_velocities, target_velocities):
    """Velocity mean absolute error.

    Args:
        pred_velocities: predicted velocities (any scale, e.g. [0,1] or [0,127])
        target_velocities: target velocities, broadcastable to pred shape
    Returns:
        Scalar MAE >= 0.
    """
    return F.l1_loss(pred_velocities.float(), target_velocities.float())


def duration_iou_loss_fn(pred_durations, target_durations, eps: float = 1e-8):
    """Duration IoU loss: 1 - mean IoU of predicted vs target note durations.

    Treats each duration as a 1-D interval [0, d]; IoU of overlapping
    half-open intervals starting at 0 is min/max.

    Args:
        pred_durations: non-negative predicted durations, any shape
        target_durations: non-negative target durations, same shape
    Returns:
        Scalar loss in [0, 1].
    """
    pred = pred_durations.float().clamp(min=0)
    target = target_durations.float().clamp(min=0)
    inter = torch.minimum(pred, target)
    union = torch.maximum(pred, target)
    iou = (inter + eps) / (union + eps)
    return 1.0 - iou.mean()
