# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F


def _stft_input_2d(t: torch.Tensor) -> torch.Tensor:
    """Coerce (B, 1, T) → (B, T) for torch.stft; raise on non-mono."""
    if t.dim() == 3:
        if t.shape[1] != 1:
            raise ValueError(f"loss expects mono stems (B, 1, T), got shape {tuple(t.shape)}")
        return t.squeeze(1)
    if t.dim() != 2:
        raise ValueError(f"loss expects (B, 1, T) or (B, T), got shape {tuple(t.shape)}")
    return t


class MultiResolutionSTFTLoss(nn.Module):
    """MR-STFT loss for perceptual stem fidelity (3 resolutions).

    Per resolution: log-magnitude L1 + normalized spectral convergence
    (||Y|-|X||_F / ||Y||_F), with a Hann window — matching the standard
    MR-STFT formulation (Yamamoto et al., 2020). Each resolution's STFT is
    computed once and reused for both terms (M-3).
    """

    def __init__(self, resolutions=None):
        super().__init__()
        self.resolutions = resolutions or [(2048, 512), (1024, 256), (512, 128)]

    def forward(self, x, y):
        # torch.stft accepts only 1D/2D input; flatten (B, C, T) -> (B, T)
        x = _stft_input_2d(x)
        y = _stft_input_2d(y)
        # pad so the largest resolution window fits
        max_fft = max(r[0] for r in self.resolutions)
        if x.shape[-1] < max_fft:
            pad = max_fft - x.shape[-1]
            x = F.pad(x, (0, pad))
            y = F.pad(y, (0, pad))
        loss = x.new_zeros(())
        for n_fft, hop in self.resolutions:
            window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
            X_mag = torch.stft(x, n_fft, hop, window=window, return_complex=True).abs()
            Y_mag = torch.stft(y, n_fft, hop, window=window, return_complex=True).abs()
            # Log-magnitude loss
            loss = loss + F.l1_loss(torch.log(X_mag + 1e-8), torch.log(Y_mag + 1e-8))
            # Spectral convergence: normalized Frobenius magnitude distance
            loss = loss + torch.norm(Y_mag - X_mag, p="fro") / (torch.norm(Y_mag, p="fro") + 1e-8)
        return loss / len(self.resolutions)


class PerceptualAudioLoss(nn.Module):
    """Composite loss: MR-STFT + spectral flatness + crest factor + onset alignment.

    Transcription components are optional and only active when the
    corresponding targets are supplied by the caller (C-3): pitch CE
    (class-index / single-hot targets) or multi-label pitch BCE (multi-hot
    targets — chords) requires ``target_pitches``; velocity MAE requires
    ``target_velocities``. Duration IoU and expression losses remain
    unwired — no dataset currently emits those targets (see TODO ARCH-11.6).
    """

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

        # Onset detection: BCE is the optimised term (differentiable everywhere).
        # The hard F1 below is a *metric* only - thresholding has zero gradient,
        # so as a loss term it contributed nothing to the update (H-4).
        onset_bce_loss = 0
        onset_f1_metric = 0
        if pred_onsets is not None and target_onsets is not None:
            onset_bce_loss = onset_bce_loss_fn(pred_onsets, target_onsets, is_logits=True)
            with torch.no_grad():
                onset_f1_metric = onset_f1_loss_fn(
                    pred_onsets.detach(), target_onsets.detach(), is_logits=True
                )

        # Pitch loss: multi-hot targets (any frame labeling >1 pitch — a
        # chord) train via multi-label BCE (pitch_bce_loss_fn); class-index /
        # single-hot targets train via 128-way cross-entropy, unchanged.
        pitch_ce_loss = 0
        if pred_pitch_logits is not None and target_pitches is not None:
            # A multi-hot chord grid is (B, T, V); a class-index grid is (B, T).
            # Both halves of this test matter: `dim() == 3` alone would push an
            # unbatched (T, V) grid into cross-entropy, and a width test alone would
            # read a (B, 128)-frame class-index target as multi-hot.
            n_classes = pred_pitch_logits.shape[-1]
            if target_pitches.dim() == 3 and target_pitches.shape[-1] == n_classes:
                pitch_ce_loss = pitch_bce_loss_fn(pred_pitch_logits, target_pitches)
            else:
                pitch_ce_loss = pitch_ce_loss_fn(pred_pitch_logits, target_pitches)

        # Velocity MAE (masked to frames where a velocity target exists)
        velocity_mae_loss = 0
        if pred_velocities is not None and target_velocities is not None:
            mask = (target_velocities > 0).to(pred_velocities.dtype)
            velocity_mae_loss = velocity_mae_loss_fn(pred_velocities, target_velocities, mask=mask)

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
            + loss_cfg.get("onset_bce_weight", loss_cfg.get("onset_f1_weight", 1.0))
            * onset_bce_loss
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
            "onset_bce": _scalar(onset_bce_loss),  # the optimised onset term
            "onset_f1": _scalar(onset_f1_metric),  # metric only (no gradient)
            "pitch": _scalar(pitch_ce_loss),  # CE (single-label) or BCE (chords)
            "pitch_ce": _scalar(pitch_ce_loss),  # same number, PR #13's key name
            "velocity_mae": _scalar(velocity_mae_loss),
            "duration_iou": _scalar(duration_iou_loss),
            "alignment": _scalar(alignment_loss),
        }

    def _spectral_flatness_loss(self, x, y):
        """Penalize unnatural spectral smoothing (preserves transient detail).

        Flatness is computed ACROSS FREQUENCY bins per frame (dim=1 of the
        (B, F, T_f) STFT output) — H-3. The previous implementation averaged
        over time (the last dim), which measured temporal stationarity
        instead of spectral shape.
        """
        x = _stft_input_2d(x)
        y = _stft_input_2d(y)
        if x.shape[-1] < 2048:
            pad = 2048 - x.shape[-1]
            x = F.pad(x, (0, pad))
            y = F.pad(y, (0, pad))

        def flatness(spec):
            # spec: (B, F, T_f) — geometric/arithmetic mean over F per frame
            geom = torch.exp(torch.mean(torch.log(spec + 1e-8), dim=1))
            arith = torch.mean(spec, dim=1)
            return geom / (arith + 1e-8)

        window = torch.hann_window(2048, device=x.device, dtype=x.dtype)
        x_spec = torch.stft(x, 2048, 512, window=window, return_complex=True).abs()
        y_spec = torch.stft(y, 2048, 512, window=window, return_complex=True).abs()
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

        Compares the energy at predicted onset frames against the energy at
        target onset frames — pointwise on the shared frame grid (H-1).
        The previous implementation broadcast (B, T, 1) onsets against a
        (B, 1, T) energy envelope, materializing an O(T²) tensor whose
        global average compared two different statistics.
        """
        if target_onsets is None:
            return pred_onsets.new_zeros(())
        # Stem energy pooled onto the STFT frame grid (M-4: samples vs frames)
        hop = int((self.cfg.get("audio") or {}).get("hop_length", 512))
        energy = torch.mean(stem**2, dim=1)  # (B, T_samples)
        energy_frames = F.avg_pool1d(
            energy.unsqueeze(1), kernel_size=hop, stride=hop, ceil_mode=True
        ).squeeze(
            1
        )  # (B, T_f)

        pred = pred_onsets.reshape(pred_onsets.shape[0], -1)
        tgt = target_onsets.reshape(target_onsets.shape[0], -1).clamp(0, 1).float()

        n = min(pred.shape[-1], tgt.shape[-1], energy_frames.shape[-1])
        if n < 2:
            return pred.new_zeros(())
        pred = pred[..., :n]
        energy_frames = energy_frames[..., :n]

        probs = torch.sigmoid(pred)
        if probs.min() >= 0 and probs.max() <= 1 and float(energy_frames.abs().max()) <= 1e-6:
            return pred.new_zeros(())

        # Zero-mean cosine distance between the predicted onset curve and the
        # (frame-grid) energy envelope. Reducing to two scalars and taking their
        # MSE - the previous form - is minimised by shifting the mean
        # prediction and says nothing about *where* onsets are; an all-constant
        # prediction now scores 1.0 (worst case).
        def _unit(x):
            x = x - x.mean(dim=-1, keepdim=True)
            return x / (x.norm(dim=-1, keepdim=True) + 1e-6)

        cos = (_unit(probs) * _unit(energy_frames)).sum(dim=-1).mean()
        return 1.0 - cos


def onset_bce_loss_fn(
    pred_onsets, target_onsets, pos_weight: float | None = None, is_logits: bool | None = None
):
    """Differentiable onset objective (BCE-with-logits).

    ``pos_weight`` counteracts the strong class imbalance of onset targets
    (typically ~1 % of frames are onsets); ~100 is a reasonable starting point.

    Args:
        pred_onsets: logits (or probabilities when ``is_logits=False``)
        target_onsets: binary targets, same element count
        pos_weight: optional weight for the positive class
        is_logits: None → infer from the value range
    """
    if is_logits is None:
        is_logits = bool(pred_onsets.min() < 0 or pred_onsets.max() > 1)
    logits = pred_onsets.reshape(-1).float()
    if not is_logits:
        logits = torch.logit(logits.clamp(1e-8, 1 - 1e-8))
    target = target_onsets.reshape(-1).float().clamp(0, 1)
    n = min(logits.numel(), target.numel())
    if n == 0:
        return pred_onsets.new_zeros(())
    weight = None
    if pos_weight is not None:
        weight = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits[:n], target[:n], pos_weight=weight)


def onset_f1_loss_fn(
    pred_onsets,
    target_onsets,
    threshold: float = 0.5,
    eps: float = 1e-8,
    is_logits: bool | None = None,
):
    """Onset F1 loss: 1 - F1 of binarized predictions vs binary targets.

    Shape-robust: inputs may be (B, T, 1), (B, T), or (B, 1, T, 1) — they are
    flattened and paired elementwise (padding/truncation to the predicted
    frame grid is expected to happen at the caller for unequal lengths).

    Args:
        pred_onsets: probabilities or logits (see ``is_logits``)
        target_onsets: binary targets, same number of elements
        is_logits: True → apply sigmoid first; False → inputs are already
            probabilities; None (default) → infer from the data range
            (M-2: prefer passing this explicitly; inference from data can
            misclassify raw logits that happen to land in [0, 1]).
    Returns:
        Scalar loss in [0, 1].
    """
    if is_logits is None:
        is_logits = bool(pred_onsets.min() < 0 or pred_onsets.max() > 1)
    probs = torch.sigmoid(pred_onsets) if is_logits else pred_onsets

    pred_flat = probs.reshape(-1)
    target_flat = target_onsets.reshape(-1).float()
    if pred_flat.numel() != target_flat.numel():
        raise ValueError(
            "onset_f1_loss_fn: prediction and target element counts differ "
            f"({pred_flat.numel()} vs {target_flat.numel()}); align the frame "
            "grids at the caller instead of silently truncating."
        )

    pred_bin = (pred_flat >= threshold).float()
    target_bin = (target_flat >= threshold).float().clamp(0, 1)

    tp = (pred_bin * target_bin).sum()
    fp = (pred_bin * (1 - target_bin)).sum()
    fn = ((1 - pred_bin) * target_bin).sum()

    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return 1.0 - f1


def pitch_ce_loss_fn(pred_pitch_logits, target_pitches, ignore_index: int = -100):
    """128-way pitch cross-entropy loss.

    Args:
        pred_pitch_logits: (B, T, V) pitch logits. (No layout guessing:
            a (B, V, T) tensor must be transposed by the caller — M-1.)
        target_pitches: (B, T) longs in [0, V), or one-hot (B, T, V).
            Values outside [0, V) (including any negative sentinel such as
            -100) are mapped to ``ignore_index`` instead of being silently
            relabeled to a valid class.
    Returns:
        Scalar cross-entropy loss.
    """
    logits = pred_pitch_logits
    if logits.dim() != 3:
        raise ValueError(
            f"pitch_ce_loss_fn expects pred_pitch_logits (B, T, V); got {tuple(logits.shape)}"
        )
    vocab = logits.shape[-1]
    logits = logits.reshape(-1, vocab)
    targets = target_pitches

    if targets.dim() == 4 and targets.shape[1] == 1:
        targets = targets.squeeze(1)  # collated (B, 1, T, V) one-hot
    if targets.dim() == 3:
        if targets.shape[-1] != vocab:
            raise ValueError(f"one-hot target vocab {targets.shape[-1]} != logits vocab {vocab}")
        valid = targets.sum(dim=-1) > 0
        targets = targets.argmax(dim=-1).masked_fill(~valid, ignore_index)
    elif targets.dim() != 2:
        targets = targets.reshape(targets.shape[0], -1) if targets.dim() > 2 else targets

    flat = targets.reshape(-1).long()
    out_of_range = (flat < 0) | (flat >= vocab)
    flat = torch.where(out_of_range, torch.full_like(flat, ignore_index), flat)
    if (flat != ignore_index).sum() == 0:
        # All targets ignored (e.g. one-hot rows that are entirely zero):
        # F.cross_entropy would return NaN over an empty reduction — return a
        # zero that stays connected to the graph instead.
        return logits.sum() * 0.0
    return F.cross_entropy(logits, flat, ignore_index=ignore_index)


def pitch_bce_loss_fn(pred_pitch_logits, target_multi_hot):
    """Multi-label pitch BCE loss for POLYPHONIC (chord) targets.

    Args:
        pred_pitch_logits: (B, T, V) pitch logits, decoded per-pitch
            (sigmoid threshold) at inference — a frame may activate several
            pitches at once.
        target_multi_hot: (B, T, V) multi-hot float/long targets. Frames
            whose row sums to 0 have no pitch label and are excluded from
            the loss (mirrors ``pitch_ce_loss_fn``'s ignore_index rows).
            ``_prepare_pitch_target`` returns this format when any frame
            labels more than one pitch (a chord).
    Returns:
        Scalar loss. Per frame, the positive class terms are weighted by
        ``(V - n_active) / n_active`` so a 3-note chord is balanced against
        its 125 negatives exactly like a single note is against its 127 —
        without this, BCE over 128 classes is dominated by negatives and
        collapses toward "predict nothing".
    """
    logits = pred_pitch_logits
    if logits.dim() != 3:
        raise ValueError(
            f"pitch_bce_loss_fn expects pred_pitch_logits (B, T, V); got {tuple(logits.shape)}"
        )
    vocab = logits.shape[-1]
    targets = target_multi_hot
    if targets.dim() == 4 and targets.shape[1] == 1:
        targets = targets.squeeze(1)  # collated (B, 1, T, V)
    if targets.dim() != 3 or targets.shape[-1] != vocab:
        raise ValueError(
            f"pitch_bce_loss_fn expects multi-hot (B, T, {vocab}) targets; "
            f"got {tuple(targets.shape)} — class-index targets belong to pitch_ce_loss_fn"
        )

    valid = targets.sum(dim=-1) > 0  # (B, T): frames with at least one label
    if valid.sum() == 0:
        # All frames unlabeled: keep the result connected to the graph.
        return logits.sum() * 0.0

    logits_v = logits[valid]  # (M, V)
    targets_v = targets[valid].float().clamp(0.0, 1.0)
    counts = targets_v.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (M, 1)
    pos_weight = (vocab - counts) / counts  # broadcast over the vocab dim
    bce = F.binary_cross_entropy_with_logits(
        logits_v, targets_v, pos_weight=pos_weight, reduction="none"
    )
    return bce.mean()


def velocity_mae_loss_fn(pred_velocities, target_velocities, mask=None):
    """Velocity mean absolute error.

    Args:
        pred_velocities: predicted velocities (any scale, e.g. [0,1] or [0,127])
        target_velocities: target velocities, same shape as predictions
        mask: optional boolean/float mask, broadcastable to the prediction
            shape; only masked frames contribute (velocity targets are only
            defined at onset frames).
    Returns:
        Scalar MAE >= 0.
    """
    pred = pred_velocities.float().reshape(-1)
    target = target_velocities.float().reshape(-1)
    if pred.numel() != target.numel():
        raise ValueError(
            "velocity_mae_loss_fn: prediction and target element counts differ "
            f"({pred.numel()} vs {target.numel()})"
        )
    if mask is None:
        return F.l1_loss(pred, target)
    m = mask.float().reshape(-1)
    if m.numel() != pred.numel():
        raise ValueError(f"velocity_mae_loss_fn: mask element count {m.numel()} != {pred.numel()}")
    if m.sum() == 0:
        return pred.sum() * 0.0  # zero, but connected to the graph
    return F.l1_loss(pred[m > 0], target[m > 0])


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
