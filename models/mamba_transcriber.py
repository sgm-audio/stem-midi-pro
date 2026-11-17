# SPDX-License-Identifier: Apache-2.0
import logging
import math

import torch
import torch.nn as nn

from models._mamba_compat import MAMBA_BACKEND, Mamba

logger = logging.getLogger(__name__)


class MambaTranscriber(nn.Module):
    """
    Mono audio-to-MIDI with expression detection.

    Input:  stem (B, C=1, T)
    Outputs: onset_logits (B, T, 1), pitch_logits (B, T, V=128),
             velocity (B, T, 1), expression (B, T, E), confidence (B, T)

    Decoder honesty: the pitch head emits (B, T, V=128) logits which the
    downstream ``_logits_to_midi`` decoder reads PER-PITCH (sigmoid over
    ``transcriber.pitch_threshold``) — polyphonic frames emit chords, and a
    frame trained only on single labels still yields the argmax pitch via
    the fallback. Whether the MODEL actually predicts chords depends on the
    training data: multi-hot pitch targets (any frame labeling >1 note)
    train through ``pitch_bce_loss_fn``; single-pitch targets keep the
    128-way cross-entropy. The expression head exists but is untrained
    (ARCH-11.4.4); its output is not encoded into MIDI by the orchestrator.

    The mixers are stacked as pre-norm residual blocks (``h = h + mix(norm(h))``
    plus a final LayerNorm). Bare mixer stacking multiplied the activation RMS
    by ~0.02 per block, so the 8-layer stack collapsed to a constant and every
    head emitted a fixed value (C-1 in AUDIT_REPORT.md).
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg  # needed by _mel_spectrogram (sample_rate etc.)
        d_model = cfg["transcriber"]["d_model"]

        n_layers = cfg["transcriber"].get("n_layers", cfg["transcriber"].get("n_layer"))
        if not n_layers or int(n_layers) <= 0:
            raise ValueError(
                "transcriber config must define a positive 'n_layers' (or legacy "
                f"'n_layer') key, got {n_layers!r}"
            )
        n_layers = int(n_layers)

        # Input projection: mel spectrogram → Mamba dim
        self.n_mels = cfg["audio"]["n_mels"]
        self.n_fft = cfg["audio"]["n_fft"]
        self.hop_length = cfg["audio"]["hop_length"]
        self.register_buffer("mel_window", torch.hann_window(self.n_fft))
        # Mel filter bank built once (C-2.9 / PERF-7.2) — not per forward call.
        self.register_buffer(
            "mel_basis",
            self._create_mel_basis(
                sr=cfg["audio"]["sample_rate"],
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                fmin=0,
                fmax=cfg["audio"]["sample_rate"] // 2,
            ),
        )
        self.input_proj = nn.Linear(self.n_mels, d_model)

        # Pre-norm partners for the residual stack (C-1).
        self.block_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)

        gradient_checkpointing = bool(
            (cfg.get("training") or {}).get("gradient_checkpointing", False)
        )

        # Mamba backbone (valid mamba-ssm kwargs only; CPU reference impl)
        self.mamba = nn.ModuleList(
            [
                Mamba(
                    d_model=d_model,
                    d_state=cfg["transcriber"]["d_state"],
                    d_conv=int(cfg["transcriber"].get("d_conv", 4)),
                    expand=int(cfg["transcriber"].get("expand", 2)),
                    gradient_checkpointing=gradient_checkpointing,
                )
                for _ in range(n_layers)
            ]
        )

        # Multi-task heads
        self.onset_head = nn.Linear(d_model, 1)
        self.pitch_head = nn.Linear(d_model, cfg["transcriber"]["pitch_vocab_size"])
        self.velocity_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Normalized velocity [0,1]
        )
        self.expression_head = nn.Linear(d_model, len(cfg["transcriber"]["expression_heads"]))

        # Confidence head: ensemble uncertainty estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 128), nn.Dropout(0.1), nn.Linear(128, 1), nn.Sigmoid()
        )

        # Store config for inference
        self.onset_threshold = cfg["transcriber"].get("onset_threshold", 0.5)

        logger.info("MambaTranscriber built: backend=%s n_layers=%d", MAMBA_BACKEND, n_layers)

    def forward(self, stem: torch.Tensor):
        if stem.dim() != 3 or stem.shape[1] != 1:
            raise ValueError(f"MambaTranscriber expects stem (B, 1, T), got {tuple(stem.shape)}")
        if stem.shape[-1] < self.n_fft:
            raise ValueError(
                f"Stem length {stem.shape[-1]} is shorter than n_fft={self.n_fft}; "
                "torch.stft cannot produce a frame. Pad or supply longer audio."
            )

        # Mel spectrogram input: (B, C, T) → (B, T, n_mels)
        mel = self._mel_spectrogram(stem.squeeze(1)).transpose(1, 2)
        hidden = self.input_proj(mel)

        # Pre-norm residual Mamba stack (C-1)
        for norm, block in zip(self.block_norms, self.mamba, strict=True):
            hidden = hidden + block(norm(hidden))
        hidden = self.final_norm(hidden)

        # Task-specific outputs
        onset_logits = self.onset_head(hidden)  # (B, T, 1)
        pitch_logits = self.pitch_head(hidden)  # (B, T, V)
        velocity = self.velocity_head(hidden)  # (B, T, 1)
        expression = torch.sigmoid(self.expression_head(hidden))  # (B, T, E)

        # Confidence: Monte Carlo dropout ensemble (eval only — see C-2.10)
        if not self.training:
            confidence = self.mc_dropout_eval(hidden, n_samples=5)  # (B, T)
        else:
            confidence = self.confidence_head(hidden).squeeze(-1)

        return onset_logits, pitch_logits, velocity, expression, confidence

    def mc_dropout_eval(self, hidden: torch.Tensor, n_samples: int = 5) -> torch.Tensor:
        """
        Eval-only MC-dropout confidence estimate (C-2.10).

        Explicitly enables the Dropout inside ``confidence_head`` while in eval
        mode so the ensemble is non-degenerate (otherwise all samples would be
        identical because dropout is inactive in eval). The head is restored to
        eval mode afterwards. Never used during training.

        The score is **ensemble disagreement**, not the ensemble mean: returning
        ``samples.mean()`` produced a number that looked like a probability but
        whose spread was measured at ~5e-4, carrying no uncertainty (H-3). For
        Bernoulli outputs with standard deviation ``sd`` (max 0.5), confidence is
        ``1 - 2*sd`` clamped to [0, 1]: 1.0 when every sample agrees, 0.0 when
        they disagree maximally.
        """
        head = self.confidence_head
        head.train()  # activate Dropout
        try:
            samples = torch.stack([head(hidden) for _ in range(n_samples)])
        finally:
            head.eval()
        std = samples.std(dim=0, unbiased=False)
        return (1.0 - 2.0 * std).clamp(0.0, 1.0).squeeze(-1)

    def _mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Differentiable mel-spectrogram for end-to-end training."""
        # Compute STFT
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.mel_window,
            return_complex=True,
            pad_mode="reflect",
        )
        mag = torch.abs(spec)  # (B, F, T)

        # Convert to mel scale using the precomputed basis buffer (C-2.9)
        mel_spec = torch.matmul(self.mel_basis, mag)  # (B, n_mels, T)

        # Apply log compression. Bounded on BOTH sides: the floor stops
        # log(0) on digital silence, and the ceiling stops a corrupt or
        # DC-heavy stem from feeding thousands of nats into the SSM blocks.
        # Unbounded above, a loud stem reached log-mel 19.36 and the
        # recurrence overflowed fp32 to NaN -- which, combined with the
        # finiteness guard added elsewhere, made a completely silent input
        # raise RuntimeError instead of returning an empty transcription
        # (review §3).
        log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-8, max=1e3))

        return log_mel_spec

    def _create_mel_basis(self, sr, n_fft, n_mels, fmin, fmax):
        """Create mel filter bank matrix"""
        # Mel scale boundaries (math.log10 on python scalars — torch.log10
        # requires a Tensor argument)
        mel_fmin = 2595 * math.log10(1 + fmin / 700)
        mel_fmax = 2595 * math.log10(1 + fmax / 700)

        # Equally spaced in mel scale
        mel_points = torch.linspace(mel_fmin, mel_fmax, n_mels + 2)

        # Convert back to Hz
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)

        # Bin indices
        bin = torch.floor((n_fft + 1) * hz_points / sr)

        # Create filter bank
        fbank = torch.zeros((n_mels, n_fft // 2 + 1))
        for j in range(1, n_mels + 1):
            left = int(bin[j - 1])
            center = int(bin[j])
            right = int(bin[j + 1])

            for i in range(left, center):
                if center - left > 0:
                    fbank[j - 1, i] = (i - left) / (center - left)
            for i in range(center, right):
                if right - center > 0:
                    fbank[j - 1, i] = (right - i) / (right - center)

        return fbank
