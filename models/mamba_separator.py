# SPDX-License-Identifier: Apache-2.0
import logging

import torch
import torch.nn as nn

from models._mamba_compat import MAMBA_BACKEND, Mamba

logger = logging.getLogger(__name__)


class MambaSeparator(nn.Module):
    """
    Phase-coherent guitar/bass separation using Mamba SSM.

    Input:  audio (B, C=1, T) mono waveform
    Output: guitar_stem / bass_stem / residual (B, 1, T), None (no state
            cache yet - see C-2.5), metrics (B, 2) =
            [centroid_separation_proxy, phase_coherence]

    NOTE: Streaming inference with SSM state caching is not yet implemented.
    The original prototype used activation concatenation (not real SSM state),
    which was structurally incorrect. See C-2.5 in TODO.md for the full design
    when real SSM state caching is added. For now, each forward call is
    independent - overlap-add in main.py works at the waveform level only.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.hop_length = cfg["audio"]["hop_length"]
        self.n_fft = cfg["audio"]["n_fft"]
        d_model = cfg["separator"]["d_model"]

        n_layers = cfg["separator"].get("n_layers", cfg["separator"].get("n_layer"))
        if not n_layers or int(n_layers) <= 0:
            raise ValueError(
                "separator config must define a positive 'n_layers' (or legacy "
                f"'n_layer') key, got {n_layers!r}"
            )
        n_layers = int(n_layers)

        gradient_checkpointing = bool(
            (cfg.get("training") or {}).get("gradient_checkpointing", False)
        )
        if MAMBA_BACKEND == "pure-torch" and gradient_checkpointing:
            logger.info(
                "Separator: gradient checkpointing enabled for the pure-torch "
                "Mamba blocks (training memory bounded per block)."
            )

        # STFT encoder (fixed, no grad). persistent=False keeps these out of
        # state_dict, so checkpoint compatibility is unaffected.
        self.register_buffer("window", torch.hann_window(self.n_fft))
        self.register_buffer(
            "freqs",
            torch.linspace(0, self.n_fft // 2, self.n_fft // 2 + 1).view(1, -1, 1),
            persistent=False,
        )

        # Learnable projection: spectral bins → Mamba dimension
        self.input_proj = nn.Sequential(
            nn.Linear(self.n_fft // 2 + 1, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )

        # Mamba backbone - valid mamba-ssm kwargs only (no causal_conv1d_impl /
        # selective_scan_impl / use_fast_path; CPU uses the reference impl).
        self.mamba_blocks = nn.ModuleList(
            [
                Mamba(
                    d_model=d_model,
                    d_state=cfg["separator"]["d_state"],
                    d_conv=cfg["separator"]["d_conv"],
                    expand=cfg["separator"]["expand"],
                    gradient_checkpointing=gradient_checkpointing,
                )
                for _ in range(n_layers)
            ]
        )

        # Multi-head output: guitar, bass, residual masks
        self.mask_heads = nn.ModuleDict(
            {
                "guitar": nn.Linear(d_model, self.n_fft // 2 + 1),
                "bass": nn.Linear(d_model, self.n_fft // 2 + 1),
                "residual": nn.Linear(d_model, self.n_fft // 2 + 1),
            }
        )

        # Phase coherence head (auxiliary loss)
        self.phase_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        logger.info("MambaSeparator built: backend=%s n_layers=%d", MAMBA_BACKEND, n_layers)

    def forward(self, audio: torch.Tensor):
        if audio.dim() != 3:
            raise ValueError(
                f"MambaSeparator expects audio (B, C, T), got shape {tuple(audio.shape)}"
            )
        _B, C, T = audio.shape
        if C != 1:
            raise ValueError(f"MambaSeparator expects mono input (C=1), got C={C}")
        if self.n_fft > T:
            raise ValueError(
                f"Input length {T} is shorter than n_fft={self.n_fft}; "
                "torch.stft cannot produce a frame. Pad or supply longer audio."
            )

        # STFT: (B, C, T) → (B, F, T_frames)
        spec = torch.stft(
            audio.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        mag = torch.abs(spec)  # (B, F, T_f) - cached, reused by the metric below
        expected_bins = self.n_fft // 2 + 1
        if mag.shape[1] != expected_bins:
            raise ValueError(f"STFT freq bins {mag.shape[1]} != n_fft//2+1 ({expected_bins})")

        # Project to Mamba dim: (B, T_f, F) → (B, T_f, D)
        x = mag.transpose(1, 2)  # (B, T_f, F)
        hidden = self.input_proj(x)  # (B, T_f, D)
        del mag  # input_proj output kept; raw magnitude no longer needed
        if hidden.shape[-1] != self.mamba_blocks[0].d_model:
            raise ValueError(
                f"hidden dim {hidden.shape[-1]} != d_model {self.mamba_blocks[0].d_model}"
            )

        # Mamba blocks
        for block in self.mamba_blocks:
            hidden = block(hidden)

        # Apply masks directly to the complex spectrogram: mask_mag * spec is
        # equivalent to (mask_mag * mag) * exp(i*phase). Reuses the cached STFT.
        # Process ONE stem at a time - compute its mask, apply, ISTFT, and drop
        # everything but the reduced centroid - so peak memory stays O(two
        # spectrograms) instead of O(three masks + three magnitudes + spec)
        # on the 4-8 GB CPU target.
        stems = {}
        centroids = {}
        for name, head in self.mask_heads.items():
            mask = head(hidden)  # (B, T_f, F)
            mask_mag = torch.sigmoid(mask).transpose(1, 2)  # (B, F, T_f)
            del mask
            masked_spec = mask_mag * spec  # (B, F, T_f) complex
            del mask_mag
            if name in ("guitar", "bass"):
                # reduced immediately to a (B, T_f) centroid vector - full
                # magnitude spectrograms are NOT retained (C-1 follow-up)
                centroids[name] = self._spectral_centroid(torch.abs(masked_spec))
            stem = torch.istft(
                masked_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                length=T,
            )
            stems[name] = stem.unsqueeze(1)  # (B, 1, T)
            del masked_spec
        del spec

        phase_score = self.phase_head(hidden.mean(dim=1))  # (B, 1)

        # Perceptual metric (spectral centroid separation proxy - NOT real SI-SDR).
        # No additional STFT (C-2.6 / PERF-7.1).
        centroid_metric = self._centroid_separation(centroids["guitar"], centroids["bass"])
        metrics = torch.stack([centroid_metric, phase_score.squeeze(-1)], dim=-1)

        return (
            stems["guitar"],
            stems["bass"],
            stems["residual"],
            None,  # new_state_cache placeholder (C-2.5: real SSM state TBD)
            metrics,
        )

    def _centroid_separation(
        self, guitar_centroid: torch.Tensor, bass_centroid: torch.Tensor
    ) -> torch.Tensor:
        """
        Spectral centroid separation heuristic - NOT SI-SDR.

        Takes per-frame centroid vectors ``(B, T_f)`` reduced inside
        ``forward`` (no additional STFT; no full magnitude retention —
        C-2.6 / PERF-7.1 / C-1). "Are the stems spectrally distinct?"
        True SI-SDR requires a ground-truth reference and is computed
        only during training (via PerceptualAudioLoss).
        Renamed from _estimate_si_sdr per C-2.6 / TODO.md.
        """
        separation = torch.abs(guitar_centroid - bass_centroid).mean(dim=-1)
        return torch.clamp(10 * torch.log10(separation + 1e-8) + 20, min=0, max=30)

    def _spectral_centroid(self, mag: torch.Tensor) -> torch.Tensor:
        """Spectral centroid per frame from a magnitude spectrogram (B, F, T_f)."""
        freqs = self.freqs.to(dtype=mag.dtype)
        centroid = (mag * freqs).sum(dim=1) / (mag.sum(dim=1) + 1e-8)
        return centroid  # (B, T_frames)
