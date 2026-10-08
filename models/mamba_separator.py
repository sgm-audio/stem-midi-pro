# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn

from models._mamba_compat import Mamba


class MambaSeparator(nn.Module):
    """
    Phase-coherent guitar/bass separation using Mamba SSM.

    Input:  audio (B, C=1, T) mono waveform
    Output: guitar_stem / bass_stem / residual (B, 1, T), None (no state
            cache yet — see C-2.5), metrics (B, 2) =
            [centroid_separation_proxy, phase_coherence]

    NOTE: Streaming inference with SSM state caching is not yet implemented.
    The original prototype used activation concatenation (not real SSM state),
    which was structurally incorrect. See C-2.5 in TODO.md for the full design
    when real SSM state caching is added. For now, each forward call is
    independent — overlap-add in main.py works at the waveform level only.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.hop_length = cfg["audio"]["hop_length"]
        self.n_fft = cfg["audio"]["n_fft"]
        d_model = cfg["separator"]["d_model"]

        # STFT encoder (fixed, no grad)
        self.register_buffer("window", torch.hann_window(self.n_fft))

        # Learnable projection: spectral bins → Mamba dimension
        self.input_proj = nn.Sequential(
            nn.Linear(self.n_fft // 2 + 1, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )

        # Mamba backbone — valid mamba-ssm kwargs only (no causal_conv1d_impl /
        # selective_scan_impl / use_fast_path; CPU uses the reference impl).
        # nn.Sequential for a cleaner, JIT-friendly forward (PERF-7.10).
        self.mamba_blocks = nn.Sequential(
            *[
                Mamba(
                    d_model=d_model,
                    d_state=cfg["separator"]["d_state"],
                    d_conv=cfg["separator"]["d_conv"],
                    expand=cfg["separator"]["expand"],
                )
                for _ in range(cfg["separator"].get("n_layers", cfg["separator"].get("n_layer")))
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

    def forward(self, audio: torch.Tensor):
        assert audio.dim() == 3, f"expected audio (B, C, T), got {tuple(audio.shape)}"
        _B, C, T = audio.shape
        assert C == 1, f"expected mono input (C=1), got C={C}"

        # STFT: (B, C, T) → (B, F, T_frames)
        spec = torch.stft(
            audio.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        mag = torch.abs(spec)  # (B, F, T_f) — cached, reused by the metric below
        assert (
            mag.shape[1] == self.n_fft // 2 + 1
        ), f"STFT freq bins {mag.shape[1]} != n_fft//2+1 ({self.n_fft // 2 + 1})"

        # Project to Mamba dim: (B, T_f, F) → (B, T_f, D)
        x = mag.transpose(1, 2)  # (B, T_f, F)
        hidden = self.input_proj(x)  # (B, T_f, D)
        assert (
            hidden.shape[-1] == self.mamba_blocks[0].d_model
        ), f"hidden dim {hidden.shape[-1]} != d_model {self.mamba_blocks[0].d_model}"

        # Mamba blocks
        hidden = self.mamba_blocks(hidden)

        # Generate masks + phase coherence score
        masks = {k: head(hidden) for k, head in self.mask_heads.items()}  # (B, T_f, F)

        # Apply masks directly to the complex spectrogram: mask_mag * spec is
        # equivalent to (mask_mag * mag) * exp(i*phase). Reuses the cached STFT.
        stems = {}
        masked_mags = {}
        for name, mask in masks.items():
            mask_mag = torch.sigmoid(mask).transpose(1, 2)  # (B, F, T_f)
            masked_spec = mask_mag * spec  # (B, F, T_f) complex
            masked_mags[name] = torch.abs(masked_spec)  # cached for the metric
            stem = torch.istft(
                masked_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                length=T,
            )
            stems[name] = stem.unsqueeze(1)  # (B, 1, T)

        phase_score = self.phase_head(hidden.mean(dim=1))  # (B, 1)

        # Perceptual metric (spectral centroid separation proxy — NOT real SI-SDR).
        # Uses the cached masked magnitudes — no additional STFT (C-2.6 / PERF-7.1).
        centroid_metric = self._centroid_separation(masked_mags["guitar"], masked_mags["bass"])
        metrics = torch.stack([centroid_metric, phase_score.squeeze(-1)], dim=-1)

        return (
            stems["guitar"],
            stems["bass"],
            stems["residual"],
            None,  # new_state_cache placeholder (C-2.5: real SSM state TBD)
            metrics,
        )

    def _centroid_separation(
        self, guitar_mag: torch.Tensor, bass_mag: torch.Tensor
    ) -> torch.Tensor:
        """
        Spectral centroid separation heuristic — NOT SI-SDR.

        Takes cached masked-magnitude spectrograms ``(B, F, T_f)`` from
        ``forward`` (so no additional STFT is computed; see C-2.6 / PERF-7.1).
        This is a lightweight proxy for "are the stems spectrally distinct?"
        True SI-SDR requires a ground-truth reference and is computed
        only during training (via PerceptualAudioLoss).
        Renamed from _estimate_si_sdr per C-2.6 / TODO.md.
        """
        guitar_centroid = self._spectral_centroid(guitar_mag)  # (B, T_f)
        bass_centroid = self._spectral_centroid(bass_mag)  # (B, T_f)
        separation = torch.abs(guitar_centroid - bass_centroid).mean(dim=-1)
        return torch.clamp(10 * torch.log10(separation + 1e-8) + 20, min=0, max=30)

    def _spectral_centroid(self, mag: torch.Tensor) -> torch.Tensor:
        """Spectral centroid per frame from a magnitude spectrogram (B, F, T_f)."""
        freqs = torch.linspace(
            0, self.n_fft // 2, self.n_fft // 2 + 1, device=mag.device, dtype=mag.dtype
        )
        centroid = (mag * freqs.view(1, -1, 1)).sum(dim=1) / (mag.sum(dim=1) + 1e-8)
        return centroid  # (B, T_frames)
