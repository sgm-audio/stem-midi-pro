import torch
import torch.nn as nn
from mamba_ssm import Mamba
from nemo.core.classes import Module, typecheck
from nemo.core.neural_types import AudioSignal, NeuralType, ChannelType


class MambaSeparator(Module):
    """
    Phase-coherent guitar/bass separation using Mamba SSM.
    Designed for streaming inference with overlap-add.
    """

    @property
    def input_types(self):
        return {
            "audio": NeuralType(("B", "C", "T"), AudioSignal()),
            "state_cache": typecheck.Optional(
                NeuralType(("B", "L", "D"), ChannelType())
            ),
        }

    @property
    def output_types(self):
        return {
            "guitar_stem": NeuralType(("B", "C", "T"), AudioSignal()),
            "bass_stem": NeuralType(("B", "C", "T"), AudioSignal()),
            "residual": NeuralType(("B", "C", "T"), AudioSignal()),
            "new_state_cache": typecheck.Optional(
                NeuralType(("B", "L", "D"), ChannelType())
            ),
            "metrics": NeuralType(
                ("B"), ChannelType()
            ),  # SI-SDR estimate, phase coherence
        }

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.hop_length = cfg["audio"]["hop_length"]
        self.n_fft = cfg["audio"]["n_fft"]

        # STFT encoder (fixed, no grad)
        self.register_buffer("window", torch.hann_window(self.n_fft))

        # Learnable projection: spectral bins → Mamba dimension
        self.input_proj = nn.Sequential(
            nn.Linear(self.n_fft // 2 + 1, cfg["separator"]["d_model"]),
            nn.GELU(),
            nn.LayerNorm(cfg["separator"]["d_model"]),
        )

        # Mamba backbone with selective scan
        self.mamba_blocks = nn.ModuleList(
            [
                Mamba(
                    d_model=cfg["separator"]["d_model"],
                    d_state=cfg["separator"]["d_state"],
                    d_conv=cfg["separator"]["d_conv"],
                    expand=cfg["separator"]["expand"],
                    use_fast_path=True,  # CUDA kernel fusion
                    causal_conv1d_impl=cfg["separator"]["causal_conv1d_impl"],
                    selective_scan_impl=cfg["separator"]["selective_scan_impl"],
                )
                for _ in range(cfg["separator"]["n_layer"])
            ]
        )

        # Multi-head output: guitar, bass, residual masks
        self.mask_heads = nn.ModuleDict(
            {
                "guitar": nn.Linear(cfg["separator"]["d_model"], self.n_fft // 2 + 1),
                "bass": nn.Linear(cfg["separator"]["d_model"], self.n_fft // 2 + 1),
                "residual": nn.Linear(cfg["separator"]["d_model"], self.n_fft // 2 + 1),
            }
        )

        # Phase coherence head (auxiliary loss)
        self.phase_head = nn.Sequential(
            nn.Linear(cfg["separator"]["d_model"], 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # Output: phase coherence score [0,1]
        )

    @typecheck()
    def forward(self, audio: torch.Tensor, state_cache: torch.Tensor = None):
        B, C, T = audio.shape

        # STFT: (B, C, T) → (B, C, F, T_frames)
        spec = torch.stft(
            audio.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        mag = torch.abs(spec)  # (B, F, T)
        phase = torch.angle(spec)

        # Project to Mamba dim: (B, T, F) → (B, T, D)
        x = mag.transpose(1, 2)  # (B, T, F)
        hidden = self.input_proj(x)

        # Streaming state handling
        new_state_cache = None
        if state_cache is not None:
            # Inject cached SSM state for chunk continuity
            hidden = torch.cat([state_cache, hidden], dim=1)

        # Mamba selective scan blocks
        for block in self.mamba_blocks:
            hidden = block(hidden)
            # Optional: gradient checkpointing for long sequences
            # hidden = torch.utils.checkpoint.checkpoint(block, hidden)

        # Extract new state cache for next chunk (last L tokens)
        if state_cache is not None:
            new_state_cache = hidden[:, -state_cache.shape[1] :, :]
            hidden = hidden[:, state_cache.shape[1] :, :]  # Trim prepended cache

        # Generate masks + phase coherence score
        masks = {k: head(hidden) for k, head in self.mask_heads.items()}  # (B, T, F)
        phase_score = self.phase_head(hidden.mean(dim=1))  # (B, 1)

        # Apply masks + inverse STFT
        stems = {}
        for name, mask in masks.items():
            mask_mag = torch.sigmoid(mask)  # Soft mask [0,1]
            masked_spec = mask_mag.unsqueeze(1) * torch.exp(1j * phase)  # (B, 1, F, T)
            stem = torch.istft(
                masked_spec.squeeze(1),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                length=T * self.hop_length,  # Ensure exact output length
            )
            stems[name] = stem.unsqueeze(1)  # (B, 1, T)

        # Compute perceptual metrics (for confidence injection)
        si_sdr_est = self._estimate_si_sdr(stems["guitar"], stems["bass"])
        metrics = torch.stack([si_sdr_est, phase_score.squeeze(-1)], dim=-1)

        return (
            stems["guitar"],
            stems["bass"],
            stems["residual"],
            new_state_cache,
            metrics,
        )

    def _estimate_si_sdr(
        self, guitar: torch.Tensor, bass: torch.Tensor
    ) -> torch.Tensor:
        """Lightweight SI-SDR proxy for confidence scoring (no ground truth)."""
        # Use spectral centroid separation as proxy
        guitar_centroid = self._spectral_centroid(guitar)
        bass_centroid = self._spectral_centroid(bass)
        separation = torch.abs(guitar_centroid - bass_centroid).mean(dim=-1)
        # Normalize to approximate dB scale
        return torch.clamp(10 * torch.log10(separation + 1e-8) + 20, min=0, max=30)

    def _spectral_centroid(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute spectral centroid per frame for separation heuristic."""
        spec = torch.stft(
            audio.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        mag = torch.abs(spec)
        freqs = torch.linspace(
            0, self.n_fft // 2, self.n_fft // 2 + 1, device=audio.device
        )
        centroid = (mag * freqs.unsqueeze(0).unsqueeze(-1)).sum(dim=1) / (
            mag.sum(dim=1) + 1e-8
        )
        return centroid  # (B, T_frames)
