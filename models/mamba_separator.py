import torch
import torch.nn as nn
from mamba_ssm import Mamba
from nemo.core.classes import Module, typecheck
from nemo.core.neural_types import AudioSignal, NeuralType, ChannelType, TimeType


class MambaSeparator(Module):
    """
    Phase-coherent guitar/bass separation using Mamba SSM.

    NOTE: Streaming inference with SSM state caching is not yet implemented.
    The original prototype used activation concatenation (not real SSM state),
    which was structurally incorrect. See C-2.5 in TODO.md for the full design
    when real SSM state caching is added. For now, each forward call is
    independent — overlap-add in main.py works at the waveform level only.
    """

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'C', 'T'), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "guitar_stem": NeuralType(('B', 'C', 'T'), AudioSignal()),
            "bass_stem": NeuralType(('B', 'C', 'T'), AudioSignal()),
            "residual": NeuralType(('B', 'C', 'T'), AudioSignal()),
            "new_state_cache": typecheck.Optional(NeuralType(('B', 'L', 'D'), ChannelType())),
            "metrics": NeuralType(('B'), ChannelType()),
        }

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.hop_length = cfg['audio']['hop_length']
        self.n_fft = cfg['audio']['n_fft']
        d_model = cfg['separator']['d_model']

        # STFT encoder (fixed, no grad)
        self.register_buffer("window", torch.hann_window(self.n_fft))

        # Learnable projection: spectral bins → Mamba dimension
        self.input_proj = nn.Sequential(
            nn.Linear(self.n_fft // 2 + 1, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        # Mamba backbone — valid kwargs only (no causal_conv1d_impl etc.)
        self.mamba_blocks = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=cfg['separator']['d_state'],
                d_conv=cfg['separator']['d_conv'],
                expand=cfg['separator']['expand'],
                use_fast_path=True,
            ) for _ in range(cfg['separator']['n_layer'])
        ])

        # Multi-head output: guitar, bass, residual masks
        self.mask_heads = nn.ModuleDict({
            "guitar": nn.Linear(d_model, self.n_fft // 2 + 1),
            "bass": nn.Linear(d_model, self.n_fft // 2 + 1),
            "residual": nn.Linear(d_model, self.n_fft // 2 + 1),
        })

        # Phase coherence head (auxiliary loss)
        self.phase_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    @typecheck()
    def forward(self, audio: torch.Tensor):
        B, C, T = audio.shape

        # STFT: (B, C, T) → (B, F, T_frames)
        spec = torch.stft(
            audio.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        mag = torch.abs(spec)      # (B, F, T_f)
        phase = torch.angle(spec)  # (B, F, T_f)

        # Project to Mamba dim: (B, T_f, F) → (B, T_f, D)
        x = mag.transpose(1, 2)                     # (B, T_f, F)
        hidden = self.input_proj(x)                 # (B, T_f, D)
        assert hidden.shape[-1] == self.mamba_blocks[0].d_model, (
            f"hidden dim {hidden.shape[-1]} != d_model {self.mamba_blocks[0].d_model}"
        )

        # Mamba blocks
        for block in self.mamba_blocks:
            hidden = block(hidden)

        # Generate masks + phase coherence score
        masks = {k: head(hidden) for k, head in self.mask_heads.items()}  # (B, T_f, n_fft//2+1)
        phase_score = self.phase_head(hidden.mean(dim=1))  # (B, 1)

        # Apply masks + inverse STFT
        stems = {}
        for name, mask in masks.items():
            mask_mag = torch.sigmoid(mask)
            phase_aligned = phase.transpose(1, 2)        # (B, T_f, F)
            masked_spec = mask_mag * torch.exp(1j * phase_aligned)
            masked_spec = masked_spec.transpose(1, 2).contiguous()  # (B, F, T_f)
            stem = torch.istft(
                masked_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                length=T,
            )
            stems[name] = stem.unsqueeze(1)  # (B, 1, T)

        # Perceptual metric (spectral centroid separation proxy — NOT real SI-SDR)
        centroid_metric = self._centroid_separation(stems["guitar"], stems["bass"])
        metrics = torch.stack([centroid_metric, phase_score.squeeze(-1)], dim=-1)

        return (
            stems["guitar"],
            stems["bass"],
            stems["residual"],
            None,       # new_state_cache placeholder (C-2.5: real SSM state TBD)
            metrics,
        )

    def _centroid_separation(self, guitar: torch.Tensor, bass: torch.Tensor) -> torch.Tensor:
        """
        Spectral centroid separation heuristic — NOT SI-SDR.

        This is a lightweight proxy for "are the stems spectrally distinct?"
        True SI-SDR requires a ground-truth reference and is computed
        only during training (via PerceptualAudioLoss).
        Renamed from _estimate_si_sdr per C-2.6 / TODO.md.
        """
        guitar_centroid = self._spectral_centroid(guitar)   # (B, T_f)
        bass_centroid = self._spectral_centroid(bass)       # (B, T_f)
        separation = torch.abs(guitar_centroid - bass_centroid).mean(dim=-1)
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
        freqs = torch.linspace(0, self.n_fft // 2, self.n_fft // 2 + 1, device=audio.device)
        centroid = (mag * freqs.unsqueeze(0).unsqueeze(-1)).sum(dim=1) / (mag.sum(dim=1) + 1e-8)
        return centroid  # (B, T_frames)