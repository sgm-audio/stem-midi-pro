import torch
import torch.nn as nn
from mamba_ssm import Mamba
from nemo.core.classes import Module, typecheck
from nemo.core.neural_types import AudioSignal, NeuralType, ChannelType, LogitsType


class MambaTranscriber(Module):
    """
    Polyphonic audio-to-MIDI with expression detection.
    Outputs: onset logits, pitch logits, velocity, expression CCs.
    """

    @property
    def input_types(self):
        return {"stem": NeuralType(("B", "C", "T"), AudioSignal())}

    @property
    def output_types(self):
        return {
            "onset_logits": NeuralType(("B", "T", 1), LogitsType()),
            "pitch_logits": NeuralType(
                ("B", "T", "V"), LogitsType()
            ),  # V=128 MIDI notes
            "velocity": NeuralType(("B", "T", 1), ChannelType()),
            "expression": NeuralType(
                ("B", "T", "E"), ChannelType()
            ),  # E=expression types
            "confidence": NeuralType(("B", "T"), ChannelType()),
        }

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        d_model = cfg["transcriber"]["d_model"]

        # Input projection: mel spectrogram → Mamba dim
        self.n_mels = cfg["audio"]["n_mels"]
        self.input_proj = nn.Linear(self.n_mels, d_model)

        # Mamba backbone
        self.mamba = nn.Sequential(
            *[
                Mamba(
                    d_model=d_model,
                    d_state=cfg["transcriber"]["d_state"],
                    d_conv=4,
                    expand=2,
                    use_fast_path=True,
                )
                for _ in range(cfg["transcriber"]["n_layer"])
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
        self.expression_head = nn.Linear(
            d_model, len(cfg["transcriber"]["expression_heads"])
        )

        # Confidence head: ensemble uncertainty estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 128), nn.Dropout(0.1), nn.Linear(128, 1), nn.Sigmoid()
        )

        # Store config for inference
        self.onset_threshold = cfg["transcriber"].get("onset_threshold", 0.5)

    @typecheck()
    def forward(self, stem: torch.Tensor):
        # Mel spectrogram input: (B, C, T) → (B, T, n_mels)
        mel = self._mel_spectrogram(stem.squeeze(1)).transpose(1, 2)
        hidden = self.input_proj(mel)

        # Mamba processing
        for block in self.mamba:
            hidden = block(hidden)

        # Task-specific outputs
        onset_logits = self.onset_head(hidden)  # (B, T, 1)
        pitch_logits = self.pitch_head(hidden)  # (B, T, V)
        velocity = self.velocity_head(hidden)  # (B, T, 1)
        expression = torch.sigmoid(self.expression_head(hidden))  # (B, T, E)

        # Confidence: Monte Carlo dropout ensemble (inference-time)
        if not self.training:
            conf_samples = []
            for _ in range(5):  # 5-sample MC dropout
                conf_samples.append(self.confidence_head(hidden))
            confidence = torch.stack(conf_samples).mean(dim=0).squeeze(-1)  # (B, T)
        else:
            confidence = self.confidence_head(hidden).squeeze(-1)

        return onset_logits, pitch_logits, velocity, expression, confidence

    def _mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Differentiable mel-spectrogram for end-to-end training."""
        # STFT parameters
        n_fft = 2048
        hop_length = 512

        # Compute STFT
        spec = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft, device=audio.device),
            return_complex=True,
            pad_mode="reflect",
        )
        mag = torch.abs(spec)  # (B, F, T)

        # Convert to mel scale
        mel_basis = self._create_mel_basis(
            sr=self.cfg["audio"]["sample_rate"],
            n_fft=n_fft,
            n_mels=self.n_mels,
            fmin=0,
            fmax=self.cfg["audio"]["sample_rate"] // 2,
        ).to(audio.device)

        mel_spec = torch.matmul(mel_basis, mag)  # (B, n_mels, T)

        # Apply log compression
        log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-8))

        return log_mel_spec

    def _create_mel_basis(self, sr, n_fft, n_mels, fmin, fmax):
        """Create mel filter bank matrix"""
        # Mel scale boundaries
        mel_fmin = 2595 * torch.log10(1 + fmin / 700)
        mel_fmax = 2595 * torch.log10(1 + fmax / 700)

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
