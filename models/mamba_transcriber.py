# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class MambaTranscriber(nn.Module):
    """
    Polyphonic audio-to-MIDI with expression detection.
    Outputs: onset logits, pitch logits, velocity, expression CCs.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        d_model = cfg['transcriber']['d_model']

        # Input projection: mel spectrogram → Mamba dim
        self.n_mels = cfg['audio']['n_mels']
        self.n_fft = cfg['audio']['n_fft']
        self.hop_length = cfg['audio']['hop_length']
        self.register_buffer("mel_window", torch.hann_window(self.n_fft))

        # Pre-compute mel filter bank once (C-2.9: was per-forward Python loop)
        mel_basis = self._build_mel_basis(
            sr=cfg['audio']['sample_rate'],
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=0,
            fmax=cfg['audio']['sample_rate'] // 2,
        )
        self.register_buffer("mel_basis", mel_basis)

        self.input_proj = nn.Linear(self.n_mels, d_model)

        # Mamba backbone
        self.mamba = nn.Sequential(*[
            Mamba(
                d_model=d_model,
                d_state=cfg['transcriber']['d_state'],
                d_conv=4,
                expand=2,
                use_fast_path=True
            ) for _ in range(cfg['transcriber']['n_layer'])
        ])

        # Multi-task heads
        self.onset_head = nn.Linear(d_model, 1)
        self.pitch_head = nn.Linear(d_model, cfg['transcriber']['pitch_vocab_size'])
        self.velocity_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Normalized velocity [0,1]
        )
        self.expression_head = nn.Linear(d_model, len(cfg['transcriber']['expression_heads']))

        # Confidence head: MC-dropout ensemble (eval-only intent made explicit)
        self.mc_dropout_eval = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Store config for inference
        self.onset_threshold = cfg['transcriber'].get('onset_threshold', 0.5)

    def forward(self, stem: torch.Tensor):
        B, C, T = stem.shape
        assert C == 1, f"Expected mono input (C=1), got C={C}"

        # Mel spectrogram input: (B, C, T) → (B, T, n_mels)
        mel = self._mel_spectrogram(stem.squeeze(1)).transpose(1, 2)
        hidden = self.input_proj(mel)

        # Mamba processing
        for block in self.mamba:
            hidden = block(hidden)

        # Task-specific outputs
        onset_logits = self.onset_head(hidden)  # (B, T, 1)
        pitch_logits = self.pitch_head(hidden)  # (B, T, V)
        velocity = self.velocity_head(hidden)   # (B, T, 1)
        expression = torch.sigmoid(self.expression_head(hidden))  # (B, T, E)

        # Confidence: Monte Carlo dropout ensemble (eval-only)
        if not self.training:
            conf_samples = []
            for _ in range(5):  # 5-sample MC dropout
                conf_samples.append(self.mc_dropout_eval(hidden))
            confidence = torch.stack(conf_samples).mean(dim=0).squeeze(-1)  # (B, T)
        else:
            confidence = self.mc_dropout_eval(hidden).squeeze(-1)

        return onset_logits, pitch_logits, velocity, expression, confidence

    def _mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Differentiable mel-spectrogram for end-to-end training."""
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.mel_window,
            return_complex=True,
            pad_mode='reflect'
        )
        mag = torch.abs(spec)  # (B, F, T)

        # Use pre-computed mel basis (C-2.9: no per-call Python loop)
        mel_basis = self.mel_basis.to(audio.device)
        mel_spec = torch.matmul(mel_basis, mag)  # (B, n_mels, T)

        # Apply log compression
        log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-8))
        return log_mel_spec

    @staticmethod
    def _build_mel_basis(sr, n_fft, n_mels, fmin, fmax):
        """Build mel filter bank matrix (called once in __init__)."""
        mel_fmin = 2595 * torch.log10(torch.tensor(1.0 + fmin / 700))
        mel_fmax = 2595 * torch.log10(torch.tensor(1.0 + fmax / 700))

        mel_points = torch.linspace(mel_fmin, mel_fmax, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_indices = torch.floor((n_fft + 1) * hz_points / sr)

        fbank = torch.zeros((n_mels, n_fft // 2 + 1))
        for j in range(1, n_mels + 1):
            left = int(bin_indices[j - 1])
            center = int(bin_indices[j])
            right = int(bin_indices[j + 1])
            for i in range(left, center):
                if center - left > 0:
                    fbank[j - 1, i] = (i - left) / (center - left)
            for i in range(center, right):
                if right - center > 0:
                    fbank[j - 1, i] = (right - i) / (right - center)
        return fbank