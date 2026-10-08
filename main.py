#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
StemMidiModel — plain PyTorch orchestrator for Stem+MIDI Pro
(separation → transcription → confidence injection → quality routing).
"""

import os

import numpy as np
import torch
import torch.nn as nn

# CPU inference tuning (PERF-7.4 / PERF-7.9). Override thread count via OMP_NUM_THREADS.
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)))
torch.set_float32_matmul_precision("high")  # allows TF32/AVX-512 matmul where available

# Import our custom components (after torch thread config above — intentional E402)
from models.confidence_injector import ConfidenceInjector  # noqa: E402
from models.losses import PerceptualAudioLoss  # noqa: E402
from models.mamba_separator import MambaSeparator  # noqa: E402
from models.mamba_transcriber import MambaTranscriber  # noqa: E402
from utils.quality_gates import ProcessingReport, route_by_quality  # noqa: E402


class StemMidiModel(nn.Module):
    """
    Orchestrates separation, transcription, confidence injection, and quality routing.

    Input:  audio (B, C=1, T); optional training targets
            target_guitar / target_bass (B, C, T),
            target_onsets (B, T, 1), target_pitch (B, T, V)
    Output: Dict with guitar_stem, bass_stem, midi_metadata,
            processing_report, routing_decision.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config

        # Initialize core components
        self.separator = MambaSeparator(self.cfg)
        self.transcriber = MambaTranscriber(self.cfg)
        self.confidence_injector = ConfidenceInjector(
            onset_threshold=self.cfg["transcriber"].get("onset_threshold", 0.5),
            min_confidence=self.cfg["quality_gates"]["min_confidence"],
        )
        self.loss_fn = PerceptualAudioLoss(self.cfg)

        # For training: store targets for loss computation

    def forward(
        self,
        audio: torch.Tensor,
        target_guitar: torch.Tensor | None = None,
        target_bass: torch.Tensor | None = None,
        target_onsets: torch.Tensor | None = None,
        target_pitch: torch.Tensor | None = None,
    ) -> dict:
        """
        Full forward pass: audio → stems → MIDI + confidence + quality routing.
        Pure inference calls (eval mode, no targets) run under
        ``torch.inference_mode()`` (PERF-7.3).
        """
        if not self.training and target_guitar is None:
            with torch.inference_mode():
                return self._forward_impl(audio)
        return self._forward_impl(audio, target_guitar, target_bass, target_onsets, target_pitch)

    def _forward_impl(
        self,
        audio: torch.Tensor,
        target_guitar: torch.Tensor | None = None,
        target_bass: torch.Tensor | None = None,
        target_onsets: torch.Tensor | None = None,
        target_pitch: torch.Tensor | None = None,
    ) -> dict:
        """Implementation of forward(); see forward() for the public contract."""
        _B, _C, _T = audio.shape

        # 1. Separation (returns None in place of a state cache — real SSM
        # state passing is not implemented; see C-2.5)
        guitar_stem, bass_stem, _residual, _state_cache, sep_metrics = self.separator(audio)

        # 2. Transcription on guitar stem (could also do bass)
        onset_logits, pitch_logits, velocity, _expression, confidence = self.transcriber(
            guitar_stem
        )

        # 3. Convert logits to MIDI events (simplified for MVP)
        midi_events = self._logits_to_midi(onset_logits, pitch_logits, velocity, confidence)

        # 4. Inject confidence metadata and compute alignment scores
        stem_energy = torch.mean(guitar_stem**2, dim=1).squeeze(-1)  # (B, T)
        enhanced_midi = self.confidence_injector.inject_midi_metadata(
            midi_events, confidence.squeeze(-1), stem_energy
        )

        # 5. Compute processing report for quality gating
        report = ProcessingReport(
            si_sdr=sep_metrics[0, 0].item(),  # Average over batch
            phase_coherence=sep_metrics[0, 1].item(),
            avg_confidence=confidence.mean().item(),
            artifact_flags=self._detect_artifacts(guitar_stem, bass_stem),
            low_confidence_notes=enhanced_midi["summary"]["low_confidence_count"],
            config=self.cfg,
        )

        # 6. Get routing decision based on quality
        routing = route_by_quality(report)

        # Prepare outputs
        outputs = {
            "guitar_stem": guitar_stem,
            "bass_stem": bass_stem,
            "midi_metadata": enhanced_midi,  # Contains MIDI events + summary
            "processing_report": report,
            "routing_decision": routing,
        }

        # 7. Compute loss if targets provided (training mode)
        if target_guitar is not None and target_bass is not None:
            loss, loss_dict = self.loss_fn(
                pred_stems=[guitar_stem, bass_stem],
                target_stems=[target_guitar, target_bass],
                pred_onsets=onset_logits,
                target_onsets=self._prepare_onset_target(target_onsets, onset_logits),
            )
            outputs["loss"] = loss
            outputs["loss_dict"] = loss_dict

        return outputs

    @staticmethod
    def _prepare_onset_target(
        target_onsets: torch.Tensor | None, onset_logits: torch.Tensor
    ) -> torch.Tensor | None:
        """Coerce dataset onset targets onto the model's (B, T_frames) grid.

        Dataset items carry ``target_onsets`` shaped ``(1, n_frames, 1)`` —
        a spurious batch dim baked into every item, with a frame count
        computed from a hardcoded hop and without STFT center padding. After
        DataLoader collation this becomes ``(B, 1, T', 1)``, which cannot
        broadcast elementwise against the model's ``(B, T, 1)`` logits.

        Normalization: flatten to ``(B, T')`` and zero-pad / truncate to the
        predicted frame count (padding = "no onset" beyond target length).
        """
        if target_onsets is None:
            return None
        t = target_onsets
        if t.dim() < 2:
            t = t.unsqueeze(0)
        t = t.reshape(t.shape[0], -1).float()  # (B, T')
        n_pred = onset_logits.shape[1]
        if t.shape[1] > n_pred:
            t = t[:, :n_pred]
        elif t.shape[1] < n_pred:
            t = torch.nn.functional.pad(t, (0, n_pred - t.shape[1]))
        return t

    def training_step(self, batch, batch_idx):
        """One training step for the plain-PyTorch loop in train.py (RF-3.1.9).

        Returns a dict with ``loss`` plus per-component values under
        ``loss_dict`` (no Lightning ``self.log`` — the caller logs).
        """
        audio = batch["audio"]
        targets = {
            "target_guitar": batch["target_guitar"],
            "target_bass": batch["target_bass"],
            "target_onsets": batch["target_onsets"],
            "target_pitch": batch["target_pitch"],
        }

        outputs = self.forward(audio, **targets)
        return {"loss": outputs["loss"], "loss_dict": outputs["loss_dict"]}

    def validation_step(self, batch, batch_idx):
        """Validation step; same return contract as training_step."""
        audio = batch["audio"]
        targets = {
            "target_guitar": batch["target_guitar"],
            "target_bass": batch["target_bass"],
            "target_onsets": batch["target_onsets"],
            "target_pitch": batch["target_pitch"],
        }

        outputs = self.forward(audio, **targets)
        return {"loss": outputs["loss"], "loss_dict": outputs["loss_dict"]}

    def _logits_to_midi(self, onset_logits, pitch_logits, velocity, confidence):
        """Convert network outputs to a FLAT MIDI event list.

        Vectorized with torch.nonzero (RF-3.1.6) instead of Python per-frame
        loops. Assumes B=1 for inference (multi-batch call sites flatten batch
        0; batch >1 still returns all batches' events with onset frames as-is).
        """
        onsets = (
            torch.sigmoid(onset_logits) > self.cfg["transcriber"].get("onset_threshold", 0.5)
        ).squeeze(-1)
        pitches = torch.argmax(pitch_logits, dim=-1)
        vel_values = (velocity * 127).long().squeeze(-1)

        hits = torch.nonzero(onsets)  # (N, 2): [batch_idx, frame]
        events = []
        for b, t in hits.tolist():
            events.append(
                {
                    "note": pitches[b, t].item(),
                    "onset_frame": t,
                    "velocity": vel_values[b, t].item(),
                    "confidence": confidence[b, t].item(),
                }
            )
        return events

    def _detect_artifacts(self, guitar: torch.Tensor, bass: torch.Tensor) -> list[str]:
        """Simple artifact detection for quality gating"""
        flags = []

        # Check for clipping
        if guitar.abs().max() > 0.99 or bass.abs().max() > 0.99:
            flags.append("clipping")

        # Check for excessive noise (simplified)
        guitar_noise = torch.mean(guitar**2, dim=[1, 2])
        bass_noise = torch.mean(bass**2, dim=[1, 2])
        if torch.any(guitar_noise > 0.1) or torch.any(bass_noise > 0.1):
            flags.append("noise_floor")

        # Check for phase cancellation: stems that are nearly perfectly
        # ANTI-correlated cancel when summed (C-2.8). Positive correlation is
        # valid (shared fundamentals); sign matters here.
        cross_corr = torch.sum(guitar * bass, dim=[1, 2])
        energy_guitar = torch.sum(guitar**2, dim=[1, 2])
        energy_bass = torch.sum(bass**2, dim=[1, 2])
        phase_corr = cross_corr / (torch.sqrt(energy_guitar * energy_bass) + 1e-8)
        if torch.any(phase_corr < -0.9):  # near-perfect anti-correlation = cancellation
            flags.append("phase_cancellation")

        return flags if flags else ["none"]

    def _get_quantized_self(self) -> "StemMidiModel":
        """Lazily build and cache an INT8 dynamic-quantized clone (PERF-7.7).

        Dynamic quantization is applied to nn.Linear layers only, which keeps
        masks/logits numerically close to the fp32 model while cutting CPU
        matmul cost. The fp32 original is left untouched.
        """
        quantized = getattr(self, "_quantized_model", None)
        if quantized is None:
            import copy

            clone = copy.deepcopy(self)
            clone.eval()
            quantized = torch.ao.quantization.quantize_dynamic(
                clone, {nn.Linear}, dtype=torch.qint8
            )
            # Bypass nn.Module.__setattr__ so the clone is NOT registered as a
            # submodule (keeps it out of state_dict / parameters()).
            object.__setattr__(self, "_quantized_model", quantized)
        return quantized

    def process_audio_file(self, audio_path: str, quantize: bool = False) -> dict:
        """
        Inference method for production use
        Handles file I/O and returns user-ready package

        Args:
            audio_path: path to the input audio file.
            quantize: when True, run inference with an INT8 dynamic-quantized
                clone of the model (PERF-7.7/7.8). Output shapes are identical
                to the fp32 path.
        """
        if quantize:
            return self._get_quantized_self().process_audio_file(audio_path, quantize=False)

        # Load and preprocess audio (would use audio_io.py in full implementation)
        audio, sr = self._load_audio(audio_path)
        audio = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)  # (1, 1, T)

        # Ensure correct sample rate
        if sr != self.cfg["audio"]["sample_rate"]:
            audio = torch.nn.functional.interpolate(
                audio,
                size=int(audio.shape[-1] * self.cfg["audio"]["sample_rate"] / sr),
                mode="linear",
                align_corners=False,
            )

        # Run inference
        with torch.inference_mode():
            outputs = self.forward(audio)

        # Format for user delivery
        return {
            "stems": {
                "guitar": outputs["guitar_stem"].squeeze().cpu().numpy(),
                "bass": outputs["bass_stem"].squeeze().cpu().numpy(),
            },
            "midi": outputs["midi_metadata"],
            "report": outputs["processing_report"],
            "routing": outputs["routing_decision"],
            "sample_rate": self.cfg["audio"]["sample_rate"],
        }

    def process_audio_streaming(self, audio_path: str, chunk_seconds: float = 2.0) -> dict:
        """
        Streaming inference: process audio in chunks.

        NOTE: chunks are processed independently and recombined with
        overlap-add at the waveform level; true SSM state passing between
        chunks is not implemented yet (see C-2.5 / ARCH-11.8.1 in TODO.md).

        Args:
            audio_path: Path to input audio file
            chunk_seconds: Duration of each chunk in seconds (default 2.0)

        Returns:
            Same format as process_audio_file (aggregated outputs)
        """
        audio, sr = self._load_audio(audio_path)
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)

        target_sr = self.cfg["audio"]["sample_rate"]
        if sr != target_sr:
            audio_t = torch.nn.functional.interpolate(
                audio_t,
                size=int(audio_t.shape[-1] * target_sr / sr),
                mode="linear",
                align_corners=False,
            )
            sr = target_sr

        chunk_len = int(chunk_seconds * sr)
        overlap_len = int(chunk_len * self.cfg["audio"].get("overlap_ratio", 0.5))
        hop_len = chunk_len - overlap_len
        total_len = audio_t.shape[-1]

        all_guitar = []
        all_bass = []
        all_residual = []
        all_confidence = []
        all_onsets = []

        n_chunks = 0

        self.eval()
        with torch.inference_mode():
            pos = 0
            while pos < total_len:
                end = min(pos + chunk_len, total_len)
                chunk = audio_t[:, :, pos:end]

                if chunk.shape[-1] < chunk_len:
                    pad_len = chunk_len - chunk.shape[-1]
                    chunk = torch.nn.functional.pad(chunk, (0, pad_len))

                # C-2.5: separator has no real SSM state passing yet; chunks are
                # processed independently and recombined by overlap-add.
                guitar, bass, residual, _state_cache, _sep_metrics = self.separator(chunk)

                trim_len = end - pos if pos + chunk_len > total_len else hop_len
                all_guitar.append(guitar[:, :, :trim_len])
                all_bass.append(bass[:, :, :trim_len])
                all_residual.append(residual[:, :, :trim_len])

                trimmed = guitar[:, :, :trim_len]
                onsets, _pitches, velocity, _expression, confidence = self.transcriber(trimmed)
                all_onsets.append(onsets)
                all_confidence.append(confidence)

                pos += hop_len
                n_chunks += 1

        full_guitar = torch.cat(all_guitar, dim=-1)
        full_bass = torch.cat(all_bass, dim=-1)

        final_metrics = torch.tensor([0.0, 0.0])

        onset_logits = torch.cat(all_onsets, dim=1)
        pitch_logits = torch.zeros((1, onset_logits.shape[1], 128))
        velocity = torch.zeros((1, onset_logits.shape[1], 1))
        confidence = torch.cat(all_confidence, dim=1)

        midi_events = self._logits_to_midi(onset_logits, pitch_logits, velocity, confidence)
        stem_energy = torch.mean(full_guitar**2, dim=1).squeeze(-1)
        enhanced_midi = self.confidence_injector.inject_midi_metadata(
            midi_events, confidence.squeeze(-1), stem_energy
        )

        report = ProcessingReport(
            si_sdr=final_metrics[0].item(),
            phase_coherence=final_metrics[1].item(),
            avg_confidence=confidence.mean().item(),
            artifact_flags=self._detect_artifacts(full_guitar, full_bass),
            low_confidence_notes=enhanced_midi["summary"]["low_confidence_count"],
            config=self.cfg,
        )

        routing = route_by_quality(report)

        return {
            "stems": {
                "guitar": full_guitar.squeeze().cpu().numpy(),
                "bass": full_bass.squeeze().cpu().numpy(),
            },
            "midi": enhanced_midi,
            "report": report,
            "routing": routing,
            "sample_rate": sr,
            "chunks_processed": n_chunks,
        }

    def _load_audio(self, path: str) -> tuple[np.ndarray, int]:
        """Load audio file using soundfile (or librosa as fallback)."""
        try:
            import soundfile as sf

            audio, sr = sf.read(path)
        except Exception:
            import librosa

            audio, sr = librosa.load(path, sr=None, mono=True)
        if audio.ndim > 1 and audio.shape[1] > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), sr


def load_from_checkpoint(checkpoint_path: str, config_path: str | None = None) -> StemMidiModel:
    """Load model weights from a plain PyTorch checkpoint.

    Expects either a raw ``state_dict`` or a dict with a ``"state_dict"`` key
    (e.g. saved by ``train.py``). Config must come from ``config_path``.
    """
    if config_path is None:
        raise ValueError(
            "config_path is required when loading from checkpoint "
            "(config cannot be extracted from checkpoint in this prototype)"
        )

    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = StemMidiModel(config=config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    return model


def cli() -> None:
    """Console-script entry point (pyproject ``stem-midi-cli``)."""
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.environ.get("MODEL_CONFIG_PATH", "configs/model_config.yaml"),
        help="Path to model config YAML (or set MODEL_CONFIG_PATH)",
    )
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint (.pt) to load")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.checkpoint:
        # Load from checkpoint
        model = load_from_checkpoint(args.checkpoint, args.config)
    else:
        # Initialize new model
        model = StemMidiModel(config)

    # Process audio
    result = model.process_audio_file(args.audio)

    # Print summary
    print("Processing complete!")
    print(f"Quality: {result['report'].quality_tier.value}")
    print(f"Centroid separation (proxy, not SI-SDR): {result['report'].si_sdr:.1f}dB")
    print(f"Avg Confidence: {result['report'].avg_confidence:.0%}")
    print(f"Routing: {result['routing']['action']}")


if __name__ == "__main__":
    cli()
