#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""
Stem+MIDI Pro — Production model wrapper.
CPU-only, bare-metal i5 target. No NeMo, no PyTorch Lightning.

Backends (config ``backends.separator`` / ``backends.transcriber``):
  separator:   mamba | demucs | fake
  transcriber: mamba | basic_pitch | fake
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.backends.factory import build_separator, build_transcriber
from models.confidence_injector import ConfidenceInjector
from models.losses import PerceptualAudioLoss
from utils.quality_gates import ProcessingReport, route_by_quality

# Perf: set thread count at import time
torch.set_num_threads(os.cpu_count() or 1)
torch.set_float32_matmul_precision("high")


class StemMidiModel(nn.Module):
    """
    Stem+MIDI Pro model wrapper.
    Integrates pluggable separation/transcription backends, confidence injection,
    and quality routing.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config

        self.separator = build_separator(self.cfg)
        self.transcriber = build_transcriber(self.cfg)
        # Register nn.Module backends so .parameters() / .to() / train() work
        if isinstance(self.separator, nn.Module):
            self.add_module("separator_module", self.separator)
        if isinstance(self.transcriber, nn.Module):
            self.add_module("transcriber_module", self.transcriber)

        self.confidence_injector = ConfidenceInjector(
            onset_threshold=self.cfg.get("transcriber", {}).get("onset_threshold", 0.5),
            min_confidence=self.cfg["quality_gates"]["min_confidence"],
        )
        self.loss_fn = PerceptualAudioLoss(self.cfg)

    def forward(
        self,
        audio: torch.Tensor,
        target_guitar: Optional[torch.Tensor] = None,
        target_bass: Optional[torch.Tensor] = None,
        target_onsets: Optional[torch.Tensor] = None,
        target_pitch: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Full forward pass: audio → stems → MIDI (guitar+bass) → report → routing."""
        B, C, T = audio.shape

        # 1. Separation
        sep = self.separator.separate(audio)
        guitar_stem = sep["guitar"]
        bass_stem = sep["bass"]
        residual = sep["residual"]
        sep_metrics = sep["metrics"]

        # 2. Transcription — both stems
        guitar_tr = self._transcribe_stem(guitar_stem, instrument="guitar")
        bass_tr = self._transcribe_stem(bass_stem, instrument="bass")

        # 3. Events (prefer backend events; else logits → MIDI)
        guitar_events = self._events_from_transcribe(guitar_tr)
        bass_events = self._events_from_transcribe(bass_tr)

        # 4. Confidence metadata per stem
        g_conf = guitar_tr["confidence"]
        b_conf = bass_tr["confidence"]
        if g_conf.dim() == 3:
            g_conf = g_conf.squeeze(-1)
        if b_conf.dim() == 3:
            b_conf = b_conf.squeeze(-1)

        g_energy = torch.mean(guitar_stem**2, dim=1).squeeze(-1)
        b_energy = torch.mean(bass_stem**2, dim=1).squeeze(-1)
        # inject_midi_metadata expects confidence/energy as 1-D for batch-0 style lists
        enhanced_guitar = self.confidence_injector.inject_midi_metadata(
            guitar_events, g_conf[0] if g_conf.dim() > 1 else g_conf, g_energy[0] if g_energy.dim() > 1 else g_energy
        )
        enhanced_bass = self.confidence_injector.inject_midi_metadata(
            bass_events, b_conf[0] if b_conf.dim() > 1 else b_conf, b_energy[0] if b_energy.dim() > 1 else b_energy
        )

        avg_conf = float((g_conf.mean() + b_conf.mean()) / 2.0)
        low_notes = (
            enhanced_guitar["summary"]["low_confidence_count"]
            + enhanced_bass["summary"]["low_confidence_count"]
        )

        # Backward-compatible flat view (guitar primary) + per-stem keys
        enhanced_midi = {
            "midi_events": enhanced_guitar["midi_events"],
            "summary": {
                "avg_confidence": avg_conf,
                "low_confidence_count": low_notes,
                "alignment_warnings": (
                    enhanced_guitar["summary"].get("alignment_warnings", 0)
                    + enhanced_bass["summary"].get("alignment_warnings", 0)
                ),
            },
            "guitar": enhanced_guitar,
            "bass": enhanced_bass,
        }

        # 5. Processing report
        report = ProcessingReport(
            si_sdr=float(sep_metrics[0, 0].item()),
            phase_coherence=float(sep_metrics[0, 1].item()),
            avg_confidence=avg_conf,
            artifact_flags=self._detect_artifacts(guitar_stem, bass_stem),
            low_confidence_notes=low_notes,
            config=self.cfg,
        )

        # 6. Routing
        routing = route_by_quality(report)

        outputs = {
            "guitar_stem": guitar_stem,
            "bass_stem": bass_stem,
            "residual": residual,
            "midi_metadata": enhanced_midi,
            "processing_report": report,
            "routing_decision": routing,
        }

        # 7. Loss (training only — uses guitar onset path)
        onset_logits = guitar_tr.get("onset_logits")
        if target_guitar is not None and target_bass is not None:
            loss, loss_dict = self.loss_fn(
                pred_stems=[guitar_stem, bass_stem],
                target_stems=[target_guitar, target_bass],
                pred_onsets=onset_logits,
                target_onsets=target_onsets,
            )
            outputs["loss"] = loss
            outputs["loss_dict"] = loss_dict

        return outputs

    def _transcribe_stem(self, stem: torch.Tensor, instrument: str) -> Dict[str, Any]:
        if hasattr(self.transcriber, "set_instrument"):
            self.transcriber.set_instrument(instrument)
        return self.transcriber.transcribe(stem)

    def _events_from_transcribe(self, tr: Dict[str, Any]) -> List[List[Dict]]:
        if tr.get("events") is not None:
            return tr["events"]
        return self._logits_to_midi(
            tr["onset_logits"],
            tr["pitch_logits"],
            tr["velocity"],
            tr["confidence"],
        )

    def _logits_to_midi(self, onset_logits, pitch_logits, velocity, confidence):
        """Convert network outputs to MIDI event list (vectorized)."""
        onsets = (
            torch.sigmoid(onset_logits) > self.cfg["transcriber"].get("onset_threshold", 0.5)
        ).squeeze(-1)
        pitches = torch.argmax(pitch_logits, dim=-1)
        vel_values = (velocity * 127).long().squeeze(-1)

        events = []
        for b in range(onsets.shape[0]):
            onset_mask = onsets[b]
            indices = torch.nonzero(onset_mask, as_tuple=False).squeeze(-1)
            batch_events = []
            for t in indices:
                batch_events.append(
                    {
                        "note": pitches[b, t].item(),
                        "onset_frame": t.item(),
                        "velocity": vel_values[b, t].item(),
                        "confidence": confidence[b, t].item(),
                    }
                )
            events.append(batch_events)
        return events

    def _detect_artifacts(self, guitar: torch.Tensor, bass: torch.Tensor) -> List[str]:
        """Simple artifact detection for quality gating."""
        flags = []

        if guitar.abs().max() > 0.99 or bass.abs().max() > 0.99:
            flags.append("clipping")

        guitar_noise = torch.mean(guitar**2, dim=[1, 2])
        bass_noise = torch.mean(bass**2, dim=[1, 2])
        if torch.any(guitar_noise > 0.1) or torch.any(bass_noise > 0.1):
            flags.append("noise_floor")

        # C-2.8: Anti-correlation (≈ -1) = cancellation; positive correlation is fine
        cross_corr = torch.sum(guitar * bass, dim=[1, 2])
        energy_guitar = torch.sum(guitar**2, dim=[1, 2])
        energy_bass = torch.sum(bass**2, dim=[1, 2])
        phase_corr = cross_corr / (torch.sqrt(energy_guitar * energy_bass) + 1e-8)
        if torch.any(phase_corr < -0.7):
            flags.append("phase_cancellation")

        return flags if flags else ["none"]

    def process_audio_file(self, audio_path: str, quantize: bool = False) -> Dict:
        """Inference method for production use.

        Args:
            audio_path: Path to input audio.
            quantize: If True, run INT8 dynamic quantization on nn.Linear modules
                for this call (CPU path). Custom Mamba ops stay in floating point.
        """
        audio, sr = self._load_audio(audio_path)
        audio = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)

        if sr != self.cfg["audio"]["sample_rate"]:
            audio = torch.nn.functional.interpolate(
                audio,
                size=int(audio.shape[-1] * self.cfg["audio"]["sample_rate"] / sr),
                mode="linear",
                align_corners=False,
            )

        infer_model: nn.Module = self
        if quantize:
            # Dynamic quant on Linear only when nn.Module backends are present.
            has_modules = isinstance(self.separator, nn.Module) or isinstance(
                self.transcriber, nn.Module
            )
            if has_modules:
                import copy

                qmodel = copy.deepcopy(self)
                qmodel = torch.ao.quantization.quantize_dynamic(
                    qmodel, {nn.Linear}, dtype=torch.qint8
                )
                infer_model = qmodel

        with torch.inference_mode():
            outputs = infer_model.forward(audio)

        midi_meta = outputs["midi_metadata"]
        return {
            "stems": {
                "guitar": outputs["guitar_stem"].squeeze().cpu().numpy(),
                "bass": outputs["bass_stem"].squeeze().cpu().numpy(),
            },
            "midi": midi_meta,
            "midi_guitar": midi_meta.get("guitar", midi_meta),
            "midi_bass": midi_meta.get("bass", {"midi_events": [], "summary": {}}),
            "report": outputs["processing_report"],
            "routing": outputs["routing_decision"],
            "sample_rate": self.cfg["audio"]["sample_rate"],
        }

    def process_audio_streaming(self, audio_path: str, chunk_seconds: float = 2.0) -> Dict:
        """
        Streaming inference: process audio in chunks.
        NOTE: True SSM state caching not yet implemented (C-2.5).
        This path concatenates outputs — works for short files but not real streaming.
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

                sep = self.separator.separate(chunk)
                guitar = sep["guitar"]
                bass = sep["bass"]

                trim_len = end - pos if pos + chunk_len > total_len else hop_len
                all_guitar.append(guitar[:, :, :trim_len])
                all_bass.append(bass[:, :, :trim_len])

                pos += hop_len
                n_chunks += 1

        full_guitar = torch.cat(all_guitar, dim=-1)
        full_bass = torch.cat(all_bass, dim=-1)

        # Transcribe full concatenated stems (chunked sep only)
        guitar_tr = self._transcribe_stem(full_guitar, instrument="guitar")
        bass_tr = self._transcribe_stem(full_bass, instrument="bass")
        guitar_events = self._events_from_transcribe(guitar_tr)
        bass_events = self._events_from_transcribe(bass_tr)

        g_conf = guitar_tr["confidence"]
        b_conf = bass_tr["confidence"]
        if g_conf.dim() == 3:
            g_conf = g_conf.squeeze(-1)
        if b_conf.dim() == 3:
            b_conf = b_conf.squeeze(-1)
        g_energy = torch.mean(full_guitar**2, dim=1).squeeze(-1)
        b_energy = torch.mean(full_bass**2, dim=1).squeeze(-1)

        enhanced_guitar = self.confidence_injector.inject_midi_metadata(
            guitar_events,
            g_conf[0] if g_conf.dim() > 1 else g_conf,
            g_energy[0] if g_energy.dim() > 1 else g_energy,
        )
        enhanced_bass = self.confidence_injector.inject_midi_metadata(
            bass_events,
            b_conf[0] if b_conf.dim() > 1 else b_conf,
            b_energy[0] if b_energy.dim() > 1 else b_energy,
        )
        avg_conf = float((g_conf.mean() + b_conf.mean()) / 2.0)
        low_notes = (
            enhanced_guitar["summary"]["low_confidence_count"]
            + enhanced_bass["summary"]["low_confidence_count"]
        )
        enhanced_midi = {
            "midi_events": enhanced_guitar["midi_events"],
            "summary": {
                "avg_confidence": avg_conf,
                "low_confidence_count": low_notes,
                "alignment_warnings": (
                    enhanced_guitar["summary"].get("alignment_warnings", 0)
                    + enhanced_bass["summary"].get("alignment_warnings", 0)
                ),
            },
            "guitar": enhanced_guitar,
            "bass": enhanced_bass,
        }

        report = ProcessingReport(
            si_sdr=0.0,
            phase_coherence=0.0,
            avg_confidence=avg_conf,
            artifact_flags=self._detect_artifacts(full_guitar, full_bass),
            low_confidence_notes=low_notes,
            config=self.cfg,
        )

        routing = route_by_quality(report)

        return {
            "stems": {
                "guitar": full_guitar.squeeze().cpu().numpy(),
                "bass": full_bass.squeeze().cpu().numpy(),
            },
            "midi": enhanced_midi,
            "midi_guitar": enhanced_guitar,
            "midi_bass": enhanced_bass,
            "report": report,
            "routing": routing,
            "sample_rate": sr,
            "chunks_processed": n_chunks,
        }

    def _load_audio(self, path: str) -> Tuple[np.ndarray, int]:
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


def load_from_checkpoint(checkpoint_path: str, config_path: str) -> StemMidiModel:
    """Load model from a PyTorch checkpoint.

    Accepts either ``{"model_state_dict": ...}`` (train.py format) or a bare
    state_dict. Uses ``map_location="cpu"`` and ``weights_only=True``.
    """
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model = StemMidiModel(config)
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {checkpoint_path!r}: {e}") from e
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"state_dict mismatch loading {checkpoint_path!r}: {e}"
        ) from e
    return model


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--quantize", action="store_true", help="Enable INT8 dynamic quantization")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.checkpoint:
        model = load_from_checkpoint(args.checkpoint, args.config)
    else:
        model = StemMidiModel(config)

    result = model.process_audio_file(args.audio, quantize=args.quantize)

    print("Processing complete!")
    print(f"Quality: {result['report'].quality_tier.value}")
    print(f"SI-SDR: {result['report'].si_sdr:.1f}dB")
    print(f"Avg Confidence: {result['report'].avg_confidence:.0%}")
    print(f"Routing: {result['routing']['action']}")