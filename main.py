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
from models.backends.factory import build_separator, build_transcriber  # noqa: E402
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
        # A freshly constructed module has random parameters. Forward passes
        # remain available for training/unit tests, but file/stream inference
        # must not present those weights as production output. The checkpoint
        # loader marks a model as ready after state_dict validation.
        # This flag is set only by a validated checkpoint load or by a
        # backend that explicitly attests to its own pretrained weights; it is
        # not accepted as a user-editable config value.
        self._weights_loaded = False
        self._allow_untrained_inference = bool(config.get("allow_untrained_inference", False))

        # Initialize the configured backend. With no ``backends`` section the
        # native Mamba modules remain the backwards-compatible default. A
        # configured CPU profile (for example Demucs + Basic Pitch) must not be
        # silently ignored: the factory is the single construction path.
        if self.cfg.get("backends"):
            self.separator = build_separator(self.cfg)
            self.transcriber = build_transcriber(self.cfg)
        else:
            self.separator = MambaSeparator(self.cfg)
            self.transcriber = MambaTranscriber(self.cfg)
        # Pretrained third-party backends are safe without a project checkpoint
        # only when both constructors explicitly attest that their own weights
        # are available. Native Mamba/fake backends never make that claim.
        self._weights_loaded = self._weights_loaded or (
            bool(getattr(self.separator, "weights_loaded", False))
            and bool(getattr(self.transcriber, "weights_loaded", False))
        )
        self.confidence_injector = ConfidenceInjector(
            onset_threshold=self.cfg["transcriber"].get("onset_threshold", 0.5),
            min_confidence=self.cfg["quality_gates"]["min_confidence"],
        )
        self.loss_fn = PerceptualAudioLoss(self.cfg)

        # For training: store targets for loss computation

    @property
    def weights_loaded(self) -> bool:
        """Whether a validated checkpoint has been loaded into this model."""
        return self._weights_loaded

    def mark_weights_loaded(self) -> None:
        """Mark this instance safe for file inference after loading weights."""
        self._weights_loaded = True

    def _ensure_inference_ready(self) -> None:
        """Reject accidental production use of random initialization."""
        if self._weights_loaded or self._allow_untrained_inference:
            return
        raise RuntimeError(
            "No trained checkpoint is loaded. Set MODEL_CHECKPOINT_PATH (or pass "
            "--checkpoint) to a validated checkpoint before processing audio. "
            "Set allow_untrained_inference=true only for explicit development/tests; "
            "random weights are not usable production output."
        )

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

        # 1. Separation. Native Mamba modules return a tuple; pluggable
        # backends return a typed dict. Normalize both contracts here.
        guitar_stem, bass_stem, _residual, _state_cache, sep_metrics = self._run_separator(audio)

        # 2. Transcription on both separated stems. The guitar stream is used
        # for the primary output and the bass stream gets its own independent
        # event list, so bass.mid is no longer a copy of guitar.mid.
        guitar_transcription = self._run_transcriber(guitar_stem, instrument="guitar")
        bass_transcription = self._run_transcriber(bass_stem, instrument="bass")

        guitar_onset = guitar_transcription["onset_logits"]
        guitar_pitch = guitar_transcription["pitch_logits"]
        guitar_confidence = guitar_transcription["confidence"]
        bass_confidence = bass_transcription["confidence"]

        guitar_events = self._transcription_to_events(guitar_transcription, batch_index=0)
        bass_events = self._transcription_to_events(bass_transcription, batch_index=0)

        # 3. Inject note-level confidence and frame-aligned energy into each
        # stream independently.
        guitar_energy = self._frame_energy(guitar_stem, guitar_confidence.shape[1])
        bass_energy = self._frame_energy(bass_stem, bass_confidence.shape[1])
        guitar_midi = self.confidence_injector.inject_midi_metadata(
            guitar_events, guitar_confidence[0], guitar_energy[0]
        )
        bass_midi = self.confidence_injector.inject_midi_metadata(
            bass_events, bass_confidence[0], bass_energy[0]
        )
        enhanced_midi = {
            **guitar_midi,
            "guitar": guitar_midi,
            "bass": bass_midi,
            "bass_midi_events": bass_midi["midi_events"],
        }

        # 4. Compute processing report for quality gating. The first field is
        # a separation proxy unless a future backend supplies a reference-based
        # metric; it is intentionally exposed as a proxy by ProcessingReport.
        artifact_flags = self._detect_artifacts(guitar_stem, bass_stem)
        if not self._weights_loaded and "untrained_weights" not in artifact_flags:
            artifact_flags.append("untrained_weights")
        total_notes = len(guitar_midi["midi_events"]) + len(bass_midi["midi_events"])
        avg_confidence = (
            (
                guitar_midi["summary"]["avg_confidence"] * len(guitar_midi["midi_events"])
                + bass_midi["summary"]["avg_confidence"] * len(bass_midi["midi_events"])
            )
            / total_notes
            if total_notes
            else 0.0
        )
        report = ProcessingReport(
            separation_proxy=float(sep_metrics[:, 0].mean().item()),
            phase_coherence=float(sep_metrics[:, 1].mean().item()),
            avg_confidence=avg_confidence,
            artifact_flags=artifact_flags,
            low_confidence_notes=(
                guitar_midi["summary"]["low_confidence_count"]
                + bass_midi["summary"]["low_confidence_count"]
            ),
            model_weights_loaded=self._weights_loaded,
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
                pred_onsets=guitar_onset,
                target_onsets=self._prepare_onset_target(target_onsets, guitar_onset),
                pred_pitch_logits=guitar_pitch,
                target_pitches=self._prepare_pitch_target(target_pitch, guitar_pitch),
            )
            outputs["loss"] = loss
            outputs["loss_dict"] = loss_dict

        return outputs

    def _run_separator(self, audio: torch.Tensor):
        """Normalize native and pluggable separator contracts."""
        if hasattr(self.separator, "separate"):
            result = self.separator.separate(audio)
            return (
                result["guitar"],
                result["bass"],
                result["residual"],
                None,
                result["metrics"],
            )
        guitar, bass, residual, state_cache, metrics = self.separator(audio)
        return guitar, bass, residual, state_cache, metrics

    def _run_transcriber(self, stem: torch.Tensor, instrument: str) -> dict:
        """Normalize native and backend transcriber outputs to a dict."""
        if hasattr(self.transcriber, "set_instrument"):
            self.transcriber.set_instrument(instrument)
        if hasattr(self.transcriber, "transcribe"):
            result = self.transcriber.transcribe(stem)
        else:
            result = self.transcriber(stem)
        if isinstance(result, dict):
            return result
        onset, pitch, velocity, expression, confidence = result
        return {
            "onset_logits": onset,
            "pitch_logits": pitch,
            "velocity": velocity,
            "expression": expression,
            "confidence": confidence,
        }

    def _transcription_to_events(self, transcription: dict, batch_index: int = 0) -> list[dict]:
        """Prefer events supplied by a backend, otherwise decode model logits."""
        backend_events = transcription.get("events")
        if backend_events is not None:
            if not backend_events or batch_index >= len(backend_events):
                return []
            return [dict(event) for event in backend_events[batch_index]]
        return self._logits_to_midi(
            transcription["onset_logits"],
            transcription["pitch_logits"],
            transcription["velocity"],
            transcription["confidence"],
            transcription.get("expression"),
            batch_index=batch_index,
        )

    @staticmethod
    def _frame_energy(stem: torch.Tensor, n_frames: int) -> torch.Tensor:
        """Return a frame-aligned mean-square energy envelope."""
        if n_frames <= 0:
            return stem.new_zeros((stem.shape[0], 0))
        energy = torch.mean(stem**2, dim=1, keepdim=True)
        return torch.nn.functional.adaptive_avg_pool1d(energy, n_frames).squeeze(1)

    @staticmethod
    def _prepare_pitch_target(
        target_pitch: torch.Tensor | None, pitch_logits: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Normalize one-hot/data-loader pitch targets to the prediction grid."""
        if target_pitch is None or pitch_logits is None:
            return None
        target = target_pitch.float()
        if target.dim() == 4:
            target = target.reshape(target.shape[0], -1, target.shape[-1])
        elif target.dim() == 2:
            target = target.unsqueeze(-1)
        n_pred = pitch_logits.shape[1]
        if target.shape[1] > n_pred:
            target = target[:, :n_pred]
        elif target.shape[1] < n_pred:
            target = torch.nn.functional.pad(target, (0, 0, 0, n_pred - target.shape[1]))
        return target

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

    def _logits_to_midi(
        self,
        onset_logits,
        pitch_logits,
        velocity,
        confidence,
        expression=None,
        batch_index: int = 0,
    ):
        """Convert one batch of network outputs to a flat MIDI event list."""
        onsets = (
            torch.sigmoid(onset_logits) > self.cfg["transcriber"].get("onset_threshold", 0.5)
        ).squeeze(-1)
        pitches = torch.argmax(pitch_logits, dim=-1)
        vel_values = (velocity * 127).long().squeeze(-1)
        frame_hits = torch.nonzero(onsets[batch_index], as_tuple=False).flatten()

        events = []
        for frame in frame_hits.tolist():
            event = {
                "note": int(pitches[batch_index, frame].item()),
                "onset_frame": int(frame),
                "velocity": int(vel_values[batch_index, frame].item()),
                "confidence": float(confidence[batch_index, frame].item()),
            }
            if expression is not None:
                values = expression[batch_index, frame].detach().cpu().flatten().tolist()
                names = self.cfg["transcriber"].get(
                    "expression_heads", ["bend", "vibrato", "slide"]
                )
                event["expression"] = {
                    name: max(0.0, min(1.0, float(values[i])))
                    for i, name in enumerate(names)
                    if i < len(values)
                }
            events.append(event)
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
        self._ensure_inference_ready()
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
            "midi": {
                "guitar": outputs["midi_metadata"]["guitar"],
                "bass": outputs["midi_metadata"]["bass"],
            },
            "midi_guitar": outputs["midi_metadata"]["guitar"],
            "midi_bass": outputs["midi_metadata"]["bass"],
            "report": outputs["processing_report"],
            "routing": outputs["routing_decision"],
            "sample_rate": self.cfg["audio"]["sample_rate"],
        }

    def process_audio_streaming(self, audio_path: str, chunk_seconds: float = 2.0) -> dict:
        """Process an audio file in independently inferred chunks.

        Chunks restart model state because native SSM state passing is not
        implemented. Waveforms are nevertheless recombined with a normalized
        crossfade overlap-add, and local note events are translated to global
        frame positions. This is bounded-memory at the model stage but still
        loads the input and output accumulators in memory.
        """
        self._ensure_inference_ready()
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive")

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

        chunk_len = max(1, int(chunk_seconds * sr))
        overlap_ratio = float(self.cfg["audio"].get("overlap_ratio", 0.5))
        overlap_ratio = min(0.95, max(0.0, overlap_ratio))
        overlap_len = int(chunk_len * overlap_ratio)
        hop_len = chunk_len - overlap_len
        total_len = int(audio_t.shape[-1])

        guitar_sum = torch.zeros_like(audio_t)
        bass_sum = torch.zeros_like(audio_t)
        residual_sum = torch.zeros_like(audio_t)
        weight_sum = torch.zeros_like(audio_t)
        guitar_events: list[dict] = []
        bass_events: list[dict] = []
        metric_rows = []
        n_chunks = 0

        # A half-weight edge crossfade avoids discontinuities while retaining
        # non-zero coverage at the beginning/end of the full signal.
        window = torch.ones(chunk_len, dtype=audio_t.dtype)
        if overlap_len > 0:
            fade = torch.linspace(0.5, 1.0, overlap_len, dtype=audio_t.dtype)
            window[:overlap_len] = fade
            window[-overlap_len:] = torch.flip(fade, dims=[0])
        window = window.view(1, 1, -1)

        def _add_global_events(target: list[dict], local_events: list[dict], offset_frames: int):
            for event in local_events:
                copied = dict(event)
                copied["onset_frame"] = int(copied["onset_frame"]) + offset_frames
                target.append(copied)

        self.eval()
        with torch.inference_mode():
            pos = 0
            while pos < total_len:
                end = min(pos + chunk_len, total_len)
                valid_len = end - pos
                chunk = audio_t[:, :, pos:end]
                if valid_len < chunk_len:
                    chunk = torch.nn.functional.pad(chunk, (0, chunk_len - valid_len))

                guitar, bass, residual, _state_cache, sep_metrics = self._run_separator(chunk)
                metric_rows.append(sep_metrics[0].detach())
                weights = window[:, :, :valid_len]
                guitar_sum[:, :, pos:end] += guitar[:, :, :valid_len] * weights
                bass_sum[:, :, pos:end] += bass[:, :, :valid_len] * weights
                residual_sum[:, :, pos:end] += residual[:, :, :valid_len] * weights
                weight_sum[:, :, pos:end] += weights

                guitar_local = self._run_transcriber(guitar[:, :, :valid_len], "guitar")
                bass_local = self._run_transcriber(bass[:, :, :valid_len], "bass")
                offset_frames = round(pos * sr / self.cfg["audio"]["hop_length"])
                _add_global_events(
                    guitar_events,
                    self._transcription_to_events(guitar_local),
                    offset_frames,
                )
                _add_global_events(
                    bass_events,
                    self._transcription_to_events(bass_local),
                    offset_frames,
                )

                pos += hop_len
                n_chunks += 1

        weight_sum = weight_sum.clamp_min(1e-8)
        full_guitar = guitar_sum / weight_sum
        full_bass = bass_sum / weight_sum
        full_residual = residual_sum / weight_sum

        def _dedupe(events: list[dict]) -> list[dict]:
            unique: dict[tuple[int, int, int], dict] = {}
            for event in events:
                key = (
                    int(event["onset_frame"]),
                    int(event["note"]),
                    int(event.get("channel", 0)),
                )
                previous = unique.get(key)
                if previous is None or float(event.get("confidence", 0.0)) > float(
                    previous.get("confidence", 0.0)
                ):
                    unique[key] = event
            return sorted(unique.values(), key=lambda e: (e["onset_frame"], e["note"]))

        guitar_events = _dedupe(guitar_events)
        bass_events = _dedupe(bass_events)
        max_frame = max(
            [int(e["onset_frame"]) for e in guitar_events + bass_events] or [0]
        )
        n_frames = max_frame + 1 if guitar_events or bass_events else 0
        guitar_energy = self._frame_energy(full_guitar, n_frames)
        bass_energy = self._frame_energy(full_bass, n_frames)
        guitar_midi = self.confidence_injector.inject_midi_metadata(
            guitar_events, torch.tensor(
                [float(e.get("confidence", 0.0)) for e in guitar_events]
            ), guitar_energy[0]
        )
        bass_midi = self.confidence_injector.inject_midi_metadata(
            bass_events, torch.tensor(
                [float(e.get("confidence", 0.0)) for e in bass_events]
            ), bass_energy[0]
        )

        if metric_rows:
            final_metrics = torch.stack(metric_rows).mean(dim=0)
        else:
            final_metrics = torch.zeros(2)
        stream_artifacts = self._detect_artifacts(full_guitar, full_bass)
        if not self._weights_loaded and "untrained_weights" not in stream_artifacts:
            stream_artifacts.append("untrained_weights")
        stream_note_count = len(guitar_midi["midi_events"]) + len(bass_midi["midi_events"])
        stream_avg_confidence = (
            (
                guitar_midi["summary"]["avg_confidence"] * len(guitar_midi["midi_events"])
                + bass_midi["summary"]["avg_confidence"] * len(bass_midi["midi_events"])
            )
            / stream_note_count
            if stream_note_count
            else 0.0
        )
        report = ProcessingReport(
            separation_proxy=float(final_metrics[0].item()),
            phase_coherence=float(final_metrics[1].item()),
            avg_confidence=stream_avg_confidence,
            artifact_flags=stream_artifacts,
            low_confidence_notes=(
                guitar_midi["summary"]["low_confidence_count"]
                + bass_midi["summary"]["low_confidence_count"]
            ),
            model_weights_loaded=self._weights_loaded,
            config=self.cfg,
        )

        return {
            "stems": {
                "guitar": full_guitar.squeeze().cpu().numpy(),
                "bass": full_bass.squeeze().cpu().numpy(),
                "residual": full_residual.squeeze().cpu().numpy(),
            },
            "midi": {"guitar": guitar_midi, "bass": bass_midi},
            "midi_guitar": guitar_midi,
            "midi_bass": bass_midi,
            "report": report,
            "routing": route_by_quality(report),
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
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Set MODEL_CHECKPOINT_PATH "
            "to a trained checkpoint file."
        )

    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = StemMidiModel(config=config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            # A bare state_dict is itself a mapping of parameter names to
            # tensors; wrapped checkpoints use one of the two keys above.
            state_dict = ckpt
    else:
        state_dict = ckpt
    if not isinstance(state_dict, dict):
        raise ValueError(
            "Checkpoint must contain a state_dict/model_state_dict mapping or be a bare state_dict"
        )
    model.load_state_dict(state_dict)
    model.mark_weights_loaded()
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
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a trained checkpoint (.pt); required for file inference",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    checkpoint = args.checkpoint or os.environ.get("MODEL_CHECKPOINT_PATH") or config.get(
        "restore_from_path"
    )
    if checkpoint:
        model = load_from_checkpoint(checkpoint, args.config)
    elif (config.get("backends") or {}).get("weights_source") == "pretrained_dependencies":
        model = StemMidiModel(config)
        if not model.weights_loaded:
            parser.error(
                "pretrained backends did not load weights; install their dependencies "
                "or pass --checkpoint"
            )
    else:
        parser.error(
            "a trained checkpoint is required for file inference; pass --checkpoint "
            "or set MODEL_CHECKPOINT_PATH"
        )

    # Process audio
    result = model.process_audio_file(args.audio)

    # Print summary
    print("Processing complete!")
    print(f"Quality: {result['report'].quality_tier.value}")
    print(
        f"Centroid separation proxy (not SI-SDR): "
        f"{result['report'].separation_proxy:.1f}"
    )
    print(f"Avg Confidence: {result['report'].avg_confidence:.0%}")
    print(f"Routing: {result['routing']['action']}")


if __name__ == "__main__":
    cli()
