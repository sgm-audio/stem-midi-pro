#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
StemMidiModel — plain PyTorch orchestrator for Stem+MIDI Pro
(separation → transcription → confidence injection → quality routing).

Backend selection (H-9): when ``cfg["backends"]`` names a separator and/or
transcriber, the pluggable backends from ``models/backends/factory.py`` are
built and used (e.g. ``demucs``/``basic_pitch``). Otherwise the built-in
Mamba modules run (default). Backend events that already carry
``duration_frames`` are preferred over the frame-logit path.

Both stems are transcribed (H-8): guitar and bass MIDI are generated from
their own stems, not one copied to both.
"""

import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# CPU inference tuning (PERF-7.4 / PERF-7.9). Override thread count via OMP_NUM_THREADS.
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)))
torch.set_float32_matmul_precision("high")  # allows TF32/AVX-512 matmul where available

# Import our custom components (after torch thread config above — intentional E402)
from models._mamba_compat import MAMBA_BACKEND  # noqa: E402
from models.confidence_injector import ConfidenceInjector  # noqa: E402
from models.losses import PerceptualAudioLoss  # noqa: E402
from models.mamba_separator import MambaSeparator  # noqa: E402
from models.mamba_transcriber import MambaTranscriber  # noqa: E402
from utils.quality_gates import ProcessingReport, route_by_quality  # noqa: E402

logger = logging.getLogger(__name__)

# Default loud input cap (C-1 / C-6): with the chunked scan the separator no
# longer needs O(L·d_inner·N) memory, but STFT-domain buffers and stems still
# scale with length. Override via cfg["audio"]["max_input_seconds"].
DEFAULT_MAX_INPUT_SECONDS = 600


class StemMidiModel(nn.Module):
    """
    Orchestrates separation, transcription, confidence injection, and quality routing.

    Input:  audio (B, C=1, T); optional training targets
            target_guitar / target_bass (B, C, T),
            target_onsets (B, T, 1), target_pitch (B, T) / one-hot (B, T, V) /
            multi-hot (B, T, V) for chords,
            target_velocity (B, T, 1) normalized [0, 1]
    Output: Dict with guitar_stem, bass_stem, midi_metadata
            ({"guitar": …, "bass": …, "summary": …}), processing_report,
            routing_decision.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config

        # Pluggable backends (H-9): only used when explicitly configured.
        self.backends_cfg = config.get("backends") or {}
        self._use_backends = bool(
            self.backends_cfg.get("separator") or self.backends_cfg.get("transcriber")
        )

        # Initialize core components
        if self._use_backends:
            from models.backends.factory import build_separator, build_transcriber

            self.separator_backend = build_separator(config)
            self.transcriber_backend = build_transcriber(config)
            self.backend_name = (
                f"separator={self.backends_cfg.get('separator', 'mamba')},"
                f"transcriber={self.backends_cfg.get('transcriber', 'mamba')}"
            )
            logger.info("StemMidiModel using pluggable backends: %s", self.backend_name)
        else:
            self.separator = MambaSeparator(self.cfg)
            self.transcriber = MambaTranscriber(self.cfg)
            self.backend_name = f"mamba ({MAMBA_BACKEND})"
            logger.info("StemMidiModel using built-in Mamba modules (%s)", MAMBA_BACKEND)

        self.confidence_injector = ConfidenceInjector(
            onset_threshold=self.cfg["transcriber"].get("onset_threshold", 0.5),
            min_confidence=self.cfg["quality_gates"]["min_confidence"],
        )
        self.loss_fn = PerceptualAudioLoss(self.cfg)

    # ------------------------------------------------------------------
    # Component wrappers (unify the Mamba-module and pluggable-backend paths)
    # ------------------------------------------------------------------

    def _separate(self, audio: torch.Tensor):
        """Returns (guitar_stem, bass_stem, metrics (B, 2))."""
        if self._use_backends:
            out = self.separator_backend.separate(audio)
            return out["guitar"], out["bass"], out["metrics"]
        guitar, bass, _residual, _state_cache, metrics = self.separator(audio)
        return guitar, bass, metrics

    def _transcribe(self, stem: torch.Tensor) -> dict:
        """Returns dict with onset_logits / pitch_logits / velocity /
        confidence (all tensors) and optional per-batch ``events`` (backend)."""
        if self._use_backends:
            out = self.transcriber_backend.transcribe(stem)
            return {
                "onset_logits": out.get("onset_logits"),
                "pitch_logits": out.get("pitch_logits"),
                "velocity": out.get("velocity"),
                "confidence": out.get("confidence"),
                "events": out.get("events"),
            }
        onset, pitch, velocity, _expression, confidence = self.transcriber(stem)
        return {
            "onset_logits": onset,
            "pitch_logits": pitch,
            "velocity": velocity,
            "confidence": confidence,
            "events": None,
        }

    @staticmethod
    def _ensure_finite(tr: dict, context: str) -> None:
        """Fail loudly on NaN/Inf model output instead of corrupting MIDI/report."""
        for key, tensor in tr.items():
            if tensor is None or isinstance(tensor, list):
                continue
            if not torch.isfinite(tensor).all():
                raise RuntimeError(
                    f"{context}: model output '{key}' contains NaN/Inf; refusing to "
                    "emit corrupt stems/MIDI. Check input audio and model weights."
                )

    def _frame_energy(self, stem: torch.Tensor) -> torch.Tensor:
        """Sample-level RMS pooled onto the STFT frame grid (B, ceil(T/hop)).

        The confidence injector indexes ``stem_energy`` by ``onset_frame``
        (an STFT frame index), so the energy must live on the same grid (M-4).
        """
        hop = int(self.cfg["audio"]["hop_length"])
        energy = torch.mean(stem**2, dim=1)  # (B, T)
        frames = F.avg_pool1d(energy.unsqueeze(1), kernel_size=hop, stride=hop, ceil_mode=True)
        return frames.squeeze(1)  # (B, T_f)

    def forward(
        self,
        audio: torch.Tensor,
        target_guitar: torch.Tensor | None = None,
        target_bass: torch.Tensor | None = None,
        target_onsets: torch.Tensor | None = None,
        target_pitch: torch.Tensor | None = None,
        target_velocity: torch.Tensor | None = None,
    ) -> dict:
        """
        Full forward pass: audio → stems → MIDI + confidence + quality routing.
        Pure inference calls (eval mode, no targets) run under
        ``torch.inference_mode()`` (PERF-7.3).
        """
        if not self.training and target_guitar is None:
            with torch.inference_mode():
                return self._forward_impl(
                    audio, target_guitar, target_bass, target_onsets, target_pitch, target_velocity
                )
        return self._forward_impl(
            audio, target_guitar, target_bass, target_onsets, target_pitch, target_velocity
        )

    def _forward_impl(
        self,
        audio: torch.Tensor,
        target_guitar: torch.Tensor | None = None,
        target_bass: torch.Tensor | None = None,
        target_onsets: torch.Tensor | None = None,
        target_pitch: torch.Tensor | None = None,
        target_velocity: torch.Tensor | None = None,
    ) -> dict:
        """Implementation of forward(); see forward() for the public contract."""
        _B, _C, _T = audio.shape

        # 1. Separation (no state cache — real SSM state passing is C-2.5 / ARCH-11.8.1)
        guitar_stem, bass_stem, sep_metrics = self._separate(audio)

        # Transcription per stem (H-8: bass gets its own transcription)
        guitar_tr = self._transcribe(guitar_stem)
        bass_tr = self._transcribe(bass_stem)
        self._ensure_finite(guitar_tr, "guitar")
        self._ensure_finite(bass_tr, "bass")

        guitar_meta, guitar_conf = self._enhance(guitar_tr, guitar_stem)
        bass_meta, bass_conf = self._enhance(bass_tr, bass_stem)

        avg_confidence = torch.stack([guitar_conf.mean(), bass_conf.mean()]).mean().item()

        midi_metadata = {
            "guitar": guitar_meta,
            "bass": bass_meta,
            "summary": {
                "avg_confidence": avg_confidence,
                "low_confidence_count": (
                    guitar_meta["summary"]["low_confidence_count"]
                    + bass_meta["summary"]["low_confidence_count"]
                ),
                "alignment_warnings": (
                    guitar_meta["summary"]["alignment_warnings"]
                    + bass_meta["summary"]["alignment_warnings"]
                ),
            },
        }

        # 3. Processing report for quality gating (batch element 0 — B=1 is the
        #    supported inference shape; M-12)
        report = ProcessingReport(
            centroid_separation_db=sep_metrics[0, 0].item(),
            phase_coherence=sep_metrics[0, 1].item(),
            avg_confidence=avg_confidence,
            artifact_flags=self._detect_artifacts(guitar_stem, bass_stem),
            low_confidence_notes=midi_metadata["summary"]["low_confidence_count"],
            config=self.cfg,
            backend=self.backend_name,
        )

        routing = route_by_quality(report)

        outputs = {
            "guitar_stem": guitar_stem,
            "bass_stem": bass_stem,
            "midi_metadata": midi_metadata,  # {"guitar": …, "bass": …, "summary": …}
            "processing_report": report,
            "routing_decision": routing,
        }

        # 4. Compute loss if targets provided (training mode). Pitch and
        #    velocity targets are wired through (C-3); the expression and
        #    confidence heads remain untrained — no dataset emits those
        #    targets yet (see TODO ARCH-11.4.4 / ARCH-11.6).
        if target_guitar is not None and target_bass is not None:
            loss, loss_dict = self.loss_fn(
                pred_stems=[guitar_stem, bass_stem],
                target_stems=[target_guitar, target_bass],
                pred_onsets=guitar_tr["onset_logits"],
                target_onsets=self._prepare_onset_target(target_onsets, guitar_tr["onset_logits"]),
                pred_pitch_logits=guitar_tr["pitch_logits"],
                target_pitches=self._prepare_pitch_target(target_pitch, guitar_tr["pitch_logits"]),
                pred_velocities=guitar_tr["velocity"],
                target_velocities=self._prepare_velocity_target(
                    target_velocity, guitar_tr["velocity"]
                ),
            )
            outputs["loss"] = loss
            outputs["loss_dict"] = loss_dict

        return outputs

    def _enhance(self, tr: dict, stem: torch.Tensor) -> tuple[dict, torch.Tensor]:
        """Convert a transcription dict into injector-enhanced MIDI metadata."""
        confidence = tr["confidence"]
        events = None
        if tr.get("events"):
            events = tr["events"][0]  # B=1 in production paths
        elif tr["onset_logits"] is not None:
            events = self._logits_to_midi(
                tr["onset_logits"], tr["pitch_logits"], tr["velocity"], confidence
            )
        enhanced = self.confidence_injector.inject_midi_metadata(
            events or [],
            confidence.reshape(-1),
            self._frame_energy(stem).reshape(-1),
        )
        return enhanced, confidence

    # ------------------------------------------------------------------
    # Target preparation (C-3)
    # ------------------------------------------------------------------

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

    @staticmethod
    def _prepare_pitch_target(
        target_pitch: torch.Tensor | None, pitch_logits: torch.Tensor
    ) -> torch.Tensor | None:
        """Coerce pitch targets for the pitch loss (CHORD-AWARE).

        Returns one of two formats, chosen per batch:

        - **(B, T) class indices** for ``pitch_ce_loss_fn`` — when every
          labeled frame carries exactly one pitch (one-hot rows collapse to
          argmax; all-zero rows become ``ignore_index`` = -100; class-index
          tensors pass through). Out-of-range values are ignored, not
          clamped (M-1).
        - **(B, T, V) multi-hot float** for ``pitch_bce_loss_fn`` — when ANY
          frame labels more than one pitch (a chord). The previous behavior
          argmax-collapsed such rows, silently destroying all but one note
          of every chord in the training data even though
          ``utils.midi_targets.midi_to_targets`` writes one-hot entries per
          note_on (i.e. real chord frames arrive multi-hot).
        """
        if target_pitch is None:
            return None
        vocab = pitch_logits.shape[-1]
        n_pred = pitch_logits.shape[1]
        t = target_pitch
        if t.dim() == 4 and t.shape[1] == 1:
            t = t.squeeze(1)
        if t.dim() == 3 and t.shape[-1] == vocab:
            row_sum = t.sum(dim=-1)
            if (row_sum > 1).any():
                # Polyphonic (chord) labels present: keep the multi-hot grid.
                t = (t > 0).float()
                if t.shape[1] > n_pred:
                    t = t[:, :n_pred]
                elif t.shape[1] < n_pred:
                    t = F.pad(t, (0, 0, 0, n_pred - t.shape[1]), value=0.0)
                return t
            valid = row_sum > 0
            t = t.argmax(dim=-1).masked_fill(~valid, -100)
        elif t.dim() > 2:
            t = t.reshape(t.shape[0], -1)
        t = t.long()
        if t.shape[1] > n_pred:
            t = t[:, :n_pred]
        elif t.shape[1] < n_pred:
            t = F.pad(t, (0, n_pred - t.shape[1]), value=-100)
        return t

    @staticmethod
    def _prepare_velocity_target(
        target_velocity: torch.Tensor | None, velocity_pred: torch.Tensor
    ) -> torch.Tensor | None:
        """Coerce velocity targets onto the prediction grid (B, T, 1).

        Velocity targets are only defined at onset frames; padding with 0
        marks padded frames as "no target", which the masked velocity MAE
        excludes (C-3).
        """
        if target_velocity is None:
            return None
        v = target_velocity.float()
        if v.dim() == 4 and v.shape[1] == 1:
            v = v.squeeze(1)
        if v.dim() == 2:
            v = v.unsqueeze(-1)
        n_pred = velocity_pred.shape[1]
        if v.shape[1] > n_pred:
            v = v[:, :n_pred]
        elif v.shape[1] < n_pred:
            # pad the FRAME dim (-2), not the value dim (-1)
            v = F.pad(v, (0, 0, 0, n_pred - v.shape[1]), value=0.0)
        return v.reshape(velocity_pred.shape[0], -1, velocity_pred.shape[-1])

    # ------------------------------------------------------------------
    # Training/validation step contracts
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        """One training step for the plain-PyTorch loop in train.py (RF-3.1.9).

        Returns a dict with ``loss`` plus per-component values under
        ``loss_dict`` (no Lightning ``self.log`` — the caller logs).
        """
        return self._step_from_batch(batch)

    def validation_step(self, batch, batch_idx):
        """Validation step; same return contract as training_step."""
        return self._step_from_batch(batch)

    def _step_from_batch(self, batch) -> dict:
        outputs = self.forward(
            batch["audio"],
            target_guitar=batch["target_guitar"],
            target_bass=batch["target_bass"],
            target_onsets=batch["target_onsets"],
            target_pitch=batch["target_pitch"],
            target_velocity=batch.get("target_velocity"),
        )
        return {"loss": outputs["loss"], "loss_dict": outputs["loss_dict"]}

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def _logits_to_midi(self, onset_logits, pitch_logits, velocity, confidence):
        """Convert network outputs to a FLAT MIDI event list.

        Onset peak-picking (H-6): sigmoid probability is thresholded and an
        event fires at each RISING EDGE (first frame of an above-threshold
        run). A new onset can only fire after
        ``transcriber.onset_refractory_frames`` (default 4 ≈ 46 ms at 86 fps)
        consecutive below-threshold frames, so one physical onset whose
        activation plateaus for many frames — or wiggles across the
        threshold — emits ONE note_on, not several. Same-pitch duplicates
        are also dropped downstream, but without peak-picking a jittered
        argmax pitch mid-burst (e.g. 45,45,52,45) leaks a spurious extra
        note. Velocity/confidence are sampled at the crossing frame.

        Pitch decoding is CHORD-AWARE: every pitch whose sigmoid probability
        exceeds ``transcriber.pitch_threshold`` (default 0.5) at the accepted
        onset frame emits its own note_on, so polyphonic frames produce
        chords. If no pitch crosses the threshold at a fired onset, the
        argmax pitch is emitted as a fallback — a fired onset always yields
        at least one note (important for single-label-trained models whose
        runner-up logits stay low). ``pitch_threshold: null`` forces legacy
        single-argmax decoding.

        Assumes B=1 for inference; batch >1 still returns all batches' events
        with onset frames as-is. ``onset_refractory_frames: 0`` disables the
        suppression entirely (every above-threshold frame fires — the
        pre-H-6 behavior).
        """
        onset_prob = torch.sigmoid(onset_logits).squeeze(-1)  # (B, T)
        refractory = int(self.cfg["transcriber"].get("onset_refractory_frames", 4))
        candidates = torch.nonzero(
            onset_prob > self.cfg["transcriber"].get("onset_threshold", 0.5)
        )  # (N, 2): [batch_idx, frame], sorted by batch then frame
        pitches = torch.argmax(pitch_logits, dim=-1)
        vel_values = (velocity * 127).long().squeeze(-1)

        pitch_threshold = self.cfg["transcriber"].get("pitch_threshold", 0.5)
        chord_probs = None
        if pitch_threshold is not None:
            if not 0.0 < float(pitch_threshold) < 1.0:
                raise ValueError(
                    f"transcriber.pitch_threshold must be in (0, 1) or null; "
                    f"got {pitch_threshold!r}"
                )
            chord_probs = torch.sigmoid(pitch_logits)  # (B, T, V)

        events = []
        last_above: dict[int, int] = {}  # batch -> last candidate frame seen
        for b, t in candidates.tolist():
            if b not in last_above:
                accepted = True  # first above-threshold frame of the batch
            else:
                # below-threshold frames between this candidate and the
                # previous one; a rising edge needs >= refractory of them
                below_gap = t - last_above[b] - 1
                accepted = below_gap >= refractory
            last_above[b] = t
            if not accepted:
                continue

            if chord_probs is not None:
                frame_notes = (chord_probs[b, t] > pitch_threshold).nonzero().flatten().tolist()
            else:
                frame_notes = []
            if not frame_notes:
                # No pitch crossed the threshold (or multi-label decoding is
                # disabled): fall back to the argmax pitch so a fired onset
                # never produces an empty event set.
                frame_notes = [int(pitches[b, t].item())]
            for note in frame_notes:
                events.append(
                    {
                        "note": int(note),
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

    # ------------------------------------------------------------------
    # File-level inference
    # ------------------------------------------------------------------

    def _load_audio(self, path: str) -> tuple[np.ndarray, int]:
        """Load audio via soundfile with a librosa fallback (M-8).

        Both failures are reported together so a corrupt file produces one
        actionable error instead of a misleading secondary traceback.
        Raises ValueError on non-finite samples.
        """
        primary_error: Exception | None = None
        if not os.path.exists(path):
            # A missing file must surface as FileNotFoundError — not as a
            # confusing librosa secondary failure (M-8).
            raise FileNotFoundError(f"Audio file not found: {path!r}")
        try:
            import soundfile as sf

            audio, sr = sf.read(path)
        except Exception as e:
            primary_error = e
            try:
                import librosa

                audio, sr = librosa.load(path, sr=None, mono=True)
            except Exception as lib_err:
                raise ValueError(
                    f"Failed to load audio {path!r}: soundfile error: {primary_error}; "
                    f"librosa fallback error: {lib_err}"
                ) from lib_err

        if audio.ndim > 1:
            audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
        audio = audio.astype(np.float32)

        max_seconds = float(
            (self.cfg.get("audio") or {}).get("max_input_seconds", DEFAULT_MAX_INPUT_SECONDS)
        )
        duration = len(audio) / sr if sr > 0 else float("inf")
        if duration > max_seconds:
            raise ValueError(
                f"Input duration {duration:.1f}s exceeds max_input_seconds="
                f"{max_seconds:g}s (memory guard, see C-1/C-6). Split the file or "
                "raise the limit explicitly in the config."
            )
        if not np.isfinite(audio).all():
            raise ValueError(f"Audio {path!r} contains NaN/Inf samples; refusing to process.")
        return audio, sr

    def _resample_to_target(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Anti-aliased resample via librosa/soxr (H-7).

        torch.nn.functional.interpolate(mode="linear") applies no anti-alias
        filter and folds out-of-band content into the model's input band.
        """
        target_sr = int(self.cfg["audio"]["sample_rate"])
        if sr == target_sr:
            return audio
        import librosa

        return librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)

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

        audio, sr = self._load_audio(audio_path)
        audio = self._resample_to_target(audio, sr)
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)  # (1, 1, T)

        # Run inference
        with torch.inference_mode():
            outputs = self.forward(audio_t)

        # Format for user delivery (H-8: per-stem MIDI from per-stem stems)
        return {
            "stems": {
                "guitar": outputs["guitar_stem"].squeeze().cpu().numpy(),
                "bass": outputs["bass_stem"].squeeze().cpu().numpy(),
            },
            "midi": outputs["midi_metadata"],
            "midi_guitar": outputs["midi_metadata"]["guitar"],
            "midi_bass": outputs["midi_metadata"]["bass"],
            "report": outputs["processing_report"],
            "routing": outputs["routing_decision"],
            "sample_rate": self.cfg["audio"]["sample_rate"],
        }

    def process_audio_streaming(self, audio_path: str, chunk_seconds: float = 2.0) -> dict:
        """
        Chunked inference: process audio in chunks and recombine.

        Recombination is a weighted overlap-add with complementary linear
        crossfades over the overlap region (H-4). Chunks are separated
        independently — SSM state does NOT carry between chunks (C-2.5 /
        ARCH-11.8.1, planned); the crossfade hides the state-reset
        discontinuity but boundary artifacts in the STEM AUDIO remain
        possible. MIDI cannot duplicate boundary notes (ARCH-11.8.2
        resolved by design): transcription runs ONCE on the recombined
        full timeline (C-2), so per-chunk MIDI events do not exist to
        deduplicate.

        Transcription runs once on the recombined stems, so pitch/velocity/
        confidence are real model outputs on the full timeline (C-2) and the
        reported metrics are the mean of the per-chunk separator metrics (no
        more hardcoded zeros forcing the COMPLEX tier).

        Args:
            audio_path: Path to input audio file
            chunk_seconds: Duration of each chunk in seconds (default 2.0)

        Returns:
            Same format as process_audio_file (aggregated outputs)
        """
        audio, sr = self._load_audio(audio_path)
        audio = self._resample_to_target(audio, sr)
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
        sr = int(self.cfg["audio"]["sample_rate"])

        total_len = audio_t.shape[-1]
        chunk_len = max(1, int(chunk_seconds * sr))
        overlap_len = int(chunk_len * self.cfg["audio"].get("overlap_ratio", 0.5))
        hop_len = chunk_len - overlap_len
        if hop_len <= 0:
            raise ValueError(
                f"overlap_ratio {self.cfg['audio'].get('overlap_ratio')} leaves no hop "
                f"(chunk {chunk_len}, overlap {overlap_len}); reduce overlap_ratio."
            )

        out_guitar = torch.zeros(1, 1, total_len)
        out_bass = torch.zeros(1, 1, total_len)
        weight_sum = torch.zeros(1, 1, total_len)
        metrics_acc: list[torch.Tensor] = []
        n_chunks = 0

        self.eval()
        with torch.inference_mode():
            pos = 0
            while pos < total_len:
                end = min(pos + chunk_len, total_len)
                chunk = audio_t[:, :, pos:end]
                valid_len = end - pos

                if valid_len < chunk_len:
                    chunk = F.pad(chunk, (0, chunk_len - valid_len))

                guitar, bass, sep_metrics = self._separate(chunk)

                # Overlap-add weights (H-4): complementary linear crossfades
                # over the shared overlap region; normalized by accumulated
                # weights so any overlap/edge combination sums to 1.
                w = torch.ones(1, 1, valid_len)
                if pos > 0 and overlap_len > 0:
                    fi = min(overlap_len, valid_len)
                    w[:, :, :fi] *= torch.arange(1, fi + 1, dtype=torch.float32) / overlap_len
                if end < total_len and overlap_len > 0:
                    fo = min(overlap_len, valid_len)
                    ramp = torch.arange(fo - 1, -1, -1, dtype=torch.float32) / overlap_len
                    w[:, :, valid_len - fo :] *= ramp

                out_guitar[:, :, pos:end] += guitar[:, :, :valid_len] * w
                out_bass[:, :, pos:end] += bass[:, :, :valid_len] * w
                weight_sum[:, :, pos:end] += w

                metrics_acc.append(sep_metrics.detach().float().mean(dim=0))
                pos += hop_len
                n_chunks += 1

        full_guitar = out_guitar / weight_sum.clamp(min=1e-8)
        full_bass = out_bass / weight_sum.clamp(min=1e-8)

        final_metrics = torch.stack(metrics_acc).mean(dim=0)  # (2,) real per-chunk mean

        # Transcribe the recombined stems once — real logits on the full
        # timeline (C-2), no per-chunk frame-grid drift.
        guitar_tr = self._transcribe(full_guitar)
        bass_tr = self._transcribe(full_bass)
        self._ensure_finite(guitar_tr, "guitar")
        self._ensure_finite(bass_tr, "bass")
        guitar_meta, guitar_conf = self._enhance(guitar_tr, full_guitar)
        bass_meta, bass_conf = self._enhance(bass_tr, full_bass)

        avg_confidence = torch.stack([guitar_conf.mean(), bass_conf.mean()]).mean().item()

        midi_metadata = {
            "guitar": guitar_meta,
            "bass": bass_meta,
            "summary": {
                "avg_confidence": avg_confidence,
                "low_confidence_count": (
                    guitar_meta["summary"]["low_confidence_count"]
                    + bass_meta["summary"]["low_confidence_count"]
                ),
                "alignment_warnings": (
                    guitar_meta["summary"]["alignment_warnings"]
                    + bass_meta["summary"]["alignment_warnings"]
                ),
            },
        }

        report = ProcessingReport(
            centroid_separation_db=final_metrics[0].item(),
            phase_coherence=final_metrics[1].item(),
            avg_confidence=avg_confidence,
            artifact_flags=self._detect_artifacts(full_guitar, full_bass),
            low_confidence_notes=midi_metadata["summary"]["low_confidence_count"],
            config=self.cfg,
            backend=self.backend_name,
        )

        routing = route_by_quality(report)

        return {
            "stems": {
                "guitar": full_guitar.squeeze().cpu().numpy(),
                "bass": full_bass.squeeze().cpu().numpy(),
            },
            "midi": midi_metadata,
            "report": report,
            "routing": routing,
            "sample_rate": sr,
            "chunks_processed": n_chunks,
        }


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
    print(
        f"Centroid separation (proxy, not SI-SDR): {result['report'].centroid_separation_db:.1f}dB"
    )
    print(f"Backend: {result['report'].backend}")
    print(f"Avg Confidence: {result['report'].avg_confidence:.0%}")
    print(f"Routing: {result['routing']['action']}")


if __name__ == "__main__":
    cli()
