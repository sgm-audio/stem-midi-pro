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
from audio_io import AudioValidationError  # noqa: E402
from models._mamba_compat import MAMBA_BACKEND  # noqa: E402
from models.backends.factory import build_separator, build_transcriber  # noqa: E402
from models.confidence_injector import ConfidenceInjector  # noqa: E402
from models.losses import PerceptualAudioLoss  # noqa: E402
from models.mamba_separator import MambaSeparator  # noqa: E402
from models.mamba_transcriber import MambaTranscriber  # noqa: E402
from utils.quality_gates import ProcessingReport, route_by_quality  # noqa: E402

logger = logging.getLogger(__name__)

# Provenance label carried on every note that has expression data (ARCH-11.4.4).
_EXPRESSION_SOURCE = "expression_head (untrained; ARCH-11.4.4)"

# Default loud input cap (C-1 / C-6): with the chunked scan the separator no
# longer needs O(L·d_inner·N) memory, but STFT-domain buffers and stems still
# scale with length. Override via cfg["audio"]["max_input_seconds"].
DEFAULT_MAX_INPUT_SECONDS = 600

# Input-level artifact thresholds (see _detect_artifacts).
_SILENCE_LEVEL = 1e-6  # peak abs sample: digital silence
_CLIP_LEVEL = 0.999  # peak abs sample: full-scale (a clipped signal pins here)
_DC_OFFSET_LEVEL = 1e-3  # |mean|: audible DC offset / missing high-pass


def _is_silent(audio: torch.Tensor, level: float = _SILENCE_LEVEL) -> bool:
    """True when a tensor carries no signal above the digital-silence floor."""
    return bool(audio.numel() == 0 or audio.abs().max().item() <= level)


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

        # A freshly constructed module has random parameters. Forward passes
        # remain available for training/unit tests, but file/stream inference
        # must not present those weights as production output. The checkpoint
        # loader marks a model as ready after state_dict validation.
        # This flag is set only by a validated checkpoint load or by a backend
        # that explicitly attests to its own pretrained weights; it is not
        # accepted as a user-editable config value.
        self._weights_loaded = False
        self._allow_untrained_inference = bool(config.get("allow_untrained_inference", False))

        # Pluggable backends (H-9): only used when explicitly configured.
        self.backends_cfg = config.get("backends") or {}
        self._use_backends = bool(
            self.backends_cfg.get("separator") or self.backends_cfg.get("transcriber")
        )

        if self._use_backends:
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

        # Pretrained third-party backends are safe without a project checkpoint
        # only when both constructors explicitly attest that their own weights
        # are available. Native Mamba/fake backends never make that claim.
        if self._use_backends:
            self._weights_loaded = bool(
                getattr(self.separator_backend, "weights_loaded", False)
            ) and bool(getattr(self.transcriber_backend, "weights_loaded", False))

        self.confidence_injector = ConfidenceInjector(
            onset_threshold=self.cfg["transcriber"].get("onset_threshold", 0.5),
            min_confidence=self.cfg["quality_gates"]["min_confidence"],
        )
        self.loss_fn = PerceptualAudioLoss(self.cfg)

    @property
    def mamba_backend(self) -> str:
        """Which Mamba implementation is actually running (H-1).

        ``pure-torch`` is the portable fallback (selective scan in plain torch);
        ``mamba-ssm`` is the CUDA-optimised package. They are not numerically
        identical, so the answer belongs in provenance.
        """
        return MAMBA_BACKEND

    @property
    def weights_loaded(self) -> bool:
        """Whether a validated checkpoint or backend-owned weights are loaded."""
        return self._weights_loaded

    def mark_weights_loaded(self) -> None:
        """Mark this instance safe for file inference after state_dict validation."""
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

    def _transcribe(self, stem: torch.Tensor, instrument: str | None = None) -> dict:
        """Transcribe one stem, optionally selecting its backend instrument."""
        if self._use_backends:
            if instrument is not None and hasattr(self.transcriber_backend, "set_instrument"):
                self.transcriber_backend.set_instrument(instrument)
            out = self.transcriber_backend.transcribe(stem)
            return {
                "onset_logits": out.get("onset_logits"),
                "pitch_logits": out.get("pitch_logits"),
                "velocity": out.get("velocity"),
                "expression": out.get("expression"),
                "confidence": out.get("confidence"),
                "events": out.get("events"),
            }
        onset, pitch, velocity, expression, confidence = self.transcriber(stem)
        return {
            "onset_logits": onset,
            "pitch_logits": pitch,
            "velocity": velocity,
            "expression": expression,
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
        # Validate before any compute: NaN/Inf or a sub-window clip must fail as
        # a client error (AudioValidationError -> HTTP 400), not as a 500 from
        # inside the separator or as `"centroid_separation_db": nan` in a report.
        self._validate_audio(audio, "<forward input>")

        # Digital silence is a valid upload (a muted channel) but not a valid
        # model input: the untrained Mamba stack divides by a zero frame RMS and
        # emits NaN, which _ensure_finite turns into a RuntimeError (HTTP 500).
        # The stem-level silence guard in _enhance runs after that point, so
        # short-circuit here instead.
        if not self.training and target_guitar is None and _is_silent(audio):
            midi_metadata, report, routing = self._silent_result(audio)
            return {
                "guitar_stem": torch.zeros_like(audio),
                "bass_stem": torch.zeros_like(audio),
                "midi_metadata": midi_metadata,
                "processing_report": report,
                "routing_decision": routing,
            }

        # 1. Separation (no state cache — real SSM state passing is C-2.5 / ARCH-11.8.1)
        guitar_stem, bass_stem, sep_metrics = self._separate(audio)

        # Transcription per stem (H-8: bass gets its own transcription)
        guitar_tr = self._transcribe(guitar_stem, "guitar")
        bass_tr = self._transcribe(bass_stem, "bass")
        self._ensure_finite(guitar_tr, "guitar")
        self._ensure_finite(bass_tr, "bass")

        guitar_meta, _ = self._enhance(guitar_tr, guitar_stem)
        bass_meta, _ = self._enhance(bass_tr, bass_stem)

        guitar_notes = len(guitar_meta["midi_events"])
        bass_notes = len(bass_meta["midi_events"])
        total_notes = guitar_notes + bass_notes
        avg_confidence = (
            (
                guitar_meta["summary"]["avg_confidence"] * guitar_notes
                + bass_meta["summary"]["avg_confidence"] * bass_notes
            )
            / total_notes
            if total_notes
            else 0.0
        )

        midi_metadata = {
            "guitar": guitar_meta,
            "bass": bass_meta,
            "summary": {
                "avg_confidence": avg_confidence,
                "confidence_encoding": "polyphonic_key_pressure",
                "low_confidence_count": (
                    guitar_meta["summary"]["low_confidence_count"]
                    + bass_meta["summary"]["low_confidence_count"]
                ),
                "alignment_warnings": (
                    guitar_meta["summary"]["alignment_warnings"]
                    + bass_meta["summary"]["alignment_warnings"]
                ),
                "total_notes": (
                    guitar_meta["summary"]["total_notes"] + bass_meta["summary"]["total_notes"]
                ),
            },
        }

        # 3. Processing report for quality gating (batch element 0 — B=1 is the
        #    supported inference shape; M-12). The public metric is a named
        #    heuristic proxy, not reference-based SI-SDR.
        artifact_flags = self._detect_artifacts(guitar_stem, bass_stem, source=audio)
        if not self._weights_loaded and "untrained_weights" not in artifact_flags:
            artifact_flags.append("untrained_weights")
        # REPORT-1: average over the batch (upstream reached the same fix with
        # `.mean()`). `nanmean` is deliberate: if one item's metric is
        # non-finite, `.mean()` poisons the whole report and hides which item was
        # bad, while `nanmean` reports the items that have a real measurement --
        # and `_detect_artifacts`' non_finite flag still surfaces the bad one.
        report = ProcessingReport(
            separation_proxy=float(torch.nanmean(sep_metrics[:, 0].float()).item()),
            phase_coherence=float(torch.nanmean(sep_metrics[:, 1].float()).item()),
            avg_confidence=avg_confidence,
            artifact_flags=artifact_flags,
            low_confidence_notes=midi_metadata["summary"]["low_confidence_count"],
            # GATE-3: the gate must be able to refuse an empty render.
            note_count=len(midi_metadata["guitar"].get("midi_events") or [])
            + len(midi_metadata["bass"].get("midi_events") or []),
            config=self.cfg,
            model_weights_loaded=self._weights_loaded,
            backend=self.backend_name,
            weights_loaded=self.weights_loaded,
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
        """Convert a transcription dict into injector-enhanced MIDI metadata.

        A silent stem is short-circuited to zero events. The onset head fires
        on whatever numerical noise survives separating digital silence, so
        transcribing silence produced 8-128 plausible-looking note-ons that a
        DAW imports as music (review §4). This lives here, not at the call
        sites, because it is the single choke point for *both* stems -- the
        bass part previously slipped past a guitar-only guard.
        """
        confidence = tr["confidence"]
        if self._stem_is_silent(stem):
            empty = self.confidence_injector.inject_midi_metadata(
                [],
                confidence.reshape(-1),
                self._frame_energy(stem).reshape(-1),
            )
            return empty, confidence
        events = None
        if tr.get("events"):
            events = tr["events"][0]  # B=1 in production paths
        elif tr["onset_logits"] is not None:
            events = self._logits_to_midi(
                tr["onset_logits"],
                tr["pitch_logits"],
                tr["velocity"],
                confidence,
                tr.get("expression"),
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

    def _logits_to_midi(self, onset_logits, pitch_logits, velocity, confidence, expression=None):
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

        Pitch decoding is CHORD-AWARE when ``transcriber.pitch_threshold``
        is set (production configs set 0.5): every pitch whose sigmoid
        probability exceeds it at the accepted onset frame emits its own
        note_on, so polyphonic frames produce chords. If no pitch crosses
        the threshold at a fired onset, the argmax pitch is emitted as a
        fallback — a fired onset always yields at least one note (important
        for single-label-trained models whose runner-up logits stay low).
        Unset/None (the default for bare config dicts) keeps legacy
        single-argmax decoding: untrained noise-centered logits split ~50/50
        around any fixed threshold, which would otherwise decode one noisy
        frame into dozens of phantom chord notes (caught by
        test_onset_refractory_caps_note_density).

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

        # BATCH-1: this function returns ONE flat list whose events carry the raw
        # per-item `onset_frame`. For B>1 that merges separate recordings into a
        # single unattributable MIDI file -- two songs' notes interleaved, with no
        # way to tell which note came from which -- and it did so silently.
        #
        # The guard sits here, not on `forward`, for two reasons:
        #  * this loop is the only place the merging happens;
        #  * it fires only when the batch actually produces events. A B>1 call
        #    that produces no onsets (e.g. a fixture probing the batched quality
        #    report) has nothing to merge, so refusing it would break a
        #    legitimate use without protecting anything.
        n_batch = int(candidates[:, 0].max().item()) + 1 if candidates.shape[0] else 0
        if n_batch > 1:
            raise ValueError(
                f"_logits_to_midi converts one recording at a time; got batch size "
                f"{n_batch}. Batched inference would merge every item's notes into a "
                "single MIDI file. Call the model once per recording."
            )

        # Chord (polyphony) decoding, added upstream while this branch was in
        # flight. Independent of the batch guard above.
        pitch_threshold = self.cfg["transcriber"].get("pitch_threshold")
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
                event = {
                    "note": int(note),
                    "onset_frame": t,
                    "velocity": vel_values[b, t].item(),
                    "confidence": confidence[b, t].item(),
                }
                if expression is not None:
                    values = expression[b, t].detach().cpu().flatten().tolist()
                    names = self.cfg["transcriber"].get(
                        "expression_heads", ["bend", "vibrato", "slide"]
                    )
                    # The head is sigmoid, so every channel is already 0..1 with
                    # 0.5 neutral (the writer's convention). The value is tagged
                    # with where it came from: an untrained head's output must
                    # never pass for learned phrasing.
                    event["expression"] = {
                        name: max(0.0, min(1.0, float(values[i])))
                        for i, name in enumerate(names)
                        if i < len(values)
                    }
                    event["expression"]["source"] = _EXPRESSION_SOURCE
                events.append(event)
        return events

    @staticmethod
    def _stem_is_silent(stem: torch.Tensor) -> bool:
        """True when one stem is digital silence or non-finite (review §4)."""
        if not torch.isfinite(stem).all():
            return True
        return bool(torch.max(stem.abs()) < 1e-6)

    @staticmethod
    def _stems_are_silent(guitar: torch.Tensor, bass: torch.Tensor) -> bool:
        """True when both stems are unusable (silent or non-finite).

        Shared by `_detect_artifacts` and by the MIDI suppression path so the
        report flag and the emitted events can never disagree (review §4).
        """
        return StemMidiModel._stem_is_silent(guitar) and StemMidiModel._stem_is_silent(bass)

    @staticmethod
    def _noise_fraction(stem: torch.Tensor) -> float:
        """Fraction of a stem's energy sitting in the top band of the spectrum.

        A proxy for "is this broadband hiss?" rather than "is this loud?".
        Silent or near-silent stems return 0.0 -- silence is reported by
        `_stems_are_silent`, and a quiet stem must not be labelled noisy.
        """
        energy = torch.mean(stem**2)
        if not bool(torch.isfinite(energy)) or float(energy) <= 1e-10:
            return 0.0
        # (B, C, T) -> per-example 1-D spectra
        flat = stem.reshape(-1, stem.shape[-1])
        spectrum = torch.fft.rfft(flat, dim=-1).abs() ** 2  # (N, F)
        total = spectrum.sum(dim=-1)
        if not bool((total > 0).all()):
            return 0.0
        top_band = spectrum[:, spectrum.shape[-1] // 4 :].sum(dim=-1)
        return float((top_band / total).max())

    def _detect_artifacts(
        self, guitar: torch.Tensor, bass: torch.Tensor, source: torch.Tensor | None = None
    ) -> list[str]:
        """Simple artifact detection for quality gating.

        An instance method only for ``self._input_flags`` / ``self._weights_loaded``
        access; the stem logic itself is class-qualified (``StemMidiModel._…``)
        so tests can still call it unbound with ``self=None``.

        Finiteness and silence are checked FIRST (review §4): without them an
        all-NaN render raised no flag, and because `NaN >= x` is False the
        gate fell through to DRAFT. An all-silent render also raised no flag
        and scored the maximum separation proxy.

        ``source`` is the ORIGINAL input waveform. Input-level damage is not
        visible in the stems: a hard-clipped DI is scaled down by the separator
        masks (peak 1.0 in, ~0.4 out), so the stem clipping test never fired for
        exactly the uploads that are clipped, and a DC offset in the upload is
        removed by the masks while still costing the model input headroom.
        """
        flags = []
        # Class-qualified (not ``self.``): tests call this unbound with
        # ``self=None`` to exercise the flag logic without a model.
        input_flags = StemMidiModel._input_flags(source) if source is not None else []

        if not (torch.isfinite(guitar).all() and torch.isfinite(bass).all()):
            return (
                ["non_finite", "silent_output", *input_flags]
                if not (guitar.abs().max() > 0 or bass.abs().max() > 0)
                else ["non_finite", *input_flags]
            )

        # Digital silence in both stems means nothing was separated (e.g. no
        # guitar in the mix). It must not reach STUDIO via the clamped proxy.
        if StemMidiModel._stems_are_silent(guitar, bass):
            return ["silent_output", *input_flags]

        # Check for clipping
        if guitar.abs().max() > 0.99 or bass.abs().max() > 0.99:
            flags.append("clipping")

        # Noise-floor estimate (ART-1). This used to be
        #   mean(stem**2) > 0.1
        # which is a LOUDNESS test wearing a noise test's name: anything above
        # about -10 dBFS RMS was flagged, so every real recording raised
        # "noise_floor" and the artifact flag forced COMPLEX (ARCH-11.7) --
        # making STUDIO unreachable for actual music. Level alone is not noise.
        #
        # What distinguishes noise from signal is the *high-frequency* energy
        # fraction: musical content is dominated by low/mid partials, while a
        # broadband floor spreads energy into the top band. Compare the energy
        # above ~1/4 Nyquist against the total; a quiet, clean render has a
        # small ratio however loud it is.
        if (
            StemMidiModel._noise_fraction(guitar) > 0.35
            or StemMidiModel._noise_fraction(bass) > 0.35
        ):
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

        flags.extend(f for f in input_flags if f not in flags)
        return flags if flags else ["none"]

    @staticmethod
    def _input_flags(source: torch.Tensor) -> list[str]:
        """Artifact flags visible only in the original upload, not in the stems."""
        flags: list[str] = []
        src = source.reshape(-1)
        if not src.numel():
            return flags
        if not torch.isfinite(src).all():
            flags.append("non_finite")
            return flags
        peak = src.abs().max().item()
        if peak <= _SILENCE_LEVEL:
            flags.append("silence")
        elif peak >= _CLIP_LEVEL:
            flags.append("clipping")
        if abs(src.mean().item()) > _DC_OFFSET_LEVEL:
            flags.append("dc_offset")
        return flags

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

    def _finalize(
        self,
        guitar: torch.Tensor,
        bass: torch.Tensor,
        metrics: torch.Tensor,
        source: torch.Tensor | None,
    ) -> tuple[dict, ProcessingReport, dict]:
        """Transcribe both stems and build MIDI metadata + report + routing.

        Mirror of the ``_forward_impl`` tail used by the chunked file path, so a
        long upload and a short one produce the same shape of answer.
        """
        guitar_tr = self._transcribe(guitar, "guitar")
        bass_tr = self._transcribe(bass, "bass")
        self._ensure_finite(guitar_tr, "guitar")
        self._ensure_finite(bass_tr, "bass")
        guitar_meta, _ = self._enhance(guitar_tr, guitar)
        bass_meta, _ = self._enhance(bass_tr, bass)

        guitar_notes = len(guitar_meta["midi_events"])
        bass_notes = len(bass_meta["midi_events"])
        total_notes = guitar_notes + bass_notes
        avg_confidence = (
            (
                guitar_meta["summary"]["avg_confidence"] * guitar_notes
                + bass_meta["summary"]["avg_confidence"] * bass_notes
            )
            / total_notes
            if total_notes
            else 0.0
        )

        midi_metadata = {
            "guitar": guitar_meta,
            "bass": bass_meta,
            "summary": {
                "avg_confidence": avg_confidence,
                "confidence_encoding": "polyphonic_key_pressure",
                "low_confidence_count": (
                    guitar_meta["summary"]["low_confidence_count"]
                    + bass_meta["summary"]["low_confidence_count"]
                ),
                "alignment_warnings": (
                    guitar_meta["summary"]["alignment_warnings"]
                    + bass_meta["summary"]["alignment_warnings"]
                ),
                "total_notes": total_notes,
            },
        }

        m = metrics.detach().float()
        if m.dim() == 2:  # (B, 2) from _separate; (2,) from _separate_chunked
            m = m.mean(dim=0)
        artifact_flags = self._detect_artifacts(guitar, bass, source=source)
        if not self._weights_loaded and "untrained_weights" not in artifact_flags:
            artifact_flags.append("untrained_weights")
        report = ProcessingReport(
            separation_proxy=float(m[0].item()),
            phase_coherence=float(m[1].item()),
            avg_confidence=avg_confidence,
            artifact_flags=artifact_flags,
            low_confidence_notes=midi_metadata["summary"]["low_confidence_count"],
            config=self.cfg,
            model_weights_loaded=self._weights_loaded,
            backend=self.backend_name,
            weights_loaded=self.weights_loaded,
        )
        return midi_metadata, report, route_by_quality(report)

    def _validate_audio(self, audio_t: torch.Tensor, path: str = "<input>") -> None:
        """Reject audio the model cannot process, before any inference runs.

        Two inputs used to fail *silently*:

        * NaN/Inf samples survived the load and poisoned the STFT, so the report
          carried a non-finite metric and ``json.dumps`` emitted a bare ``NaN``
          token that strict parsers reject (or a 500 from deep in the separator).
        * Clips shorter than one STFT window wrap through pad/crop, producing a
          frame grid that does not correspond to any audio, i.e. notes at
          nonsense ticks instead of an error.

        Both raise ``AudioValidationError`` (a ``ValueError``), which the API
        maps to 400: the upload is invalid, the server is not.
        """
        n_fft = int(self.cfg["audio"]["n_fft"])
        sr = int(self.cfg["audio"]["sample_rate"])
        if audio_t.numel() == 0:
            raise AudioValidationError(f"Audio {path!r} is empty (0 samples).")
        if not torch.isfinite(audio_t).all():
            raise AudioValidationError(
                f"Audio {path!r} contains NaN/Inf samples; refusing to process. "
                "Check the file for corrupt frames or a broken decoder."
            )
        if audio_t.shape[-1] < n_fft:
            raise AudioValidationError(
                f"Audio {path!r} is too short: {audio_t.shape[-1]} samples "
                f"({1000.0 * audio_t.shape[-1] / sr:.1f} ms), shorter than one STFT "
                f"window (n_fft={n_fft}, {1000.0 * n_fft / sr:.1f} ms)."
            )

    def _silent_result(self, audio_t: torch.Tensor) -> tuple[dict, ProcessingReport, dict]:
        """Empty, flagged delivery for a silent input.

        Zero stems, zero notes, ``silent_output`` + ``silence`` flags. Used by
        every inference path so a muted channel gets an honest 200 instead of a
        NaN-driven 500.
        """
        hop = int(self.cfg["audio"]["hop_length"])
        frames = max(1, int(audio_t.shape[-1]) // hop)
        zeros = torch.zeros(frames)
        guitar_meta = self.confidence_injector.inject_midi_metadata([], zeros, zeros)
        bass_meta = self.confidence_injector.inject_midi_metadata([], zeros, zeros)
        midi_metadata = {
            "guitar": guitar_meta,
            "bass": bass_meta,
            "summary": {
                "avg_confidence": 0.0,
                "confidence_encoding": "polyphonic_key_pressure",
                "low_confidence_count": 0,
                "alignment_warnings": 0,
                "total_notes": 0,
            },
        }
        silence = torch.zeros_like(audio_t)
        artifact_flags = self._detect_artifacts(silence, silence, source=audio_t)
        if not self._weights_loaded and "untrained_weights" not in artifact_flags:
            artifact_flags.append("untrained_weights")
        report = ProcessingReport(
            separation_proxy=0.0,
            phase_coherence=0.0,
            avg_confidence=0.0,
            artifact_flags=artifact_flags,
            low_confidence_notes=0,
            config=self.cfg,
            model_weights_loaded=self._weights_loaded,
            backend=self.backend_name,
            weights_loaded=self.weights_loaded,
        )
        return midi_metadata, report, route_by_quality(report)

    def _separate_chunked(
        self, audio_t: torch.Tensor, chunk_seconds: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Overlapping-chunk separation with complementary crossfades (H-4).

        Returns (guitar, bass, mean per-chunk metrics (2,), n_chunks). Both file
        paths use this: a whole-file call used to hand the separator the entire
        waveform, so peak memory grew with file length (PERF-1).
        """
        sr = int(self.cfg["audio"]["sample_rate"])
        total_len = audio_t.shape[-1]
        chunk_len = max(1, int(chunk_seconds * sr))
        overlap_ratio = float(self.cfg["audio"].get("overlap_ratio", 0.5))
        overlap_len = int(chunk_len * overlap_ratio)
        hop_len = chunk_len - overlap_len
        if hop_len <= 0:
            raise ValueError(
                f"overlap_ratio {overlap_ratio} leaves no hop "
                f"(chunk {chunk_len}, overlap {overlap_len}); reduce overlap_ratio."
            )

        out_guitar = torch.zeros(1, 1, total_len)
        out_bass = torch.zeros(1, 1, total_len)
        weight_sum = torch.zeros(1, 1, total_len)
        metrics_acc: list[torch.Tensor] = []
        n_chunks = 0
        pos = 0
        while pos < total_len:
            end = min(pos + chunk_len, total_len)
            chunk = audio_t[:, :, pos:end]
            valid_len = end - pos
            if valid_len < chunk_len:
                chunk = F.pad(chunk, (0, chunk_len - valid_len))

            guitar, bass, sep_metrics = self._separate(chunk)

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

        return (
            out_guitar / weight_sum.clamp(min=1e-8),
            out_bass / weight_sum.clamp(min=1e-8),
            torch.stack(metrics_acc).mean(dim=0),
            n_chunks,
        )

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
            raise AudioValidationError(
                f"Input duration {duration:.1f}s exceeds max_input_seconds="
                f"{max_seconds:g}s (memory guard, see C-1/C-6). Split the file or "
                "raise the limit explicitly in the config."
            )
        if not np.isfinite(audio).all():
            raise AudioValidationError(
                f"Audio {path!r} contains NaN/Inf samples; refusing to process."
            )
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
        self._ensure_inference_ready()
        if quantize:
            return self._get_quantized_self().process_audio_file(audio_path, quantize=False)

        audio, sr = self._load_audio(audio_path)
        audio = self._resample_to_target(audio, sr)

        # CPU-2: honour `audio.chunk_duration_sec`. The config ships 2.0 s and the
        # README sells chunked inference for RAM-bounded hosts, but this method
        # used to load the whole file and run exactly ONE forward, so peak memory
        # grew with song length and the knob was inert (only
        # `process_audio_streaming` chunked at all).
        #
        # Chunking is output-preserving here -- measured on the real model, 6 s
        # at 2 s chunks against a single pass: identical lengths and 9e-4 / 7e-4
        # relative error on guitar/bass (the crossfade seams, H-4). Anything at or
        # below one chunk takes the single-forward path unchanged, so short
        # uploads -- and every existing unit test -- keep bit-exact behaviour.
        chunk_seconds = float(self.cfg["audio"].get("chunk_duration_sec", 0.0) or 0.0)
        target_sr = int(self.cfg["audio"]["sample_rate"])
        if chunk_seconds > 0 and len(audio) > chunk_seconds * target_sr:
            return self.process_audio_streaming(audio_path, chunk_seconds=chunk_seconds)

        audio_t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        self._validate_audio(audio_t, audio_path)
        sample_rate = int(self.cfg["audio"]["sample_rate"])

        # Long uploads are separated in overlapping chunks (PERF-1): the model
        # has a fixed receptive field, so a whole file in one pass only bought
        # memory pressure (60 s => ~2.3 GB peak before this).
        chunk_seconds = float(self.cfg["audio"].get("chunk_duration_sec", 2.0))
        chunk_len = max(1, int(chunk_seconds * sample_rate))

        self.eval()
        with torch.inference_mode():
            if audio_t.shape[-1] > chunk_len and not _is_silent(audio_t):
                guitar, bass, metrics, n_chunks = self._separate_chunked(audio_t, chunk_seconds)
                midi_metadata, report, routing = self._finalize(guitar, bass, metrics, audio_t)
                stems = {
                    "guitar": guitar.squeeze().cpu().numpy(),
                    "bass": bass.squeeze().cpu().numpy(),
                }
            else:
                outputs = self.forward(audio_t)
                stems = {
                    "guitar": outputs["guitar_stem"].squeeze().cpu().numpy(),
                    "bass": outputs["bass_stem"].squeeze().cpu().numpy(),
                }
                midi_metadata = outputs["midi_metadata"]
                report = outputs["processing_report"]
                routing = outputs["routing_decision"]
                n_chunks = 1

        # Format for user delivery (H-8: per-stem MIDI from per-stem stems)
        return {
            "stems": stems,
            "midi": midi_metadata,
            "midi_guitar": midi_metadata["guitar"],
            "midi_bass": midi_metadata["bass"],
            "report": report,
            "routing": routing,
            "sample_rate": sample_rate,
            "chunks_processed": n_chunks,
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
        self._ensure_inference_ready()
        audio, sr = self._load_audio(audio_path)
        audio = self._resample_to_target(audio, sr)
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
        sr = int(self.cfg["audio"]["sample_rate"])

        self._validate_audio(audio_t, audio_path)

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

            # Everything from here to the report is downstream of the model, so
            # it stays inside `inference_mode`. Leaving it outside rebuilt an
            # autograd graph over the RECOMBINED FULL-LENGTH stem on every
            # transcriptions' backward-supporting tensors: a 60 s upload held
            # ~4.5 MB per frame (5168 frames of 128-bin pitch logits plus the
            # mamba activations kept alive through the graph) and was OOM-killed
            # at 4 GB. Measured 875 MB -> this path, and it is what made the
            # 60 s memory probe fail on CI (torch 2.15) while `main` stayed
            # green: `main`'s file path kept `_finalize` inside
            # `inference_mode`, and this method — a copy of it — did not.
            #
            # The chunk loop above was never the problem (measured flat at
            # ~774 MB across 30 chunks); the transcribe call was.
            full_guitar = out_guitar / weight_sum.clamp(min=1e-8)
            full_bass = out_bass / weight_sum.clamp(min=1e-8)

            final_metrics = torch.stack(metrics_acc).mean(dim=0)  # (2,) real per-chunk mean

            # Transcribe the recombined stems once — real logits on the full
            # timeline (C-2), no per-chunk frame-grid drift.
            guitar_tr = self._transcribe(full_guitar, "guitar")
            bass_tr = self._transcribe(full_bass, "bass")
            self._ensure_finite(guitar_tr, "guitar")
            self._ensure_finite(bass_tr, "bass")
            guitar_meta, _ = self._enhance(guitar_tr, full_guitar)
            bass_meta, _ = self._enhance(bass_tr, full_bass)

        guitar_notes = len(guitar_meta["midi_events"])
        bass_notes = len(bass_meta["midi_events"])
        total_notes = guitar_notes + bass_notes
        avg_confidence = (
            (
                guitar_meta["summary"]["avg_confidence"] * guitar_notes
                + bass_meta["summary"]["avg_confidence"] * bass_notes
            )
            / total_notes
            if total_notes
            else 0.0
        )

        midi_metadata = {
            "guitar": guitar_meta,
            "bass": bass_meta,
            "summary": {
                "avg_confidence": avg_confidence,
                "confidence_encoding": "polyphonic_key_pressure",
                "low_confidence_count": (
                    guitar_meta["summary"]["low_confidence_count"]
                    + bass_meta["summary"]["low_confidence_count"]
                ),
                "alignment_warnings": (
                    guitar_meta["summary"]["alignment_warnings"]
                    + bass_meta["summary"]["alignment_warnings"]
                ),
                "total_notes": total_notes,
            },
        }

        stream_artifacts = self._detect_artifacts(full_guitar, full_bass, source=audio_t)
        if not self._weights_loaded and "untrained_weights" not in stream_artifacts:
            stream_artifacts.append("untrained_weights")
        report = ProcessingReport(
            separation_proxy=float(final_metrics[0].item()),
            phase_coherence=float(final_metrics[1].item()),
            avg_confidence=avg_confidence,
            artifact_flags=stream_artifacts,
            low_confidence_notes=midi_metadata["summary"]["low_confidence_count"],
            # GATE-3: the gate must be able to refuse an empty render.
            note_count=len(midi_metadata["guitar"].get("midi_events") or [])
            + len(midi_metadata["bass"].get("midi_events") or []),
            config=self.cfg,
            model_weights_loaded=self._weights_loaded,
            backend=self.backend_name,
            weights_loaded=self.weights_loaded,
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

    Accepts a raw ``state_dict``, or a dict wrapping it under ``state_dict``
    (``train.py``) or ``model_state_dict`` (``scripts/export_fine_tuned.py``,
    ``scripts/fine_tune_demucs.py``, ``scripts/verify_cpu_export.py``). Config
    must come from ``config_path``.

    CKPT-1: only ``state_dict`` used to be unwrapped, so every fine-tune export
    a user follows the docs to produce raised Missing/Unexpected key(s). A dict
    with neither key is rejected explicitly rather than silently treated as a
    state dict, which is what produced the confusing key errors.
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
            # A checkpoint dict carrying neither wrapper: accept it as a bare
            # state dict, but only if that is plausibly what it is. Feeding a
            # config/optimizer dict to `load_state_dict` produces the confusing
            # "Missing key(s)" error CKPT-1 was filed about.
            if not all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                raise ValueError(
                    f"Checkpoint {checkpoint_path!r} has neither a 'state_dict' nor a "
                    "'model_state_dict' key and does not look like a raw state dict "
                    f"(keys: {sorted(ckpt)[:8]})."
                )
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
        help="Path to a trained checkpoint (.pt); required for native-Mamba file inference",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    checkpoint = (
        args.checkpoint
        or os.environ.get("MODEL_CHECKPOINT_PATH")
        or config.get("restore_from_path")
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
    print(f"Centroid separation proxy (not SI-SDR): " f"{result['report'].separation_proxy:.1f}")
    print(f"Backend: {result['report'].backend}")
    print(f"Avg Confidence: {result['report'].avg_confidence:.0%}")
    print(f"Routing: {result['routing']['action']}")


if __name__ == "__main__":
    cli()
