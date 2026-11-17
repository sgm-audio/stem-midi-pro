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
from models._mamba_compat import MAMBA_BACKEND  # noqa: E402
from models.backends.factory import build_separator, build_transcriber  # noqa: E402
from models.confidence_injector import ConfidenceInjector  # noqa: E402
from models.losses import PerceptualAudioLoss  # noqa: E402
from models.mamba_separator import MambaSeparator  # noqa: E402
from models.mamba_transcriber import MambaTranscriber  # noqa: E402
from utils.audio import resample_audio  # noqa: E402
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

    #: Which Mamba implementation is live ("mamba-ssm" kernel vs the pure-torch
    #: reference scan). Surfaced by /model-info and the processing report so an
    #: operator can tell the two apart (BUG-HONESTY-2).
    mamba_backend: str = MAMBA_BACKEND

    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config

        # Backend selection: honour `backends:` from the config instead of
        # hard-coding Mamba. Previously the factory was bypassed entirely, so
        # configs/model_config.cpu.yaml (demucs + basic_pitch) silently built
        # Mamba, and `backends:` was dead for main/api/demo/CLI (BUG-WIRING-1).
        # The "mamba" name still constructs the modules directly so their
        # historical tuple contract and object types are unchanged.
        backends_cfg = config.get("backends") or {}
        self.backend_separator = str(backends_cfg.get("separator", "mamba")).lower()
        self.backend_transcriber = str(backends_cfg.get("transcriber", "mamba")).lower()

        self.separator = (
            MambaSeparator(self.cfg)
            if self.backend_separator == "mamba"
            else build_separator(self.cfg)
        )
        self.transcriber = (
            MambaTranscriber(self.cfg)
            if self.backend_transcriber == "mamba"
            else build_transcriber(self.cfg)
        )

        self.confidence_injector = ConfidenceInjector(
            onset_threshold=self.cfg["transcriber"].get("onset_threshold", 0.5),
            min_confidence=self.cfg["quality_gates"]["min_confidence"],
        )
        self.loss_fn = PerceptualAudioLoss(self.cfg)

        #: True once real weights are loaded (see load_from_checkpoint). Served
        #: in the report so a randomly-initialised model is never mistaken for a
        #: trained one (BUG-HONESTY-1).
        self.weights_loaded = False

        # Onset refractory window. Emitting one event per above-threshold frame
        # with no merging produced ~87 notes/second on monophonic audio
        # (BUG-MIDI-7). 50 ms allows 16th notes up to 300 BPM and caps onset
        # smearing from a single attack.
        sr = float(self.cfg["audio"]["sample_rate"])
        hop = float(self.cfg["audio"]["hop_length"])
        interval_ms = float(self.cfg["transcriber"].get("min_onset_interval_ms", 50.0))
        self.min_onset_interval_frames = max(1, round(interval_ms / 1000.0 * sr / hop))

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

    # ------------------------------------------------------------------
    # Backend plumbing: accept both the module contract (5-tuple) and the
    # pluggable-backend contract (SeparateOut / TranscribeOut dicts).
    # ------------------------------------------------------------------

    def _separate(self, audio: torch.Tensor):
        """Run the separator; normalise both backend contracts to a 5-tuple."""
        sep = getattr(self.separator, "separate", None)
        out = sep(audio) if callable(sep) else self.separator(audio)
        if isinstance(out, dict):
            guitar = out["guitar"]
            residual = out.get("residual")
            if residual is None:
                residual = torch.zeros_like(guitar)
            return guitar, out["bass"], residual, out.get("state_cache"), out["metrics"]
        return out

    def _transcribe(self, stem: torch.Tensor):
        """Run the transcriber; normalise to (logits..., events | None)."""
        tr = getattr(self.transcriber, "transcribe", None)
        out = tr(stem) if callable(tr) else self.transcriber(stem)
        if isinstance(out, dict):
            return (
                out.get("onset_logits"),
                out.get("pitch_logits"),
                out.get("velocity"),
                out.get("expression"),
                out.get("confidence"),
                out.get("events"),
            )
        onset, pitch, velocity, expression, confidence = out
        return onset, pitch, velocity, expression, confidence, None

    @staticmethod
    def _preset_events(events) -> list[dict] | None:
        """Flatten a backend's list-per-batch ``events`` payload (B=1)."""
        if events is None:
            return None
        if isinstance(events, list) and events and isinstance(events[0], list):
            return list(events[0])
        return list(events)

    def _validate_audio(self, audio: torch.Tensor) -> None:
        """Reject inputs that cannot be analysed, with a validation error.

        Both checks previously surfaced as a ``RuntimeError`` from ``torch.stft``
        deep in the transcriber, which the API mapped to HTTP 500 instead of 400
        (BUG-AUDIO-5) — or, for NaN, propagated silently into the report
        (BUG-AUDIO-4).
        """
        if not torch.isfinite(audio).all():
            raise ValueError("input audio contains non-finite samples (NaN/Inf)")
        n_fft = int(self.cfg["audio"]["n_fft"])
        if audio.shape[-1] < n_fft:
            raise ValueError(
                f"input audio is too short: {audio.shape[-1]} samples, "
                f"need at least one STFT window (n_fft={n_fft} samples)"
            )

    def _events_for(self, onset_logits, pitch_logits, velocity, confidence, preset):
        """Turn a transcriber output into a flat MIDI event list."""
        preset = self._preset_events(preset)
        if preset is not None:
            return preset
        return self._logits_to_midi(onset_logits, pitch_logits, velocity, confidence)

    def _build_outputs(
        self,
        audio: torch.Tensor,
        guitar_stem: torch.Tensor,
        bass_stem: torch.Tensor,
        sep_metrics: torch.Tensor,
        tr_guitar: tuple,
        tr_bass: tuple,
    ) -> dict:
        """Assemble the public output dict (shared by batch and chunked paths)."""
        onset_logits, pitch_logits, velocity, _expression, confidence, preset_g = tr_guitar
        _b_onset, _b_pitch, _b_velocity, _b_expr, b_confidence, preset_b = tr_bass

        # Artifact detection runs on the *input* as well as the stems: clipping
        # and DC offset are properties of what the user uploaded, and a clipped
        # input is quieter after masking (BUG-AUDIO-3), while silence has to be
        # recognised before it is transcribed into invented notes (BUG-AUDIO-1/2).
        flags = self._detect_artifacts(guitar_stem, bass_stem, audio)

        if "silence" in flags:
            # No energy in, no notes out — regardless of what the onset head says.
            guitar_events: list[dict] = []
            bass_events: list[dict] = []
        else:
            guitar_events = self._events_for(
                onset_logits, pitch_logits, velocity, confidence, preset_g
            )
            bass_events = self._events_for(
                None if preset_b is not None else _b_onset,
                _b_pitch,
                _b_velocity,
                b_confidence,
                preset_b,
            )

        conf_g = confidence if confidence is not None else torch.zeros(guitar_stem.shape[-1] // 512)
        conf_b = b_confidence if b_confidence is not None else torch.zeros_like(conf_g)

        midi_guitar = self.confidence_injector.inject_midi_metadata(
            guitar_events,
            conf_g.reshape(-1),
            torch.mean(guitar_stem**2, dim=1).reshape(-1),
        )
        midi_bass = self.confidence_injector.inject_midi_metadata(
            bass_events,
            conf_b.reshape(-1),
            torch.mean(bass_stem**2, dim=1).reshape(-1),
        )

        # `midi_metadata` keeps its historical shape (guitar events + summary)
        # and now carries the two per-stem views as *distinct* dicts: previously
        # both keys pointed at the same object, so bass.mid was a copy of
        # guitar.mid (BUG-MIDI-6).
        enhanced_midi = {**midi_guitar, "guitar": midi_guitar, "bass": midi_bass}

        report = ProcessingReport(
            si_sdr=float(sep_metrics[0, 0].item()),
            phase_coherence=float(sep_metrics[0, 1].item()),
            avg_confidence=float(conf_g.mean().item()) if conf_g.numel() else 0.0,
            artifact_flags=flags,
            low_confidence_notes=enhanced_midi["summary"]["low_confidence_count"],
            config=self.cfg,
            backend=f"{self.backend_separator}+{self.backend_transcriber}",
            weights_loaded=bool(getattr(self, "weights_loaded", False)),
        )

        return {
            "guitar_stem": guitar_stem,
            "bass_stem": bass_stem,
            "midi_metadata": enhanced_midi,
            "processing_report": report,
            "routing_decision": route_by_quality(report),
        }

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
        self._validate_audio(audio)

        # 1. Separation (returns None in place of a state cache — real SSM
        # state passing is not implemented; see C-2.5)
        guitar_stem, bass_stem, _residual, _state_cache, sep_metrics = self._separate(audio)

        # 2. Transcription. Both stems are transcribed: the API ships a
        # bass.mid, and relabelling the guitar transcription as "bass" produced
        # a file full of guitar pitches (BUG-MIDI-6).
        tr_guitar = self._transcribe(guitar_stem)
        tr_bass = self._transcribe(bass_stem)

        outputs = self._build_outputs(
            audio, guitar_stem, bass_stem, sep_metrics, tr_guitar, tr_bass
        )
        onset_logits = tr_guitar[0]

        # 3. Compute loss if targets provided (training mode)
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

        Vectorized with torch.nonzero (RF-3.1.6). Assumes B=1 for inference.

        A greedy refractory window (``transcriber.min_onset_interval_ms``,
        default 50 ms) is applied so that a single attack — which lights up
        several consecutive frames because the onset head is smooth — becomes
        one note. Without it a monophonic phrase produced ~87 notes/second
        (BUG-MIDI-7), which is neither playable nor editable.
        """
        if onset_logits is None:
            return []
        threshold = self.cfg["transcriber"].get("onset_threshold", 0.5)
        onsets = (torch.sigmoid(onset_logits) > threshold).squeeze(-1)
        pitches = torch.argmax(pitch_logits, dim=-1)
        vel_values = (velocity * 127).long().squeeze(-1)

        hits = torch.nonzero(onsets)  # (N, 2): [batch_idx, frame]
        events = []
        last_frame = -(10**9)
        gap = int(getattr(self, "min_onset_interval_frames", 1))
        for b, t in hits.tolist():
            if t - last_frame < gap:
                continue  # same attack, already emitted
            last_frame = t
            events.append(
                {
                    "note": int(pitches[b, t].item()),
                    "onset_frame": int(t),
                    "velocity": int(vel_values[b, t].item()),
                    "confidence": float(confidence[b, t].item()),
                }
            )
        return events

    def _detect_artifacts(
        self,
        guitar: torch.Tensor,
        bass: torch.Tensor,
        source: torch.Tensor | None = None,
    ) -> list[str]:
        """Detect input- and output-level defects for the quality gate.

        ``source`` is the mix that was uploaded. Checking it (rather than only
        the separated stems) is what lets the gate see the two most common bad
        uploads: digital silence, and a hard-clipped DI, which comes back
        quieter after masking and previously passed unflagged.
        """
        flags: list[str] = []

        # ---- input-level checks -------------------------------------------
        if source is not None:
            if float(source.abs().max()) < 1e-5:
                flags.append("silence")
            clipped = float((source.abs() >= 0.999).float().mean())
            if clipped > 1e-3:
                flags.append("clipping")
            if float(source.mean().abs()) > 0.02:
                flags.append("dc_offset")
        elif max(float(guitar.abs().max()), float(bass.abs().max())) < 1e-5:
            flags.append("silence")

        # ---- output-level checks ------------------------------------------
        # istft can overshoot 0 dBFS even for in-range input.
        overshoot = float(guitar.abs().max()) > 0.99 or float(bass.abs().max()) > 0.99
        if overshoot and "clipping" not in flags:
            flags.append("clipping")

        guitar_power = torch.mean(guitar**2, dim=[1, 2])
        bass_power = torch.mean(bass**2, dim=[1, 2])
        if torch.any(guitar_power > 0.1) or torch.any(bass_power > 0.1):
            flags.append("noise_floor")

        # Near-perfect ANTI-correlation cancels when summed (C-2.8). Positive
        # correlation is valid (shared fundamentals); sign matters.
        cross_corr = torch.sum(guitar * bass, dim=[1, 2])
        energy_guitar = torch.sum(guitar**2, dim=[1, 2])
        energy_bass = torch.sum(bass**2, dim=[1, 2])
        phase_corr = cross_corr / (torch.sqrt(energy_guitar * energy_bass) + 1e-8)
        if torch.any(phase_corr < -0.9):
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

    def _infer(self, audio_t: torch.Tensor) -> dict:
        """Run the pipeline, chunking long inputs to bound peak memory."""
        chunk_seconds = float(self.cfg["audio"].get("chunk_duration_sec") or 0.0)
        sr = int(self.cfg["audio"]["sample_rate"])
        chunk_len = int(chunk_seconds * sr)
        if chunk_len <= 0 or audio_t.shape[-1] <= chunk_len:
            return self.forward(audio_t)
        return self._forward_chunked(audio_t, chunk_len)

    def _forward_chunked(self, audio_t: torch.Tensor, chunk_len: int) -> dict:
        """Process ``audio_t`` in ``chunk_len`` slices and recombine.

        ``process_audio_file`` used to hand the whole file to the model, so peak
        RSS grew ~23 MB per second of audio: 60 s -> 2346 MB, ~14 GB projected
        for the 10-minute limit the API advertises (BUG-PERF-1). Chunking bounds
        the working set to one ``chunk_duration_sec`` window.

        Chunks overlap by ``audio.overlap_ratio`` and each contributes its first
        ``hop`` samples, so the concatenation reconstructs the input length.
        True SSM state passing between chunks is still not implemented (C-2.5),
        so this bounds memory but does not make the model stateful.
        """
        self._validate_audio(audio_t)
        hop_length = int(self.cfg["audio"]["hop_length"])
        overlap = int(chunk_len * float(self.cfg["audio"].get("overlap_ratio", 0.5)))
        hop = max(1, chunk_len - overlap)
        total = audio_t.shape[-1]

        g_parts: list[torch.Tensor] = []
        b_parts: list[torch.Tensor] = []
        # Per-stem transcriber outputs, concatenated along the frame axis.
        g_logits: dict[str, list[torch.Tensor]] = {"o": [], "p": [], "v": [], "e": [], "c": []}
        b_logits: dict[str, list[torch.Tensor]] = {"o": [], "p": [], "v": [], "e": [], "c": []}
        g_events: list[dict] = []
        b_events: list[dict] = []
        metrics_sum = None
        throughput_events_only = False
        pos = 0

        while pos < total:
            end = min(pos + chunk_len, total)
            chunk = audio_t[:, :, pos:end]
            valid = chunk.shape[-1]
            if valid < chunk_len:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_len - valid))

            guitar, bass, _residual, _state, metrics = self._separate(chunk)
            keep = min(valid if end >= total else hop, valid)
            g_parts.append(guitar[:, :, :keep])
            b_parts.append(bass[:, :, :keep])

            offset = pos // hop_length
            for tr, logits, events in (
                (self._transcribe(guitar[:, :, :keep]), g_logits, g_events),
                (self._transcribe(bass[:, :, :keep]), b_logits, b_events),
            ):
                if tr[0] is None:  # events-only backend (e.g. Basic Pitch)
                    throughput_events_only = True
                    events.extend(
                        {**e, "onset_frame": int(e["onset_frame"]) + offset}
                        for e in (self._preset_events(tr[5]) or [])
                    )
                else:
                    for key, part in zip(("o", "p", "v", "e", "c"), tr[:5], strict=False):
                        logits[key].append(part)

            metrics_sum = metrics if metrics_sum is None else metrics_sum + metrics
            pos += hop

        n_chunks = len(g_parts)
        guitar_stem = torch.cat(g_parts, dim=-1)
        bass_stem = torch.cat(b_parts, dim=-1)
        metrics = metrics_sum / n_chunks if metrics_sum is not None else guitar_stem.new_zeros(1, 2)

        def _joined(logits: dict[str, list[torch.Tensor]]):
            if throughput_events_only or not logits["o"]:
                return (None, None, None, None, None, None)
            joined = tuple(torch.cat(logits[k], dim=1) for k in ("o", "p", "v", "e", "c"))
            return (*joined, None)

        tr_guitar = _joined(g_logits)
        tr_bass = _joined(b_logits)
        if throughput_events_only:
            tr_guitar = (None, None, None, None, None, [g_events])
            tr_bass = (None, None, None, None, None, [b_events])

        outputs = self._build_outputs(audio_t, guitar_stem, bass_stem, metrics, tr_guitar, tr_bass)
        outputs["chunks_processed"] = n_chunks
        return outputs

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
        if not np.isfinite(audio).all():
            raise ValueError("input audio contains non-finite samples (NaN/Inf)")

        audio_t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)  # (1, 1, T)

        # Sample-rate conversion must be band-limited: linear interpolation
        # injected a -4 dB spectral image at 4.5 kHz for an 8 kHz upload
        # (BUG-AUDIO-6), corrupting the mel input the transcriber sees.
        target_sr = int(self.cfg["audio"]["sample_rate"])
        if sr != target_sr:
            audio_t = resample_audio(audio_t, sr, target_sr)
            sr = target_sr

        with torch.inference_mode():
            outputs = self._infer(audio_t)

        midi_guitar = outputs["midi_metadata"]["guitar"]
        midi_bass = outputs["midi_metadata"]["bass"]

        return {
            "stems": {
                "guitar": outputs["guitar_stem"].squeeze().cpu().numpy(),
                "bass": outputs["bass_stem"].squeeze().cpu().numpy(),
            },
            "midi": outputs["midi_metadata"],
            "midi_guitar": midi_guitar,
            "midi_bass": midi_bass,
            "report": outputs["processing_report"],
            "routing": outputs["routing_decision"],
            "sample_rate": sr,
        }

    def process_audio_streaming(self, audio_path: str, chunk_seconds: float = 2.0) -> dict:
        """Streaming/chunked inference with a bounded memory footprint.

        NOTE: chunks are processed independently and recombined at the waveform
        level; true SSM state passing between chunks is not implemented yet
        (C-2.5 / ARCH-11.8.1).

        Pitch and velocity now come from the transcriber rather than from
        ``torch.zeros``, which made every streamed note pitch 0 (C-1) at
        velocity 1 (BUG-MIDI-8).
        """
        audio, sr = self._load_audio(audio_path)
        target_sr = int(self.cfg["audio"]["sample_rate"])
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
        if sr != target_sr:
            audio_t = resample_audio(audio_t, sr, target_sr)
            sr = target_sr

        chunk_len = max(1, int(chunk_seconds * target_sr))
        self.eval()
        with torch.inference_mode():
            outputs = self._forward_chunked(audio_t, chunk_len)

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
            "sample_rate": sr,
            "chunks_processed": outputs.get("chunks_processed", 1),
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
    # train.py writes "state_dict"; accept the older "model_state_dict" too.
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.weights_loaded = True
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
