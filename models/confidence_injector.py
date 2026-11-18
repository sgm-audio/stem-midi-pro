# SPDX-License-Identifier: Apache-2.0
"""Attach note-level confidence and alignment metadata to MIDI events."""

from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger(__name__)


class ConfidenceInjector:
    """Inject confidence metadata on the same frame grid as MIDI onsets.

    JSON event metadata is canonical. The MIDI serializer represents the
    normalized confidence value as Polyphonic Key Pressure, which is
    note-specific; the historical ``cc_127_value`` field remains a JSON-only
    compatibility alias and is never emitted as MIDI CC#127.
    """

    def __init__(self, onset_threshold: float = 0.5, min_confidence: float = 0.6):
        self.onset_threshold = onset_threshold
        self.min_confidence = min_confidence

    def inject_midi_metadata(
        self, midi_events: list, confidence: torch.Tensor, stem_energy: torch.Tensor
    ) -> dict:
        """Return enhanced events and a note-level confidence summary.

        Both confidence and energy are indexed on the STFT frame grid. A note
        outside the model confidence grid receives zero confidence and zero
        energy rather than inheriting caller-provided metadata that was never
        produced by the active model.
        """
        confidence = confidence.detach().reshape(-1).float().cpu()
        stem_energy = stem_energy.detach().reshape(-1).float().cpu()

        onset_indices = (
            torch.tensor([int(e["onset_frame"]) for e in midi_events], dtype=torch.long)
            if midi_events
            else torch.empty(0, dtype=torch.long)
        )
        conf_scores = torch.zeros(len(midi_events), dtype=torch.float32)
        energies = torch.zeros(len(midi_events), dtype=torch.float32)
        valid_conf = (onset_indices >= 0) & (onset_indices < len(confidence))
        valid_energy = (onset_indices >= 0) & (onset_indices < len(stem_energy))
        if valid_conf.any():
            conf_scores[valid_conf] = confidence[onset_indices[valid_conf]]
        if valid_energy.any():
            energies[valid_energy] = stem_energy[onset_indices[valid_energy]]
        # Backends must provide normalized confidence; clamp defensively so a
        # malformed backend cannot emit invalid metadata or skew the gate.
        conf_scores.clamp_(0.0, 1.0)

        alignment_scores = conf_scores * torch.sigmoid(energies * 10)
        enhanced_events = []
        for i, event in enumerate(midi_events):
            conf_score = float(conf_scores[i].item())
            energy = float(energies[i].item())
            alignment_score = float(alignment_scores[i].item())
            if not math.isfinite(conf_score):
                raise ValueError(
                    f"MIDI event {i}: non-finite confidence {conf_score}; the model "
                    "produced NaN/Inf output. Refusing to write corrupt metadata."
                )
            # MIDI-4: `int()` truncates, so confidence 0.999 -> 126 and the top
            # of the 0-127 range ("fully confident") is unreachable. `round()`
            # makes 0.999 -> 127 and 0.5 -> 64. Upstream introduced this
            # `confidence_value` clamp on the Polyphonic Key Pressure lane with
            # the same truncation, so the fix applies here too -- rounding only
            # the writer changes nothing for a model-produced event, because
            # this line has already written the integer.
            confidence_value = max(0, min(127, round(conf_score * 127)))

            enhanced_events.append(
                {
                    **event,
                    "confidence": conf_score,
                    "confidence_value": confidence_value,
                    # Deprecated JSON aliases; never serialized as MIDI CC#127.
                    "cc_127_value": confidence_value,
                    "confidence_cc_value": confidence_value,
                    "stem_energy": energy,
                    "alignment_score": alignment_score,
                    # Review is a confidence decision; energy remains a
                    # diagnostic alignment score and must not change the flag.
                    "needs_review": conf_score < self.min_confidence,
                    "quantization_suggestion": "grid" if alignment_score > 0.85 else "human",
                }
            )

        note_confidences = conf_scores
        low_confidence_count = (
            int((note_confidences < self.min_confidence).sum().item())
            if len(note_confidences) > 0
            else 0
        )
        return {
            "midi_events": enhanced_events,
            "summary": {
                "confidence_encoding": "polyphonic_key_pressure",
                "confidence_encoding_note": (
                    "Standard MIDI note-specific pressure; DAW interpretation varies. "
                    "Use JSON confidence fields when unsupported."
                ),
                # GATE-2: `low_confidence_count` below counts NOTES, not frames.
                # Upstream reached the same conclusion independently
                # (`note_confidences` is indexed by note onset), so this branch
                # keeps upstream's implementation.
                "avg_confidence": (
                    float(note_confidences.mean().item()) if len(note_confidences) > 0 else 0.0
                ),
                "low_confidence_count": low_confidence_count,
                "alignment_warnings": sum(1 for e in enhanced_events if e["needs_review"]),
                "total_notes": len(enhanced_events),
            },
        }
