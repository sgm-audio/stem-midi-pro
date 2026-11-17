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
        """Return enhanced events and note-level confidence summary.

        ``stem_energy`` and ``confidence`` are expected on the STFT frame grid
        indexed by each event's ``onset_frame``. Samples outside either vector
        use the event's supplied confidence and zero energy.
        """
        confidence = confidence.detach().reshape(-1).float().cpu()
        stem_energy = stem_energy.detach().reshape(-1).float().cpu()

        onset_indices = (
            torch.tensor([int(e["onset_frame"]) for e in midi_events], dtype=torch.long)
            if midi_events
            else torch.empty(0, dtype=torch.long)
        )
        conf_scores = torch.tensor(
            [float(event.get("confidence", 0.0)) for event in midi_events],
            dtype=torch.float32,
        )
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
            confidence_value = int(max(0, min(127, conf_score * 127)))

            enhanced_events.append(
                {
                    **event,
                    "confidence": conf_score,
                    "confidence_value": confidence_value,
                    # Deprecated JSON alias; never serialized as MIDI CC#127.
                    "cc_127_value": confidence_value,
                    "confidence_cc_value": confidence_value,
                    "stem_energy": energy,
                    "alignment_score": alignment_score,
                    "needs_review": alignment_score < self.min_confidence,
                    "quantization_suggestion": "grid" if alignment_score > 0.85 else "human",
                }
            )

        # This is intentionally note-level. Counting every low-confidence
        # frame would claim that notes existed where no note was emitted.
        note_confidences = conf_scores
        return {
            "midi_events": enhanced_events,
            "summary": {
                "confidence_encoding": "polyphonic_key_pressure",
                "confidence_encoding_note": (
                    "Standard MIDI note-specific pressure; DAW interpretation varies. "
                    "Use JSON confidence fields when unsupported."
                ),
                "avg_confidence": (
                    float(note_confidences.mean().item()) if len(note_confidences) > 0 else 0.0
                ),
                "low_confidence_count": (
                    int((note_confidences < self.min_confidence).sum().item())
                    if len(note_confidences) > 0
                    else 0
                ),
                "alignment_warnings": sum(1 for e in enhanced_events if e["needs_review"]),
            },
        }
