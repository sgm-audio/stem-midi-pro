# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Confidence metadata injection for MIDI output (vectorized)."""
from __future__ import annotations

import numpy as np
import torch


class ConfidenceInjector:
    """
    Injects per-note confidence metadata into MIDI output.
    Aligns MIDI events with stem energy for cross-modal validation.
    """

    def __init__(self, onset_threshold: float = 0.5, min_confidence: float = 0.6):
        self.onset_threshold = onset_threshold
        self.min_confidence = min_confidence

    def inject_midi_metadata(
        self,
        midi_events: list,
        confidence: torch.Tensor,
        stem_energy: torch.Tensor,
    ) -> dict:
        """
        Args:
            midi_events: List of [{'note', 'onset_frame', ...}] per batch item
            confidence: (T,) tensor of per-frame confidence scores
            stem_energy: (T,) tensor of stem RMS energy for alignment
        Returns:
            dict with MIDI + metadata for DAW import
        """
        # Vectorized: extract all onset frames at once
        if not midi_events:
            return {
                "midi_events": [],
                "summary": {
                    "avg_confidence": 0.0,
                    "low_confidence_count": 0,
                    "alignment_warnings": 0,
                },
            }

        # Flatten batch-0 events (most common case)
        events = midi_events[0] if isinstance(midi_events, list) and len(midi_events) > 0 else midi_events

        if not events:
            return {
                "midi_events": [],
                "summary": {
                    "avg_confidence": float(torch.mean(confidence).item()) if confidence.numel() > 0 else 0.0,
                    "low_confidence_count": int((confidence < self.min_confidence).sum().item()),
                    "alignment_warnings": 0,
                },
            }

        # Vectorized: gather onset indices
        onset_frames = torch.tensor(
            [e["onset_frame"] for e in events], dtype=torch.long, device=confidence.device
        )

        # Clamp to valid range
        valid_mask = onset_frames < len(confidence)
        onset_frames = onset_frames.clamp(0, len(confidence) - 1)

        # Gather confidence and energy vectors
        conf_scores = confidence[onset_frames].cpu().numpy()
        energy_vals = stem_energy[onset_frames].cpu().numpy() if len(stem_energy) > 0 else confidence[onset_frames].cpu().numpy() * 0

        # Vectorized alignment: low confidence + low energy = artifact
        alignment_scores = conf_scores * (1.0 / (1.0 + np.exp(-energy_vals * 10)))

        enhanced_events = []
        for i, event in enumerate(events):
            conf = float(conf_scores[i]) if valid_mask[i] else 0.0
            energy = float(energy_vals[i]) if valid_mask[i] else 0.0
            alignment = float(alignment_scores[i])

            enhanced_events.append({
                **event,
                "confidence": conf,
                "stem_energy": energy,
                "alignment_score": alignment,
                "needs_review": alignment < self.min_confidence,
                "cc_127_value": int(conf * 127),
                "quantization_suggestion": "grid" if alignment > 0.85 else "human",
            })

        return {
            "midi_events": enhanced_events,
            "summary": {
                "avg_confidence": float(torch.mean(confidence).item()) if confidence.numel() > 0 else 0.0,
                "low_confidence_count": int((confidence < self.min_confidence).sum().item()) if confidence.numel() > 0 else 0,
                "alignment_warnings": sum(1 for e in enhanced_events if e["needs_review"]),
            },
        }
