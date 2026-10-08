# SPDX-License-Identifier: Apache-2.0
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
        self, midi_events: list, confidence: torch.Tensor, stem_energy: torch.Tensor
    ) -> dict:
        """
        Args:
            midi_events: List of {'note', 'onset_frame', 'duration', ...}
            confidence: (T,) tensor of per-frame confidence scores
            stem_energy: (T,) tensor of stem RMS energy for alignment
        Returns:
            dict with MIDI + metadata for DAW import
        """
        # Normalise to 1-D: callers pass (T,) or (B=1, T); flatten handles both
        confidence = confidence.reshape(-1)
        stem_energy = stem_energy.reshape(-1)

        # Vectorized lookup of per-event confidence / energy / alignment scores
        onset_indices = (
            torch.tensor([e["onset_frame"] for e in midi_events], dtype=torch.long)
            if midi_events
            else torch.empty(0, dtype=torch.long)
        )
        conf_scores = torch.zeros(len(midi_events))
        energies = torch.zeros(len(midi_events))
        valid_conf = (onset_indices >= 0) & (onset_indices < len(confidence))
        valid_energy = (onset_indices >= 0) & (onset_indices < len(stem_energy))
        if valid_conf.any():
            conf_scores[valid_conf] = confidence[onset_indices[valid_conf]]
        if valid_energy.any():
            energies[valid_energy] = stem_energy[onset_indices[valid_energy]]
        # Cross-modal alignment penalty: low confidence + low energy = likely artifact
        alignment_scores = conf_scores * torch.sigmoid(energies * 10)

        enhanced_events = []
        for i, event in enumerate(midi_events):
            conf_score = conf_scores[i].item()
            energy = energies[i].item()
            alignment_score = alignment_scores[i].item()

            enhanced_events.append(
                {
                    **event,
                    "confidence": conf_score,
                    "stem_energy": energy,
                    "alignment_score": alignment_score,
                    "needs_review": alignment_score < self.min_confidence,
                    # DAW-friendly metadata
                    "cc_127_value": int(conf_score * 127),  # Confidence as MIDI CC#127
                    "quantization_suggestion": "grid" if alignment_score > 0.85 else "human",
                }
            )

        return {
            "midi_events": enhanced_events,
            "summary": {
                "avg_confidence": torch.mean(confidence).item() if len(confidence) > 0 else 0.0,
                "low_confidence_count": (
                    (confidence < self.min_confidence).sum().item() if len(confidence) > 0 else 0
                ),
                "alignment_warnings": sum(1 for e in enhanced_events if e["needs_review"]),
            },
        }
