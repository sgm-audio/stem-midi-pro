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
        enhanced_events = []
        for event in midi_events:
            onset_idx = event["onset_frame"]
            conf_score = (
                confidence[onset_idx].item() if onset_idx < len(confidence) else 0.0
            )
            energy = (
                stem_energy[onset_idx].item() if onset_idx < len(stem_energy) else 0.0
            )

            # Cross-modal alignment penalty: low confidence + low energy = likely artifact
            alignment_score = (
                conf_score
                * (1.0 / (1.0 + torch.exp(-torch.tensor(energy * 10)))).item()
            )

            enhanced_events.append(
                {
                    **event,
                    "confidence": conf_score,
                    "stem_energy": energy,
                    "alignment_score": alignment_score,
                    "needs_review": alignment_score < self.min_confidence,
                    # DAW-friendly metadata
                    "cc_127_value": int(conf_score * 127),  # Confidence as MIDI CC#127
                    "quantization_suggestion": "grid"
                    if alignment_score > 0.85
                    else "human",
                }
            )

        return {
            "midi_events": enhanced_events,
            "summary": {
                "avg_confidence": torch.mean(confidence).item()
                if len(confidence) > 0
                else 0.0,
                "low_confidence_count": (confidence < self.min_confidence).sum().item()
                if len(confidence) > 0
                else 0,
                "alignment_warnings": sum(
                    1 for e in enhanced_events if e["needs_review"]
                ),
            },
        }
