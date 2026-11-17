# SPDX-License-Identifier: Apache-2.0
import torch


class ConfidenceInjector:
    """
    Inject per-note confidence metadata into MIDI output.

    Confidence is kept as a normalized JSON value plus a 0-127 MIDI value.
    ``utils.midi`` serializes that value as Polyphonic Key Pressure, which is
    note-specific. The historical ``cc_127_value`` field remains as a JSON
    compatibility alias but is no longer emitted as MIDI CC#127.
    """

    def __init__(self, onset_threshold: float = 0.5, min_confidence: float = 0.6):
        self.onset_threshold = onset_threshold
        self.min_confidence = min_confidence

    def inject_midi_metadata(
        self, midi_events: list, confidence: torch.Tensor, stem_energy: torch.Tensor
    ) -> dict:
        """
        Args:
            midi_events: List of {'note', 'onset_frame', 'duration_frames', ...}
            confidence: (T,) tensor of per-frame confidence scores
            stem_energy: (T,) frame-level RMS/energy envelope for alignment
        Returns:
            dict with MIDI events and note-level metadata for DAW import.
        """
        # Normalize to 1-D. Detaching and moving to CPU also makes metadata
        # creation safe when inference is later run on an accelerator.
        confidence = confidence.detach().reshape(-1).float().cpu()
        stem_energy = stem_energy.detach().reshape(-1).float().cpu()

        # Vectorized lookup of per-event confidence / energy / alignment scores.
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

        # Cross-modal alignment penalty: low confidence + low energy = likely
        # artifact. Energy is expected to be a frame-level envelope, not a
        # sample-index lookup.
        alignment_scores = conf_scores * torch.sigmoid(energies * 10)

        enhanced_events = []
        for i, event in enumerate(midi_events):
            conf_score = float(conf_scores[i].item())
            energy = float(energies[i].item())
            alignment_score = float(alignment_scores[i].item())
            confidence_value = int(max(0, min(127, conf_score * 127)))

            enhanced_events.append(
                {
                    **event,
                    "confidence": conf_score,
                    "confidence_value": confidence_value,
                    # Deprecated JSON alias. It is deliberately not serialized
                    # as MIDI CC#127; see utils/midi.py.
                    "cc_127_value": confidence_value,
                    "stem_energy": energy,
                    "alignment_score": alignment_score,
                    "needs_review": alignment_score < self.min_confidence,
                    "quantization_suggestion": "grid" if alignment_score > 0.85 else "human",
                }
            )

        # This is intentionally note-level. Counting every low-confidence
        # frame made the report claim to count notes that did not exist.
        note_confidences = conf_scores
        return {
            "midi_events": enhanced_events,
            "summary": {
                # Polyphonic Key Pressure is a standard MIDI message, but DAW
                # support for using it as a confidence overlay varies. The
                # JSON fields remain the canonical interoperable metadata.
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
