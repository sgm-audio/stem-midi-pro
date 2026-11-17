# SPDX-License-Identifier: Apache-2.0
"""Per-note confidence metadata for MIDI events.

Aligns note events with the model's frame grid and the stem's frame-level
energy, and produces per event:

* ``confidence``      — the model's per-frame confidence at the onset.
* ``stem_energy`` / ``alignment_score`` — the stem energy on that frame and
                        ``confidence * sigmoid(energy * 10)``, kept as
                        diagnostics (a note on near-silence is the classic
                        hallucination signature).
* ``needs_review``    — a **confidence** decision (``confidence < min_confidence``),
                        per ARCH-11.6.2 "low-confidence notes are flagged for
                        human review". Deriving it from ``alignment_score`` made
                        a confident pianissimo passage always flagged and let
                        the same events flip on gain alone.
* ``confidence_cc_value`` (legacy alias ``cc_127_value``) and
  ``quantization_suggestion`` — consumed by ``utils/midi.py``; the writer picks
  the actual controller (CC#120-127 are Channel Mode messages).

Both ``confidence`` and ``stem_energy`` must be on the **frame grid** (hop
length), not the sample grid: indexing a per-sample energy array with a frame
index looked at audio ~86x earlier than the note (H-3). A length mismatch and
non-finite values both raise rather than silently mis-aligning or writing NaN.
"""

import logging
import math

import torch

logger = logging.getLogger(__name__)


class ConfidenceInjector:
    """
    Injects per-note confidence metadata into MIDI output.
    Aligns MIDI events with stem energy for cross-modal validation.

    ``stem_energy`` must be indexed on the SAME grid as ``onset_frame`` — i.e.
    per STFT frame, not per audio sample (M-4). Callers pool sample-level RMS
    onto the frame grid before calling.
    """

    def __init__(self, onset_threshold: float = 0.5, min_confidence: float = 0.6):
        self.onset_threshold = onset_threshold
        self.min_confidence = min_confidence

    @staticmethod
    def _require_finite(values: torch.Tensor, name: str) -> None:
        """Refuse non-finite metadata instead of writing corrupt MIDI/JSON."""
        if values.numel() and not torch.isfinite(values).all():
            raise ValueError(
                f"{name} contains non-finite values; the model produced NaN/Inf "
                "output. Refusing to write corrupt metadata."
            )

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

        if confidence.numel() != stem_energy.numel():
            raise ValueError(
                "confidence and stem_energy must share the frame grid: "
                f"got {confidence.numel()} vs {stem_energy.numel()} frames "
                "(pool sample-rate energy by hop_length before calling)"
            )
        self._require_finite(confidence, "confidence")
        self._require_finite(stem_energy, "stem_energy")

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
        # Cross-modal diagnostic: low confidence + low energy = likely artifact
        alignment_scores = conf_scores * torch.sigmoid(energies * 10)

        enhanced_events = []
        for i, event in enumerate(midi_events):
            conf_score = conf_scores[i].item()
            energy = energies[i].item()
            alignment_score = alignment_scores[i].item()

            if not math.isfinite(conf_score):
                raise ValueError(
                    f"MIDI event {i}: non-finite confidence {conf_score}; the model "
                    "produced NaN/Inf output. Refusing to write corrupt metadata."
                )

            # Confidence goes to a general-purpose CC by default — CC#120-127
            # are Channel Mode messages per the MIDI spec (H-5). The writer
            # (utils/midi.py) chooses the actual controller number;
            # "cc_127_value" is kept as a legacy alias.
            cc_value = int(conf_score * 127)

            enhanced_events.append(
                {
                    **event,
                    "confidence": conf_score,
                    "stem_energy": energy,
                    "alignment_score": alignment_score,
                    # The review flag is a CONFIDENCE decision (ARCH-11.6.2):
                    # "low-confidence notes are flagged for human review".
                    # Energy stays in alignment_score as a diagnostic.
                    "needs_review": conf_score < self.min_confidence,
                    # DAW-friendly metadata
                    "confidence_cc_value": cc_value,
                    "cc_127_value": cc_value,  # legacy alias (pre-H-5 name)
                    "quantization_suggestion": "grid" if alignment_score > 0.85 else "human",
                }
            )

        # Count NOTES, not frames: `(confidence < thr).sum()` counts STFT frames,
        # so 1 s of audio reported ~86 "low confidence notes" no matter how many
        # notes were emitted, and the number grew with file length. The reviewer
        # listens to notes.
        low_confidence_notes = sum(1 for e in enhanced_events if e["needs_review"])
        return {
            "midi_events": enhanced_events,
            "summary": {
                "avg_confidence": torch.mean(confidence).item() if len(confidence) > 0 else 0.0,
                "low_confidence_count": low_confidence_notes,
                "alignment_warnings": low_confidence_notes,
                "total_notes": len(enhanced_events),
            },
        }
