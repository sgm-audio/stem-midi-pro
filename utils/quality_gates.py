# SPDX-License-Identifier: Apache-2.0
import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Artifact flag meaning. Anything other than "none" forces COMPLEX.
# "non_finite" and "silent_output" exist because without them an all-NaN
# render was classified DRAFT (every comparison against NaN is False) and an
# all-silent render reached STUDIO (the separation proxy clamps to 30 dB).
KNOWN_ARTIFACT_FLAGS = {
    "none",
    "clipping",
    "noise_floor",
    "phase_cancellation",
    "non_finite",
    "silent_output",
}


class QualityTier(Enum):
    STUDIO = "studio"  # Confidence >=0.85, no artifacts, no low-confidence notes
    DRAFT = "draft"  # 0.70 <= confidence < 0.85, no artifacts
    COMPLEX = "complex"  # <0.70, artifact flags, invalid metrics, or low-conf notes


@dataclass
class ProcessingReport:
    # Renamed from `si_sdr` per C-2.6: this value is the spectral-centroid
    # separation HEURISTIC from MambaSeparator (how spectrally distinct the
    # stems are, in dB) — NOT true SI-SDR, which requires ground truth and is
    # only computed during training (ARCH-11.3.2).
    centroid_separation_db: float
    phase_coherence: float
    avg_confidence: float
    artifact_flags: list[str]
    low_confidence_notes: int
    config: dict[str, Any] | None = None
    # Which Mamba implementation / backend stack produced this result (M-7):
    # "mamba (pure-torch)" vs "mamba (mamba-ssm)" vs a backend pair like
    # "separator=demucs,transcriber=basic_pitch".
    backend: str | None = None
    # How many note events the render actually contains (GATE-3). `None` means
    # "caller did not supply it" -- reports built by older call sites (and test
    # doubles) keep working, and the empty-render rule then simply does not
    # apply. Supplying it is what lets the gate refuse to sell an empty MIDI.
    note_count: int | None = None
    # True when any metric is NaN/Inf: the render is unusable regardless of
    # what the other numbers say (review §4).
    invalid: bool = field(default=False, init=False)

    def __post_init__(self):
        gates = (self.config or {}).get("quality_gates", {})
        self.studio_confidence_threshold = gates.get("studio_confidence_threshold", 0.85)
        self.draft_confidence_threshold = gates.get("draft_confidence_threshold", 0.70)
        # A NaN/Inf anywhere means the model blew up. This must be checked
        # BEFORE any threshold comparison: `NaN >= x` is always False, so a
        # blown-up render used to skip the STUDIO branch and land on DRAFT via
        # confidence alone ("refine in our editor") instead of being reported
        # as a failure (review §4).
        self.invalid = not all(
            isinstance(v, (int, float)) and math.isfinite(float(v))
            for v in (self.centroid_separation_db, self.phase_coherence, self.avg_confidence)
        )
        # Normalise flags so an unknown flag can never silently bypass the
        # artifact check, and record the failure explicitly.
        self.artifact_flags = [
            f if f in KNOWN_ARTIFACT_FLAGS else f"unknown:{f}" for f in (self.artifact_flags or [])
        ]
        if self.invalid and "non_finite" not in self.artifact_flags:
            self.artifact_flags.append("non_finite")

        # The centroid value is a saturation-prone proxy, not SI-SDR (C-2.6),
        # and it is NOT sufficient on its own: one FFT bin of separation
        # reports exactly 20.00 dB and clears the shipped `min: 20` (review §9).
        # It is therefore never the *only* test -- see `quality_tier`, which
        # also demands confidence, zero low-confidence notes and no artifacts.
        #
        # But it stays *active*, because all three shipped configs document it
        # and an inert knob is its own bug class (AUDIT GATE tests). A missing,
        # non-numeric or non-finite threshold disables just this half.
        raw_threshold = gates.get("min_centroid_separation_db")
        self.min_centroid_separation_db = (
            float(raw_threshold)
            if isinstance(raw_threshold, (int, float)) and math.isfinite(float(raw_threshold))
            else None
        )
        legacy = gates.get("min_si_sdr")
        if legacy is not None:
            warnings.warn(
                f"quality_gates.min_si_sdr={legacy} is ignored: the reported "
                "value is a centroid-separation proxy, not SI-SDR (review §9). "
                "Studio quality is gated on confidence, low-confidence notes "
                "and artifacts.",
                DeprecationWarning,
                stacklevel=2,
            )

    @property
    def quality_tier(self) -> QualityTier:
        # Invalid metrics or any artifact flag force COMPLEX regardless of
        # confidence (ARCH-11.7): stellar numbers cannot make a clipped,
        # non-finite or silent render studio- or draft-quality.
        if self.invalid:
            return QualityTier.COMPLEX
        real_flags = [f for f in self.artifact_flags if f != "none"]
        if real_flags:
            return QualityTier.COMPLEX
        # GATE-3: a render with no notes is not a deliverable, however good its
        # numbers look. `draft` is a promise that there is something to edit and
        # `studio`/direct_download is a promise that there is something to use,
        # so an empty MIDI must fall through to COMPLEX (fallback + refund
        # options) rather than being sold.
        if self.note_count is not None and self.note_count <= 0:
            return QualityTier.COMPLEX
        # STUDIO requires ALL of: confidence at the studio bar, no individually
        # unreliable notes, and (when configured) the separation threshold.
        # Requiring several independent signals is the fix for §9 -- one FFT bin
        # of centroid separation used to be enough on its own.
        separation_ok = (
            self.centroid_separation_db >= self.min_centroid_separation_db
            if self.min_centroid_separation_db is not None
            else True
        )
        if (
            self.avg_confidence >= self.studio_confidence_threshold
            and self.low_confidence_notes == 0
            and separation_ok
        ):
            return QualityTier.STUDIO
        elif self.avg_confidence >= self.draft_confidence_threshold:
            return QualityTier.DRAFT
        else:
            return QualityTier.COMPLEX

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe view (no NaN/Inf) for `processing_report.json` (review §12).

        `json.dumps` happily writes bare `NaN`/`Infinity`, which is invalid
        RFC 8259 and throws in `JSON.parse`. Non-finite numbers become `None`
        and the failure is recorded in `invalid` / the flags.
        """

        def _safe(x: Any) -> Any:
            try:
                xf = float(x)
            except (TypeError, ValueError):
                return None
            return xf if math.isfinite(xf) else None

        return {
            "centroid_separation_db": _safe(self.centroid_separation_db),
            "centroid_separation_is_proxy": True,
            "phase_coherence": _safe(self.phase_coherence),
            "avg_confidence": _safe(self.avg_confidence),
            "artifact_flags": list(self.artifact_flags),
            "low_confidence_notes": int(self.low_confidence_notes),
            "note_count": None if self.note_count is None else int(self.note_count),
            "quality_tier": self.quality_tier.value,
            "invalid": self.invalid,
            "backend": self.backend,
        }


def route_by_quality(report: ProcessingReport) -> dict[str, Any]:
    """
    Returns user-facing routing decision + UI prompts.
    """
    if report.quality_tier == QualityTier.STUDIO:
        return {
            "action": "direct_download",
            "badge": "✨ Studio Quality",
            "message": "Your stems and MIDI are ready for professional use.",
            "editor_launch": False,
        }
    elif report.quality_tier == QualityTier.DRAFT:
        return {
            "action": "editor_launch",
            "badge": "✏️ Draft Quality",
            "message": (
                f"MIDI confidence: {report.avg_confidence:.0%}. Refine in our editor before export."
            ),
            "editor_launch": True,
            "editor_presets": {
                "highlight_low_confidence": True,
                "default_quantization": "human" if report.avg_confidence < 0.80 else "grid",
            },
        }
    else:  # COMPLEX
        return {
            "action": "fallback_options",
            "badge": "⚠️ Complex Material",
            "message": "Heavy processing detected. Choose your path:",
            "options": [
                {"id": "raw", "label": "Download raw output (free)", "confidence_overlay": True},
                {"id": "human", "label": "Human-reviewed refinement (+$4.99, 24h)", "upsell": True},
                {"id": "cancel", "label": "Cancel with full credit refund", "refund": True},
            ],
        }
