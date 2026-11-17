# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from enum import Enum
from typing import Any


class QualityTier(Enum):
    STUDIO = "studio"  # Confidence ≥0.85, SI-SDR ≥20dB
    DRAFT = "draft"  # 0.70 ≤ confidence < 0.85
    COMPLEX = "complex"  # <0.70 or artifact flags


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
    # Whether TRAINED weights produced this report, or the HF repo's
    # randomly-initialised defaults (H-2 / HONESTY-2). None = unknown (a
    # report built by hand, e.g. in a unit test).
    weights_loaded: bool | None = None

    def __post_init__(self):
        gates = (self.config or {}).get("quality_gates", {})
        self.studio_confidence_threshold = gates.get("studio_confidence_threshold", 0.85)
        self.draft_confidence_threshold = gates.get("draft_confidence_threshold", 0.70)
        # `min_si_sdr` is the pre-rename key (C-2.6); accepted as legacy alias.
        self.min_centroid_separation_db = gates.get(
            "min_centroid_separation_db", gates.get("min_si_sdr", 20.0)
        )

    @property
    def quality_tier(self) -> QualityTier:
        # Artifact flags force COMPLEX regardless of confidence/centroids
        # (ARCH-11.7): stellar numbers cannot make a clipped or
        # phase-cancelling render studio- or draft-quality.
        real_flags = [f for f in self.artifact_flags if f != "none"]
        if real_flags:
            return QualityTier.COMPLEX
        if (
            self.avg_confidence >= self.studio_confidence_threshold
            and self.centroid_separation_db >= self.min_centroid_separation_db
        ):
            return QualityTier.STUDIO
        elif self.avg_confidence >= self.draft_confidence_threshold:
            return QualityTier.DRAFT
        else:
            return QualityTier.COMPLEX


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
