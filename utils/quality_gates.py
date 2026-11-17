# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from enum import Enum
from typing import Any


class QualityTier(Enum):
    STUDIO = "studio"  # Heuristic confidence/proxy threshold; not a measured studio grade.
    DRAFT = "draft"  # 0.70 <= confidence < 0.85 or proxy below the studio threshold.
    COMPLEX = "complex"  # <0.70 or artifact flags.


@dataclass
class ProcessingReport:
    """Processing metrics and the conservative routing decision.

    ``separation_proxy`` is intentionally named as a proxy. ``si_sdr`` and
    ``centroid_separation_db`` are deprecated compatibility aliases and are
    never presented as true reference-based SI-SDR measurements.
    """

    separation_proxy: float | None = None
    phase_coherence: float = 0.0
    avg_confidence: float = 0.0
    artifact_flags: list[str] | None = None
    low_confidence_notes: int = 0
    config: dict[str, Any] | None = None
    model_weights_loaded: bool | None = None
    si_sdr: float | None = None  # deprecated input/output compatibility alias
    centroid_separation_db: float | None = None  # deprecated compatibility alias
    backend: str | None = None

    def __post_init__(self):
        if self.separation_proxy is None:
            self.separation_proxy = float(
                self.centroid_separation_db
                if self.centroid_separation_db is not None
                else (self.si_sdr or 0.0)
            )
        proxy = float(self.separation_proxy)
        self.separation_proxy = proxy
        # Compatibility aliases for older callers. Routing always uses proxy.
        self.si_sdr = proxy
        self.centroid_separation_db = proxy

        if self.artifact_flags is None:
            self.artifact_flags = ["none"]
        else:
            self.artifact_flags = list(self.artifact_flags)
        if self.model_weights_loaded is False and "untrained_weights" not in self.artifact_flags:
            self.artifact_flags.append("untrained_weights")

        gates = (self.config or {}).get("quality_gates", {})
        self.studio_confidence_threshold = gates.get("studio_confidence_threshold", 0.85)
        self.draft_confidence_threshold = gates.get("draft_confidence_threshold", 0.70)
        self.min_separation_proxy = gates.get(
            "min_separation_proxy",
            gates.get("min_centroid_separation_db", gates.get("min_si_sdr", 20.0)),
        )
        # Deprecated config aliases for callers that inspect the object.
        self.min_si_sdr = self.min_separation_proxy
        self.min_centroid_separation_db = self.min_separation_proxy

    @property
    def quality_tier(self) -> QualityTier:
        # Artifact flags force COMPLEX regardless of confidence/proxy.
        real_flags = [f for f in self.artifact_flags if f != "none"]
        if real_flags:
            return QualityTier.COMPLEX
        if (
            self.avg_confidence >= self.studio_confidence_threshold
            and self.separation_proxy >= self.min_separation_proxy
        ):
            return QualityTier.STUDIO
        if self.avg_confidence >= self.draft_confidence_threshold:
            return QualityTier.DRAFT
        return QualityTier.COMPLEX


def route_by_quality(report: ProcessingReport) -> dict[str, Any]:
    """Return honest user-facing routing metadata.

    There is no built-in editor, human-review service, payment flow, or refund
    system in this repository. Routing therefore points the user to their DAW
    instead of advertising nonexistent services.
    """
    if report.quality_tier == QualityTier.STUDIO:
        return {
            "action": "direct_download",
            "badge": "Heuristic quality gate passed",
            "message": (
                "Your stems and MIDI passed the configured heuristic gate. "
                "Verify them against the source before final use."
            ),
            "editor_launch": False,
        }
    if report.quality_tier == QualityTier.DRAFT:
        return {
            "action": "manual_review",
            "badge": "Draft quality",
            "message": (
                f"MIDI confidence: {report.avg_confidence:.0%}. "
                "Review and refine the output in your DAW."
            ),
            "editor_launch": False,
            "review_in_daw": True,
            "editor_presets": {
                "highlight_low_confidence": True,
                "default_quantization": "human" if report.avg_confidence < 0.80 else "grid",
            },
        }
    return {
        "action": "manual_review",
        "badge": "Complex material",
        "message": "Artifacts or low confidence were detected; review the raw output in your DAW.",
        "review_in_daw": True,
        "options": [
            {"id": "raw", "label": "Download raw output", "confidence_overlay": True},
            {"id": "human", "label": "Review and edit in your DAW"},
            {"id": "cancel", "label": "Discard this output"},
        ],
    }
