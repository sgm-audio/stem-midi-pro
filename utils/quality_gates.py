from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any

class QualityTier(Enum):
    STUDIO = "studio"    # Confidence ≥0.85, SI-SDR ≥20dB
    DRAFT = "draft"      # 0.70 ≤ confidence < 0.85
    COMPLEX = "complex"  # <0.70 or artifact flags

@dataclass
class ProcessingReport:
    si_sdr: float
    phase_coherence: float
    avg_confidence: float
    artifact_flags: List[str]
    low_confidence_notes: int
    
    @property
    def quality_tier(self) -> QualityTier:
        # Import config thresholds (in practice, these would come from a config)
        studio_confidence_threshold = 0.85
        draft_confidence_threshold = 0.70
        min_si_sdr = 20.0
        
        if self.avg_confidence >= studio_confidence_threshold and self.si_sdr >= min_si_sdr:
            return QualityTier.STUDIO
        elif self.avg_confidence >= draft_confidence_threshold:
            return QualityTier.DRAFT
        else:
            return QualityTier.COMPLEX

def route_by_quality(report: ProcessingReport) -> Dict[str, Any]:
    """
    Returns user-facing routing decision + UI prompts.
    """
    if report.quality_tier == QualityTier.STUDIO:
        return {
            "action": "direct_download",
            "badge": "✨ Studio Quality",
            "message": "Your stems and MIDI are ready for professional use.",
            "editor_launch": False
        }
    elif report.quality_tier == QualityTier.DRAFT:
        return {
            "action": "editor_launch",
            "badge": "✏️ Draft Quality",
            "message": f"MIDI confidence: {report.avg_confidence:.0%}. Refine in our editor before export.",
            "editor_launch": True,
            "editor_presets": {
                "highlight_low_confidence": True,
                "default_quantization": "human" if report.avg_confidence < 0.80 else "grid"
            }
        }
    else:  # COMPLEX
        return {
            "action": "fallback_options",
            "badge": "⚠️ Complex Material",
            "message": "Heavy processing detected. Choose your path:",
            "options": [
                {"id": "raw", "label": "Download raw output (free)", "confidence_overlay": True},
                {"id": "human", "label": "Human-reviewed refinement (+$4.99, 24h)", "upsell": True},
                {"id": "cancel", "label": "Cancel with full credit refund", "refund": True}
            ]
        }