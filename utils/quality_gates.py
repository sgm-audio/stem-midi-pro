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
    # Provenance and input-level flags remain distinct from output artifacts.
    "untrained_weights",
    "silence",
    "dc_offset",
    # 0 transcribed notes: nothing to deliver, whatever the metrics say.
    "empty_render",
}


class QualityTier(Enum):
    STUDIO = "studio"  # Confidence >=0.85, no artifacts, no low-confidence notes
    DRAFT = "draft"  # 0.70 <= confidence < 0.85, no artifacts
    COMPLEX = "complex"  # <0.70, artifact flags, invalid metrics, or low-conf notes


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
    # Content-calibrated confidence (CONF-1): ``avg_confidence`` scaled by the
    # evidence that the stems actually carry notes (tonality x crest factor).
    # This is the ONLY confidence the tier decision reads — the raw head is
    # uninformative and, measured, ranks a clipped square above clean music.
    # ``None`` means "not calibrated" and falls back to ``avg_confidence`` so
    # hand-built reports keep working.
    calibrated_confidence: float | None = None
    artifact_flags: list[str] | None = None
    low_confidence_notes: int = 0
    # Number of notes in the render. ``None`` = unknown (a hand-built report);
    # 0 means there is nothing to deliver, which must never grade as STUDIO or
    # DRAFT however good the metrics look (GATE-3).
    total_notes: int | None = None
    config: dict[str, Any] | None = None
    model_weights_loaded: bool | None = None
    si_sdr: float | None = None  # deprecated input/output compatibility alias
    centroid_separation_db: float | None = None  # deprecated compatibility alias
    backend: str | None = None
    # Whether TRAINED weights produced this report, or the repo's
    # randomly-initialised defaults (HONESTY-1). None = unknown (a report built
    # by hand, e.g. in a unit test).
    weights_loaded: bool | None = None
    # How many note events the render actually contains (GATE-3). `None` means
    # "caller did not supply it" -- reports built by older call sites (and test
    # doubles) keep working, and the empty-render rule then does not apply.
    # Supplying it is what lets the gate refuse to sell an empty MIDI.
    note_count: int | None = None
    # True when any metric is NaN/Inf: the render is unusable regardless of
    # what the other numbers say (review §4).
    invalid: bool = field(default=False, init=False)

    def __post_init__(self):
        # Provenance matters for CONF-1's corroboration rule: only a crosstalk
        # reading (0..1) says anything about whether the stems were separated,
        # and a legacy centroid value inherited from an older call site must not
        # be mistaken for one. Recorded here, where the fallback happens.
        self._proxy_is_crosstalk = self.separation_proxy is not None
        if self.separation_proxy is None:
            self.separation_proxy = float(
                self.centroid_separation_db
                if self.centroid_separation_db is not None
                else (self.si_sdr or 0.0)
            )
        proxy = float(self.separation_proxy)
        self.separation_proxy = proxy
        # Compatibility aliases for older callers. Routing never uses proxy.
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

        # A NaN/Inf anywhere means the model blew up. This must be checked
        # BEFORE any threshold comparison: `NaN >= x` is always False, so a
        # blown-up render used to skip the STUDIO branch and land on DRAFT via
        # confidence alone ("refine in our editor") instead of being reported
        # as a failure (review §4).
        self.invalid = not all(
            isinstance(v, int | float) and math.isfinite(float(v))
            for v in (self.centroid_separation_db, self.phase_coherence, self.avg_confidence)
        )
        # Normalise flags so an unknown flag can never silently bypass the
        # artifact check, and record the failure explicitly.
        self.artifact_flags = [
            f if f in KNOWN_ARTIFACT_FLAGS else f"unknown:{f}" for f in (self.artifact_flags or [])
        ]
        if self.invalid and "non_finite" not in self.artifact_flags:
            self.artifact_flags.append("non_finite")
        if self.calibrated_confidence is None:
            self.calibrated_confidence = float(self.avg_confidence)
        if self.total_notes == 0 and "empty_render" not in self.artifact_flags:
            self.artifact_flags.append("empty_render")

        # The centroid value is a saturation-prone proxy, not SI-SDR (C-2.6):
        # one FFT bin of separation reports exactly 20.00 dB, which clears
        # `min_si_sdr: 20`. Renaming it (done on main) makes the report honest
        # but does not make the number discriminating, so it is kept for
        # observability and removed from the tier decision (review §9).
        # The threshold attributes remain as compatibility metadata only.
        self.min_separation_proxy = gates.get(
            "min_separation_proxy",
            gates.get("min_centroid_separation_db", gates.get("min_si_sdr", 20.0)),
        )
        # GATE-4: the reported separation number is now MASK CROSSTALK, where
        # LOWER IS BETTER (0.0 = every stem's energy is in bins it dominates,
        # 0.667 = a flat one-third mask that separates nothing). It therefore
        # needs its own ceiling rather than the old higher-is-better floor.
        #
        # Default 0.60 sits below the analytic no-separation value (0.667) and
        # above the untrained model's measured 0.47: a separator that does worse
        # than a flat mask is refused, while an untrained-but-not-degenerate one
        # is not punished for a threshold calibrated on trained weights we do not
        # ship. `None` disables the gate.
        self.max_separation_crosstalk = gates.get("max_separation_crosstalk", 0.60)
        if self.max_separation_crosstalk is not None:
            if not 0.0 < float(self.max_separation_crosstalk) <= 1.0:
                raise ValueError(
                    "quality_gates.max_separation_crosstalk must be in (0, 1] or null; "
                    f"got {self.max_separation_crosstalk!r}"
                )
            self.max_separation_crosstalk = float(self.max_separation_crosstalk)
        self.min_centroid_separation_db = gates.get("min_centroid_separation_db")
        self.min_si_sdr = self.min_separation_proxy
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

    def _crosstalk_ceiling_for_corroboration(self) -> float:
        """The crosstalk value a render must beat to be corroborated.

        A render whose separation is worse than a flat one-third mask (0.667)
        has no corroboration available, so the ceiling is the configured
        `max_separation_crosstalk`. When the gate is disabled (`null`) there is
        no independent evidence at all, and nothing may be corroborated: the
        ceiling becomes 0.0, which no real render achieves, so STUDIO falls back
        to requiring the DRAFT bar and can never be self-certified.
        """
        if self.max_separation_crosstalk is None:
            return 0.0
        return float(self.max_separation_crosstalk)

    @property
    def separation_proxy_db(self) -> float:
        """The same number as ``centroid_separation_db``, under a functional name.

        Kept for callers written against the pre-C-2.6 name; it reads the
        renamed field, so it cannot drift from the dataclass again.
        """
        return self.centroid_separation_db

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
        # numbers look. `draft` promises there is something to edit and
        # `studio`/direct_download promises there is something to use, so an
        # empty MIDI falls through to COMPLEX (fallback + refund options).
        if self.note_count is not None and self.note_count <= 0:
            return QualityTier.COMPLEX
        # GATE-4: a separation so poor that it is worse than handing every stem
        # the same flat mask means the stems are not usable as stems. Checked
        # before the confidence tiers because a confident transcription of a
        # non-separation is still not a separation.
        if (
            self.max_separation_crosstalk is not None
            and self.separation_proxy is not None
            and math.isfinite(float(self.separation_proxy))
            # Only meaningful for the crosstalk metric; the legacy centroid
            # proxy uses the opposite (higher-is-better) orientation and would
            # be misread here.
            and float(self.separation_proxy) <= 1.0
            and float(self.separation_proxy) > self.max_separation_crosstalk
        ):
            return QualityTier.COMPLEX
        # CONF-1: STUDIO is not a claim about BELIEVING the render, it is a
        # claim about the render being right -- so it must be corroborated by
        # something other than the confidence head.
        #
        # The head had no loss term anywhere in the repo, so it kept its random
        # initialisation and reported ~0.9555 for music, white noise, silence
        # and DC alike (measured spread across content classes: 0.00079) while
        # STUDIO's bar was 0.85. A single unsupervised number clearing its own
        # threshold is self-certification; it cannot distinguish "the model is
        # confident because it is right" from "the head outputs 0.95 for
        # everything".
        #
        # `corroborated` is the independent evidence, and it is the separation
        # gate (GATE-4 crosstalk): a render cannot be studio quality unless the
        # stems were actually separated. Confidence is still REQUIRED, so this
        # only ever removes unearned badges -- it never raises a tier.
        #
        # Only a CROSSTALK value (0..1) can corroborate anything. A report that
        # carries the legacy centroid proxy (>= 20 dB is a typical reading) has
        # no independent evidence attached to it, so it cannot corroborate --
        # and it must not be silently punished either: an absent measurement is
        # not a bad measurement. That distinction matters because `route_by_quality`
        # is a public helper and older callers construct reports with
        # `centroid_separation_db` directly (see tests/test_quality_gates.py).
        proxy = self.separation_proxy
        is_crosstalk = getattr(self, "_proxy_is_crosstalk", False) and (
            proxy is not None and math.isfinite(float(proxy)) and 0.0 <= float(proxy) <= 1.0
        )
        if is_crosstalk:
            corroborated = float(proxy) <= self._crosstalk_ceiling_for_corroboration()
        else:
            # The report carries a legacy centroid reading (or was built by an
            # older call site that never set `separation_proxy`). That is not
            # evidence of separation, but it is also not evidence AGAINST it, so
            # the render keeps the pre-CONF-1 behaviour rather than being
            # downgraded for a field it never had.
            corroborated = True
        # The tier input itself is `calibrated_confidence` (upstream CONF-1),
        # not the raw head: a render whose average confidence clears the bar
        # while dozens of notes are individually doubtful is not studio-ready
        # (review §9), and an uncalibrated head is not evidence at all.
        if (
            self.calibrated_confidence >= self.studio_confidence_threshold
            and self.low_confidence_notes == 0
            and corroborated
        ):
            return QualityTier.STUDIO
        if self.calibrated_confidence >= self.draft_confidence_threshold:
            return QualityTier.DRAFT
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
            "calibrated_confidence": _safe(self.calibrated_confidence),
            "artifact_flags": list(self.artifact_flags),
            "low_confidence_notes": int(self.low_confidence_notes),
            # GATE-3: two names, both truthful. `total_notes` is upstream's
            # canonical count and drives the `empty_render` flag; `note_count`
            # is this branch's stricter carry-through, which also rejects a
            # negative count. Kept as an alias rather than dropped so existing
            # reports and guards keep working.
            "max_separation_crosstalk": self.max_separation_crosstalk,
            "note_count": None if self.note_count is None else int(self.note_count),
            "total_notes": None if self.total_notes is None else int(self.total_notes),
            "quality_tier": self.quality_tier.value,
            "invalid": self.invalid,
            "backend": self.backend,
        }


def route_by_quality(report: ProcessingReport) -> dict[str, Any]:
    """Return honest user-facing routing metadata.

    There is no built-in editor, human-review service, payment flow, or refund
    system in this repository. Routing therefore points the user to their DAW
    instead of advertising nonexistent services.
    """
    # Nothing to deliver: no tier can make an empty render useful, so this is
    # checked before the tiers (GATE-3).
    if "empty_render" in (report.artifact_flags or []):
        return {
            "action": "fallback_options",
            "badge": "No notes detected",
            "message": (
                "Transcription found no notes in this upload. Download the raw stems and "
                "check the input, or try a longer/cleaner excerpt."
            ),
            "review_in_daw": True,
            "options": [
                {"id": "raw", "label": "Download raw output", "confidence_overlay": True},
                {"id": "cancel", "label": "Discard this output"},
            ],
        }
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
