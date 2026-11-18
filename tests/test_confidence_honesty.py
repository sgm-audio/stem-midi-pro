# SPDX-License-Identifier: Apache-2.0
"""CONF-1: the number every tier depends on must be trained, and must not self-certify.

Two separate defects, both measured on the tree this branch started from:

1. **The head was never trained.** ``models/losses.py`` had loss terms for
   onset, pitch, velocity, duration and alignment -- and none for confidence,
   while ``train.py`` optimises exactly what the loss returns. So the head kept
   its random initialisation forever. Measured mean confidence over 2 s inputs:

       clean music 0.95543 | clipped 0.95609 | white noise 0.95566 | silence 0.95571

   A spread of 0.00079 across content classes, with STUDIO's bar at 0.85:
   every render, including digital silence, was comfortably "studio confident".

2. **A single unsourced number authorised the badge.** Even once trained, a
   model can be confidently wrong, so the tier decision cannot rest on the head
   alone. Studio now additionally requires corroboration -- the GATE-4 crosstalk
   gate -- so confidence is a necessary condition that can only ever REMOVE a
   badge, never manufacture one.

What this file cannot do is prove the retrained head is well-calibrated: that
needs trained weights, which this repo does not ship (``weights_loaded`` is
False for every run here). It proves the two things that are structural.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="the loss and the head are torch modules")

GATES = {
    "studio_confidence_threshold": 0.85,
    "draft_confidence_threshold": 0.70,
    "max_separation_crosstalk": 0.60,
}


# ---------------------------------------------------------------------------
# 1. the head is trainable
# ---------------------------------------------------------------------------


def test_confidence_loss_term_exists_and_has_a_gradient():
    """A head with no loss term is a random projection of the hidden state.

    The target is the model's own per-frame onset correctness, so the term is
    observable and needs no new labels.
    """
    import yaml

    from models.losses import PerceptualAudioLoss

    with open("configs/model_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    loss_fn = PerceptualAudioLoss(cfg)

    b, t = 1, 32
    stems = [torch.randn(b, 1, 1024), torch.randn(b, 1, 1024)]
    targets = [torch.randn(b, 1, 1024), torch.randn(b, 1, 1024)]
    confidence = torch.full((b, t), 0.5, requires_grad=True)
    onset_pred = torch.randn(b, t, 1)
    onset_target = (torch.rand(b, t, 1) > 0.5).float()

    total, parts = loss_fn(
        stems,
        targets,
        pred_onsets=onset_pred,
        target_onsets=onset_target,
        pred_confidence=confidence,
        pred_onsets_for_confidence=onset_pred,
        target_onsets_for_confidence=onset_target,
    )
    assert "confidence" in parts, "the loss no longer reports a confidence term"
    assert parts["confidence"] > 0.0, "the confidence term did not activate"
    total.backward()
    assert confidence.grad is not None, "no gradient reached the confidence head"
    assert float(confidence.grad.abs().sum()) > 0.0, "gradient is identically zero"


def test_optimising_the_confidence_term_moves_it_towards_correctness():
    """It is not enough for a term to exist; it has to change the prediction.

    Trains a single confidence vector against a fixed onset target and checks
    that it stops being a constant and lands near per-frame correctness.
    """
    import yaml

    from models.losses import PerceptualAudioLoss

    with open("configs/model_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    loss_fn = PerceptualAudioLoss(cfg)

    b, t = 1, 48
    torch.manual_seed(0)
    stems = [torch.randn(b, 1, 1024), torch.randn(b, 1, 1024)]
    targets = [torch.randn(b, 1, 1024), torch.randn(b, 1, 1024)]
    confidence = torch.full((b, t), 0.5, requires_grad=True)
    onset_pred = torch.randn(b, t, 1)
    onset_target = (torch.rand(b, t, 1) > 0.5).float()
    with torch.no_grad():
        # (B, T, 1) -> (B, T) to match the confidence tensor.
        correct = ((torch.sigmoid(onset_pred) > 0.5).float() == onset_target).float().squeeze(-1)

    # lr 0.5 diverges on this single-vector problem (measured err 0.59); 0.05 converges.
    optimiser = torch.optim.SGD([confidence], lr=0.05)
    for _ in range(200):
        optimiser.zero_grad()
        total, _parts = loss_fn(
            stems,
            targets,
            pred_onsets=onset_pred,
            target_onsets=onset_target,
            pred_confidence=confidence,
            pred_onsets_for_confidence=onset_pred,
            target_onsets_for_confidence=onset_target,
        )
        total.backward()
        optimiser.step()

    spread = float(confidence.max() - confidence.min())
    assert spread > 0.1, (
        f"confidence stayed near-constant (spread {spread:.3f}) after 200 steps; "
        "the term is not driving the head"
    )
    error = float((confidence.detach() - correct).abs().mean())
    assert error < 0.20, (
        f"confidence did not move towards per-frame correctness (mean |conf - "
        f"correct| = {error:.3f}, started at ~0.5)"
    )


# ---------------------------------------------------------------------------
# 2. an untrained head cannot authorise a studio badge
# ---------------------------------------------------------------------------


def test_an_untrained_head_is_not_informative_about_content():
    """Documents the failure this branch fixes, and would catch a regression.

    Runs the shipped architecture with random weights (what every test and
    every install without a checkpoint has) and measures confidence on music
    versus silence. A trained head separates these; an unsupervised one reports
    the same number, which is precisely why confidence alone cannot gate a tier.
    """
    import yaml

    from models.mamba_transcriber import MambaTranscriber

    with open("configs/model_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    torch.manual_seed(0)
    transcriber = MambaTranscriber(cfg).eval()

    sr = 44100
    t = torch.arange(2 * sr) / sr
    cases = {
        "music": 0.30 * torch.sin(2 * torch.pi * 110 * t)
        + 0.15 * torch.sin(2 * torch.pi * 330 * t),
        "silence": torch.zeros_like(t),
    }
    scores = {}
    for name, signal in cases.items():
        with torch.inference_mode():
            _o, _p, _v, _e, conf = transcriber(signal.reshape(1, 1, -1).float())
        scores[name] = float(conf.mean())

    spread = abs(scores["music"] - scores["silence"])
    # The point is not a threshold on the model -- it is that the SPREAD is tiny,
    # so a tier decision cannot be justified by this number alone.
    assert spread < 0.05, (
        f"untrained confidence differs by {spread:.3f} between music and silence "
        f"({scores}); if this ever becomes discriminating, revisit whether the "
        "corroboration requirement in quality_gates is still needed"
    )
    assert all(v > 0.85 for v in scores.values()), (
        f"an untrained head no longer clears the studio threshold ({scores}); the "
        "self-certification this fix removed would be back"
    )


# ---------------------------------------------------------------------------
# 3. studio requires corroboration
# ---------------------------------------------------------------------------


def _tier(avg_confidence: float, crosstalk: float | None, ceiling: float | None = 0.60) -> str:
    from utils.quality_gates import ProcessingReport

    gates = dict(GATES, max_separation_crosstalk=ceiling)
    report = ProcessingReport(
        separation_proxy=crosstalk,
        phase_coherence=0.6,
        avg_confidence=avg_confidence,
        artifact_flags=["none"],
        low_confidence_notes=0,
        note_count=10,
        total_notes=10,
        config={"quality_gates": gates},
    )
    return report.quality_tier.value


def test_high_confidence_with_a_bad_separation_is_not_studio():
    """The exact failure mode: 0.96 confidence, worse than a flat mask."""
    assert _tier(0.96, 0.90) == "complex"


def test_high_confidence_with_a_good_separation_is_studio():
    """Corroboration must not make STUDIO unreachable."""
    assert _tier(0.96, 0.30) == "studio"


def test_confidence_is_still_required():
    """Corroboration is additional, never a substitute for confidence."""
    assert _tier(0.50, 0.10) == "complex"
    assert _tier(0.75, 0.10) == "draft"


def test_disabling_the_separation_gate_removes_the_corroboration_not_the_bar():
    """With no independent evidence available, nothing may self-certify."""
    assert _tier(0.96, 0.10, ceiling=None) == "draft", (
        "with the separation gate disabled a confident render still reached studio, "
        "which is self-certification"
    )


def test_corroboration_can_only_ever_lower_a_tier():
    """Sweeping crosstalk at fixed content: the tier is monotone NON-INCREASING.

    Higher crosstalk means worse separation, so the tier may only fall as the
    sweep advances; it must never rise.
    """
    order = {"complex": 0, "draft": 1, "studio": 2}
    tiers = [order[_tier(0.96, cx)] for cx in (0.0, 0.3, 0.59, 0.61, 0.9)]
    assert tiers == sorted(tiers, reverse=True), (
        f"the tier rose as separation got worse: {tiers} for crosstalk "
        "(0.0, 0.3, 0.59, 0.61, 0.9)"
    )


def test_a_real_untrained_pipeline_never_reaches_studio():
    """End-to-end: the shipped config with random weights is not studio quality.

    This is the integration-level version of the same claim -- whatever the
    report says, an uncheckpointed model must not hand out the badge that
    promises a usable render.
    """
    import tempfile

    import soundfile as sf
    import yaml

    from main import StemMidiModel

    with open("configs/model_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["allow_untrained_inference"] = True
    torch.manual_seed(0)
    model = StemMidiModel(cfg).eval()

    sr = int(cfg["audio"]["sample_rate"])
    t = np.arange(3 * sr) / sr
    signal = 0.3 * np.sin(2 * np.pi * 110 * t) + 0.15 * np.sin(2 * np.pi * 330 * t)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
        path = fh.name
    sf.write(path, signal.astype(np.float32), sr, subtype="FLOAT")
    out = model.process_audio_file(path)
    tier = out["report"].quality_tier.value
    assert tier != "studio", (
        f"an untrained model was graded {tier}: confidence {out['report'].avg_confidence:.3f} "
        f"with crosstalk {out['report'].separation_proxy:.3f}"
    )
