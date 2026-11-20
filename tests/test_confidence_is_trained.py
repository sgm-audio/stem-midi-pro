# SPDX-License-Identifier: Apache-2.0
"""CONF-1: the confidence term must be reachable from the real training path.

`tests/test_confidence_honesty.py` proves the term works in isolation. That is
not enough, and this file exists because it was not: the term was added to
`PerceptualAudioLoss` but `StemMidiModel._forward_impl` never passed it the
confidence arguments, so with the shipped training loop the head would still
have received no gradient and kept its random initialisation forever -- the
exact defect CONF-1 reports, reintroduced one layer up behind a green test.

So this test asserts the wiring, not the formula:

* `confidence` appears in the loss dict of a real `forward()` call;
* its value is non-zero;
* `backward()` puts a gradient on every `confidence_head` weight.

A term that does not train anything is indistinguishable from a missing one.
"""

from __future__ import annotations

import pytest
import torch
import yaml


@pytest.fixture(scope="module")
def model():
    from main import StemMidiModel

    with open("configs/model_config.cpu.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Native Mamba modules: the confidence head lives on them, and the
    # pretrained-backend path replaces it entirely.
    cfg.pop("backends", None)
    torch.manual_seed(0)
    m = StemMidiModel(cfg)
    m.train()
    return m, cfg


@pytest.fixture(scope="module")
def batch(model):
    _m, cfg = model
    b, t = 1, cfg["audio"]["sample_rate"]
    onsets = torch.zeros(b, 200)
    onsets[0, ::10] = 1.0
    return {
        "audio": torch.randn(b, 1, t) * 0.1,
        "target_guitar": torch.randn(b, 1, t) * 0.05,
        "target_bass": torch.randn(b, 1, t) * 0.05,
        "target_onsets": onsets,
    }


def test_the_confidence_term_reaches_the_loss_dict(model, batch):
    """A term that is never called cannot train anything."""
    m, _ = model
    out = m.forward(
        batch["audio"],
        target_guitar=batch["target_guitar"],
        target_bass=batch["target_bass"],
        target_onsets=batch["target_onsets"],
    )
    assert "confidence" in out["loss_dict"], (
        "the confidence term is missing from the loss dict: `_forward_impl` is not "
        "passing pred_confidence / onset targets to PerceptualAudioLoss, so the head "
        "is unsupervised exactly as CONF-1 describes"
    )
    assert float(out["loss_dict"]["confidence"]) > 0.0, (
        f"confidence term is {float(out['loss_dict']['confidence'])}: zero means it "
        "contributes nothing to the objective"
    )


def test_the_confidence_head_receives_gradient(model, batch):
    """The end-to-end claim: an optimiser stepping this loss moves the head."""
    m, _ = model
    m.zero_grad(set_to_none=True)
    out = m.forward(
        batch["audio"],
        target_guitar=batch["target_guitar"],
        target_bass=batch["target_bass"],
        target_onsets=batch["target_onsets"],
    )
    out["loss"].backward()
    norms = [
        float(p.grad.norm())
        for p in m.transcriber.confidence_head.parameters()
        if p.grad is not None
    ]
    assert norms, "no confidence_head parameter received a gradient at all"
    assert all(n > 0.0 for n in norms), f"a confidence_head weight has zero gradient: {norms}"


def test_a_mismatched_confidence_target_raises_instead_of_broadcasting():
    """(B, T, 1) vs (B, T) broadcasts to (B, T, T) rather than failing.

    That silent broadcast is how the term first went useless. It must raise.
    """
    from models.losses import _as_bt

    with pytest.raises(ValueError, match="must be"):
        _as_bt(torch.zeros(2, 3, 4, 5), "target")
    assert _as_bt(torch.zeros(2, 3, 1), "t").shape == (2, 3)
    assert _as_bt(torch.zeros(2, 3), "t").shape == (2, 3)
