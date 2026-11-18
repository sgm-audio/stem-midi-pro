#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify the API's honesty and error contracts (HONESTY-1, API-6.7).

Run: ``python3 scripts/verify_api.py``

Ported from PR #13, whose version asserted a hard gate (``load_model()`` must
refuse to serve a randomly-initialised network unless ``ALLOW_UNTRAINED_MODEL``
is set). This repo has not adopted that gate: it *discloses* provenance instead
— ``/model-info`` and ``processing_report.json`` carry ``weights_loaded`` and
``mamba_backend`` (see ``main.StemMidiModel``), so a client can tell a trained
render from an untrained one. The script checks what the code actually does, and
prints an explicit warning while the hard gate is still absent.

Checks:

* a model built without a checkpoint reports ``weights_loaded is False``, and
  ``load_from_checkpoint`` flips it to ``True``,
* OOM-shaped exceptions are recognised (the handler maps them to 503),
* a NaN/Inf metric cannot reach ``processing_report.json`` as a bare token.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, ".")

import torch
import yaml

import api
from main import StemMidiModel, load_from_checkpoint


def check_untrained_provenance() -> None:
    """A checkpoint-less model must say so; provenance must flip on load."""
    # model_config.cpu.yaml names demucs/basic_pitch (the [pretrained] extras),
    # which are not installed here; model_config.yaml is the built-in Mamba stack.
    with open("configs/model_config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = StemMidiModel(cfg).eval()
    print(
        f"untrained model : weights_loaded={model.weights_loaded} backend={model.mamba_backend!r}"
    )
    if not model.weights_loaded:
        print(
            "                  service still SERVES it: provenance is disclosed, and there is no "
            "hard gate (ALLOW_UNTRAINED_MODEL does not exist here)."
        )

    path = os.path.join(tempfile.mkdtemp(), "ckpt.pt")
    torch.save({"state_dict": model.state_dict()}, path)
    reloaded = load_from_checkpoint(path, "configs/model_config.yaml")
    print(f"after checkpoint: weights_loaded={reloaded.weights_loaded}")
    assert reloaded.weights_loaded is True, "load_from_checkpoint did not mark trained weights"

    # /model-info must expose the same two facts. Checked through the app, not by
    # reading source: a route that stops returning them is the failure mode.
    from fastapi.testclient import TestClient

    api.model = model
    api.model_config = cfg
    api._model_ready = True
    info = TestClient(api.app).get("/model-info")
    assert info.status_code == 200, info.text
    body = info.json()
    assert body["weights_loaded"] is False, body
    assert body["mamba_backend"] == model.mamba_backend, body
    print(f"/model-info     : weights_loaded={body['weights_loaded']} backend={body['backend']!r}")


def check_oom_mapping() -> None:
    """Allocator failures must be distinguishable from ordinary bugs."""
    handlers = api._oom_exceptions()
    print(f"oom handlers    : {[h.__name__ for h in handlers]}")
    assert MemoryError in handlers, "MemoryError is not mapped to a retryable 503"


def check_json_guard() -> None:
    """A non-finite metric must become null, not a bare NaN token."""
    from utils.quality_gates import ProcessingReport

    report = ProcessingReport(
        centroid_separation_db=float("nan"),
        phase_coherence=float("-inf"),
        avg_confidence=0.87,
        artifact_flags=["none"],
        low_confidence_notes=0,
    )
    payload = report.as_dict()
    text = json.dumps(payload, allow_nan=False)  # raises if a NaN slipped through
    print(f"json guard      : valid RFC 8259 ({len(text)} bytes)")
    print(f"                  invalid={payload.get('invalid')} metrics={payload.get('metrics')}")
    assert payload.get("invalid") is True, "a non-finite metric was not marked invalid"


if __name__ == "__main__":
    check_untrained_provenance()
    check_oom_mapping()
    check_json_guard()
    print("verify_api: OK")
