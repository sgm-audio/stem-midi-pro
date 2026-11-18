# SPDX-License-Identifier: Apache-2.0
"""Load checkpoints written before the pre-norm residual stack (C-1 fallout).

The fix for C-1 wrapped every Mamba mixer in ``h = h + mix(norm(h))`` and added
a final ``LayerNorm`` to both the separator and the transcriber. That added two
parameter groups to ``state_dict`` — ``block_norms.*.{weight,bias}`` and
``final_norm.{weight,bias}`` — so every checkpoint saved before the change fails
to load, with a wall of "missing key(s)" lines that says nothing about what to
do about it.

``load_state_dict_compat`` distinguishes the two cases that matter:

* **Nothing but the new norms is missing** -> the old network is a strict subset
  of the new one, and the LayerNorms have exactly one sane initialisation
  (identity: weight 1, bias 0, which makes ``h + mix(norm(h))`` reduce to
  ``h + mix(h)`` for the first step, i.e. the old behaviour). The missing
  entries are filled in and the load is accepted, and the caller is told.
* **Anything else is missing, unexpected, or the wrong shape** -> raised, with
  the offending keys listed. A silent ``strict=False`` here would hide a real
  architecture mismatch, which is worse than the failure it papers over.

``scripts/migrate_checkpoint.py`` wraps this for the offline case so a training
pipeline can be repaired once instead of on every load.
"""

from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn as nn

# Parameter groups the C-1 fix introduced. Keys are matched on the *module*
# component, so they work for both a bare separator and the orchestrator's
# ``separator_backend.<module>.<group>.<idx>.weight`` layout.
_NEW_GROUP_RE = re.compile(r"(?:^|\.)(?:block_norms|final_norm)\.")
_IDENTITY_SUFFIXES = (".weight", ".bias")


def _is_new_norm_key(key: str) -> bool:
    """True for a parameter the pre-norm residual stack added."""
    return bool(_NEW_GROUP_RE.search(key))


def is_legacy_state_dict(state_dict: dict[str, Any], module: nn.Module) -> bool:
    """True when ``state_dict`` predates the pre-norm residual stack."""
    expected = set(module.state_dict())
    present = set(state_dict)
    missing = expected - present
    if not missing:
        return False
    new_keys = {k for k in missing if _is_new_norm_key(k)}
    return new_keys == missing and bool(new_keys)


def fill_identity_norms(state_dict: dict[str, Any], module: nn.Module) -> list[str]:
    """Add identity tensors for the new norm parameters, in place.

    Only keys that exist in ``module`` but not in ``state_dict`` are added, and
    only when they belong to the pre-norm groups; anything else is left for
    :func:`load_state_dict_compat` to complain about. Returns the added keys.
    """
    filled = state_dict
    added: list[str] = []
    current = module.state_dict()
    for key, value in current.items():
        if key in filled or not _is_new_norm_key(key):
            continue
        if not key.endswith(_IDENTITY_SUFFIXES):
            continue
        identity = torch.ones_like(value) if key.endswith(".weight") else torch.zeros_like(value)
        filled[key] = identity
        added.append(key)
    return added


def load_state_dict_compat(
    module: nn.Module,
    state_dict: dict[str, Any],
    *,
    allow_legacy: bool = True,
) -> dict[str, Any]:
    """Load ``state_dict`` into ``module``, filling identity norms if needed.

    Returns a small report: ``{"legacy": bool, "filled": [...], "missing": [...]}``
    so a caller (or the CLI) can print what happened. Raises ``RuntimeError``
    for any incompatibility that is not the known pre-norm upgrade, and
    ``RuntimeError`` (not a silent pass) when ``allow_legacy`` is False.
    """
    if not isinstance(state_dict, dict):
        raise TypeError(f"state_dict must be a mapping, got {type(state_dict).__name__}")
    if not state_dict:
        raise RuntimeError("Checkpoint state_dict is empty; refusing to load it")

    expected = module.state_dict()
    missing = [k for k in expected if k not in state_dict]
    unexpected = [k for k in state_dict if k not in expected]
    shape_mismatch = [
        f"{k}: checkpoint {tuple(state_dict[k].shape)} vs model {tuple(expected[k].shape)}"
        for k in expected
        if k in state_dict
        and hasattr(state_dict[k], "shape")
        and tuple(state_dict[k].shape) != tuple(expected[k].shape)
    ]

    filled: list[str] = []
    legacy = bool(missing) and all(_is_new_norm_key(k) for k in missing)
    if legacy and allow_legacy:
        if not all(k.endswith(_IDENTITY_SUFFIXES) for k in missing):
            raise RuntimeError(
                "Checkpoint is missing pre-norm parameters that cannot be filled "
                f"automatically: {sorted(missing)}"
            )
        state_dict = dict(state_dict)
        filled = fill_identity_norms(state_dict, module)
        missing = [k for k in expected if k not in state_dict]
    elif legacy:
        raise RuntimeError(
            "Checkpoint predates the pre-norm residual stack (missing "
            f"{len(missing)} parameter(s) such as {sorted(missing)[:3]}) and "
            "allow_legacy=False. Re-export it with scripts/migrate_checkpoint.py."
        )

    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "Checkpoint does not match the model: "
            f"{len(missing)} missing, {len(unexpected)} unexpected, "
            f"{len(shape_mismatch)} shape mismatch. "
            f"missing={sorted(missing)[:8]} unexpected={sorted(unexpected)[:8]} "
            f"shape={shape_mismatch[:8]}"
        )

    module.load_state_dict(state_dict, strict=True)
    return {"legacy": legacy, "filled": filled, "missing": []}
