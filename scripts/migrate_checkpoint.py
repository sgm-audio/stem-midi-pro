#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Migrate a checkpoint written before the pre-norm residual stack (C-1).

Usage::

    python3 scripts/migrate_checkpoint.py old.pt new.pt --config configs/model_config.yaml

The separator and transcriber gained ``block_norms.*`` / ``final_norm.*``
parameters when the Mamba stacks were changed to ``h = h + mix(norm(h))`` plus a
final LayerNorm. A checkpoint from before that change therefore misses exactly
those tensors; this script adds them with the identity initialisation (weight 1,
bias 0) and rewrites the file, so the result loads with ``strict=True``.

Nothing else is touched: a checkpoint that is missing or has mismatched *other*
keys is rejected, because that means a different architecture, not an upgrade.
Only run inference with the migrated weights if you accept that the new norms
start at identity; fine-tuning is what makes them meaningful.
"""

from __future__ import annotations

import argparse
import sys

import torch


def _extract(ckpt: object) -> tuple[dict, str | None]:
    """Return (state_dict, wrapper_key) from a checkpoint object."""
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key], key
        return ckpt, None
    raise SystemExit(f"Unsupported checkpoint type: {type(ckpt).__name__}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="checkpoint to migrate")
    parser.add_argument("destination", help="where to write the migrated checkpoint")
    parser.add_argument(
        "--config", required=True, help="model config yaml (defines the architecture)"
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="also fill *any* missing key with zeros (dangerous; off by default)",
    )
    args = parser.parse_args(argv)

    import yaml

    from main import StemMidiModel
    from models.checkpoint_compat import fill_identity_norms, is_legacy_state_dict

    with open(args.config) as f:
        config = yaml.safe_load(f)
    model = StemMidiModel(config)

    ckpt = torch.load(args.source, map_location="cpu", weights_only=True)
    state_dict, wrapper = _extract(ckpt)

    if not is_legacy_state_dict(state_dict, model):
        if not args.allow_missing:
            print(
                "Nothing to migrate: the checkpoint already matches the current "
                "architecture (or it is broken in a way this tool will not guess at).",
                file=sys.stderr,
            )
            return 1
        reference = model.state_dict()
        for key, value in reference.items():
            if key not in state_dict:
                state_dict[key] = torch.zeros_like(value)
                print(f"filled {key} with zeros")
    else:
        added = fill_identity_norms(state_dict, model)
        print(f"filled {len(added)} identity LayerNorm parameter(s)")

    model.load_state_dict(state_dict)  # strict: proves the migration worked

    if wrapper is None:
        out = state_dict
    else:
        out = dict(ckpt)
        out[wrapper] = state_dict
    torch.save(out, args.destination)
    print(f"wrote {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
