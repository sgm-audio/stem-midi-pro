# ADR 0004: Consolidate on the v1 Production Path

**Status:** Accepted (TODO D-STRAT-1, Section 1)
**Date:** 2026-06

## Context

The repo contained two parallel implementations: the v1 path (`main.py`, `api.py`, `models/*`, `train.py`, `data/datasets.py`) and Mamba-3 experiment files at the root (`model.py`, `train_mamba3.py`, `utils/losses_mamba3.py`, a vendored `mamba/` clone). Having both at top level confused imports, tooling, and docs.

## Decision

The v1 path is the **only production path**. All Mamba-3 experiment code moves under `/research/mamba3_per_track/`; the vendored mamba-ssm clone moves to `/research/mamba-ssm-reference/`. Dataset classes were renamed to drop redundant qualifiers (`CanadianAudioDataset` → `AudioDataset`, `get_canadian_data_loaders` → `get_data_loaders`); the Canadian-artist mission lives in curation policy/config, not identifiers.

## Consequences

- One obvious place for each concern; CI, `check_syntax.py`, and `verify_structure.py` updated accordingly.
- Research code is clearly marked as-is and unsupported (see `research/README.md`).
- Old class names (`Slakh2100CADataset`, `MUSDBIndieDataset`) are retired everywhere including docs.
- Future Mamba-3 work proceeds under `/research/` without destabilizing production.
