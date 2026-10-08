# ADR 0005: Mamba-3 Research Moved to `/research/`

**Status:** Accepted (TODO D-STRAT-1, R-1.1.1–1.1.12)
**Date:** 2026-06

## Context

Mamba-3 per-track processing was an experimental direction with its own model (`per_track_processor.py`), training script, losses, datasets, and tests. It is promising but not the shipping architecture, and its presence at the repo root made the production path ambiguous.

## Decision

All Mamba-3 code lives under `research/mamba3_per_track/` with its own `SETUP.md`, config, datasets module, and tests. It shares the repo's Apache-2.0 license but is **provided as-is with no support guarantees** and no API-stability promise. A top-level `research/README.md` indexes both research subdirectories.

## Consequences

- Production docs and onboarding no longer trip over experimental files.
- Research can iterate (breaking changes allowed) without touching `main.py` / `api.py`.
- Promotion of a research result to production is an explicit, reviewable move (a new ADR would be required).
- Contributors must add `sys.path` or editable installs to run research tests — acceptable overhead for isolation.
