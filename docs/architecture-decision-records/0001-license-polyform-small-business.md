# ADR 0001: License — Polyform Small Business License 1.0.0

**Status:** **Superseded by [ADR 0006](0006-relicense-apache-2.0.md)** (2026-10-08)
**Date:** 2026-06

## Context

Stem+MIDI Pro's mission is to serve indie and Canadian creators with indie-friendly pricing, while still being able to charge larger organizations. Pure permissive licenses (MIT/Apache) allow free large-company commercial use; copyleft (GPL/AGPL) scares commercial adopters; proprietary hurts the indie mission.

## Decision

License the codebase and trained model weights under the **Polyform Small Business License 1.0.0**: free for individuals and for companies under $1M annual revenue AND under 50 employees; a paid license is required above either threshold. Full text at `/LICENSE`; plain-English summaries in `README.md` and `LICENSE_NOTICE.md`; generated outputs carry the same license, stated in the upload prompt (`user_content/rights_usage_prompt.md`). Contact: licensing@stem-midi-pro.example.
The Mamba-3 research code under `/research/` uses the same license, provided as-is without support.

## Consequences

- Indie users (the target market) pay nothing — mission aligned.
- Commercial licensing revenue remains possible above the thresholds.
- Requires all source files to carry an SPDX header (`PolyForm-Small-Business-1.0.0`) and explicit `NOTICE` attributions for third-party deps (mamba-ssm Apache-2.0, librosa ISC, PyTorch BSD-3-Clause, etc.).
- Not OSI-approved; we accept that trade-off for the revenue threshold mechanics.
