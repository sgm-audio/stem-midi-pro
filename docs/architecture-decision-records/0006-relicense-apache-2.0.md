# ADR 0006: Relicense to Apache-2.0

**Status:** Accepted
**Date:** 2026-10-08
**Supersedes:** [ADR 0001 — Polyform Small Business](0001-license-polyform-small-business.md)

## Context

ADR 0001 chose the Polyform Small Business License 1.0.0 to enable
indie-friendly pricing with a paid tier for larger companies. Two things
changed since that decision:

1. **Upstream moved first.** `origin/main` settled on Apache-2.0 through
   four license-only commits (net effect: `a31911f Revert license to
   Apache-2.0 (contains third-party Apache-2.0 code)`). Keeping PolyForm
   locally would create a permanent fork in licensing between the local
   tree and the canonical remote.
2. **Third-party licensing friction.** The core dependency `mamba-ssm` is
   Apache-2.0. Shipping a *more restrictive* PolyForm wrapper around
   Apache-2.0 code complicates distribution (PolyForm is source-available,
   not OSI-approved, and adds revenue/employee thresholds that conflict
   with downstream redistribution of Apache-2.0 components).

## Decision

Adopt the **Apache License 2.0** for the codebase, the trained model
weights, and the research code under `/research/` (vendored upstream code
under `research/mamba-ssm-reference/` keeps its own Apache-2.0 license).

Concretely:

- `/LICENSE` is the full Apache-2.0 text (identical to upstream's).
- All `.py` SPDX headers read `# SPDX-License-Identifier: Apache-2.0`.
- `pyproject.toml` declares `license = { text = "Apache-2.0" }`.
- `LICENSE_NOTICE.md` summarizes Apache-2.0 in plain English.
- `NOTICE` continues to carry third-party attributions.
- ADR 0001 and TODO Section 5 are marked superseded (kept for history).

## Consequences

- Simpler compliance story: one permissive license for the project and its
  Apache-2.0 dependencies.
- No paid-license revenue path; the indie-friendly positioning shifts from
  "free under thresholds" to "free for everyone, forever".
- OSI-approved → distributable in Linux distros and downstream products.
- Generated output claims in `user_content/` no longer assert licensing of
  user audio beyond what the input rights already were.
- Any PolyForm-era distributions already made remain under the terms they
  were received under; this change applies going forward.
