# Remediation plan — AUDIT-test-suite.md open defects

**Branch:** `arena/2fcfcea6-stem-midi-pro` (base `f3888a5`)
**Date:** 2026-11-17

`AUDIT-test-suite.md` locks 40 open defects as strict-xfail markers; `TODO.md` §18.1
lists them. This file records which ones I am fixing, the order, and what I am
deliberately leaving open with a reason.

Baseline on this branch before any change: **3 failed, 270 passed, 2 skipped,
42 xfailed**. The 3 failures are pre-existing and all trace to my own PR #17, which
landed in `f3888a5` against a tree whose gate tests predate the §9 change:

| Failing test | Cause |
|---|---|
| `test_audio_edge_cases.py::test_hard_clipped_input_is_flagged` | `_detect_artifacts` became an instance method calling `self._stems_are_silent`, but the audit suite calls it unbound (`_detect_artifacts(None, g, b)`). Regression from §4. |
| `test_audio_edge_cases.py::test_correlated_stems_are_not_phase_cancellation` | same `AttributeError` |
| `test_quality_gate_adversarial.py::test_thresholds_come_from_the_config_not_from_hardcoded_defaults` | §9 removed the proxy from the tier decision, so `min_centroid_separation_db` no longer re-tiers. |

## Wave 1 — my own regressions and the P0 delivery bugs

| ID | Sev | Fix |
|---|---|---|
| — | reg | `_detect_artifacts` back to a `staticmethod` (pure function of the two stems). |
| — | reg | Resolve the config-vs-§9 contradiction: **keep the configured threshold active** and make it robust instead of removing it (see below). |
| **ZIP-1** | P0 | `sf.write(..., subtype="FLOAT")` in `create_response_zip` — float32 stems are currently quantised to PCM_16 and hard-clipped at ±1.0 *in the delivery layer*. |
| **JSON-1** | P0 | Serialise `processing_report.json` with `allow_nan=False` after sanitising non-finite values to `null`. Today a NaN metric emits bare `NaN`, which `JSON.parse` rejects — the whole download is unparseable. |
| **IO-2** | P1 | `open_and_validate` rejects a 0-frame WAV with a 400 instead of letting torch raise `Expected 2D or 3D … got [1,1,0]` as a 500. |

### The §9 decision (why not "delete the knob")

My §9 change removed the proxy from the tier decision entirely. That is defensible in
isolation, but this tree ships `min_centroid_separation_db: 20.0` in all three configs
and the audit's gate test asserts that changing it changes the tier. Leaving it inert
means a documented knob does nothing — the exact class of bug the audit is about.

Revised rule, which satisfies both:

* The **proxy is still the wrong thing to gate on** (one FFT bin ⇒ 20.00 dB), so it must
  not be the *only* gate. Confidence + `low_confidence_notes == 0` + no artifact flags
  remain required, unchanged from §9.
* The configured threshold is **honoured when present and finite**, so an operator who
  sets 10.0 or 25.0 gets what they asked for, and the shipped yaml is not a lie.
  A `None`/absent/NaN threshold disables that half of the test (documented).

## Wave 2 — the remaining P0 correctness bugs

| ID | Sev | Fix |
|---|---|---|
| **MIDI-5** | P0 | `create_response_zip` passes the model's `hop_length`/`sample_rate` to the writer instead of module defaults. |
| **CKPT-1** | P0 | `load_from_checkpoint` accepts `state_dict` *and* `model_state_dict`, errors on neither. |
| **ALIGN-1** | P0 | `stem_energy` pooled over the hop, not per sample. |
| **GATE-3** | P0 | `ProcessingReport` carries `note_count`; a 0-note render can never be STUDIO/`direct_download`. |
| **CPU-1/2/3** | P0 | Long input routed through the chunked path. |

## Wave 3 — P1s with small, well-specified fixes

`MIDI-4` (round not truncate), `GATE-2/REPORT-1`, `SHIM-1`, `FAKE-1`,
`DOC-1`/`DOC-2`, `GATE-1/ART-1`, `IO-4`, `BATCH-1`, `REPRO-1`.

## Deliberately left open (and why)

| ID | Why not now |
|---|---|
| **TRANS-1** (content gate) | Needs a no-note logit and retraining; a threshold on the untrained head would be a guess. The density half is already fixed by H-6. |
| **CONF-1** (confidence calibration) | Requires trained weights; calibrating an untrained head is meaningless. This is the audit's own #1 residual gap. |
| **GATE-4** (separation metric) | Needs a reference-free separation measure (mask sum / residual energy / MR-STFT) and a corpus to set a threshold against. Picking a new formula without data would repeat the original mistake. |
| **SEP-1** (stems sum to mix) | Architectural: requires `residual = mix − guitar − bass`, which changes what the residual stem *is* and invalidates checkpoints/training. |
| **CPU-1/2/3** if chunking changes numbers | Streaming alters the render; needs a quality comparison the suite does not yet have. Would fix only if it is output-preserving. |
| **DOC-1/DOC-2** | Trivial and safe — done in wave 3. |
| **PERF-1** | Test-only rewrite; lower value than the product bugs. |
| **EXPR-1** | Either carry expression into the SMF or delete the docs claim. Needs a product decision (do we ship expression?), not a unilateral call. |

## Verification standard

For each fix: the locking `known_bug` marker is deleted, the test must pass, and the
suite must not regress. Every marker reason string already contains the measurement, so
the before/after is recorded by construction.
