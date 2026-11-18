# Remediation plan and outcome — `AUDIT-test-suite.md` open defects

**Branch:** `arena/2fcfcea6-stem-midi-pro` (base `f3888a5`)
**Date:** 2026-11-17

`AUDIT-test-suite.md` locks 40 open defects as strict-xfail markers; `TODO.md` §18.1
lists them. This file records which ones were fixed, in what order, and which are
deliberately left open with a reason.

## Result

| | Tests | Failed | xfailed |
|---|---|---|---|
| Baseline on this branch | 315 | **3** | 42 |
| Final | **344** | **0** | **6** |

**36 defects closed**, 6 deliberately left open (all architectural — see the table
at the end). `ruff check .` and `black --check .` are clean.

Every fix followed the same rule: the locking `known_bug` marker is deleted, the
test passes unmarked, and `TODO.md` §18.1 records the row as `[x]` with a
pass-today guard in `tests/test_audit_remediation_guards.py` naming the id.

The 3 baseline failures were all self-inflicted by PR #17 landing in `f3888a5`
against a tree whose gate tests predate its §9 change. They are listed here
because a reviewer should be able to see that the branch started red:

| Failing test | Cause |
|---|---|
| `test_hard_clipped_input_is_flagged` | `_detect_artifacts` had become an instance method calling `self._stems_are_silent`, but the audit suite calls it unbound with `None` as self. |
| `test_correlated_stems_are_not_phase_cancellation` | same `AttributeError` |
| `test_thresholds_come_from_the_config_not_from_hardcoded_defaults` | §9 removed the proxy from the tier decision, so `min_centroid_separation_db` no longer re-tiered anything. |

## Wave 1 — regressions and delivery-layer P0s

| ID | Sev | What was wrong → what it does now |
|---|---|---|
| — | reg | `_detect_artifacts` is a `staticmethod` again (a pure function of two stems, which is the contract the audit tests rely on). |
| — | reg | **§9 was reversed in part.** Removing the proxy outright made `min_centroid_separation_db: 20.0` — shipped in all three configs — inert. The threshold is active again but is no longer *sufficient*: STUDIO now needs confidence **and** `low_confidence_notes == 0` **and** the configured separation. Keeps §9's point (one FFT bin cannot buy a Studio badge) while the config still means something. |
| **ZIP-1** | P0 | `sf.write(..., subtype="FLOAT")`. Without it libsndfile picks PCM_16, so float32 stems were quantised to 1/32768 and anything outside [-1, 1] was hard-clipped *in the delivery layer* (measured: a 3.0 peak decoded as exactly 1.0). |
| **JSON-1** | P0 | `processing_report.json` is built from an explicit documented key set, non-finite floats become `null`, serialised with `allow_nan=False`. A NaN metric used to emit bare `NaN`, which `JSON.parse` rejects — the whole download was unreadable. |
| **MIDI-5** | P0 | `create_response_zip` forwards the pipeline's `hop_length`/`sample_rate` instead of letting the writer fall back to 48 kHz-era defaults (a 16 kHz/hop-128 model put every event ~2.75× off in time). `process_audio_file` now also reports its hop grid. |
| **IO-2** | P1 | A 0-frame WAV is refused in `open_and_validate` (400) instead of dying inside torch as `Expected 2D or 3D … got [1,1,0]` → HTTP 500 with internals in the body. |
| **IO-3** | P0 | Non-finite samples are refused in `open_and_validate` (for **every** caller, not just `load_native_mono`) **and** in `MambaSeparator.forward` (direct/training callers). An all-NaN upload used to return 200 with `artifact_flags=['none']` and deliver a constant −1.0 full-scale “stem”. |
| **IO-4** | P1 | Anti-phase stereo (L = −R) mean-downmixes to exact silence; a full-level input became an all-zero track and the user got an empty render blamed on “complex material”. The downmix RMS is now compared against the per-channel RMS and the load is refused below 5 %. Measured: identical channels / in-phase 0.6× / hard-panned single channel all still load; only true anti-phase is rejected. |
| **DOC-1** | P1 | Confidence is **CC#16**, not CC#127, in `USER_GUIDE.md` (×3), `API_DOCUMENTATION.md` (×2), `DEVELOPMENT_GUIDE.md` and `ARCHITECTURE.md`. CC#120–127 are MIDI Channel Mode messages; `MIDI_CONFIDENCE_CC=127` is documented as the opt-in. Users following `USER_GUIDE.md` step 3 were filtering a controller that is never written. |
| **DOC-2** | P2 | The Processing Report Schema table documents `backend` and `invalid`, and states that floats may be `null` rather than NaN. |

### The §9 decision (why the knob came back)

My §9 change removed the proxy from the tier decision entirely. That is defensible
in isolation, but this tree ships `min_centroid_separation_db: 20.0` in all three
configs and the audit's gate test asserts that changing it changes the tier.
Leaving it inert means a documented knob does nothing — the exact class of bug the
audit is about.

Revised rule, which satisfies both:

* The **proxy is still the wrong thing to gate on** (one FFT bin ⇒ 20.00 dB), so it
  must not be the *only* gate. Confidence + `low_confidence_notes == 0` + no
  artifact flags remain required, unchanged from §9.
* The configured threshold is **honoured when present and finite**, so an operator
  who sets 10.0 or 25.0 gets what they asked for and the shipped yaml is not a lie.
  A `None`/absent/NaN threshold disables that half of the test (documented).

## Wave 2 — correctness P0s

| ID | Sev | What was wrong → what it does now |
|---|---|---|
| **GATE-3** | P0 | `ProcessingReport` carries `note_count`, and `note_count == 0` is COMPLEX regardless of metrics. 0 notes + confidence 0.95 + 25 dB used to route to `direct_download` / “✨ Studio Quality”, i.e. the user paid for an empty MIDI. `None` (not supplied) leaves routing unchanged, so existing call sites keep working. |
| **CKPT-1** | P0 | `load_from_checkpoint` accepts `state_dict` **and** `model_state_dict` — the key `scripts/export_fine_tuned.py`, `fine_tune_demucs.py` and `verify_cpu_export.py` all save under, so every fine-tune export a user produced by following the docs raised Missing/Unexpected keys. A dict with neither is rejected with a clear message instead of being mistaken for a state dict. |
| **ALIGN-1** | P0 | **Already fixed** by PR #17's M-4 work: `_frame_energy` pools per-sample energy onto the hop grid and both injector call sites use it. The marker described code that no longer existed, and the test additionally indexed `midi_metadata["midi_events"]` when the real shape is per-stem. Marker removed, fixture corrected, no product change needed. |
| **REPORT-1** | P1 | The batched report read `metrics[0, 0]` under an “Average over batch” comment, stamping item 1's garbage with item 0's numbers. Now `torch.nanmean` across the batch — `nanmean` so one non-finite item does not poison the whole report. |
| **CPU-1** | P0 | **Already fixed, assertion was stale.** The scan runs in `Mamba.SCAN_CHUNK` (256-frame) windows, but the test modelled the pre-chunking `O(L)` shape and kept reporting 162 GB for 5 minutes. Measured on the real block at the shipped separator shape: 493 MB @ 256 frames → 561 MB @ 4096 frames — **flat across a 16× length increase**. The old “linear in song length” test asserted the bug; it is now split into “linear inside one window” and “constant past it”. |
| **CPU-2** | P0 | `process_audio_file` ignored `audio.chunk_duration_sec` (shipped as 2.0, sold in the README) and ran one monolithic forward. It now routes input longer than one chunk through `process_audio_streaming`; anything at or below one chunk keeps the bit-exact single-forward path. **Verified output-preserving before doing it**: 6 s at 2 s chunks vs one pass → identical lengths, 9e-4 / 7e-4 relative error (the H-4 crossfade seams). |

## Wave 3 — smaller, well-specified P1s

| ID | What was wrong → what it does now |
|---|---|
| **MIDI-4** | `int(conf * 127)` truncated, so 0.999 → 126 and a DAW reading the top of the range as “fully confident” never saw it. Both call sites now `round()` — the injector **and** the writer fallback, which is the trap the audit flagged (rounding only the writer changes nothing for a model-produced event because the injector has already written the integer). Four legacy tests encoded `int(0.5*127) == 63` as the contract and were corrected. |
| **GATE-2** | `low_confidence_count` was `(confidence < thr).sum()` over the confidence *tensor* — a count of analysis **frames** — but it is published to users and compared against tier thresholds as “low confidence notes”. One second is ~86 frames, so the number grew with file length and meant nothing. It now counts note events. |
| **ART-1** | The “noise floor” test was `mean(stem**2) > 0.1`, a **loudness** test: anything above about −10 dBFS RMS raised the flag, and any artifact flag forces COMPLEX before the thresholds are consulted — so STUDIO was unreachable for real music and a clean 0.5-amplitude sine scored worse than digital silence. It is now a high-band energy fraction (above ¼ Nyquist / total). Measured: clean sine at 0.5 and 0.9 → 0.000, `['none']`; white noise at 0.05 → 0.744, `['noise_floor']`. Level no longer participates, so a quiet broadband floor is still caught. |
| **SHIM-1** | The pure-torch fallback's `**_ignored` swallowed anything, so a config carrying `causal_conv1d_impl=`/`selective_scan_impl=` — the exact mechanism that hid C-2.1 on CPU — constructed happily and only failed on a CUDA box. It now mirrors the real `Mamba.__init__` signature from `research/mamba-ssm-reference/`: every upstream kwarg is accepted, anything else raises `TypeError` naming the offender. **The audit's test was wrong here**: it listed `use_fast_path` as rejectable, but that is a real mamba-ssm parameter, so it is now covered by an accept-every-upstream-kwarg test. Rejecting a valid parameter would have been a worse bug than the one being fixed. |
| **FAKE-1** | `FakeSeparatorBackend` returned 12.0 / 0.5 for *any* input, so gate tests through it asserted a fixture — and 12.0 is below the shipped `min_centroid_separation_db: 20.0`, pinning the fake path to COMPLEX and making the studio/draft branches untestable. Metrics are now computed from the two stems (silence 25.0/0.0 vs a tone 30.0/1.0). |
| **REPRO-1** | Eval-time MC-dropout drew masks from the **global** RNG, so the same file gave different confidence on every retry (measured max |Δconf| = 0.0065) — enough to move a CC, flip `needs_review` at the threshold, and change the quality tier between two runs of the same paid request. The estimate now runs inside `torch.random.fork_rng` with a fixed seed: verified 0.0 difference across calls while still differing from a plain eval forward, so the 5-mask ensemble C-2.10 relies on is genuinely non-degenerate. |
| **MIDI-7** | **Already fixed** — the reader applies the tempo map per tick. Measured on the case the reason string claimed was broken: frame 215 (2.496 s), not the 120-BPM answer of frame 129. The *test* was broken: `_write_midi` appended every extra tempo event immediately after the initial one, so `time=480` was consumed as a rest before the first note and pushed the second note off the grid. Fixed to merge events by absolute tick. |
| **BATCH-1** | `_logits_to_midi` flattens every item's events into one list with raw per-item frame indices, so a B=2 call returned a single MIDI containing two songs' notes interleaved and unattributable. It now raises when the batch actually produced events. **The guard's placement took three attempts** because BATCH-1 and REPORT-1 genuinely conflict: REPORT-1 needs a B=2 call to build a report, and the first two placements broke it. It sits after peak-picking, where events exist to concatenate — a batched call that produced no onsets has nothing to merge. |
| **PERF-1** | The only memory/latency gate proved nothing: it built the 64-wide toy fixture (12× smaller per layer than the shipped 768×12), forwarded 1 s, and its psutil branch read *current* RSS while its `resource` fallback read the *peak*. It now runs the shipped config for 8 s, measures only true peak RSS, and a second gate covers the chunked path. Budget 2500 → 4096 MB is a **measurement**, not a concession: interpreter+torch ≈ 410 MB, model build ≈ 663 MB, chunked 6 s pipeline ≈ 2260 MB — 2500 was below what the real pipeline needs. |

## Deliberately left open (6)

All six are blocked on something this branch cannot supply, not on effort.

| ID | Why not now |
|---|---|
| **TRANS-1** (content gate) | Needs a no-note logit and retraining. A threshold on the untrained head would be a guess. The density half is already fixed by H-6. |
| **CONF-1** (confidence calibration) | Needs trained weights. The audit's own measurement — 0.516 for a clipped square vs 0.300 for clean music — is a statement about an **untrained head**; recalibrating it would be fitting noise. This is the audit's #1 residual gap and it stays open until there are weights. |
| **GATE-4** (separation metric) | The proxy (`clamp(10·log10(|Δcentroid|)+20, 0, 30)`) is not a separation metric and *rewards* distortion. Replacing it needs a reference-free measure **and a corpus** to set a threshold against; picking a formula without data is how the original mistake happened. |
| **SEP-1** (stems sum to mix) | Architectural: requires `residual = mix − guitar − bass`, which changes what the residual stem *is* and invalidates checkpoints and training. |
| **CPU-3** (cross-chunk SSM state) | Carrying SSM state across chunks changes what the model outputs, so it needs its own quality validation. Note that CPU-2 was done **because** chunking was measured output-preserving (9e-4 relative error); CPU-3 does not have that property. |
| **EXPR-1** (expression head) | Either carry expression into the SMF or delete the docs claim. That is a product decision (do we ship expression?), not a unilateral call. |

## Things the audit got wrong, recorded so they are not “fixed” later

* **ALIGN-1, MIDI-7 and CPU-1 were not bugs in the product** — all three were fixed
  code carrying a stale marker, and in each case the *test* was the defect (wrong
  dict shape, a fixture that mis-timed its own notes, a formula modelling
  pre-chunking behaviour). Checking the code before trusting the reason string is
  what avoided three unnecessary rewrites.
* **SHIM-1's test was wrong** about `use_fast_path` (a real upstream kwarg).
* **REPORT-1 and BATCH-1 conflict** at the guard level; resolving it required
  putting the batch guard where events are concatenated, not at the entry point.
* **Two audit tests asserted frame counts as note counts** and **four more**
  asserted truncation as the CC contract — those encoded the bug and were rewritten.


## Merge with `main` (`4b6daf1`) — what changed underneath this branch

While this branch was in flight, `main` gained three substantial parallel
changesets (`#15`, `#16` and a reconciliation merge) that rewrote several of the
same files. The merge was done by hand, file by file, and the interesting part is
**which of my fixes were already solved upstream** — checked by reading main's
code, not by assuming either side:

| Finding | Outcome after the merge |
|---|---|
| **MIDI-5** | Already fixed upstream (hop/sample_rate threaded in `api.py`). Took theirs; dropped mine. |
| **CKPT-1** | Already fixed upstream. Merged my clearer error for a dict carrying neither key on top. |
| **GATE-2** | Already fixed upstream (`note_confidences` indexed by note). Took theirs. |
| **GATE-3** | Not on main. `note_count` + empty-render rule kept. |
| **REPORT-1** | Fixed on both sides independently. Kept upstream's structure with my `nanmean` (so one non-finite item cannot poison the whole report). |
| **ZIP-1** | Not on main — still writing no subtype. Kept mine. |
| **JSON-1** | **Still live on main**: `json.dumps(report_json, indent=2)` with no `allow_nan=False`. Kept mine, applied to upstream's expanded schema. |
| **MIDI-4** | Upstream moved confidence to Polyphonic Key Pressure but reintroduced `int()` in *both* `confidence_injector.py` and `_clamp_midi`. Re-applied rounding on both. |
| **DOC-1** | **Superseded**: upstream removed the confidence CC entirely (correctly — CC#120-127 are Channel Mode). My CC#16 fix described a design that no longer exists; took theirs and rewrote the guard to assert the polytouch lane. |
| **DOC-2** | Still needed: upstream's schema table omitted `invalid`/`backend` that their own `api.py` emits. Added. |
| **ART-1** | Not on main — still `mean(stem**2) > 0.1`. Kept my high-band energy fraction. |
| **SHIM-1, FAKE-1, REPRO-1, IO-2, IO-3, IO-4, BATCH-1, PERF-1, GATE-4** | Not on main. All kept. |
| **§9 knob** | **Upstream made a different, documented decision**: renamed to `min_separation_proxy`, marked it "deprecated metadata only", and removed it from the tier. That is not the inert-knob bug (the name is honest and the docs say so). Took theirs and dropped my re-activation. |

### The 60 s memory probe: a real defect hiding behind an "environment note"

An earlier revision of this section blamed the sandbox for
`test_one_minute_upload_fits_in_2gb` stalling after ~400 tests. That was wrong,
and the correction is worth recording because the *shape* of the mistake is
general: a test that fails only inside a long run looks like an environment
problem, so the probe got deselected instead of read.

CI (torch 2.15, 16 GB) failed it outright, together with three other memory
tests that all reported ~14 GB — the same number, which is impossible for
independent tests and was the clue. Two separate defects:

**1. `process_audio_streaming` transcribed outside `inference_mode`.** The chunk
loop was correctly wrapped, but the block closed *before* `_transcribe` ran on
the recombined full-length stem, so the transcribe call built an autograd graph:
every mamba activation plus the `(1, T, 128)` pitch logits stayed reachable via
`grad_fn`. On `main` this was invisible because `process_audio_file` kept
`_finalize` **inside** `inference_mode`; the streaming method — a near-copy —
did not. CPU-2 (honouring `chunk_duration_sec` in `process_audio_file`) routed
long files into that latent bug and exposed it. Measured before/after on 60 s at
44.1 kHz:

| | peak RSS | time |
|---|---|---|
| transcribe inside `inference_mode` | **898 MB** | ~25 s |
| outside (the bug) | **OOM-killed at 4 GB** | killed |

A control run showed the severity of the retention: one grad-enabled forward of
the 60 s stem was still running after 188 s (versus ~2 s inside
`inference_mode`), because autograd also disables the fast path used for the
mamba scan. The chunk loop itself was never the problem — it measured flat at
774 MB across all 30 chunks.

The new guard asserts the *structural* signature rather than a RSS threshold:
inside `inference_mode` a tensor reports `is_inference=True` /
`requires_grad=False`, while the pre-fix call site produced
`requires_grad=True, is_inference=False, grad_fn=ViewBackward0`. Verified by
reproducing that exact control flow.

**2. `ru_maxrss` is a monotonic process-wide high-water mark.** Every memory
test read the same never-decreasing number, so once any earlier test reached
14 GB, all four memory gates inherited it and each blamed its own code path
("60 s upload grew peak RSS by 12434 MB" was not about the upload). Fixed with
`tests/memprobe.py`, which reads `VmHWM` from `/proc/self/status` and resets it
through `/proc/self/clear_refs`, so a test measures *its own* peak; non-Linux
spots skip rather than assert on a monotonic value.

After both fixes the whole suite, including the 60 s probe, runs in **one
process**: 429 passed, 4 skipped, 6 xfailed, 0 failed in 42 s — versus an OOM
kill at the same point before. CI's exact command
(`pytest -m "not cuda" --strict-deps --fail-on-skip`) passes in 44 s. The probe
is no longer deselected anywhere.


## Verification standard

For each fix: the locking `known_bug` marker is deleted, the test passes unmarked,
`TODO.md` §18.1 keeps the id as `[x]` (the id is the join key between the commit,
the marker's reason string and the guard), and
`tests/test_audit_remediation_guards.py` holds a pass-today guard naming that id.
Every marker's reason string already contains the measurement, so the before/after
is recorded by construction.
