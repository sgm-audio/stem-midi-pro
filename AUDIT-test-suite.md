# Test-Suite Audit — Stem+MIDI Pro

**Scope:** `tests/` (all 24 pre-existing files), `pytest.ini`, `Makefile`,
`.github/workflows/ci.yml`, `.codecov.yml`, and every production module they claim to
cover: `main.py`, `api.py`, `audio_io.py`, `utils/{midi,midi_targets,quality_gates,confidence_injector,audio_io}.py`,
`models/{mamba_separator,mamba_transcriber,_mamba_compat}.py`, `models/backends/*`,
`configs/*.yaml`, `data/datasets.py`, `losses.py`, `train.py`, `scripts/*`, `demo.py`.
**Date:** 2026-11-17 · **Branch:** `arena/265e3d3e-stem-midi-pro` · **Author:** Arena agent (skeptical test-engineer brief)

> **Current-branch reconciliation note (PR #15):** This document is a historical audit
> baseline, not a current capability claim. The merged branch retains its strict honesty
> layer and known-bug inventory, but its MIDI output contract intentionally supersedes
> the audit's legacy CC#16 discussion: confidence is canonical in JSON and serialized
> as Polyphonic Key Pressure; CC#127 is not emitted. Current test totals and evidence
> are recorded in `VERIFICATION_MATRIX.md`.

---

## 0. Verdict

The pre-existing suite is **not a safety net for anything a user touches**. It is a
well-built harness around a tiny synthetic fixture: 2 166 lines, 98 tests, and every
test passes — on a model with **no trained weights**, on **sine waves**, with the
real Mamba modules reached in only two places, with **zero** assertions about the
*MIDI content* of the two headline deliverables, and with a CI step that cannot fail
(`continue-on-error: true`).

Its own greenness is the bug. Measured proof, now locked by strict xfails:

| The suite claims to cover | What it actually catches |
| --- | --- |
| "MIDI generation" (`test_midi_generation.py`) | note **counts**, not pitches/timings/durations. The writer's tick math is off by **60×** and no test noticed |
| "quality gates" (`test_quality_gates.py`) | enum values and routing dict keys. A 0-note render is `STUDIO`; a square wave scores 24.75 dB of "separation" |
| "real Mamba path" (`test_mamba_separator.py`) | yes, 8192 samples of white noise, shape-only |
| "checkpoint loading" (`test_checkpoint_load.py`) | gated behind `importorskip("mamba_ssm")` — the only test that could see the `state_dict` vs `model_state_dict` mismatch, skipped on every CPU box |
| "dataset loaders" (`test_dataset_loaders.py`) | gated behind `importorskip("torchaudio")`, a package **nothing in the repo imports** |
| "backends" (`test_backends.py`) | `StemMidiModel` never calls the backend factory, so the config key `backends:` is dead and the fake-backend tests exercise the real modules |
| "API integration" | `/process` has **never been posted to** in the suite; the ZIP the customer downloads was never opened |
| "low-memory CPU target" | one RSS sample after a 1-second forward on `mock_config`, against a 2 500 MB budget |

After this audit: **259 tests — 192 passing, 65 strict-xfails carrying the bug
report in their reason string, 1 legitimately skipped, 0 failures**, and the CI
correctness gate can actually go red.

```
$ make test-cpu            # pytest -m "not cuda" --strict-deps --fail-on-skip
192 passed, 1 skipped, 1 deselected, 65 xfailed in 13.5s
```
(against `6ab1a1e`). After merging PR #11 the same command reads:
```
$ pytest -q -m "not cuda" --strict-deps --fail-on-skip
254 passed, 1 skipped, 1 deselected, 40 xfailed, exit 0 in 14.7s
```
40 of the original 65 locks survive, 25 became XPASS failures because #11 fixed the
underlying defects, and 11 of the 33 TODO IDs are now marked resolved. See §0.1.
```

---

## 0.1 Reconciliation with `main` after the audit (PR #11 merged into this branch)

The audit and its suite were written against `6ab1a1e`. `origin/main` has since landed
PR #11 (`00cd285`), which fixes a third of what I indicted. This section is the diff
between "what the audit said" and "what the suite locks today", written because the
numbers in §0-§5 are otherwise misleading to a reader who starts here.

**Fixed upstream, markers deleted, tests kept as pass-today guards** (each of these
turned into an XPASS-strict failure on the merged tree first — the mechanism this audit
proposed, doing its job): `MIDI-1` (tick math: `ticks_per_beat * 1e6 / tempo_us`),
`MIDI-2` (per-note timelines + sort by absolute tick), `MIDI-3` (confidence now rides
immediately *with* its `note_on`, and moved off CC#127 onto CC#16 = General Purpose 1 —
which the merge then superseded: a channel CC cannot differ between simultaneous notes,
so the shipped lane is per-note Polyphonic Key Pressure and confidence writes no CC),
`MIDI-6` (reader applies the parsed tempo), `STEM-1` (H-8 transcribes each stem,
`midi_metadata = {guitar, bass, summary}`), `API-1` (`_stem_midi_data` selects per stem),
`IO-1` (`models/mamba_separator.py:101-108` validates rank/channels/length and raises
`ValueError` before `torch.stft` can die), `BACK-1`/`BACK-2` (`main` builds backends from
`cfg["backends"]` via `models.backends.factory` and consumes the `SeparateOut` /
`TranscribeOut` dicts), `CFG-2` (`mock_config` is 512/128, so the real separator runs on
the fast config), and half of `CFG-3`.

**New defect found while reconciling, fixed on this branch:** `CFG-3`'s other half.
`main` wired `training.gradient_checkpointing` into `models/_mamba_compat.py` but called
`torch.utils.checkpoint.checkpoint(...)` without importing the submodule — `import torch`
does not bind it — so the CPU memory mitigation raised `AttributeError` on first use and
`main`'s own `tests/test_review_fixes.py::test_gradient_checkpointing_training_path` was
reddening merged main while its CI step is `continue-on-error`. This branch adds the one
missing import and guards it with
`tests/test_memory_ceiling.py::test_gradient_checkpointing_call_resolves_in_a_fresh_interpreter`,
which deliberately probes in a **subprocess**: inside a shared session, `torch.utils.checkpoint`
becomes importable for the whole run as soon as *any* test imports it, so an in-process
probe passes depending on test order. Verified both ways: commenting the import out reddens
the guard, restoring it greens it. This is the only non-test file this audit touches; drop
that commit if you want the fix to travel separately.

**Two IDs added:** `MIDI-7` (the reader honours one tempo for the whole file and ignores a
`set_tempo` in the middle of a tempo map — measured: a note at tick 1440 of a 120→60 BPM
file lands on frame 129 instead of 215, a one-second drift) and `EXPR-1` (the expression
head is discarded and no `pitch_bend`/CC1 ever reaches the SMF, while the README promises
per-note vibrato and slide).

**Two of the audit's own claims were wrong, and are retracted here.** Both were caught by
the merged tree, not by me in advance, which is the argument for running the suite against
`main` rather than reasoning about it:

1. *"parsed note onsets must be non-decreasing"* — not a product bug. The writer resolves
   same-pitch re-articulation as **release-then-strike inside one tick**, so a re-parsed
   note list is in *closing* order, not striking order, and multi-track files interleave.
   `assert ons == sorted(ons)` and `parsed["notes"][1]` ("the second note") are both
   meaningless against this design; the guards now compare onsets as a multiset and select
   notes by pitch.
2. *"PPQ propagation must be exact to 1e-3 s"* — also not a bug. 96-PPQ quantisation moves a
   0.232292 s onset to 0.234375 s, i.e. 2.08 ms = 0.2 ticks. A tick-grid tolerance
   (`<= 2` ticks at the file's PPQ) is the musically honest criterion; "fixing" the writer
   to beat 1e-3 would be *wrong*, since SMF ticks are integers.

One refinement the audit should have made: `MIDI-4` is not a one-line `int() -> round()` in
`utils/midi.py`. `models/confidence_injector.py:72` computes `int(conf_score * 127)` and
hands the writer an already-truncated `confidence_cc_value`, so rounding only the writer
changes nothing for a model-produced event. The `xfail` reason now names both call sites.


### …and PR #14 landed during the reconciliation (same mechanism, second round)

`ee8dd62` renamed `ProcessingReport.si_sdr` → `centroid_separation_db` (C-2.6, with
`min_si_sdr` kept as a legacy config key) and added H-6 onset peak-picking. The first run
against it produced 12 failures here, all of them *this suite's* fault, and one of them a
finding worth the merge: `api.py`'s `processing_report.json` ships a `backend` key
(M-7) that the "Processing Report Schema" table does not document (**DOC-2**), and
`USER_GUIDE.md` / `API_DOCUMENTATION.md` still instruct users to filter notes on
**CC#127** although #11 moved confidence to CC#16 (**DOC-1**). Both were locked; DOC-1 is
now **resolved**, and its lock was replaced by
`tests/test_api_zip.py::test_the_documented_confidence_lane_is_the_one_the_writer_emits`,
which reads a rendered file and fails if any guide names a CC the writer does not emit.
DOC-2 remains locked.

Adaptations: `tests/doubles.py`'s `StubReport` and the report fixtures in
`tests/test_api_zip.py` / `tests/test_quality_gate_adversarial.py` follow the renamed field
(and `tests/test_quality_gate_adversarial.py` gained a lock that the legacy `min_si_sdr`
config key still re-tiers, because #14 kept it as an alias for shipped yaml files);
`test_logits_to_midi_threshold_pitch_and_velocity` had its onset frames spaced past the new
refractory window; and `TRANS-1` is re-scoped — its density half is fixed (a saturated
onset field now yields exactly **1** event, `onset_refractory_frames: 0` restores 86/s),
while the missing content gate is not, so the id stays open with new measurements.

How this was diagnosed at all: the Actions log host is not reachable from the review
sandbox, so `.github/workflows/ci.yml` now converts a failing test run's `FAILED`/`ERROR`
lines plus the interpreter/torch/numpy/pytest versions into ≤12 check annotations. Nine
lines, one API call — and it is the reason a red `continue-on-error` step can never again
hide a broken `test_review_fixes.py` the way #11's did.


## 1. Baseline re-verification (the brief asked me not to trust the stated numbers)

Stated: `pytest -q -m "not cuda"` → **96 passed, 2 skipped, 0 failed** on Python 3.13.
Measured here (same repo, `6ab1a1e`): **90 passed, 8 skipped** after installing the
full dependency set, and **2 failed, 72 passed, 20 skipped** on a leaner env.

| Delta | Cause |
| --- | --- |
| 96 → 90 tests | This sandbox runs **Python 3.11.2**, not the pinned 3.13; `requirements.txt` has no hard pins, so version-guarded/optional paths (`torchaudio` absent, `librosa` present/absent) changed collection. The gap is *not* reproducible without the interpreter note. |
| 2 → 8 skips | Two `importorskip` gates (`torchaudio` in `test_dataset_loaders.py`, `mamba_ssm` in `test_checkpoint_load.py`) plus per-test optional skips. **Both gates are invalid**: `data/datasets.py` never imports torchaudio, and `models/_mamba_compat.py` exists precisely so the product needs no mamba-ssm. I removed them; 6 permanent skips became 6 real tests, and one of them immediately surfaced a checkpoint-format bug (CKPT-1). |
| `-m "not cuda"` | A **no-op filter**: `grep -rn "@pytest.mark.cuda" tests/` → 0 matches at HEAD. It excluded nothing, and the "2 skipped" were unrelated. `cuda` is now a real marker used by the mamba-ssm parity job. |

Also verified as stated: `demo.py` synthesises its own audio (`generate_demo_audio`),
there are no trained weights in the repo (`weights/` absent, no `*.ckpt`/`*.pt`), and
a fake backend exists (`models/backends/fake_backend.py`).

**Consequence for anyone re-running this:** every "the suite is green" claim must name
the interpreter, whether `torchaudio`/`mamba_ssm` are installed, and whether
`--strict-deps` was on. Without that, 90 and 96 are both "green".

---

## 2. Requirement 1 — coverage of the critical paths

### 2.1 Real Mamba separator / transcriber inference

*Before:* `test_mamba_separator.py` (3 tests) and `test_mamba_transcriber.py` (4)
build the modules from a downsampled config and assert output **shapes** and
`torch.isfinite`. `test_losses.py`, `test_checkpoint_load.py` (2 of 9),
`test_api_integration.py` reach the real classes; everything else uses `mock_model`
(a hand-rolled `nn.Module` returning zeros).

*Assessment:* the shapes are the least informative property of this model — a
transposed head or an off-by-one hop changes nothing about them. The one
semantically meaningful old test (`test_separator_output_is_finite_and_bounded`) is
still worth keeping and now has company.

*Added (all in the real path, no mocks):*

| Test | Asserts |
| --- | --- |
| `test_audio_edge_cases.py::test_separator_survives_the_adversarial_corpus[10 cases]` | for silence, DC, impulse, hard-clipped square, 8× overflow, −140 dB, pink/white noise, pluck line, 12-voice chord: finite output, exact length `== T`, no stem louder than the input peak, metrics inside `[0, 30]` |
| `test_audio_edge_cases.py::test_sub_window_input_raises_typed_error` *(resolved by #11)* | T ≤ 2·n_fft must fail with a domain error (IO-1: today it is `RuntimeError: Argument #4: Padding size…` — from both separator *and* transcriber) |
| `test_audio_edge_cases.py::test_property_scale_invariance_is_bounded`, `…_length_contract`, `…_mask_sum_is_bounded` | seeded property checks over random filters/noise: masks stay in `[0,1]`, the frame law `1 + T//hop` holds, stems never exceed the input |
| `test_backend_contracts.py::test_mamba_block_is_causal` | feeding a changed tail must leave the prefix **bit-identical**. This is the only test in the repo that can catch a non-causal conv/scan, which would make every onset anticipatory (and it passes today: maxdiff 0.0) |
| `test_backend_contracts.py::test_mamba_block_is_stateless_between_calls` | no hidden recurrent state survives a forward |
| `test_backend_contracts.py::test_separator_and_transcriber_build_with_no_mamba_ssm_installed` | the CPU promise itself: build + run with no optional dep present, and the module-level `MAMBA_BACKEND` string must match reality |
| `test_backend_contracts.py::test_inference_on_the_same_file_is_bitwise_reproducible` | same file twice → same notes, same confidences, same tier. **Fails today**: the confidence head differs by 6.46e-3 between identical calls while every other head is bit-identical (REPRO-1) |
| `test_pipeline_integrity.py::*` (17) | wiring-level, using injected doubles: batch merge, per-stem aliasing, energy indexing, report aggregation, stems-sum-to-mix, expression-head reach |

### 2.2 MIDI round-trip correctness (onset / pitch / expression / confidence)

*Before:* 3 tests over `build_midi_from_events`, asserting `len(note_on) == N` and
that the bytes start with `MThd`. Nothing parsed the SMF. `test_midi_targets.py`
exercised `midi_to_targets` on hand-built `MidoFile` objects — i.e. **the writer and
the reader were never run against each other**, which is the single highest-value
gap in the repo: the training labels and the product deliverable share that
boundary.

*Now:* 64 tests across `test_midi_semantics.py` (writer semantics: absolute ticks,
ordering, overlaps, CC#127 attribution, duration, PPQ, tempo map, dangling
note-offs), `test_midi_target_roundtrip.py` (writer → reader → targets for hop ∈
{512, 256, 128}, one-hot pitch rows, durations, tempo changes, the API's grid
forwarding) and `test_api_zip.py` (the MIDI *inside the download*).

Findings locked there — each of these is invisible to any count-based assertion:

* **MIDI-1 (P0).** *(resolved by PR #11; now a pass-today guard.)* `utils/midi.py:65` computes `ticks_per_second =
  ticks_per_beat * (60 / (tempo_us / 1e6))` = **57 600** where the correct value at
  120 BPM/480 PPQ is **960**. A 1-second render becomes a 60-second SMF (2-second
  upload → `guitar.mid` claims 119.94 s). `midi_to_targets` recovers **0 of 4**
  onsets from a file the project's own writer produced. Every DAW import, every
  `processing_report`, and the whole Slakh-derived label pipeline sits on this line.
* **MIDI-2.** a single running cursor + `max(0, delta)` — an onset inside the
  previous note's sustain is silently moved to the end of that sustain, and unsorted
  input collapses onto duplicate ticks.
* **MIDI-3/4.** CC#127 is written *after* the note_off (note *i* carries note
  *i-1*'s confidence, note 0 carries none), and `int(c*127)` truncation means 0.999
  → 126, so "fully confident" is unreachable.
* **MIDI-5.** `api.create_response_zip` calls the writer with its module defaults
  (hop 512 / 44.1 kHz) while the frames come from the configured grid → any other
  config shifts every note (2.75× at 16 kHz/hop 128).
* **MIDI-6.** *(resolved by PR #11; the mid-file tempo map is still open as MIDI-7.)* `midi_to_targets` parses `set_tempo` and then `pass`es: non-120-BPM
  sources land on the wrong frames. Silent label corruption.
* **EXPR-1.** *(resolved on `main` — the head is carried, named from
  `transcriber.expression_heads`, labelled `source: expression_head (untrained;
  ARCH-11.4.4)`, and written as pitch-wheel/CC1/CC74 only under
  `include_expression=True`.)* The measurement that opened it: `_forward_impl` discarded
  the head, the writer emitted no pitch-bend/CC1, and the only non-note messages in a
  render were CC#127 — which the merged writer no longer emits either.

### 2.3 Quality-gate decisions

*Before:* `test_quality_gates.py` (7 tests) asserts the tier for hand-built reports
and that `min_si_sdr`/thresholds exist in the dataclass. Correct as far as it goes;
it never asks whether the **inputs** mean anything, and the gates are the payment
decision (free download vs editor vs +$4.99 human review).

*Added:* `test_quality_gate_adversarial.py` (9) tries to buy a Studio badge: 0-note
renders, silence, clipping, white noise, and the fake backend's constants;
plus `test_pipeline_integrity.py::test_empty_render_is_never_offered_as_studio_or_draft`
and `test_low_confidence_notes_counts_notes_not_frames`. Four new bugs (GATE-3,
GATE-4, CONF-1, FAKE-1) — see §5.

### 2.4 Low-memory behaviour (the 4–8 GB target)

*Before:* `test_perf_smoke.py` (1 test, `slow`): `mock_config`, one forward of
`(1, 1, 44100)`, `rss < 2500 MB`, `elapsed < 30 s`, and **two branches that measure
different quantities** (psutil → current RSS; `resource` → process peak). Nothing
connects the shipped shape, the frame grid, and the "one forward per file"
implementation.

*Added:* `test_memory_ceiling.py` (13): the analytic ceiling at the shipped config,
the linear-in-length guard, the chunking dispatch (with a **control** test proving
the plumbing works), a two-process VmHWM comparison, an OOM→503 contract test, a
streaming-length contract, and a source-level indictment of the existing perf test.

### 2.5 Failure when `mamba-ssm` is absent

*Before:* "handling" meant **skipping**. `models/_mamba_compat.py` is the actual CPU
product, and it had no test of its own: no causality test, no state-leak test, no
kwarg validation (its fallback signature ends in `**_ignored`, which is *exactly* how
TODO C-2.1 stayed invisible on CPU), and no parity check against the real kernels.
`test_checkpoint_load.py` skipped 7 of 9 tests when mamba-ssm was missing, hiding
CKPT-1.

*Now:* the shim is tested directly (`test_backend_contracts.py`), the skip is
impossible for the tier that covers it (§4), and the CUDA-only parity job exists as a
template with a real marker (`requires_mamba_ssm`, `cuda`) that is enforced when the
package *is* importable.

---

## 3. Requirement 2 — the synthetic-only / mock-only inventory

Files whose green status proves nothing on its own (each now has a real counterpart):

1. **`test_midi_generation.py`, `test_midi_roundtrip.py`, `test_midi_targets.py`** —
   counts and headers; never compare tick values, never read a file the writer made.
2. **`test_api_integration.py`, `test_api_endpoints.py`, `test_api_auth.py`,
   `test_api_metrics.py`, `test_api_validation.py`** — `/live`, `/ready`, auth,
   metrics, validation. `TestClient` is imported in four files and `/process` is
   **never posted**; `create_response_zip` is never called; the ZIP is never unzipped.
3. **`test_backends.py`** — `test_stem_midi_model_with_fake_backends` and
   `test_process_audio_file_fake` assert *key presence* on a model that ignores the
   `backends` config entirely (BACK-1, *honoured since PR #11*), so they exercise the real
   Mamba while being
   named for the fake path. `FakeBackend`'s metrics are constants (FAKE-1).
4. **`test_confidence_injector.py`, `test_quality_gates.py`, `test_losses.py`** —
   correct unit tests over synthetic tensors; none of them reaches a decision that a
   user can perceive (they pass while GATE-*/CONF-1/ALIGN-1 are live).
5. **`test_dataset_loaders.py`** — permanently skipped on torchaudio (not a
   dependency). `test_checkpoint_load.py` — 7/9 tests permanently skipped on
   mamba-ssm.
6. **`test_mamba_separator.py` / `test_mamba_transcriber.py`** — real modules,
   white noise, shape-only. No silence, no clipping, no sub-window input, no
   long file, no polyphony, no determinism, no causality.
7. **`demo.py` + `test_main.py`** — the "end-to-end" evidence in the README is a
   5-second synthetic sine rendered by an untrained model, and the test asserts the
   output dict's keys.
8. **`.github/workflows/ci.yml`** — `continue-on-error: true` on the test step, and
   the coverage gate (`--cov-fail-under=70`) fused into the same step, so neither the
   tests nor coverage could ever fail the build. Fixed: correctness is a hard step;
   coverage stays a separate, still-permissive step until the number is reachable.
9. **`pytest -m "not cuda"`** — filtered nothing (marker unused at HEAD).

Numerical edge cases with **zero** coverage at HEAD: silence, DC, sub-window
(shorter than one STFT) files, 0-frame files, NaN/Inf, >1 full-scale overflow,
anti-phase stereo, extreme polyphony, 8–600-second lengths, non-120-BPM MIDI,
duplicate/unsorted event ticks, and repeated inference on identical input.

---

## 4. Requirement 4 — keeping the suite honest about the skip

The rule I wrote into `tests/honesty.py` and `tests/conftest.py`:

> A test may skip only for a reason the project *cannot* fix on that host, and every
> skip must be visible, categorised, and — for tiers declared must-run — fatal.

Mechanics:

1. **No dependency gate outside one place.** `require(name, purpose)` is the only
   allowed way to skip for a missing package. `importorskip` on `torch`/`torchaudio`/
   `mamba_ssm` in tests is banned (the two existing cases were removed). At import
   time it uses `pytest.skip(..., allow_module_level=True)` via `module_level=True`.
2. **Two tiers, opposite privileges.**
   * `real_mamba` — the pure-torch Mamba/separator/transcriber path. It needs **no
     optional package**, so it must run everywhere. A skip or "not-run" is a
     **suite-honesty violation** that flips the exit code, with or without flags.
   * `requires_mamba_ssm` — CUDA-kernel parity + real-package builds. Skips freely on
     CPU hosts; if `importlib.import_module("mamba_ssm")` succeeds and the test still
     skipped, the session fails (`--require-mamba-ssm` makes even the *absence* of the
     package a hard error, for the GPU job).
3. **`--strict-deps`** (CI default): an `optional-dep:` skip becomes a
   `BROKEN DEPENDENCY GATE` failure, so "we skipped 20 tests" is only possible when
   the maintainer explicitly opts into a convenience run.
4. **`--fail-on-skip`**: any skip whose reason is not in
   `honesty.ALLOWED_SKIP_REASONS` (`optional-dep:`, `codec-unavailable:`,
   `artifact-unavailable:`) fails. The allowlist is deliberately 3 entries long:
   every entry is a hole in coverage and needs a reviewer to add one.
5. **Capability table on every run** (`pytest_terminal_summary`):
   `real_mamba:passed 40 · real_mamba:xfailed 9 · requires_mamba_ssm:skipped 1 · mamba_ssm importable: False`
   — the skip count is reported *per tier*, not as one number, so a suite cannot hide
   behind "2 skipped".
6. **Deselection is not a violation.** The gate prunes to the items that survived
   `-m`/`-k`/explicit paths (verified: `pytest -k one_test` → exit 0, no violations),
   so partial local runs stay usable while an unexecuted must-run test still fails.
7. **Bugs are strict xfails, never skips.** `known_bug(id, detail)` =
   `xfail(strict=True)` with `TODO <ID>: <measured evidence>` as the reason, which is
   what CI prints. Fixing the bug turns the run red with an XPASS, forcing the marker
   deletion. `STEM_KNOWN_BUGS=fail` reports all 65 as failures for release gates
   (verified: exit 1 with 5 lines for one file).
8. **Every xfail has a non-xfail plumbing guard.** An xfail is "confirmed" by *any*
   failure, including a broken harness, so each measurement-based known-bug is paired
   with a passing smoke test (`test_peak_rss_probe_runs_at_all` caught exactly this
   class of self-deception while I was writing it).

Verification I ran for the mechanism itself (not just the tests):

```
scratch test @pytest.mark.real_mamba + pytest.skip()
  → SUITE-HONESTY VIOLATIONS … must-run tier ['real_mamba'] did not execute → exit 1
scratch test require("definitely_not_installed")
  → default: 1 skipped (exit 0) · --strict-deps: FAILED "BROKEN DEPENDENCY GATE" (exit 1)
```

CI (`test-cpu`) now runs `pytest -m "not cuda" --strict-deps --fail-on-skip` with **no
`continue-on-error`**, and a commented `mamba-ssm` job template (self-hosted GPU
runner) runs `-m "cuda or requires_mamba_ssm" --require-mamba-ssm` — enabled by
deleting the comments when such a runner exists, not by adding a skip.

---

## 5. Requirement 3+5 — new tests, and why each was needed

Full index: **§18 of `TODO.md`** lists all defect IDs (22 still open, 11 resolved by
PR #11 plus 1 by this branch — see §0.1) with severity, one-line
root cause, and the file:line to fix. Below: the rationale per new file, in the
brief's own frame (bad stems · useless MIDI · OOM · dishonest success).

| File (tests) | Highest-value assertions | Why the old suite was weak here |
| --- | --- | --- |
| `test_midi_semantics.py` (19 ✓ / 10 ✗) | absolute seconds and ticks from `parse_midi`, not message counts; `note_on`↔`note_off` balance; CC#127 attaches to the note it describes; `duration_frames` honoured; PPQ and tempo changes respected; velocity clamped not wrapped | every old MIDI test stopped at "does mido parse it". The 60× tick bug produces a perfectly valid SMF — that is why it survived |
| `test_midi_target_roundtrip.py` (9 ✓ / 10 ✗) | writer→reader identity for 3 hop sizes; one-hot pitch rows; recovered onsets == requested frames ±1; duration survives the circle; the API forwards the model's grid | the two halves of the same boundary were tested in isolation, so a self-consistent pair of bugs (writer slow, reader assumes 120 BPM) looked fine on both sides |
| `test_pipeline_integrity.py` (7 ✓ / 10 ✗) | with injected doubles: per-stem MIDI is actually per-stem; `guitar+bass+residual ≈ mix`; energy indexed by frame not sample; `low_confidence_notes` counts notes; batched reports aggregate; a 0-note render is never `direct_download`; expression reaches the file | model-level wiring was untestable (assigning to an `nn.Module` child raises `TypeError`, so nobody stubbed it) — `tests/doubles.inject()` exists for exactly that |
| `test_audio_edge_cases.py` (35 ✓ / 6 ✗) | the 10-case adversarial corpus on the **real** separator/transcriber; the `T → frames` length contract; masks bounded; NaN/Inf isolation; 0-frame and anti-phase files; `CFG-2` pinned (mock_config cannot drive `torch.istft`) | zero numerical edge cases existed; the shared `mock_config` grid was silently incompatible with the module under test |
| `test_backend_contracts.py` (7 ✓ / 2 skip / 8 ✗) | factory validation and no-silent-downgrade; `backends:` config actually reaches `StemMidiModel`; protocol dicts vs 5-tuple unpack; shim causality, statelessness, kwarg strictness; run-to-run reproducibility; CUDA parity template | the whole `models/backends/` layer was tested as a library, never as the wiring it claims to be; the shim had no test at all; `-m "not cuda"` was decorative |
| `test_api_zip.py` (7 ✓ / 9 ✗) | the 5-entry archive; stems decodable, mono, right sr, finite, in range, *not* copies; per-stem MIDI content; MIDI duration ≈ audio duration; `?quantize=true` reaches the model; `processing_report.json` is strict JSON; empty/non-finite uploads are 400, not 200/500 | nothing had ever posted a file and unzipped the answer. Model-free on purpose: a hand-built stub model isolates the seam, so a red test here means the boundary, not the weights |
| `test_memory_ceiling.py` (7 ✓ / 6 ✗) | 271 MB/audio-second of scan activations at the shipped shape (0.54 GB for 2 s, 8.1 GB for 30 s, 16.3 GB for 60 s) vs a 4 GB budget; `chunk_duration_sec` ignored by `process_audio_file` but honoured by a control case; VmHWM +124 MB (2→16 s) monolithic vs +32 MB chunked; OOM ⇒ 503 + `Retry-After`; a mid-stream OOM must not return a short render; streaming render length matches input | the only perf gate measured 1 second of a toy model, once, and its two branches disagree about what "RSS" means |
| `test_quality_gate_adversarial.py` (4 ✓ / 5 ✗) | thresholds come from config; artifact flags are an absolute veto; badge ⇄ action ⇄ editor flags agree; and the four attempts to buy a badge: 0 notes, a fake backend's constants, a clipped square rated *above* a clean sine (17.75 vs 0.00 dB), and `avg_confidence` that scores clipping 0.5164 > music 0.2998 | the gates were tested as a pure function of numbers that no test checked the provenance of |

Supporting infrastructure (needed to make the above possible, not tests in
themselves): `tests/honesty.py` (146), `tests/synth.py` (314), `tests/doubles.py`
(132), `tests/conftest.py` (457, rewritten).

Two deliberate design choices worth flagging to reviewers:

* **No parametrisation across a bug/feature boundary in gate tests.** A
  `known_bug`-marked `parametrize` where one case legitimately passes becomes an
  `XPASS(strict)` failure that hides the real verdict; the confidence test therefore
  evaluates all four signals in one body and reports one verdict.
* **Known bugs must not be silently re-labelable.** The `TODO <ID>` string is greppable
  in both directions: `grep -rn "TODO GATE-3" tests/` finds the locking tests, and
  `TODO.md` §18 holds the fix entry. Closing a bug = deleting one decorator, which is
  enforced by the strict xfail.

---

## 6. What I tried to indict and could not

A skeptical audit has to record the near-misses; three of these were my own
hypotheses that measurement killed:

* **Mel basis all-zero rows** (would zero out some frequency bands forever) — not
  present. Now kept as a *passing* guard, because it is the kind of thing that silently
  reappears.
* **`json.dumps(midi_metadata)`** at the API boundary — serialises fine.
* **Chunk-boundary clicks** in `process_audio_streaming` — 6 chunks over 6 s, 4.3 %
  relative error vs the monolithic render, no discontinuity spike at the seams. (The
  *quality* of that approximation is a separate, untested question: no test compares
  chunked vs single-pass output at all, which is why 4.3 % is a measured number in
  this document rather than a test — that's the next file to write.)
* **Non-causality of the fallback scan** — measured prefix-invariance: maxdiff 0.0.
  Kept as a passing regression guard (`test_mamba_block_is_causal`), not as a bug.
* **`filterwarnings = error` in `pytest.ini`** — I tested it and *rejected* it: with
  the pinned-but-unpinned dependency set it raises third-party noise
  (`StarletteDeprecationWarning`, `pytest-asyncio`'s loop-scope warning) in ~every API
  test, i.e. a config change that makes the suite red for the wrong reason. The
  numerical-precision worries it would have caught are instead covered explicitly by
  the NaN/Inf corpus in `test_audio_edge_cases.py`.

---

## 7. Residual gaps (what a reader should not conclude)

1. **Every model test runs on untrained weights.** No repo test can therefore say
   "the stems are good", only "the render is finite, bounded, self-consistent and
   not obviously degenerate". The moment a checkpoint exists, add a nightly tier:
   10 real MUSDB/Slakh excerpts, golden `note_count`/`onset-F1`/`SI-SDR` bands, and
   `pytest.mark.artifact_unavailable` skips (the allowlist prefix is already there).
   Until then, **`si_sdr ≥ 20` and `confidence ≥ 0.85` are unreachable numbers in
   production** (STUDIO can only fire through the fake path or a mis-set gate) —
   that is the single most commercially important fact in this document.
2. **65 known bugs, 0 fixed.** I changed no production code except test infrastructure,
   CI and the Makefile. The P0 set — MIDI-1, MIDI-5, MIDI-6, STEM-1, API-1, TRANS-1,
   GATE-3, CONF-1, IO-3, ZIP-1, CKPT-1, ALIGN-1, CPU-1/2 — is ordered in `TODO.md` §18.
3. **`hypothesis` is not a dependency.** The property-style tests are seeded
   hand-rolled generators (5 seeds × parametrised shapes) with `pytest.mark.property`.
   Adding `hypothesis` would widen the frame-law and mask-boundedness checks cheaply.
4. **Coverage is not measured here.** `.codecov.yml` still targets a number the
   pre-existing suite could not reach; the CI step is now honest (separate, still
   permissive). Re-measure after §18 lands — coverage of `utils/midi.py` was the
   misleading metric (high, and worthless) before this work.
5. **Environment caveat:** all measurements in this document come from
   Python 3.11.2, `torch 2.5.1+cpu`, no `torchaudio`, no `mamba_ssm`, 2 cores / 3 GB.
   The `real_mamba` tier is *designed* to be the same everywhere (pure-torch shim),
   so the 65 xfails should reproduce on 3.13 — but the 90-vs-96 baseline gap in §1 is
   exactly what an unrecorded interpreter looks like, so record the interpreter next
   to any "green" claim.
6. **The `requires_mamba_ssm` tier has never been executed.** Its two tests (shim-vs-kernel
   parity, real-package checkpoint round-trip) have never run anywhere, because no
   environment here has mamba-ssm. They are *written to* be correct, and they will
   either pass or fail loudly on the first GPU run — but I cannot claim more than that,
   and neither can the suite: the CI gate that makes that true is §4's item 2.

---

## 8. Reviewer checklist (to keep this suite honest)

* A new skip in a test needs either an `ALLOWED_SKIP_REASONS` prefix (a PR review
  question: why is this unfixable on CI?) or a `require()` call. Nothing else.
* A test that asserts `len(events) == N` for MIDI is a *lint* candidate: assert tick
  positions or seconds (`tests.synth.parse_midi` returns both).
* Any test touching the separator/transcriber must use `pipeline_config`
  (2048 / hop 512) — `mock_config` cannot drive `torch.istft` (CFG-2) and mixes units
  with `utils.midi.HOP_LENGTH`.
* Stubs for `nn.Module` children go through `tests.doubles.inject()`.
* A new defect gets a `known_bug("<ID>", "<measured evidence>")` **plus** a §18 TODO
  entry **plus** a non-xfail smoke check of its own harness if it measures anything.
* CI correctness step keeps `--strict-deps --fail-on-skip` and never
  `continue-on-error`. If the suite must go green, fix a bug or delete a marker —
  don't re-add permissiveness.
