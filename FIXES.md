# Fix Log — audit findings (2026-11-17)

Companion to [AUDIT_REPORT.md](AUDIT_REPORT.md). Every row was *measured*; the "Before" column is the
audit's own reproduction, the "After" column is a fresh run on this branch **rebased onto `main`**
(which now contains PR #11 and PR #14 — two independent fix passes). Only the delta this branch adds
is claimed as new work; where `main` had already fixed something, the row says so and the number is
re-measured on the merged tree.

Environment for every measurement: 2 threads, CPU only, `mamba_ssm` not installed → pure-PyTorch
fallback (`models/_mamba_compat.py`), `configs/model_config.yaml`.

Reproduce with:

```bash
python3 -m pytest -q                                  # 357 passed, 4 skipped, 40 xfailed
python3 scripts/verify_numerics.py                    # block-RMS ladder, masks, mixture consistency
python3 scripts/verify_midi.py                        # clock, polyphony, durations, CC
python3 scripts/verify_api.py                         # OOM mapping, untrained gate, JSON guard
```

## Critical / High

| # | Finding | Fix on this branch | Evidence: before → after (merged tree) |
|---|---------|--------------------|----------------------------------------|
| C-1 | Hidden state collapses across the Mamba stack; masks pinned at 0.5 | **Added here** (not in `main`): pre-norm residual stack — per-block `LayerNorm`, `x + block(norm(x))`, plus a final `LayerNorm` — in separator *and* transcriber | separator per-block RMS `1.0 → 4.5e-2 → … → 7.3e-20` → `1.00 → 1.00 → … → 1.01` (12 blocks); transcriber `1.33 → 1.34` |
| C-1 | Masks carry no information | **Added here**: `separator.mask_normalization: softmax` (default) over guitar/bass/residual | mask mean/std `0.500084 / ~0.000` → guitar `0.337 / 0.148`, bass `0.331 / 0.144`, residual `0.332 / 0.144` |
| C-1 | Stems do not reconstruct the mix | Follows from softmax masks | `||guitar+bass+residual − mix|| / ||mix||` = **1.15e-7** |
| C-3 | MIDI clock 60x slow (3 s of audio → 179.8 s of MIDI) | `main` had `ticks_per_second = ticks_per_beat · 1e6 / tempo_us`; kept, and the writer now honours `duration_frames` | two events spanning **34.83 s** of audio → MIDI ends at **35.29 s (1.01x)**. (With random weights the peak-picker emits a single note, so an end-to-end file-length check needs a trained model.) |
| C-3 | Chords serialised into arpeggios (`+120`-tick stagger) | `main` had already merged events into one time-sorted stream; kept | 3 simultaneous notes → **3 note-ons at tick 0** |
| C-3 | Confidence written to CC#127 (`Poly Mode On`, a Channel Mode message) | row written before the merge; **the merged lane is per-note Polyphonic Key Pressure with no confidence CC at all** (see “Merge onto `main” below) — main's CC#16/`MIDI_CONFIDENCE_CC` convention did not survive it | controllers `{127}` → `{}`; polytouch `[(60, 114), (64, 50)]` |
| C-3 | A channel CC cannot express per-note confidence (simultaneous notes share the channel) | **Added here**: `confidence note=<n> value=<v>` marker track (`include_confidence_track`) | 2 simultaneous notes → 2 marker messages, one per note |
| C-3 | Fixed 1/16-note durations | **Added here**: `duration_frames` honoured; `_logits_to_midi` derives durations from inter-onset gaps (clamped to `min_note_seconds`/`max_note_seconds`) | 400-frame note → 4458 ticks = 400 × ticks/frame; 20-frame note → 222 ticks (±1 from rounding) @120 BPM |
| C-4 | `O(T)` scan memory (23 MB per second of audio; 180 s OOM-killed) | `main` ships the chunked state-carrying scan (`Mamba.SCAN_CHUNK`, default 256, checkpointed in training); `_mamba_compat.py` taken wholesale | 300 s single pass: **2546 MB peak, completes** (audit: 180 s killed) |
| C-4 | Long files unbounded in `process_audio_file` | `main` (PR #16) separates in overlapping chunks (`audio.chunk_duration_sec`, default 2 s); **added here**: frame-segmented transcription (`inference.segment_seconds`, default 60 s) so a long stem is not decoded in one pass, and `_match_frame_grid` keeps the injector's grid consistent | 5 min chunked: **104.0 s / 1598 MB / 300 chunks**; 10 min: **173.6 s / 2224 MB / 600 chunks**; 5 min in one chunk: **69.0 s / 2164 MB**. Chunking costs ~1.5x wall time versus a single pass; memory is the binding constraint. |
| H-6 | One physical onset emitted several note_ons (activation plateau, jittered argmax) | `main` (PR #14) added rising-edge peak-picking with `onset_refractory_frames`; taken as-is and combined with the duration estimate | 3 s of random-weight audio: 259 note_ons → **1** (the plateau collapses); durations still follow onset spacing |
| — | `ProcessingReport.separation_proxy_db` raised `AttributeError` after the C-2.6 rename (the property still read `self.si_sdr`) | **Fixed here**: the property reads `centroid_separation_db` | `report.separation_proxy_db` returns the number instead of raising (`process_audio_file` hit this) |
| H-4 | Pitch/velocity heads got no gradient | `forward()` passes pitch + velocity targets; datasets emit `target_velocity`; Slakh labels come from `all_src.mid` (`main`'s `midi_to_targets`) | loss dict reports non-zero `pitch_ce` / `velocity_mae` from real targets |
| H-4 | Onset "F1 loss" had zero gradient (it binarised the logits) | **Added here**: BCE-with-logits is the optimised term (`onset_bce`), hard F1 is a **metric only** | `onset_bce` present and finite in every training step; F1 still reported |
| H-4 | Cross-modal term compared two scalars (constant-shift gradient) | **Added here**: zero-mean cosine distance between predicted onsets and the energy envelope | an all-constant prediction now scores 1.0 (worst case) instead of ~0 |
| untrained model (API) | API served a random-weight model silently | **Added here** (absent in `main`): `load_model()` refuses untrained weights unless `ALLOW_UNTRAINED_MODEL=1` / `inference.allow_untrained_model`; `trained_weights` is surfaced in `/ready`, `/model-info` and the report | `/ready` → **503 `No trained weights available`**; with the override, `trained_weights: false` |
| API-6.9 | NaN/Inf reached `processing_report.json` | **Added here**: `_finite()` + `allow_nan=False`, and `ConfidenceInjector` raises on non-finite model output instead of writing it | NaN/Inf report → `{"centroid_separation_db": 0.0, "phase_coherence": 0.0, …}`, no `NaN`/`Infinity` token in the file |
| API-6.7 | CPU allocator OOM surfaced as 500 | **Added here**: `_is_oom_error()` also matches `DefaultCPUAllocator` / `alloc_cpu.cpp` / "out of memory" → 503 + `Retry-After` | `RuntimeError("DefaultCPUAllocator: can't allocate memory…")` → **503** |
| H-3 | Per-sample energy indexed with a frame index (~86x early) | `main`'s two-pass writer plus `_frame_energy()` hop-grid pooling; the injector raises on a grid mismatch, and `_enhance` pads/truncates to the confidence frame count (STFT gives `1 + T//hop` frames) | `ValueError: confidence and stem_energy must share the frame grid` (new test); a 3 s clip runs without the off-by-one |
| M-1 | Backend selection silent | `MAMBA_BACKEND=auto|pure-torch|mamba-ssm`; an explicit request that cannot be satisfied raises; `_log_backend_info()` logs impl/scan chunk | `mamba_impl=pure-torch, scan_chunk=256` in logs, `/model-info`, report |
| M-3 | Targets aligned at a hard-coded 120 BPM; chords overwrote each other | `build_tempo_map` / `tick_to_seconds` from every `set_tempo`; multi-hot pitch targets | tick 960 @120 BPM → frame **86**, @60 BPM → frame **172**; two notes on one frame → `pitch.sum() == 2.0` |
| Docs | CC#127 claims, stale test counts, "10-minute maximum" without memory evidence, unqualified SI-SDR | README/ARCHITECTURE/API/PRD docs updated to the merged conventions (CC#16, `centroid_separation_db`, measured memory table) | docs cite the numbers in this file |

## Still open (deliberately not claimed as fixed)

| # | Item | Why it is still open |
|---|------|----------------------|
| C-2.5 | Streaming SSM state cache | Segments are processed independently; seams are cross-faded but no SSM state crosses a boundary. Documented in `MambaSeparator.forward` and `process_audio_file`. |
| H-5 | Polyphony | The pitch head is still a 128-way argmax (one note per frame). Writer and targets are polyphony-ready (independent note-on/off, multi-hot targets, per-note marker), so a multi-label head can drop in. |
| H-2 | Expression head | Untrained; its channels are unnamed. `_logits_to_midi` passes an `expression` vector through, but the writer only emits bend/vibrato when an event carries those keys (`include_expression=False` by default). |
| M-4 | No tempo detection | MIDI is written on a fixed grid at `tempo_us` (default 120 BPM); the summary says `"timing": "fixed grid (no tempo detection)"`. |
| — | No trained checkpoint ships with the repo | Quality numbers and the MIDI content of an untrained model are meaningless; the API now refuses to pretend otherwise. Concrete consequence measured here: with random weights the merged peak-picker emits **1** note for 3 s of audio, so no end-to-end musical claim is possible. |
| — | `dt` clamping | The merged code relies on the upstream `dt_proj.bias` init (`softplus(bias) ∈ [1e-3, 0.1]`, measured `0.00104 … 0.0884`) rather than a runtime clamp. Extreme inputs (±1e3) stay finite (tested), but if a future data path drives huge `dt` inputs, add a clamp. |
| — | Breaking checkpoints | `block_norms.*` / `final_norm.*` keys were added to the separator and transcriber, so checkpoints saved before this branch need `strict=False`. |

## Test suite

`pytest -q` → **357 passed, 4 skipped, 40 xfailed, 0 failed** (`main` alone: 316 passed, 40 xfailed).
The 4 skips are `mamba_ssm`-only checkpoint tests. `ruff check .` and `black --check .` are clean.

New/rewritten coverage on this branch:

- `tests/test_numerics.py` — block-RMS ladder, mask content, mixture consistency, silence, `dt` init
  range, chunked-vs-unchunked scan equality, MC-dropout uncertainty.
- `tests/test_midi_targets.py` — tempo honoured (120/60 BPM, mid-file tempo change), multi-hot chords.
- `tests/test_midi_roundtrip.py` — rewritten to the merged contract (absolute clock, chord
  simultaneity, durations, retrigger truncation, the opt-in confidence marker track).
- `tests/test_midi_generation.py` — the note-specific confidence lane and the absence of any
  Channel-Mode write (see "Merge onto `main`" below for which lane that is).
- `tests/test_confidence_injector.py` — frame-grid mismatch, non-finite input rejected, relative
  energy review flags.

## Merge onto `main` (PR #13 head `50e2d46` onto `main` @ `4b6daf1`, 2026-11-18)

While this PR was open, `main` took PR #12 (a second test audit), #15, #16 and #17 — including a
**rewrite of `utils/midi.py`** and of `models/confidence_injector.py`. Rebasing was not enough: the
result was 15 conflicted files, 26 hunks. Every conflict was resolved semantically, and the two
branches' product fixes are both in the tree. Where the two sides had different opinions about the
same contract, the table records which one won and why.

| Where | PR #13 | `main` | Resolution |
|-------|--------|--------|-----------|
| `utils/midi.py` confidence lane | CC#16 + companion `marker` track, written unconditionally | PR #15 replaced the CC lane with note-specific **Polyphonic Key Pressure** | `main`'s lane (a channel controller is shared by every note on the channel; poly pressure is per-note). #13's marker track survives as `include_confidence_track` (**default off**, because `main`'s contract is one track per render — asserted by `tests/test_midi_roundtrip.py::test_confidence_marker_track_is_opt_in`) |
| `utils/midi.py` timing | `tempo_us` parameter feeding both `set_tempo` and the tick divisor | hard-coded `DEFAULT_TEMPO_US` in both, while its own docstring promised `ticks_per_beat * 1e6 / tempo_us` | **#13's plumbing**, on `main`'s scheduler: the two constants could disagree with each other and both contradicted a 60 BPM request |
| `utils/midi.py` expression | `include_expression=False`, expression as a raw `vector` | `_expression_messages()` maps `bend`/`vibrato`/`slide` to pitch-wheel / CC1 / CC74, always on | `main`'s named mapping, with #13's mean-over-the-note windowing kept in `_logits_to_midi` |
| `models/confidence_injector.py` | rich docstring, frame-grid mismatch check, `_require_finite` | the same three behaviours, converged independently | `main`'s code + #13's docstring (re-worded for the poly-pressure lane); the duplicate `alignment_scores` line and the CC#16 sentence dropped |
| `main.py:_transcribe` | expression tensor threaded to `_logits_to_midi` | backend branch returned `"expression": None` | the auto-merge produced a **duplicate dict key** (`"expression": out.get(...)`, then `"expression": None`) that silently re-broke EXPR-1; the dead key is deleted. Same for the native branch's second `"expression"` entry |
| `main.py` long input | `_infer_segmented` (whole inference in 60 s windows) + `_transcribe_segmented` | `_separate_chunked` + `_finalize` on the separator | `main`'s chunked separator (finer bound) **and** #13's transcriber bound adopted inside `_forward_impl` *and* `_finalize`; `_transcribe_segmented` now takes `instrument` so a per-stem backend still gets it, keeps `expression` across the concat, and refuses to merge per-segment `events` (their frames are segment-local — merging them would misplace every note after the first seam) |
| `api.py` readiness | refuse untrained weights unless `ALLOW_UNTRAINED_MODEL` / `inference.allow_untrained_model`; report `trained_weights` | refuse unless a checkpoint **or** an attested `weights_source: pretrained_dependencies` backend pair; report `weights_loaded` / `checkpoint_loaded` | `main`'s stricter refusal, with #13's override checked first (so a smoke test can still boot and sees `trained_weights: false`) and all three spellings emitted from one value |
| `api.py` report | `_finite()` + `allow_nan=False`, `centroid_separation_db` only | `separation_proxy` + documented `si_sdr` alias + `metric_semantics` + `confidence_encoding` | `main`'s key set (no third alias for a deprecated name) with #13's sanitisation on every metric — `tests/test_api_zip.py::test_report_in_the_zip_is_the_pipeline_report_not_a_rederivation` pins the key set exactly |
| `configs/*.yaml` | `inference: {segment_seconds, allow_untrained_model}`, `min_centroid_separation_db` | `min_separation_proxy` (the gate's canonical name) | both: `main`'s key name, `min_centroid_separation_db` documented as an accepted alias, #13's `inference` block added (both keys are read) |
| `models/_mamba_compat.py` | `from torch.utils.checkpoint import checkpoint` | `import torch.utils.checkpoint` + qualified call | `main`'s form (it is what the CFG-3 guard checks textually), call site kept qualified |
| `scripts/verify_midi.py` | verified the CC#16 default and CC#127 opt-in | — | **rewritten against the merged contract**: polytouch per note, no CC#120-127 ever, `include_confidence_track=True` for the marker lane, and a `tempo_us=60 BPM` check |

Re-measured on the merged tree, same environment (2 threads, `mamba_ssm` absent,
`configs/model_config.yaml`, random weights):

```bash
python3 -m pytest -q -m "not cuda" --strict-deps --fail-on-skip
# 375 passed, 1 skipped, 3 deselected, 36 xfailed  (main @ 4b6daf1: 349 / 1 / 3 / 39)
python3 scripts/verify_numerics.py   # mixture consistency 1.15e-07, chunked-vs-1-pass 1.49e-08
python3 scripts/verify_midi.py       # polytouch [(60, 114), (64, 50)], 0 channel CCs, tempo honoured
python3 scripts/verify_api.py        # untrained gate refuses; ALLOW_UNTRAINED_MODEL=1 -> trained_weights: false
```

The 36 xfails are `main`'s still-open defects (`TODO.md` §18.1), not this branch's: three locks
this audit set are now satisfied and were converted into pass-today guards — SEP-1
(`test_stems_sum_back_to_the_mix`), JSON-1 (`test_zip_report_is_strict_json`) and, for the docs
half, DOC-1 — while MIDI-5 and CKPT-1 had already been fixed on `main` by PR #15/#16.

## Merge onto `main` (round 2: `f216fef` + `origin/main` @ `fda6071`, 2026-11-18)

PR #13's head was merged once (round 1, `f216fef`) and `main` moved again before the
review finished, so the branch was re-merged onto `fda6071`: 13 files, 26 hunks, one of
them an add/add on all four `scripts/verify_*` + `tests/test_numerics.py` pairs (both
branches had rewritten the same files from scratch). Nothing was resolved by deleting a
side's semantics; where both sides were right, both survive, and where one side was
wrong it is recorded here rather than quietly dropped.

| Site | `main` @ `fda6071` | This branch | Resolution |
|---|---|---|---|
| `utils/midi.py` signature | `include_expression=False` (PR #18: the head is untrained, so an opt-in) | `tempo_us`, `include_confidence_track` | **union of all three** parameters; the expression block is gated by main's flag, the marker collection stays, and both defaults keep main's one-track contract |
| `utils/midi.py` docstring | one paragraph per behaviour | two paragraphs, merged badly | rewritten once, no duplicated paragraphs; states that the marker track is opt-in *because* `len(mid.tracks) == 1` is pinned |
| `main.py::_logits_to_midi` | chord-aware decoding (`pitch_threshold` → one `note_on` per pitch above threshold, argmax fallback) | gap-derived `duration_frames`, `events.sort`, expression averaged over the note | **both**: chord decoding drives emission, `duration_frames` comes from inter-onset spacing, expression is the mean over that note and keeps main's `source` label. A chord's notes share one duration - stated in the docstring |
| `main.py` velocity | `(velocity * 127).long()` (truncates) | `squeeze(-1).clamp(0, 1) * 127 → round → clamp(1, 127)` | **kept the hardened mapping** (a 0 velocity is a `note_off`); main's new chord test expected `63` for 0.5 and now expects `64`, with the reason inline |
| `models/losses.py` multi-hot test | `target_pitches.dim() == 3` | `shape[-1] == n_classes` | **both conditions** (`dim() == 3 and shape[-1] == n_classes`): either alone misclassifies a `(T, V)` unbatched grid or a `(B, 128)`-frame class-index target |
| `models/losses.py` `loss_dict` | `"pitch"`, `"onset_f1"` | `"pitch_ce"`, `"onset_bce"`, `"onset_f1"` (metric-only) | main's key names stay (its `test_review_fixes.py:1026` asserts `loss_dict["pitch"] > 0`); `onset_bce`/`pitch_ce` are kept as extra keys, since main's single `onset_f1` reported the no-gradient metric as if it were the loss term (H-4) |
| `models/losses.py` `pitch_bce_loss_fn` | new, validating, chord-weighted | new, simpler squeeze-based | **main's only.** The merge auto-kept both definitions in one file (`def` twice, second shadowing the first) - #13's copy is deleted, and an AST duplicate-def sweep now runs as part of this merge's checks |
| `models/mamba_transcriber.py` | "chords are NOT modeled" replaced by the PR #19 decoder note | pre-norm residual stack note (C-1) | main's text, then #13's stack note appended - both are true of the merged module |
| `configs/model_config*.yaml` | `transcriber.pitch_threshold: 0.5` | `transcriber.min/max_note_seconds`, `quality_gates.min_separation_proxy`, top-level `inference:` | **both key groups** in all three files (YAML-validated after the merge, not by eye) |
| `utils/midi_targets.py` | private `_build_tempo_map` + `build_tempo_map = _build_tempo_map` alias block | public def + private aliases at the end | the auto-merge produced `build_tempo_map = _build_tempo_map` *above* the definition - an `ImportError`/`NameError` on any import of `utils.midi_targets`. Alias block deleted, back-compat aliases kept after the definitions |
| `scripts/verify_midi.py` | main's (clock, polyphony, confidence-polytouch, expression, durations, reader tempo) - docstring still promised "CC#16 default, `confidence_cc=127`", which main's own writer has no parameter for | writer `tempo_us` round trip, marker track, no-CC assertion | **main's script + this branch's `check_writer_params()`**, and the stale header rewritten. Its `check_confidence` printed `(expected {64: 51})` next to `{64: 50}` because it re-derived with `round()` while the writer truncates; it now asserts the writer's mapping and prints both spellings |
| `scripts/verify_api.py` | untrained *provenance* only; states "there is no hard gate" | untrained *refusal* gate + `ALLOW_UNTRAINED_MODEL` | main's script, with a `check_untrained_gate()` appended: it asserts `load_model()` refuses **and** that the override serves with `trained_weights=False`. A script that only printed the flag would pass whether or not the gate existed |
| `scripts/verify_numerics.py` | hand-replays the *bare* stack (`for block in sep.mamba_blocks: h = block(h)`, sigmoid masks) to show the C-1/SEP-2 collapse and ~0.5 mixture consistency | walks the model's real normed residual stack + softmax masks | **walks the real modules.** Main's replay described a structure the merged model no longer has, so it was reporting numbers for a dead bug and passing |
| `tests/test_numerics.py` | 7 tests, incl. `@known_bug("SEP-2")` on the hand-replayed ladder | 8, incl. `test_mixture_consistency` | main's 7 as the base (SEP-1's mixture pin stays in `test_pipeline_integrity.py::test_stems_sum_back_to_the_mix`, one pin per defect). The two hand-replay tests now drive the real stack, and the ladder test gained a structural assertion (`len(sep.block_norms) == len(sep.mamba_blocks)`) so a future replay/forward drift fails loudly |
| `README.md` | five bullets incl. "~4x realtime … 1.8 GB peak RSS" | benchmark subsection + a note that main's figure was wrong by 15x | **main's bullets verbatim**, and the audit-machine table stays - but its "~0.25–0.3x realtime" claim was **this branch's arithmetic error** (300 s of audio in 104.0 s is 2.9x *faster* than realtime, not 4x slower). Corrected: the measurements now corroborate main's ~4x (4.35x one pass, 2.9x chunked) and the "271 MB/audio-second" activation ceiling is left in `AUDIT-test-suite.md`, where it is defined, instead of being conflated with end-to-end peak RSS |
| `TODO.md` §18.1 | `EXPR-1` closed, `SEP-2` opened, `MIDI-5` open with no lock, MIDI-6 citing a test name that has never existed | `MIDI-5`/`CKPT-1`/`SEP-1`/`JSON-1`/`DOC-1` closed | main's list as the base; this branch's closures grafted by id; `SEP-2` closed because main's own cure is in the tree; `MIDI-5` re-closed; the stale citation repaired; header counts re-measured (never copied) |

### Two defects the merge itself found

1. **`utils/midi_targets.py` did not import.** Git auto-merged main's
   `build_tempo_map = _build_tempo_map` alias *above* the function that HEAD defines, so
   `import utils.midi_targets` raised `NameError` - and because `utils/__init__.py`
   re-exports it, every test module importing anything from `utils.*` failed at
   collection. No conflict marker was involved: a clean, silent, tree-breaking merge. The
   first thing that surfaced it was `tests/test_pipeline_integrity.py` erroring on
   import, which is why the suite is run before *and* after the doc pass of every merge here.
2. **A stale "known bug" is worse than no known bug.** Both `main`'s
   `scripts/verify_numerics.py` and `tests/test_numerics.py` proved the C-1/SEP-2
   collapse by re-implementing the buggy stack inline instead of calling the model. After
   PR #13 landed the normed residual stack, that replay kept producing the collapse, so
   the "bug is still here" evidence was self-generated. The merged versions drive the real
   `forward` and assert the structure they replay; `SEP-2` is closed on that basis.

### Was anything from PR #13 lost? (checked against its final tip)

`#13` was force-pushed once more before it was closed: its head is now `535ee86`
(“Add SPDX headers to the verification scripts”) on top of `985549f` + `ad50bc7`,
whereas this branch merged `50e2d46`. `git diff 535ee86 HEAD` over the product files
lists 92 removed lines, and each one is a line this merge deliberately replaced rather
than dropped: the pre-chord `_logits_to_midi` body, the truncating velocity mapping, the
unconditional `_expression_messages` call (now gated by `include_expression`),
`DEFAULT_TEMPO_US` hard-coded in the writer (now the `tempo_us` parameter), the duplicated
`pitch_bce_loss_fn`, and `weights_loaded`/`checkpoint_loaded` reading only the attribute
(now `getattr(model, "weights_loaded", _model_trained)`, one value for three spellings).
The one commit that was new since this branch read #13 — the SPDX headers — is already
satisfied: the tracked-file sweep reports 0 missing headers, which is what CI's
`license` job checks.

### Re-measured on the round-2 merge

```bash
$ .venv/bin/python -m pytest -q -m "not cuda" --strict-deps --fail-on-skip
381 passed, 1 skipped, 3 deselected, 35 xfailed in 57.13s        # exit 0
$ .venv/bin/python -m ruff check .        # All checks passed!
$ .venv/bin/python -m black --fast --check .   # 88 files unchanged
$ for f in scripts/verify_{midi,api,numerics}.py; do .venv/bin/python $f; done   # all exit 0
$ for f in $(git ls-files '*.py' | grep -v '^research/'); do head -3 "$f" | grep -q SPDX || echo "$f"; done
                                                              # no output = 0 missing SPDX headers
```

`main` @ `fda6071` on the same interpreter, without the honesty flags: 362 passed /
1 skipped / 3 deselected / 39 xfailed over 405 collected (measured in a throwaway
worktree with the same venv). The delta is +19 tests (PR #19's chord/expression coverage
plus this branch's guards) and −4 xfails. The four are exactly the locks this branch turns
into pass-today guards because the fixes are in the merged tree: `SEP-1` and `JSON-1` (PR
#13), `DOC-1` (PR #15's docs plus the new writer-vs-doc guard) and `SEP-2` (PR #13's
pre-norm residual stack). `EXPR-1`, `MIDI-5` and `CKPT-1` are documentation closures on
`main`, not markers - main had already converted those locks itself.

