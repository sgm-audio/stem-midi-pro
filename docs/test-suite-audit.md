# Test-suite audit — `stem-midi-pro`

**Scope:** `tests/` at `6ab1a1e` (branch `arena/c200cbcb-stem-midi-pro`), read against
`main.py`, `api.py`, `models/*`, `utils/*`, `data/datasets.py`.
**Question asked:** does this suite protect MIDI usability and CPU practicality, or
does it only prove that a sine wave survives a forward pass?
**Answer (as found):** it mostly proved the latter. The suite was green and
*honest about skipping* `mamba_ssm`, but not honest about **output correctness**:
it passed 96 tests while shipping MIDI no DAW could use (notes placed **60× too
late**, chords serialised into arpeggios), a `bass.mid` that was a copy of
`guitar.mid`, silence transcribed as ~87 notes/second, and a pipeline needing
**~14 GB** for the ten-minute upload its own API advertises.

**Status: 20 defects found, all 20 resolved.** Resolution is split by *where*
it landed, because `main` moved under this branch: an independent review pass
(merged as PRs #11/#14) fixed the MIDI writer, tempo tracking, sample-rate
conversion, the streaming path and the backend wiring — rows marked
**upstream** in §3.1. The rest — input validation, silence/clipping/DC handling,
chunked file processing, review-gate semantics and weights provenance — landed
in this branch (**here**). Every defect is guarded by a test that failed on the
old code. The last one — expression output (BUG-MIDI-9 / EXPR-1) — was resolved
by porting PR #13's expression work: the head's channels are now carried per
note and labelled as untrained, and the writer emits pitch-bend/CC1 on explicit
opt-in (`include_expression=True`), because writing an untrained head's output by
default would fabricate expression. Everything below is the original audit, with resolution status
recorded in §3.1 and §6.

---

## 0. Verification method

Everything below was measured, not inferred. Environment: Python 3.11 + torch
2.15 CPU wheel (the repo targets 3.13; the behaviour audited is pure Python/torch
and version-independent), 2 vCPU, 3.9 GB RAM, `mamba_ssm` absent
(`MAMBA_BACKEND == "pure-torch"`).

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt
PYTHONPATH=. .venv/bin/python -m pytest -q -m "not cuda"
# at audit time (6ab1a1e): 96 passed, 2 skipped  (with torchaudio)
# defects pinned:          119 passed, 8 skipped, 33 xfailed
# after main's own review suite landed: 336 passed, 1 skipped, 42 xfailed
# (the xfails are main's `known_bug` pins, strict: they fail the run when fixed)
```

The 33 `xfail` pins did their job: each turned green exactly when its defect was
fixed, forcing the marker to be deleted rather than left behind. Eight of them
went green because `main` fixed the defect independently (same root cause, same
fix), which is why those rows are attributed upstream rather than here.

**First finding is about the headline number itself.** `96 passed, 2 skipped` is
only reproducible with `torchaudio` installed. Without it the suite reports
`90 passed, 8 skipped` and *still exits 0* — six tests of `data/datasets.py`
(the only coverage of that module) silently evaporate. Worse, without
`fastapi`, `tests/test_checkpoint_load.py::test_lifespan_calls_module_load_from_checkpoint`
does not skip — it **fails**, because it imports `api` at module level while every
other API test uses `importorskip`. The suite is therefore green-or-red depending
on which extras a contributor happens to have; "96 passed" is not a stable
contract. (`importorskip("fastapi")` has been added to that test in this branch;
the `torchaudio` gap still needs the CI install to change.)

---

## 1. Coverage of the critical paths

Legend: **✅ real** = exercises production code with meaningful assertions;
**⚠️ thin** = reaches production code but asserts almost nothing;
**❌ absent** = no test reaches this behaviour.

| Critical path | Existing coverage | Verdict |
|---|---|---|
| Mamba **separator** inference | `test_mamba_separator.py`: 5 tests on `torch.randn(1,1,8192)`, asserts shapes + `isfinite` + `guitar != bass` | ⚠️ thin — no musical input, no silence/clip/length, no metric meaning |
| Mamba **transcriber** inference | `test_mamba_transcriber.py`: shapes, `[0,1]` bounds, finiteness | ⚠️ thin — never asserts a *pitch* or an *onset* is meaningful |
| Mamba **end-to-end** (`StemMidiModel`) | via `mock_model` in most tests, always at `d_model=32` | ⚠️ thin — never at production dims, never checked for artefact plausibility |
| **MIDI** onset timing | count, CC#127, ordering, velocity clamp **+ `test_midi_timing.py` (absolute ticks, chords, tempo, round-trip)** | ✅ covered — was absent at audit time |
| **MIDI** pitch correctness | `test_midi_usability.py` (per-stem registers, bass range), `test_midi_to_targets_keeps_every_chord_pitch` | ✅ covered |
| **MIDI** expression (bend/vibrato/slide) | `expression_head` output shape only | ✅ written on request (`include_expression=True`), labelled untrained (BUG-MIDI-9 → EXPR-1) |
| **MIDI** confidence | CC#127 == `int(conf*127)`; `needs_review` presence **+ `test_quality_gate_truthfulness.py` (flags vs gain, notes vs frames, summary agreement)** | ✅ covered |
| **Quality gates** | `test_quality_gates.py` (pure functions) **+ `test_quality_gate_truthfulness.py` (report built by real inference, provenance)** | ✅ covered |
| **Low memory / 4–8 GB target** | `test_perf_smoke.py` (1 s) **+ chunking test + 60 s RSS test** | ✅ covered for realistic input |
| **`mamba_ssm` absent** | `test_mamba_backend_honesty.py` (blocked-module subprocess, signature, causality, provenance) | ✅ covered |
| **Real backend path** (`demucs`, `basic_pitch`, `fake`, `MambaSeparatorBackend`) | `test_backends.py` + `test_backend_wiring.py`; configs now reach `StemMidiModel` | ✅ wiring covered (demucs/basic_pitch themselves remain uninstalled in CI) |

### 1.1 Mamba separator/transcriber: real, but only as a shape test

`test_mamba_separator.py` and `test_mamba_transcriber.py` do construct the real
classes (no fake backend) — that is better than it first looks. But every input
is `torch.randn`, and every assertion is a shape or `torch.isfinite`. Since the
models are **untrained**, that is the only thing that *can* be asserted — which
is precisely the point: the tests cannot detect a broken model, only a broken
tensor. Two consequences:

* Nothing checks the **separation metric** is a separation metric. `si_sdr` is
  `clamp(10*log10(centroid_gap)+20, 0, 30)`; measured on 1 s of audio it returns
  20.15 for *heavily clipped noise*, 20.92 for a *square wave*, 2.95 for a clean
  sine and 0.00 for *digital silence*. A 20 dB "SI-SDR" for clipped noise is a
  gate input that means nothing, and `min_si_sdr: 20.0` is compared against it.
* Nothing asserts the **state cache is absent**. `separator(...)` returns `None`
  in its place; a test that asserts `state is None` (present in the new suite)
  documents C-2.5 as a contract instead of a comment.

### 1.2 MIDI round-trip: the gap that hides a 60× error

The two MIDI files contain 16 tests and not one of them asserts an absolute tick
value. `test_multi_event_ordering` asserts `times == sorted(times)` — which is
satisfied by *any* monotone drift, including the 60× one. The event spacing
chosen by the fixtures (`frame_step=86 ≈ 1 s`) also happens to be wider than the
default note duration, so the chord-collapse bug never triggers in the fixtures
either. Two independent, user-visible defects live in that blind spot (§3,
BUG-MIDI-1/2). On top of that: **expression heads are never written to MIDI**
(no pitch-bend, no CC#1), so the "expression detection" half of the product is
untested *and* undelivered.

### 1.3 Quality gates: correct arithmetic, meaningless inputs

`test_quality_gates.py` is genuinely good unit coverage of `quality_tier` and
`route_by_quality` — including the artefact-forces-COMPLEX rule. The weakness is
upstream: every report is hand-built, so nothing verifies that the numbers fed
in come from the audio. Measured: a **silent** file yields
`artifact_flags=['none']`, and a **hard-clipped** file yields `['noise_floor']`
(not `clipping`). The gate is therefore structurally unable to reject the two
most common bad uploads.

### 1.4 Low-memory behaviour (fixed and covered, see §3.1)

`test_perf_smoke.py` measures RSS on **1 second** of audio with **test-sized**
models (`d_model=32`). The production config is 62.2 M params
(`d_model=768`, 12 separator layers). Measured scaling with the real config:

| audio | wall | peak RSS | realtime factor |
|---|---|---|---|
| 1 s | 0.24 s | 827 MB | 0.24× |
| 10 s | 2.69 s | 1033 MB | 0.27× |
| 30 s | 8.9 s | 1667 MB | 0.30× |
| 60 s | 17.9 s | 2346 MB | 0.30× |

≈ **23 MB per second of audio, linear**, because `process_audio_file` never
chunks: `cfg["audio"]["chunk_duration_sec"]` is read *only* by
`process_audio_streaming`. A 4-minute track projects to ≈6.4 GB (OOM on the 4 GB
target), a 10-minute one to ≈14.5 GB. The CPU figure (0.30× realtime) is
acceptable-ish; the memory curve is not, and no test sees it.

### 1.5 `mamba_ssm` absent (now asserted, see §4.5)

The fallback works — `import main` succeeds with `mamba_ssm` blocked, and
`MAMBA_BACKEND` correctly reports `"pure-torch"`. `MAMBA_BACKEND` was
**written and never read** — no log line, no `/model-info` field, no report
field. It is now exposed as `StemMidiModel.mamba_backend` and
`/model-info.mamba_backend`.
An operator cannot tell the reference scan from the CUDA kernel, and the two
differ by ~10× in throughput. Meanwhile `test_checkpoint_load.py` guards its two
real tests with `importorskip("mamba_ssm")`, which is the right call but means
the CUDA path is exercised by **zero tests** on every CPU-only runner. See §5
for the policy that fixes this without faking the package.

---

## 2. Tests that only ever see synthetic data, mocks, or the fake backend

| Test / module | What it actually exercises | Why it is weak |
|---|---|---|
| `conftest.py::audio`, `stereo_audio`, `temp_*_file` | 0.3-amplitude 220/110 Hz sine | one spectral shape, no transients, no noise, no dynamics, no silence |
| `conftest.py::mock_model` (used by ~25 tests) | real Mamba at `d_model=32` | tiny dims hide the memory/CPU curve; random weights hide output quality |
| `test_headline`: `test_backends.py` (all 7) | `fake` backend + factory | fake returns fixed `0.6/0.4` gains, `si_sdr=12.0`, one middle-C — asserts nothing about the real backends |
| `test_backends.py::test_stem_midi_model_with_fake_backends` | **misnamed**: builds `StemMidiModel`, which ignores `cfg["backends"]` and instantiates `MambaSeparator` | the name promises fake-backend coverage that does not exist (BUG-WIRING-1) |
| `test_api_integration.py::_FakeModel` + `np.zeros(4410)` stems | API plumbing | exercises zip/JSON formatting against **zeroed** audio; cannot catch anything about content |
| `test_quantization.py::FakeModel` | quantize flag plumbing | never compares quantized vs fp32 **numerics**, despite the name |
| `test_api_auth.py::test_process_requires_api_key` | auth | monkeypatches `_do_process`; no inference |
| `test_dataset_loaders.py` (6 tests) | `synthetic=True` path only | `_get_real_item` is a placeholder that calls `_get_synthetic_item`, so the "real dataset" path has **no** implementation and no test; also skipped entirely when `torchaudio` is missing |
| `test_losses.py` | `torch.randn` tensors | asserts finite + range; no test that a *perfect* prediction gives ~0 loss or that a wrong one is larger (the only assertion that would catch a sign/scale error) |
| `test_mamba_separator/transcriber.py` | `torch.randn` | shapes/finiteness only (see §1.1) |
| `test_perf_smoke.py` | 1 s, test dims | see §1.4 |
| `test_midi_*.py` | hand-written event dicts | never driven from model output; no absolute timing |

**Missing numerical/edge cases, in priority order:** silent input; input below
one STFT window; hard-clipped input; DC offset; NaN/Inf samples; sample-rate
mismatch (8 k / 48 k uploads) and its aliasing; extreme polyphony; long files
(memory); non-120 BPM MIDI; MIDI with overlapping notes; empty/zero-note
transcription; and *any* comparison between what the model produced and what a
musician would consider plausible.

---

## 3. Defects the audit found (these are what the new tests pin)

Each is reproduced deterministically by a test, marked
`xfail(strict=True)` with the ID below, so CI stays green while the defect is
open and **goes red the moment it is fixed** (forcing the marker's removal).

| ID | Defect | Evidence |
|---|---|---|
| **BUG-MIDI-1** | `utils/midi.py` computes `ticks_per_second = ticks_per_beat * (60 / sec_per_beat)` = 480×120 = **57600** instead of 480×2 = 960. Every note is 60× too late; a 2 s phrase is written 2 minutes into the file | frame 100 → tick **66873**, expected 1115 (exactly 60.00×) |
| **BUG-MIDI-2** | The delta cursor advances past each `note_off`, then `max(0, delta)` clamps. Simultaneous pitches are serialised | 4-note chord → ticks `{33436, 33556, 33676, 33796}` instead of one tick |
| **BUG-MIDI-3** | Because of 1, `build_midi_from_events` → `midi_to_targets` (the project's own writer into its own reader) is 60× misaligned | onset frames return 60× off |
| **BUG-MIDI-4** | `midi_to_targets` ignores `set_tempo`; assumes 120 BPM | a 60 BPM file places a 1 s note at frame 43 instead of 86 |
| **BUG-MIDI-6** | `process_audio_file` sets `midi_guitar` and `midi_bass` to the **same dict**; the transcriber only ever runs on the guitar stem | `bass.mid` contains pitch 84 (C6) and the identical note list |
| **BUG-MIDI-7** | `_logits_to_midi` emits one event per frame with no refractory window and no onset merging | **86 notes in 1 s** of monophonic audio |
| **BUG-MIDI-8** | `process_audio_streaming` replaces `pitch_logits` and `velocity` with `torch.zeros` before `_logits_to_midi` | every streamed note is pitch **0** (C-1) at velocity 1 |
| **BUG-MIDI-9** | `expression_head` output is discarded; no pitch-bend/CC ever written | grep + `_logits_to_midi` signature |
| **BUG-GATE-1** | `summary.low_confidence_count` (confidence-driven) contradicts per-note `needs_review` (alignment-driven) | summary says 0, 86/86 events flagged |
| **BUG-GATE-3** | `needs_review = conf * sigmoid(energy*10) < min_confidence` — **loudness drives review**, so a confident pianissimo passage is always flagged | same events+gain: quiet `[T,T,T,T]`, loud `[F,F,F,F]` |
| **BUG-GATE-4** | `low_confidence_count` counts low-confidence **frames** but is reported as low-confidence **notes** | 2 notes over a 64-frame grid → reports 64 |
| **BUG-GATE-2** | `summary` has no `total_notes`; `ProcessingReport` has no backend/weights provenance | `KeyError: 'total_notes'` |
| **BUG-AUDIO-1** | No energy gate: silence is transcribed at face value | 1 s of zeros → 86 confident note events |
| **BUG-AUDIO-2** | Silence is not flagged: `artifact_flags=['none']` | measured |
| **BUG-AUDIO-3** | Clipping is detected on the model's **output** only, so a clipped input comes back as `noise_floor` | measured `['noise_floor']` |
| **BUG-AUDIO-4** | DC offset unflagged; a single NaN sample propagates into `ProcessingReport` and then into `processing_report.json` as a bare `NaN` literal (invalid JSON) | `si_sdr=nan`, `json.dumps(..., allow_nan=False)` raises |
| **BUG-AUDIO-5** | Audio shorter than `n_fft/2` raises `RuntimeError` from `torch.stft` reflect padding → API answers **500** where it should answer 400 | 1/200/1000-sample clips |
| **BUG-AUDIO-6** | Resampling is `F.interpolate(mode="linear")` — a triangle filter with no anti-imaging | 3.5 kHz tone upsampled 8 k→44.1 k keeps a **−4.4 dB image** at 4.5 kHz |
| **BUG-PERF-1** | `process_audio_file` never chunks (see §1.4) | 6 s file → one 264 600-sample forward with `chunk_duration_sec=2.0` |
| **BUG-WIRING-1** | `StemMidiModel.__init__` bypasses `models.backends.factory` → `backends:` in every config is dead for `main`/`api`/`demo`/CLI, and `configs/model_config.cpu.yaml` (demucs + basic_pitch) silently builds Mamba | `type(model.separator).__name__ == "MambaSeparator"` with `separator="fake"` |
| **BUG-WIRING-2** | No backend/weights provenance in `ProcessingReport`, `/model-info`, or logs | `getattr(report, "backend", None) is None` |
| **BUG-HONESTY-1/2** | With no checkpoint present, `api.load_model` serves a **randomly-initialised** 62 M-param model with HTTP 200; `MAMBA_BACKEND` is never surfaced | `checkpoints/` holds only a README |

> Note on BUG-WIRING-1: it means `configs/model_config.cpu.yaml` — described as
> the production CPU config — has never been the configuration under test, and
> `tests/test_backends.py`'s "fake backend" model test has always been a Mamba
> test. This single wiring defect explains most of the "everything is synthetic"
> feeling of the suite.

### 3.1 Resolution status

"Fixed in" distinguishes the independent review pass that landed on `main`
(PRs #11/#14: `H-*`/`M-*`/`C-*` items) from the work in this branch. Where both
sides fixed the same defect, the upstream implementation is the one kept — it is
strictly better (e.g. the confidence controller is CC#16, not CC#127, because
120–127 are Channel Mode messages; same-pitch overlaps are resolved rather than
only sorted). "Verified by" names the test that failed before the change.

| ID | Fixed in | Fix | Verified by |
|---|---|---|---|
| BUG-MIDI-1 | upstream | `utils/midi.py`: `ticks_per_second = ticks_per_beat * 1e6 / tempo_us` → 960, not 57 600 | `test_onset_tick_matches_audio_time`, `test_midi_length_matches_audio_length` |
| BUG-MIDI-2 | upstream | Two-pass writer: absolute onsets sorted by `(tick, kind)`, `note_off` before `note_on`, same-pitch overlaps truncated (H-6) | `test_chord_notes_share_one_tick`, `test_chord_after_short_note_keeps_its_position` |
| BUG-MIDI-3 | upstream | Falls out of MIDI-1 | `test_own_writer_reads_back_at_the_same_frames` |
| BUG-MIDI-4 | upstream | `utils/midi_targets.py`: running `set_tempo` per track, advance before applying | `test_midi_to_targets_honours_file_tempo` |
| BUG-MIDI-6 | upstream (H-8) | `main.py`: both stems transcribed; `midi_metadata["guitar"]`/`["bass"]` are distinct dicts | `test_bass_midi_contains_no_guitar_range_notes`, `test_guitar_and_bass_midi_are_different_files` |
| BUG-MIDI-7 | upstream (H-6) | Rising-edge peak-picking with `transcriber.onset_refractory_frames` (default 4 ≈ 46 ms) | `test_note_density_is_musical` |
| BUG-MIDI-8 | upstream (C-2) | Streaming recombines the stems, then transcribes once on the full timeline | `test_streaming_notes_are_not_all_pitch_zero` |
| BUG-MIDI-9 | **here** | `_transcribe` carries the expression head; `_logits_to_midi` names its channels (`bend`, `vibrato`, `slide`) and labels each event `source: expression_head (untrained)`; `build_midi_from_events(include_expression=True)` writes pitch-wheel/CC1/CC74, off by default | `test_expression_head_output_reaches_the_midi_when_opted_in`, `test_expression_is_mapped_to_standard_midi_messages`, `scripts/verify_midi.py` |
| BUG-GATE-1/3 | **here** | `needs_review = confidence < min_confidence`; alignment kept for diagnostics only | `test_review_flag_is_independent_of_output_gain`, `test_summary_counts_agree_with_events` |
| BUG-GATE-4 | **here** | `low_confidence_count` counts notes, not frames | `test_low_confidence_count_counts_notes_not_frames` |
| BUG-GATE-2 | **here** | `summary["total_notes"]` added to both stem payloads and the overall summary | `test_summary_exposes_total_note_count` |
| BUG-AUDIO-1/2 | **here** | `silence` flag from the *input*; silent input short-circuits to zero stems/zero notes before the separator | `test_silence_emits_no_note_events_even_with_a_confident_head`, `test_silence_is_flagged_not_silent_pass` |
| BUG-AUDIO-3 | **here** | Input-peak `clipping` flag (stems are scaled down by the masks, which is why the stem test missed it) + `dc_offset` | `test_hard_clipped_input_is_flagged`, `test_dc_offset_is_flagged_or_removed` |
| BUG-AUDIO-4 | **here** | `_validate_audio` raises `AudioValidationError` on non-finite input, at `forward()` as well as the file path | `test_nan_input_does_not_silently_produce_nan_report`, `test_report_serialises_to_strict_json` |
| BUG-AUDIO-5 | **here** | `_validate_audio` rejects `< n_fft` samples; `AudioValidationError` is a `ValueError`, which `api.py` maps to 400 | `test_sub_window_audio_raises_a_validation_error`, `test_single_stft_window_is_accepted` |
| BUG-AUDIO-6 | upstream (H-7) | `_resample_to_target` goes through `librosa.resample` (polyphase) | `test_upsampling_does_not_inject_spectral_images` (image **−4.4 dB → −148.7 dB**) |
| BUG-PERF-1 | **here** | `_separate_chunked` (overlap-add, `chunk_duration_sec`) now backs *both* `process_audio_file` and `process_audio_streaming` | `test_process_audio_file_honours_chunk_duration`, `test_one_minute_upload_fits_in_2gb` |
| BUG-WIRING-1 | upstream (H-9) | `StemMidiModel` builds via `models.backends.factory` (`separator_backend`/`transcriber_backend`) | `test_stem_midi_model_uses_the_configured_backend` |
| BUG-WIRING-2 | upstream (M-7) | `ProcessingReport.backend` = `"separator=…,transcriber=…"` or `"mamba (pure-torch)"` | `test_report_identifies_the_backend_that_produced_it` |
| BUG-HONESTY-1 | **here** | `ProcessingReport.weights_loaded` (set by `load_from_checkpoint`), exposed in `/model-info` | `test_report_records_weights_provenance` |
| BUG-HONESTY-2 | **here** | `StemMidiModel.mamba_backend` property + `/model-info.mamba_backend` | `test_runtime_exposes_which_mamba_implementation_ran` |

Measured after the fixes, production config, 2 vCPU:

| | before | after |
|---|---|---|
| note at `onset_frame=100` | tick 66 873 | tick **1 115** |
| 4-note chord | ticks `33436/33556/33676/33796` | all at **557** |
| notes per second (monophonic) | 86 | **≤ 21.5** (one rising edge per 4 frames) |
| a continuous attack | 86 notes | **1 note** |
| silence | 86 events, no flag | **0 events, `['silence']`** |
| silence on production weights | `NaN` logits → `RuntimeError` (HTTP 500) | **200: 0 notes, `['silence']`** |
| 8 k→44.1 k image at 4.5 kHz | −4.4 dB | **−148.7 dB** |
| peak RSS, 60 s upload | 2 346 MB | **~966 MB (plateau)** |
| wall clock | 0.30× realtime | 0.35–0.46× realtime |

Chunking costs roughly 20–40% wall time on short files because overlapped regions
are processed twice; that is the deliberate trade for bounded memory, and it is
the number to watch if the CPU target gets tighter.

Two notes on the wiring fix, since it changes behaviour rather than only
correctness:

* Honouring `backends:` means `scripts/eval_smoke.py` and
  `scripts/runpod_deploy.py` — which default to `configs/model_config.cpu.yaml`
  (demucs + basic_pitch) — now fail loudly with an install hint when those
  optional extras are absent. That is the intended behaviour: previously they
  silently ran Mamba and reported success. `configs/model_config.yaml` (no
  `backends:` block) still resolves to Mamba, so `demo.py`, the CLI and the API
  default are unchanged.
* The first thing the new clipping detector did was flag the repo's own
  `synthetic_test_audio.wav`: peak exactly 1.0 with 2.3% of samples pinned. The
  flag is correct — that fixture really is clipped.

---

## 4. New tests added in this audit

Six modules, 65 tests: **32 assert behaviour that was already correct, 33 were
strict-xfail pins on the defects in §3 — all 33 now pass as ordinary
regressions**, and 2 in this set skip until `mamba_ssm` is present (they are the
CUDA-path guards). Runtime added: ~7 s (`-m "not slow"`), ~34 s with the memory
test.

```
tests/test_midi_timing.py              15  pure mido, ~0.3 s   (no torch)
tests/test_midi_usability.py           10  model → SMF artefacts
tests/test_audio_adversarial.py        16  silence/clip/NaN/short/resample/long
tests/test_quality_gate_truthfulness.py 12  gate inputs + review flags + provenance
tests/test_mamba_backend_honesty.py     8  fallback contract + real-path guard
tests/test_backend_wiring.py            4  config → model wiring
```

For each: **name — must assert — why the old suite was weak here.**

### 4.1 MIDI timing (`test_midi_timing.py`)

| Test | Must assert | Why the current suite is weak |
|---|---|---|
| `test_onset_tick_matches_audio_time` | `onset_frame=100` → tick 1115 (not 66873) | no test asserts an absolute tick; "non-decreasing" is satisfied by 60× drift |
| `test_midi_length_matches_audio_length[1,2,5s]` | last note tick ≤ `seconds × 960` | a 5 s phrase used to end at tick 287 555 = 5 minutes of timeline |
| `test_sixteenth_note_spacing_is_one_sixteenth` | 123-tick steps ±1 (integer ticks) | ratio-level check; catches "fixed the constant, broke the cursor" |
| `test_chord_notes_share_one_tick` | 4 pitches at one frame → 1 tick | fixtures space events ≈1 s apart, so overlap never occurs |
| `test_chord_after_short_note_keeps_its_position` | a chord on the next 16th stays there | the `max(0, delta)` clamp is invisible to count-based tests |
| `test_own_writer_reads_back_at_the_same_frames` | build → save → `midi_to_targets` recovers frames | the two test files never meet; targets and writer can diverge silently |
| `test_midi_to_targets_honours_file_tempo` | 60 BPM file: 1 beat → frame 86 | real corpus MIDI is not all 120 BPM |
| `test_midi_to_targets_keeps_every_chord_pitch` | 3-note chord → 3 hot pitch bins (**already holds** — kept as a guard) | would have caught a future one-hot regression |
| `test_empty_event_list_still_parses`, `test_note_count_matches_input_events`, `test_no_zero_velocity_sounding_notes`, `test_explicit_duration_frames_sets_note_length`, `test_events_are_emitted_in_the_order_given` | regressions that already pass | cheap, keep them |

> The strict-xfail discipline immediately caught a bug in *this audit*: the first
> version of the helper recomputed `TICKS_PER_SECOND` with production's formula
> and the timing tests **passed** against a 60× error. The helper is now derived
> independently (`BPM = 60e6/TEMPO_US; ticks = tpb*BPM/60`).

### 4.2 Delivered MIDI artefacts (`test_midi_usability.py`)

| Test | Must assert | Why weak before |
|---|---|---|
| `test_guitar_and_bass_midi_are_different_files` | `bass.mid` note list ≠ `guitar.mid` note list | compares **notes**, not bytes — bytes differ by track name even for a copy (a trap the first version of this test fell into) |
| `test_bass_midi_contains_no_guitar_range_notes` | pitch 84 (C6) never appears in bass | transcriber pinned to 84; proves the copy |
| `test_note_density_is_musical` | ≤ 24 notes/s of monophonic audio | was 86/s; nothing bounded event rate |
| `test_no_duplicate_notes_on_the_same_frame`, `test_every_event_is_a_playable_note` | unique (frame,pitch); velocity 1..127; confidence 0..1; ordered | guards the "note_on with velocity 0" and "unordered ticks" classes |
| `test_summary_counts_agree_with_events` | `low_confidence_count` == count of `needs_review` | report contradicts the MIDI it ships with |
| `test_summary_exposes_total_note_count` | `summary["total_notes"]` exists | key absent |
| `test_streaming_notes_are_not_all_pitch_zero` | streamed notes ≠ `{0}` | streaming zeroes the pitch head; existing streaming test only counts chunks |
| `test_streaming_reports_chunk_count_and_sample_rate`, `test_streaming_never_claims_studio_quality` | chunk count, sample rate, tier ≠ STUDIO | streaming hard-codes `[0,0]` metrics — must not be routable as studio |

### 4.3 Adversarial audio (`test_audio_adversarial.py`)

| Test | Must assert | Why weak before |
|---|---|---|
| `test_silence_emits_no_note_events_even_with_a_confident_head` | 0 events for digital silence | uses a *pinned* transcriber, so the assertion is about the pipeline, not a random seed (the first version used `mock_model` and was seed-flaky — strict xfail caught it) |
| `test_silence_is_flagged_not_silent_pass` | `silence`/`no_signal` in `artifact_flags` | silence reports `['none']` |
| `test_hard_clipped_input_is_flagged` | `clipping` flag for ±1.0-clipped input | clipping is measured post-separation only |
| `test_dc_offset_is_flagged_or_removed` | `dc_offset` flag (or HPF) | DC passes through |
| `test_nan_input_does_not_silently_produce_nan_report` | raises `ValueError`/`AssertionError` at the boundary | NaN reaches the report |
| `test_report_is_strict_json_for_nan_input` | `json.dumps(..., allow_nan=False)` succeeds | `processing_report.json` can contain `NaN`, which strict parsers reject |
| `test_sub_window_audio_raises_a_validation_error[1,200,1000]` | `AudioValidationError`/`ValueError`, i.e. HTTP 400 | `RuntimeError` → HTTP 500 |
| `test_upsampling_does_not_inject_spectral_images` | 8 k→44.1 k image at 4.5 kHz < **−30 dB** re fundamental | linear-interpolate resampler: measured **−4.4 dB** |
| `test_downsampled_length_is_preserved` | 1 s in → 1 s out (passes) | cheap regression |
| `test_twenty_four_tone_cluster_is_finite_and_bounded` | 24-partial cluster: finite stems, finite report, < 30 s | extreme polyphony never attempted |
| `test_full_scale_noise_stems_stay_finite` | 0 dBFS noise → finite (passes) | istft overshoot edge |
| `test_process_audio_file_honours_chunk_duration` | no single forward > `chunk_duration_sec`, and >1 chunk for 6 s | used to be one 264 600-sample forward |
| `test_one_minute_upload_fits_in_2gb` (`slow`) | 60 s upload grows peak RSS < 800 MB over baseline | measured the real path, not a fitted slope: `ru_maxrss` is a high-water mark, and a slope through the new plateau is allocator noise |

### 4.4 Quality-gate truthfulness (`test_quality_gate_truthfulness.py`)

| Test | Must assert | Why weak before |
|---|---|---|
| `test_gate_inputs_come_from_real_inference` | report from the model is finite/consistent; routing action matches tier | every existing gate test hand-builds the report — the integration hop was untested |
| `test_random_weights_never_reach_studio` | untrained model ⇒ tier ≠ STUDIO | **the "dishonest success" guard**: no checkpoints ship, so this is the only thing between random weights and a "direct download" badge |
| `test_nan_metrics_cannot_be_routed_as_studio` | NaN report degrades safely (passes) | NaN comparisons are False; pin it |
| `test_gate_thresholds_are_config_driven` | same numbers tier differently under different configs | thresholds were only tested at their defaults |
| `test_any_artifact_flag_blocks_studio` | any flag ⇒ COMPLEX (passes, kept) | |
| `test_review_flag_is_independent_of_output_gain` | flags identical at 1e-4 vs 1.0 stem energy | loudness drives review |
| `test_low_confidence_count_counts_notes_not_frames` | 2 notes → 2 | reports 64 |
| `test_low_confidence_notes_are_flagged`, `test_high_confidence_loud_notes_are_not_flagged`, `test_cc127_reflects_confidence_not_alignment` | flag semantics + DAW cue independence | internal consistency unverified |
| `test_events_outside_the_confidence_grid_are_handled_safely` | out-of-range onset frame → finite score, needs_review | silent zero-fill path |
| `test_report_records_weights_provenance` | report carries `weights_loaded`/backend | nothing distinguishes "no checkpoint" from "trained, doing badly" |

### 4.5 `mamba_ssm` honesty (`test_mamba_backend_honesty.py`)

| Test | Must assert | Why weak before |
|---|---|---|
| `test_backend_identity_matches_availability` | `MAMBA_BACKEND` == `"mamba-ssm"` iff importable | the flag was never asserted against reality |
| `test_importing_the_model_does_not_require_mamba_ssm` | subprocess with `sys.modules["mamba_ssm"] = None`: `import main` succeeds, reports `pure-torch` | the fallback path was tolerated, never tested |
| `test_fallback_module_is_a_drop_in_for_the_mamba_signature` | `(B,L,D) -> (B,L,D)`, kwargs used by both models accepted | version drift in `mamba-ssm` kwargs |
| `test_fallback_matches_an_independent_recurrence` | **causality**: perturbing frame 5 changes only frames ≥ 5 | proves the fallback is a sequence model, not a reshaped MLP — makes C-2.5 a real limitation |
| `test_mamba_ssm_is_not_a_hard_dependency` | no `mamba` line in `pyproject.toml`/`requirements.txt` | dependency creep guard |
| `test_real_mamba_kernel_is_actually_used` (`skipif` absent, `cuda`) | with the package installed, blocks are `mamba_ssm.Mamba` | **inverts the `importorskip`**: validates the real path on a CUDA box instead of skipping it |
| `test_real_and_fallback_paths_agree_on_shapes` (`skipif` absent, `cuda`) | both backends expose the same tensor contract | parity between paths |
| `test_runtime_exposes_which_mamba_implementation_ran` | `model.mamba_backend` == `MAMBA_BACKEND` | flag is computed and discarded |

### 4.6 Backend wiring (`test_backend_wiring.py`)

| Test | Must assert | Why weak before |
|---|---|---|
| `test_factory_selects_the_configured_backend` | control: the factory does honour config (passes) | separates "factory broken" from "factory bypassed" |
| `test_stem_midi_model_uses_the_configured_backend` | `backends.separator="fake"` ⇒ `FakeSeparatorBackend` | `test_stem_midi_model_with_fake_backends` never checked this; it has always been a Mamba test wearing a fake's name |
| `test_cpu_config_still_declares_external_backends` | cpu config names demucs/basic_pitch | config-rot guard while BUG-WIRING-1 is open |
| `test_report_identifies_the_backend_that_produced_it` | report carries a backend field | fake output (`si_sdr=12.0`, one middle-C) is indistinguishable from a real run |

---

## 5. Keeping the suite honest about `mamba_ssm`

The rule, applied throughout: **never `importorskip` the behaviour you are
testing.** Skip only on absent *hardware or optional third-party packages*, and
assert the fallback's *contract* so the "package missing" branch is covered by
real assertions rather than by silence.

1. **Two-sided guards.** When `mamba_ssm` is absent, tests assert the fallback
   (identity string, signature, causality, shape parity) and skip only the two
   `@pytest.mark.cuda` kernel tests. When it is present, those two run and fail
   if the fallback is silently preferred. There is no configuration in which the
   real path is quietly untested *and* unreported.
2. **Assert on the flag, not the comment.** `MAMBA_BACKEND` must be surfaced
   (report + `/model-info` + one startup log line). Then this line in CI logs is
   a contract, not trivia: `mamba_backend=pure-torch (reference scan, ~10x slower)`.
3. **Never fake the package.** A stub `mamba_ssm` module would let the CUDA
   tests pass in an environment that could never run them — the exact dishonesty
   the task warns about. Simulated *absence* (`sys.modules["mamba_ssm"] = None`,
   in a subprocess so it cannot leak) is legitimate; simulated presence is not.
4. **CI matrix** (recommended): one CPU job runs `-m "not cuda"` and reports the
   active backend in the summary; one self-hosted CUDA job installs
   `mamba-ssm` and runs `-m "cuda"` — that job is the only place the kernel tests
   execute, and it must be required, not allowed to fail.
5. **Make skips visible.** `-ra` is already in `addopts`; add a CI assertion that
   the skip count for `mamba_ssm` is exactly the number of tests marked
   `cuda` (2 today). A rising skip count is how the CPU path rotted to
   "everything is skipped" in the first place — same failure as the six
   `torchaudio` tests that vanish silently today.
6. **Same policy for `demucs`/`basic_pitch`.** They have no fallback and are not
   installable in CPU CI: keep them behind an explicit `integration` +
   `importorskip`, but add the *wiring* tests in §4.6 so "not installed" never
   degrades into "not tested".

---

## 6. Fix order used (items 1–8 done; item 9 partial)

Status: items 1–8 **done**; item 9 is partially done (BUG-MIDI-7/8 fixed,
BUG-AUDIO-5/6 fixed, BUG-MIDI-9 open).

| # | Action | Why first | Effort |
|---|---|---|---|
| 1 | **BUG-MIDI-1** — `ticks_per_second` must be `ticks_per_beat * BPM / 60` = `480 * (60_000_000/tempo_us) / 60` = 960, not 57 600 | every MIDI file the product has ever emitted is unusable in a DAW | 1 line + drop 5 xfail markers |
| 2 | **BUG-MIDI-2** — write note_ons from absolute ticks with a separate `note_off` event list | chords are the product's reason to exist | ~15 lines |
| 3 | **BUG-MIDI-6** — transcribe the bass stem, not a relabelled guitar; stop aliasing the dicts | users get a guitar part in `bass.mid` | ~10 lines |
| 4 | **BUG-GATE-3/4** — `needs_review` from confidence only; count *notes* not frames | the review UI currently flags every quiet note and under-reports in the summary | ~5 lines |
| 5 | **BUG-AUDIO-1/2** — reject or flag inputs with RMS below a floor before transcription | stops the "silence → 87 notes" failure | ~10 lines |
| 6 | **BUG-PERF-1** — chunk `process_audio_file` on `chunk_duration_sec` | the 4–8 GB target cannot survive a 4-minute upload | ~25 lines |
| 7 | **BUG-WIRING-1** — route `StemMidiModel` through `build_separator`/`build_transcriber` | makes the CPU config real and unlocks the fake/demucs backends for tests | ~15 lines |
| 8 | **BUG-HONESTY-1** — refuse to serve (or hard-label) output when no checkpoint is loaded | the "dishonest success" case: random weights, HTTP 200 | ~10 lines |
| 9 | **BUG-AUDIO-5/6, BUG-MIDI-7/8/9** | 400-vs-500, resampler, onset refractory, streaming pitch, expression output | medium |

That is exactly what happened: as each fix landed its strict-xfail test flipped
to XPASS (a CI failure), and the marker was deleted. Items 1–3 and 7 were
subsequently fixed upstream too (PRs #11/#14), in a better form than the pins
assumed; the branch keeps upstream's implementation and its tests. The suite now
guards the deliverable with ordinary assertions rather than pins.

---

## 7. Deliberately out of scope / still uncovered

* **No trained weights exist**, so no test can assert transcription *accuracy*
  (note F1 vs ground truth, pitch error in cents). The honest substitutes used
  here are structural: event density, pitch range per stem, silence rejection,
  and the "random weights must not be routed as studio" gate. When weights land,
  add a golden-file test: fixed 30 s WAV → assert note-F1 ≥ threshold against a
  checked-in reference MIDI, and a `mutation` test that a 1-semitone shift of the
  input shifts the MIDI by 1 semitone.
* **No real SI-SDR** (needs ground-truth stems); the centroid proxy is only
  checked for finiteness/range, not for correlation with separation quality.
  `research/` corpus + Demucs references would enable a proper SI-SDR regression.
* **`demucs` / `basic_pitch`** are not installed in CI; only their wiring and
  config are tested.
* **Expression (bend/vibrato/slide)** cannot be tested end-to-end because
  nothing writes it to MIDI yet (BUG-MIDI-9). The test to add then: a synthetic
  vibrato input must produce pitch-bend events whose direction matches.
* **API-level zip/JSON** assertions live in the new modules only for the
  NaN-serialisation case; the full `/process` path is still monkeypatched
  (`test_api_integration.py`), so `create_response_zip` has never seen real
  model output. Worth an `integration`-marked test with `TestClient` + the
  `mock_model` fixture once BUG-WIRING-1 lands.
