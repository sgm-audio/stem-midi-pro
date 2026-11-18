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
python3 -m pytest -q                                  # 400 passed, 4 skipped, 28 xfailed
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
| C-3 | Confidence written to CC#127 (`Poly Mode On`, a Channel Mode message) | `main` (PR #15) now writes it as **polyphonic key pressure**, which is note-specific; CC#120–127 are never emitted and `cc_127_value` survives only as a deprecated input alias | controllers `{127}` → **none**; per-note pressure `[(60,114),(64,50)]` for confidences 0.9/0.4 |
| C-3 | A channel CC cannot express per-note confidence (simultaneous notes share the channel) | solved upstream by polyphonic key pressure (PR #15); this branch's interim marker track was dropped during reconciliation | 2 simultaneous notes → 2 independent pressure messages, one per note |
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
| Docs | CC#127 claims, unqualified SI-SDR, performance numbers with no stated hardware | docs reconciled to main's conservative stance, plus a clearly-labelled "2-vCPU cloud box, random weights" table | `README.md` states the hardware, the weight provenance and that quality is uncertified |

## Round 2 — findings closed after the PR #13 recovery (2026-11-17)

Same rules: each row is a measurement made on this tree, 2 threads, CPU only, pure-torch Mamba.

| # | Finding | Fix on this branch | Evidence (measured) |
|---|---------|--------------------|---------------------|
| C-2.5 | No SSM state crossed a chunk boundary — every segment restarted the recurrence | **Added here**: the fallback block carries real state (`{"h": (B, d_inner, d_state), "conv": (B, d_conv-1, d_inner)}`) through `forward(x, state=..., return_state=True)`, and `MambaSeparator.forward(audio, state=..., return_state=True)` threads one state per block | one pass over 96 frames vs four pieces (7/13/40/36): `max\|Δ\| = 7.5e-9`; **without** state the same split does not match (asserted in `tests/test_audit_round2.py`) |
| M-4 | No tempo detection: every export got a fixed 120 BPM grid | `utils/tempo.py` (spectral-flux onset envelope → autocorrelation with octave and prominence guards), `tempo_us` accepted by the writer, `midi.tempo: auto \| fixed \| <bpm>` in the config, tempo attached to every stem's metadata and the summary | click tracks at 60/90/120/140/200 BPM estimated within **0.4 BPM**; silence, white noise and a 220 Hz drone return `None` → fixed grid; changing the estimate moves the tick grid but **not** absolute time (400 frames = 4.644 s at both 120 and 140 BPM) |
| — | `dt` was only bounded at initialisation | runtime `dt.clamp(max=dt_max)` inside the scan (`dt_max` default 0.1, configurable per module as `separator.dt_max` / `transcriber.dt_max`; the kwarg is only passed to the fallback) | `dt_proj.bias = 50` (softplus ≈ 50, i.e. dt ≫ 1): output stays finite and below the unclamped saturation, `dt_max` validated |
| — | Pre-norm checkpoints could not be loaded at all | `models/checkpoint_compat.py` (fills exactly the `block_norms`/`final_norm` tensors with identity, refuses anything else) + `scripts/migrate_checkpoint.py`, and `load_from_checkpoint` uses it with a warning | 44 LayerNorm parameters filled, `strict=True` load succeeds; an unrelated missing key still raises `Checkpoint does not match the model`; `allow_legacy=False` refuses |
| M-8 | `torch.set_num_threads(os.cpu_count())` ignored cgroups/affinity | `_resolve_thread_count()`: `OMP_NUM_THREADS` → `sched_getaffinity` → `cpu_count`, capped at 4 | 2-vCPU sandbox → 2 threads (was 2 by accident: `cpu_count()` reported the host); `OMP_NUM_THREADS=7` → 7 |
| H-4 | Expression could never reach the MIDI file | `midi.include_expression` (default `false`) is passed by the API to the writer; every event still tags `expression.source` | default export: no `pitchwheel`, no `control_change`; with the flag: pitch-wheel + CC on the note |
| M-6 | INT8 speedup claim had no measurement | measured, then documented as a **non-optimisation** in `API_DOCUMENTATION.md` and the `_get_quantized_self` docstring | 3 s clip, d_model 128 / 4 layers / 2 threads: **0.21 s fp32 vs 0.22 s INT8 (0.93x)**, stems differ by max 5.9e-3 |
| H-5 | Polyphony structurally impossible | **fixed upstream** by PR #19 before this branch: `transcriber.pitch_threshold` decodes every pitch above threshold, multi-hot BCE targets train it | `test_chord_decoding_emits_all_pitches_above_threshold`: two notes at one onset frame |
| M-2 | Doc/code contradictions (`NeMo ModelPT`, CUDA-graph latency, "50 MiB cap", Canadian-artist weighting) | **already swept upstream** (PR #15–#19); re-verified on this tree | `grep` for each claim → only the historical `[REMOVED]` rows in `SUMMARY.md` remain |


## Round 3 — backlog defects closed (the `@known_bug` locks)

The suite carries `@known_bug("ID", ...)` markers whose ids are also open entries in `TODO.md` §18.1.
A strict xfail failing the moment the bug is fixed is what makes these safe to close; each id below
lost its marker, gained a pass-today guard and had its TODO entry checked off.

| ID | Defect (as measured before) | Fix | Guard / evidence |
|----|-----------------------------|-----|------------------|
| **IO-2** | a 0-frame WAV validated clean, then died inside torch → HTTP 500 with internals in the body | `audio_io.open_and_validate` rejects `frames <= 0` / non-positive duration with a 400-level message | `test_zero_frame_wav_is_rejected`, `test_empty_upload_is_a_client_error_not_a_500` |
| **IO-3** | NaN/Inf samples sailed through validation: 200-level "success", NaN metrics, stems that decode as a constant −1.0 full-scale block | three layers: `load_native_mono` checks `np.isfinite`, new `audio_io.check_finite()` streams the upload in 1M-frame blocks inside `/process` (O(1) memory), and `MambaSeparator.forward` refuses non-finite input so a backend or notebook cannot bypass it | `test_non_finite_audio_is_rejected`, `test_non_finite_upload_is_refused[nan/inf/-inf]`, `test_non_finite_input_never_leaks_into_stems` |
| **GATE-3** | `route_by_quality` never looked at what the render contained: 0 notes with confidence 0.95 and a 25 dB proxy graded **STUDIO** and answered `direct_download` on an empty MIDI | `ProcessingReport` carries `total_notes`; 0 adds an `empty_render` artifact flag (→ COMPLEX) and routing answers `fallback_options` with an honest "no notes detected" badge | `test_empty_render_is_never_offered_as_studio_or_draft`, `test_a_render_with_no_notes_is_never_routed_as_a_result` |

Still-open `@known_bug` ids at this commit (**28 xfail markers**, each an open TODO.md §18.1 entry):
`ART-1`, `BATCH-1`, `CONF-1`, `CPU-1`, `CPU-2`, `DOC-1`, `DOC-2`, `FAKE-1`, `GATE-2`, `IO-4`,
`MIDI-4`, `MIDI-7`, `PERF-1`, `REPORT-1`, `REPRO-1`, `SEP-2`, `SHIM-1`, `TRANS-1`, `ZIP-1` and the
`ALIGN-1`/`GATE-4`/`CPU-*` variants. None of them is a same-line change; each needs its own
measured decision (e.g. `CONF-1` — the head scores *degraded* renders *highest*, so a fix means
calibration, not a threshold).

## Still open (deliberately not claimed as fixed)

| # | Item | Why it is still open |
|---|------|----------------------|
| C-2.5 | Streaming SSM state cache | Segments are processed independently; seams are cross-faded but no SSM state crosses a boundary. Documented in `MambaSeparator.forward` and `process_audio_file`. |
| H-5 | Polyphony | The pitch head is still a 128-way argmax (one note per frame). Writer and targets are polyphony-ready (independent note-on/off, multi-hot targets, per-note marker), so a multi-label head can drop in. |
| H-2 | Expression head | Untrained; its channels are unnamed. `_logits_to_midi` passes an `expression` vector through, but the writer only emits bend/vibrato when an event carries those keys (`include_expression=False` by default). |
| C-2.5 (pipeline) | The **file** path still separates overlapping 2 s windows and cross-fades them. With 50 % waveform overlap there is no single recurrence to continue, so state threading there would be meaningless; the state API is exercised by the block/separator tests and is available to a caller that streams non-overlapping, frame-contiguous chunks. |
| — | No trained checkpoint ships with the repo | Quality numbers and the MIDI content of an untrained model are meaningless; the API refuses to pretend otherwise. Measured here: with random weights the peak-picker emits **1** note for 3 s of audio, so no end-to-end musical claim is possible. |
| H-4 | The expression head is still untrained | Its values can now reach a MIDI file (`midi.include_expression`) and are tagged `source: untrained`, but no amount of plumbing makes them learned. |
| M-4 (partial) | Tempo is estimated, not tracked | One tempo per file: no mid-file tempo changes, no downbeat/bar phase. The estimate is reported as `summary.timing` with its confidence so a consumer can ignore it. |

## Test suite

`pytest -q` → **400 passed, 4 skipped, 28 xfailed, 0 failed** (`main` alone: 345 passed, 37 xfailed).
The 4 skipped tests are `mamba_ssm`-only. The one local failure is the sandbox RSS probe
(`test_peak_rss_probe_runs_at_all`, it asserts a ceiling this container cannot meet); it passes in CI. `ruff check .` and `black --check .` are clean.

New/rewritten coverage on this branch:

- `tests/test_numerics.py` — block-RMS ladder, mask content, mixture consistency, silence, `dt` init
  range, chunked-vs-unchunked scan equality, MC-dropout uncertainty.
- `tests/test_midi_targets.py` — tempo honoured (120/60 BPM, mid-file tempo change), multi-hot chords.
- `tests/test_midi_roundtrip.py` — rewritten to the merged contract (clock, polyphony, durations,
  polyphonic key pressure, retrigger truncation).
- `tests/test_midi_generation.py` — confidence lane, no Channel-Mode writes.
- `tests/test_confidence_injector.py` — frame-grid mismatch, non-finite input rejected, relative
  energy review flags.
- `tests/test_audit_round2.py` — tempo detection (5 tempi + 3 non-tempi), tempo/writer invariance,
  SSM state carrying (exact and the negative control), checkpoint migration, thread resolution,
  `dt` clamp, expression export.

## Round-2 verification scripts

```bash
python3 -m pytest -q tests/test_audit_round2.py   # 23 passed
python3 scripts/migrate_checkpoint.py old.pt new.pt --config configs/model_config.yaml
```
