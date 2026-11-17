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
| C-3 | Confidence written to CC#127 (`Poly Mode On`, a Channel Mode message) | `main`'s convention is kept: default **CC#16**, `MIDI_CONFIDENCE_CC` override, CC#120–127 rejected unless explicitly requested | parsed controllers `{127}` → **`{16}`**, no channel-mode writes |
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
are `mamba_ssm`-only checkpoint tests. `ruff check .` and `black --check .` are clean.

New/rewritten coverage on this branch:

- `tests/test_numerics.py` — block-RMS ladder, mask content, mixture consistency, silence, `dt` init
  range, chunked-vs-unchunked scan equality, MC-dropout uncertainty.
- `tests/test_midi_targets.py` — tempo honoured (120/60 BPM, mid-file tempo change), multi-hot chords.
- `tests/test_midi_roundtrip.py` — rewritten to the merged contract (clock, polyphony, durations,
  CC#16 default, CC#127 opt-in, retrigger truncation, marker track).
- `tests/test_midi_generation.py` — CC#16 default, explicit override, no Channel-Mode writes.
- `tests/test_confidence_injector.py` — frame-grid mismatch, non-finite input rejected, relative
  energy review flags.
