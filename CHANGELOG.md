# Changelog

All notable changes to Stem+MIDI Pro. Currently maintained manually; release automation (e.g. `git-cliff`) is planned — see TODO DOC-10.1.10.

## [Unreleased] — Production-readiness effort

Work is tracked item-by-item in [TODO.md](TODO.md); this summarizes the themes.

### Correctness pass: test-suite audit follow-through (2026-11-17)
Audit of `tests/` against what the product actually ships; the write-up and
measurements are in [`docs/test-suite-audit.md`](docs/test-suite-audit.md).
The MIDI-writer, resampler, streaming and backend-wiring findings were fixed
independently on `main`; the items below are what this branch adds on top.

**Input validation** (`main.py`, `audio_io.py`)
- `_validate_audio` rejects non-finite samples and clips shorter than one STFT
  window with `AudioValidationError`, from `forward()` as well as from the file
  path. A NaN upload used to reach `processing_report.json` as a bare `NaN`
  token, or fail as a 500 from inside the separator; the API answers 400.
- Silent input short-circuits to zero stems / zero notes before the separator
  runs. On the production config the untrained Mamba stack emits NaN logits for
  all-zero input, so a muted channel returned `RuntimeError` → HTTP 500.

**Artifact flags** (`main.py`)
- `_detect_artifacts` also takes the original input: input-peak `clipping`,
  `silence` and `dc_offset`. A clipped DI is scaled down by the separator
  masks, so the stem-level clipping test never fired for clipped uploads; a DC
  upload was removed by the masks and silently cost input headroom.

**CPU memory** (`main.py`)
- Chunked separation factored into `_separate_chunked` and used by
  `process_audio_file` as well as `process_audio_streaming`; `_finalize` builds
  MIDI + report + routing once for both. 60 s upload peak-RSS growth
  ~2 346 MB → ~190 MB, for 20–40% extra wall time on the re-processed overlap.

**Review gates** (`models/confidence_injector.py`)
- `needs_review` is `confidence < min_confidence`, not
  `confidence * sigmoid(energy*10)`: a confident quiet passage was always
  flagged and gain alone flipped the same events. Energy stays in
  `alignment_score` as a diagnostic.
- `low_confidence_count` counts notes rather than STFT frames (1 s of audio
  reported ~86 "low-confidence notes"); `summary.total_notes` added.

**Provenance** (`main.py`, `api.py`, `utils/quality_gates.py`)
- `ProcessingReport.weights_loaded` (set by `load_from_checkpoint`) and
  `StemMidiModel.mamba_backend`, both exposed by `/model-info` next to the
  existing `backend`.

**Tests** — six audit modules (`test_midi_timing`, `test_midi_usability`,
`test_audio_adversarial`, `test_quality_gate_truthfulness`,
`test_mamba_backend_honesty`, `test_backend_wiring`). Two pre-existing `main`
failures are fixed as a side effect (`_detect_artifacts` used unbound
`self`-qualified calls and raised `AttributeError` when called as
`StemMidiModel._detect_artifacts(None, …)`), and one `main` test that
contradicted the documented "the saturated centroid proxy must not decide the
tier" contract was corrected. Suite: 336 passed, 42 xfailed (known-bug),
1 skipped.

### Strategic decisions (TODO Section 0)
- Consolidated on the v1 production path: `main.py` / `api.py` / `models/*` / `train.py` / `data/datasets.py`. Mamba-3 research code moved to `research/`.
- Locked the platform: **CPU-only, bare-metal i5, Python 3.13**.
- License: **Apache-2.0** (`LICENSE`; relicensed from PolyForm Small Business 1.0.0 on 2026-10-08, see ADR 0006).

### Structural (TODO Section 1)
- Moved Mamba-3 files (`per_track_processor.py`, `train.py`, `losses.py`, datasets, tests) into `research/mamba3_per_track/`; vendored mamba-ssm clone into `research/mamba-ssm-reference/`.
- Renamed `data/canadian_datasets.py` → `data/datasets.py`; classes renamed (`AudioDataset`, `Slakh2100YourMT3Dataset`, `MUSDB18HQDataset`, `get_data_loaders`).

### API hardening (TODO Section 6)
- Upload streaming with a 50 MiB cap and `413` responses; audio validation (format, duration, sample rate).
- Optional API-key auth (`X-API-Key`) and configurable CORS allowlist.
- Template rendering endpoints with path-traversal protection.

### Documentation (TODO Section 10)
- `README.md` rewritten with an honest quickstart, CPU performance expectations, and the license summary.
- `SUMMARY.md`, `ARCHITECTURE.md`, `USER_GUIDE.md`, `API_DOCUMENTATION.md`, `DEVELOPMENT_GUIDE.md` brought in line with the actual code (features tagged `[IMPLEMENTED]`/`[PARTIAL]`/`[PLANNED]`/`[REMOVED]`; removed TensorRT-LLM / FP8 / CUDA claims).
- New `SETUP.md`, this changelog, `docs/architecture-decision-records/` (ADRs 0001–0005), and `docs/operations/` (runbook, monitoring, upgrading, rollback).
- User-facing templates in `user_content/` audited against the real processing-report keys.

### Known outstanding work
- NeMo/PyTorch-Lightning removal from the training loop (TR-12.3), dataset-class collapse (RF-3.1.1)
- True streaming SSM state passing (C-2.5/ARCH-11.8.1), coverage gate hardening (T-8.1.3/T-8.4.1)

### Maintenance pass 2026-10-07 (automated review fixes)
- Deleted the stale `stem_midi_pro/` mirror tree (root files are canonical).
- Added `validate_audio_file()` to `api.py` (thin wrapper over `audio_io.open_and_validate`, raising 400s) — un-skipped the 7 `test_api_validation.py` tests.
- `Artifact flags force COMPLEX tier` implemented in `utils/quality_gates.py` — removed 4 xfail markers (T-8.3.9).
- Added `quantize=True` INT8 dynamic-quantization path to `main.py:process_audio_file` and `?quantize=` on `/process` (PERF-7.7/7.8) — removed the xfail on `test_quantization.py`.
- Added `main()`/`cli()` console entry points wired to `pyproject.toml` `[project.scripts]`.
- Rewrote `main.py:load_from_checkpoint` to plain-PyTorch `torch.load` + `load_state_dict` (was calling a nonexistent NeMo method).
- Synced `pyproject.toml` torch pins (`>=2.5.0,<3.0.0`) with `requirements.txt`; removed the hard `mamba-ssm` dependency (CPU fallback lives in `models/_mamba_compat.py`); Dockerfile now copies `audio_io.py` (was missing — image would not boot).
- Standardized tests on `n_layers` (conftest + `test_mamba_separator.py` contract test).
- Full `ruff` cleanup (typing modernization, B006/B008/B039/B904/E501/E712/SIM fixes); `ruff check .` is green.
- **Training loop rewrite (TR-12.2/12.3/12.4, RF-3.1.9):** `train.py` is now a plain PyTorch loop (Lightning removed — it was already absent from requirements, so `import train` crashed before). Epoch train/val, `--grad-accum`, grad clipping, best/last checkpoints, early stopping, `--resume`, explicit best-checkpoint load for the final test. Verified end-to-end with a 1-epoch synthetic smoke run.
- **Fixed a latent training-path bug** found by the smoke run: dataset onset targets (`(1, T', 1)` per item → `(B, 1, T', 1)` after collation, frame count from a hardcoded hop, no STFT center padding) could not broadcast against model logits. `StemMidiModel._prepare_onset_target` now coerces them onto the model frame grid, and `onset_f1_loss_fn` is shape-robust. Regression tests in `tests/test_training_forward.py` (4 new). Suite: 86 passing.

### Relicense 2026-10-08
- Adopted **Apache-2.0** (aligning with upstream `origin/main`, which settled on Apache-2.0 because the project ships third-party Apache-2.0 code, notably `mamba-ssm`). `/LICENSE` now matches upstream exactly; SPDX headers updated to `Apache-2.0` on files touched by the relicense (`api.py`, `audio_io.py`, `utils/midi.py`, `tests/test_losses.py`, and others which previously had none or PolyForm headers and would have failed the CI license job); `pyproject.toml` license field, `LICENSE_NOTICE.md`, README/ARCHITECTURE/SUMMARY/SETUP/USER_GUIDE/research README, and `user_content/` license sections updated. A header sweep for remaining PolyForm `SPDX-License-Identifier` headers (e.g. some `scripts/`, `models/backends/`, `utils/` files) is tracked as a follow-up. ADR 0001 marked superseded; ADR 0006 records the decision; TODO D-STRAT-6/Section 5 annotated as superseded (kept for history).

Refer to `git log` for commit-level history and to [TODO.md](TODO.md) for the authoritative status of each item.
