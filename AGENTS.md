# AGENTS.md

## What this is

Stem+MIDI Pro — Mamba-SSM guitar/bass stem separation + MIDI transcription. **CPU-only, bare-metal i5 target, Python 3.13.** 96 P0/P1 items from `TODO.md` were completed 2026-09-21, plus an automated maintenance pass on 2026-10-07 — see CHANGELOG.md and the stamps in TODO.md. License: **Apache-2.0** (`/LICENSE`); every root-level `.py` carries an SPDX header. Relicensed from PolyForm Small Business on 2026-10-08 to align with upstream (ADR 0006).

## Entry points

| Command | Purpose |
|---------|---------|
| `python demo.py` | Quick smoke test with synthetic audio |
| `python main.py --audio <file>` | CLI inference (`main:cli` is the pyproject entry point). `_load_audio` in main.py is still a stub-ish path; `audio_io.py` (soundfile, native-sr, mono downmix) is the real loader used by api.py |
| `python api.py` / `uvicorn api:app` | FastAPI server on :8000 — lifespan startup, Bearer auth on `/process` (`API_KEYS` env), 500 MB cap, semaphore-1 concurrency, `/live` `/ready` `/metrics` `/warmup`, structlog-or-stdlib logging, X-Request-ID, `?quantize=true` INT8 path |
| `python check_syntax.py` | AST syntax check on all `.py` files |
| `python train.py --config configs/model_config.yaml --data-config example_data_config.yaml --output-dir ./outputs` | Plain-PyTorch training loop (no Lightning — TR-12.3 done 2026-10-07). Flags: `--strict-data`, `--resume`, `--grad-accum`, `--max-epochs` |
| `make test` / `pytest` | Test suite: `tests/` (86 tests; torch-dependent ones skip without torch) |

## State of play (post 2026-10-07 maintenance pass)

- **Done:** all Section-2 runtime bugs, real losses (`models/losses.py`: onset F1, pitch CE, velocity MAE, duration IoU, alignment), full API hardening, `utils/midi.py::build_midi_from_events`, `audio_io.py`, license files, docs rewrite, packaging (`pyproject.toml`, Dockerfile, Makefile, CI, requirements{,-dev}.txt), test infra, PLUS (2026-10-07): `stem_midi_pro/` mirror deleted, `validate_audio_file` in api.py, artifact-flags-force-COMPLEX in quality gates, INT8 `quantize` path (PERF-7.7/7.8), `api:main`/`main:cli` entry points, torch pins synced pyproject↔requirements, Dockerfile copies `audio_io.py`, `__init__` exports (RF-3.2.1–3), Lightning-free `train.py` (TR-12.2/12.3/12.4), onset-target shape normalization (train path verified end-to-end), full ruff cleanup.
- **Known issue (fixed 2026-09-21):** em-dash in MIDI marker meta text broke mido latin-1 encode → now ASCII `-`. Beware non-ASCII in mido text fields.
- **Open top items:** RF-3.1.1 (collapse dataset classes), RF-3.1.2 (shared audio/stft.py), RF-3.2.5 (utils/paths.py), API-6.13 (true streaming), API-6.19 (OTel), PERF-7.5 (vectorized ISTFT), PERF-7.11/7.12 (batch endpoint, caching), TR-12.5/12.7 (EMA, QAT), T-8.1.3/T-8.4.1 (coverage gate hardening), Section 11 P1/P2 features, **license contradiction with origin/main (Apache-2.0)**.

## Gotchas

- **`stem_midi_pro/` mirror was deleted 2026-10-07** (125 stale files that broke mypy/check_syntax). Root files are canonical.
- **`train.py`** is a custom loop now; checkpoints save under `<output-dir>/checkpoints/{best,last}.pt` with a `"state_dict"` key (`main.load_from_checkpoint` reads that shape). CPU bfloat16 autocast not implemented (P2).
- **Dataset onset targets** carry a spurious `(1, T', 1)` batch dim and a hardcoded-hop frame count; `StemMidiModel._prepare_onset_target` normalizes them — keep new loss inputs shape-robust.
- **mamba-ssm CPU path:** models use the pure-PyTorch reference impl via `models/_mamba_compat.py` (no `causal_conv1d`, no CUDA kernels). `mamba-ssm` is NOT a hard dependency (needs nvcc to build). Invalid kwargs like `causal_conv1d_impl` must never be passed.
- **Streaming:** the old fake `_state_cache` activation-concat was removed (C-2.5); chunked inference restarts state per chunk — document, don't reintroduce.
- **"si_sdr" in `ProcessingReport`** is a centroid-separation **proxy**, not real SI-SDR (needs ground truth). Don't present it as SI-SDR.
- No real datasets ship; Slakh/MUSDB loaders fall back to synthetic data unless `--strict-data`.
- **Line endings:** working tree is LF; git warns LF→CRLF on touch (no `.gitattributes`). Add one before committing.

## Architecture

```
audio → MambaSeparator (STFT → Mamba blocks (nn.Sequential) → masks → cached-STFT ISTFT)
     → MambaTranscriber (mel_basis buffer → Mamba → onset/pitch/velocity/expression + mc_dropout_eval confidence)
     → ConfidenceInjector (vectorized) → MIDI (utils/midi.py, CC#127)
     → quality_gates.route_by_quality → studio/draft/complex
```

Shapes: audio `(B, 1, T)`, spec `(B, F, T)`, mel `(B, n_mels, T)`.

## Config & deps

- `configs/model_config.yaml` — fp32, `n_layers`, no TensorRT block. `MODEL_CONFIG_PATH` env / `--config` flag.
- `requirements.txt` (prod) + `requirements-dev.txt` (tests/lint). NeMo, PyTorch-Lightning(→dev?), pretty-midi and flake8 removed from prod; ruff config lives in `pyproject.toml`.
