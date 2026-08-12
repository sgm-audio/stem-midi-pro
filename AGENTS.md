# AGENTS.md

## What this is

Stem+MIDI Pro — **CPU-only** prototype (bare-metal i5 target) for Mamba-SSM guitar/bass stem separation + MIDI transcription. `main.py` header: "No NeMo, no PyTorch Lightning." Many docs (README, ARCHITECTURE.md, DEVELOPMENT_GUIDE.md) still describe the old CUDA/FP8/TensorRT/NeMo design — treat them as aspirational. Trust `pyproject.toml`, `requirements*.txt`, `Makefile`, and `.github/workflows/ci.yml`.

## Entry points

| Command | Purpose |
|---------|---------|
| `python demo.py` | Smoke test with synthetic audio (needs `mamba_ssm` import) |
| `python main.py --audio <file>` | CLI inference; `_load_audio` is real (soundfile, librosa fallback) at `main.py:273` |
| `python api.py` / `uvicorn api:app --reload` | FastAPI on :8000. `/process` returns a ZIP: `guitar_stem.wav`, `bass_stem.wav`, `guitar.mid`, `bass.mid`, `processing_report.json` |
| `python train.py --config configs/model_config.yaml --data-config example_data_config.yaml --output-dir ./outputs` | Plain-PyTorch CPU training (bfloat16 autocast). Extra flags: `--strict-data`, `--grad-accum`, `--ema`, `--resume`, `--quantize`. No `--gpus` flag (README is stale) |
| `python check_syntax.py` / `python verify_structure.py` | AST + structure CI gates — run from repo root, they pass when run correctly |

## Commands

- **Tests**: `pytest -m "not cuda"` (CPU suite; CI + `make test`). CUDA-marked tests (`-m "cuda"`) run on a self-hosted runner only. Markers: `slow`, `cuda`, `integration` (`pytest.ini`). Coverage gate 70% (`make test-cpu`, CI).
- **Lint**: `make lint` = `ruff check .` + `black --check .` + `mypy .`. Not flake8/isort. `make format` = `ruff check --fix .` + `black .`. Line length 100.
- **Install**: `make install-dev` = `pip install -r requirements.txt -r requirements-dev.txt && pip install -e .`.
- **License gate**: every `.py` outside `research/` must carry an `SPDX-License-Identifier` header (PolyForm-Small-Business-1.0.0) — CI `license` job fails the build otherwise.

## Gotchas

- **`mamba_ssm` is still a hard import** despite "CPU-only": `models/mamba_separator.py:4` and `models/mamba_transcriber.py:5` do `from mamba_ssm import Mamba`, so `main.py`/`api.py`/`demo.py`/`train.py` won't import without the `mamba-ssm` pip package (which compiles CUDA kernels at install). Tests that import models auto-skip when it's missing.
- **Checkpoint load**: use module helper `main.load_from_checkpoint(path, config_path)` — not an instance method. API lifespan wires `MODEL_CHECKPOINT_PATH` through that helper. Accepts `{"model_state_dict": ...}` or bare state_dict.
- **`quantize=True` on inference**: `process_audio_file` deep-copies then `torch.ao.quantization.quantize_dynamic` on `nn.Linear` only. `train.py --quantize` logs a warning (QAT not implemented).
- **Dev deps alone aren't enough to run tests**: tests import `api.py`, which imports `prometheus_client` (a runtime dep). Install both `requirements.txt` and `requirements-dev.txt`.
- **Python 3.13 only**: `requires-python = ">=3.13,<3.14"` in `pyproject.toml`.
- **`research/` is vendored/experimental** — `research/mamba-ssm-reference/` (upstream mamba repo copy) and `research/mamba3_per_track/`. Excluded from ruff/black/mypy/check_syntax/package/SPDX checks. Don't lint or edit it like owned code.
- **`user_content/*.md` templates are whitelisted** in `api.py` `ALLOWED_TEMPLATES` — adding a template requires updating that set.
- **Streaming inference is fake**: `process_audio_streaming` concatenates per-chunk outputs; true SSM state caching is unimplemented (see `mamba_separator.py` docstring, TODO C-2.5).
- **API is concurrency-capped at 1** (`inference_semaphore`), Bearer auth via `API_KEYS` env, accepts wav/flac/mp3, 44.1/48 kHz, ≤600 s, ≤500 MB. Env overrides: `MODEL_CONFIG_PATH`, `MODEL_CHECKPOINT_PATH`, `CORS_ORIGINS`, `PORT`.
- **Git:** repo initialized on `main` (local only until a remote is added).
- Two model configs: `configs/model_config.yaml` (separator d_model=768, 12 layers) and `configs/model_config.cpu.yaml` (d_model=256, 6 layers — smaller RAM). Tests use a small mock config; `demo.py` runs the 768-dim model on CPU and is slow.

## Architecture

```
audio → MambaSeparator (12-layer SSM) → guitar/bass stems
     → MambaTranscriber (8-layer SSM) → onset/pitch/velocity/expression/confidence
     → ConfidenceInjector → MIDI with CC#127 metadata
     → quality_gates.route_by_quality → studio/draft/complex
```

All model I/O shapes: audio `(B, 1, T)`, spec `(B, F, T)`, mel `(B, n_mels, T)`.

## Config

- `configs/model_config.yaml` — model architecture, loss weights, quality gate thresholds (source of truth; `training.precision` is cosmetic — training hardcodes CPU bfloat16)
- `example_data_config.yaml` — dataset type (`synthetic` default / `musdb18hq` / `slakh2100_yourmt3`), root dir, batch size. Missing `root_dir` triggers synthetic fallback

## Dependencies (verified from pyproject.toml)

```
torch/torchaudio → mamba-ssm → librosa → soundfile → mido
fastapi/uvicorn → prometheus-client → structlog → opentelemetry → auraloss
```

No pytorch-lightning, nemo-toolkit, tensorrt, or torch-trt. `mamba_ssm` remains the hardest install (CUDA kernel compilation).
