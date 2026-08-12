# AGENTS.md

## What this is

Stem+MIDI Pro — **CPU-capable** guitar/bass stem separation + MIDI transcription. Production default backends: **Demucs** (stems) + **Basic Pitch** (MIDI). Optional **Mamba-SSM** backends for research/training. `main.py`: "No NeMo, no PyTorch Lightning." Many docs (README, ARCHITECTURE.md) still describe the old CUDA/FP8/TensorRT design — treat them as aspirational. Trust `pyproject.toml`, `requirements*.txt`, `Makefile`, CI.

## Entry points

| Command | Purpose |
|---------|---------|
| `python demo.py` | Fast smoke test with **fake** backends (no heavy deps) |
| `python main.py --config configs/model_config.cpu.yaml --audio <file>` | CLI inference (Demucs + Basic Pitch). Needs `pip install demucs basic-pitch` |
| `python scripts/eval_smoke.py --audio <file>` | Print quality tier + note counts |
| `python scripts/download_pretrained.py` | Warm Demucs/Basic Pitch weight caches |
| `python api.py` / `uvicorn api:app --reload` | FastAPI :8000. ZIP: stems + guitar.mid + bass.mid + report |
| `python train.py --config configs/model_config.mamba.yaml ...` | Train Mamba path only (needs `mamba-ssm` + real data) |
| `python check_syntax.py` / `python verify_structure.py` | CI structure gates |

## Commands

- **Tests**: `pytest -m "not cuda"` (CPU suite; CI + `make test`). CUDA-marked tests (`-m "cuda"`) run on a self-hosted runner only. Markers: `slow`, `cuda`, `integration` (`pytest.ini`). Coverage gate 70% (`make test-cpu`, CI).
- **Lint**: `make lint` = `ruff check .` + `black --check .` + `mypy .`. Not flake8/isort. `make format` = `ruff check --fix .` + `black .`. Line length 100.
- **Install**: `make install-dev` = `pip install -r requirements.txt -r requirements-dev.txt && pip install -e .`.
- **License gate**: every `.py` outside `research/` must carry an `SPDX-License-Identifier` header (PolyForm-Small-Business-1.0.0) — CI `license` job fails the build otherwise.

## Gotchas

- **Backends are config-driven** (`backends.separator` / `backends.transcriber`): `demucs` \| `mamba` \| `fake` and `basic_pitch` \| `mamba` \| `fake`. Factory: `models/backends/factory.py`. Tests + `demo.py` use **fake** (no heavy deps).
- **Default production config is Demucs + Basic Pitch** (`configs/model_config.yaml` and `.cpu.yaml`). Install: `pip install demucs basic-pitch` or `pip install -e ".[pretrained]"`. First Demucs run downloads weights to torch hub cache.
- **`mamba_ssm` is optional** — only required when a backend is `mamba` (`models/mamba_*.py`). Training defaults to `configs/model_config.mamba.yaml`.
- **Both guitar and bass are transcribed** (Phase 1). ZIP / `process_audio_file` expose `midi_guitar` + `midi_bass`; flat `midi` remains guitar-primary for compat.
- **Checkpoint load**: `main.load_from_checkpoint(path, config_path)` — not an instance method. API uses this for `MODEL_CHECKPOINT_PATH`. Only meaningful for Mamba `nn.Module` backends.
- **`quantize=True`**: dynamic INT8 on `nn.Linear` when Module backends present; no-op for pure demucs/basic_pitch/fake.
- **Dev deps alone aren't enough for API tests**: need `prometheus_client` from `requirements.txt`.
- **Python 3.13 only**: `requires-python = ">=3.13,<3.14"`.
- **`research/`** vendored — excluded from lint/package/SPDX.
- **Streaming** still concatenates chunks (no true SSM state).
- **API**: concurrency 1, Bearer `API_KEYS`, wav/flac/mp3, 44.1/48 kHz, ≤600 s, ≤500 MB.
- **Git:** local `main` only until remote added.
- **NeMo import adapter** not implemented yet (Phase 1.4 deferred) — use Demucs/Basic Pitch or export weights offline to `.pt` for Mamba.

## Architecture

```
audio → SeparatorBackend (demucs | mamba | fake) → guitar/bass/residual
     → TranscriberBackend (basic_pitch | mamba | fake) × {guitar, bass}
     → ConfidenceInjector → MIDI + CC#127
     → quality_gates.route_by_quality → studio/draft/complex
```

I/O shapes: audio `(B, 1, T)`.

## Config

- `configs/model_config.yaml` / `model_config.cpu.yaml` — production (demucs + basic_pitch)
- `configs/model_config.mamba.yaml` — Mamba train/infer (needs mamba-ssm + weights)
- `example_data_config.yaml` — dataset type for `train.py`

## Dependencies

```
torch/torchaudio → demucs + basic-pitch   # production default
optional: mamba-ssm                       # mamba backends only
librosa → soundfile → mido → fastapi → prometheus-client → structlog
```
