# AGENTS.md

## What this is

Stem+MIDI Pro — prototype audio AI for Mamba-SSM-based guitar/bass stem separation + MIDI transcription. Most "production" claims in docs are aspirational.

## Entry points

| Command | Purpose |
|---------|---------|
| `python demo.py` | Quick smoke test with synthetic audio |
| `python main.py --audio <file>` | CLI inference (audio loading returns DUMMY data — placeholder at `main.py:248`) |
| `python api.py` | FastAPI dev server on port 8000 (also a stub — `_load_audio` returns dummy) |
| `python check_syntax.py` | AST syntax check on all `.py` files |
| `python train.py --config configs/model_config.yaml --data-config example_data_config.yaml --output-dir ./outputs | Training loop with Canadian dataset support |
| `uvicorn api:app --reload` | Same as `python api.py` |
| `pytest` | No tests exist — there is no `tests/` directory |
| `flake8 . / black . / isort . / mypy .` | Lint/format/type-check |

## Gotchas

- **`verify_structure.py`** hardcodes `stem_midi_pro/` prefix (e.g. `stem_midi_pro/main.py`) but all files are at repo root. Running it from repo root will always fail.
- **`train.py` maps `precision: "fp8"` to `precision=16`** in PyTorch Lightning (line 79). Real FP8 needs TransformerEngine; this is a pass-through placeholder.
- **Mamba-SSM requires CUDA** — the `mamba_ssm` package compiles custom kernels (`selective_scan_fn`, `causal_conv1d`). Will not run on CPU or Mac.
- **All dataset loaders fall back to synthetic data** — `Slakh2100CADataset` and `MUSDBIndieDataset` are stubs.
- **`main.py:process_audio_file` returns dummy audio** — `_load_audio` generates random noise, doesn't use `soundfile`/`librosa`.
- **`api.py:create_placeholder_midi` returns a minimal valid MIDI file** (header + end-of-track only, no note events).
- **No git repo** is initialized.

## Architecture

```
audio → MambaSeparator (12-layer SSM) → guitar/bass stems
     → MambaTranscriber (8-layer SSM) → onset/pitch/velocity/expression/confidence
     → ConfidenceInjector → MIDI with CC#127 metadata
     → quality_gates.route_by_quality → studio/draft/complex
```

All model I/O shapes: audio `(B, 1, T)`, spec `(B, F, T)`, mel `(B, n_mels, T)`.

## Config

- `configs/model_config.yaml` — model architecture, loss weights, quality gate thresholds, TensorRT-LLM export stub
- `example_data_config.yaml` — dataset type/path/batch size for training

## Dependency chain

```
torch → pytorch-lightning → nemo-toolkit[all] → mamba-ssm → librosa → soundfile → torchaudio → fastapi → uvicorn
```

`tensorrt` + `torch-trt` in `requirements.txt` are unused. `mamba_ssm` is the hardest dependency (CUDA kernel compilation).
