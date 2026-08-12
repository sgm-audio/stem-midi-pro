# Checkpoints

| Directory | Contents |
|-----------|----------|
| `external/` | Downloaded pretrained weights (Demucs cache is usually under `~/.cache/torch`; Basic Pitch ships in-package) |
| `imported/` | NeMo → `.pt` exports (Phase 1.4, optional) |
| `finetuned/` | Your cloud training runs |

Large binaries are **gitignored** (`*.pt`, `*.ckpt`, `*.safetensors`, `*.nemo`).

## Production default

Config `backends.separator: demucs` + `backends.transcriber: basic_pitch` downloads weights on first run via the respective libraries.

```bash
pip install -e ".[pretrained]"
# or: pip install demucs basic-pitch
python scripts/download_pretrained.py   # optional warm cache
python main.py --config configs/model_config.cpu.yaml --audio path/to/mix.wav
```

## Mamba path

```bash
pip install mamba-ssm
python main.py --config configs/model_config.mamba.yaml --audio path/to/mix.wav
# Still needs a trained checkpoint — random init is not usable
```
