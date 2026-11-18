# Checkpoints

Training data: Slakh2100-YourMT3-16k (CC BY 4.0, commercial OK). Do NOT train on MUSDB18-HQ for a shippable product (non-commercial).

| Directory | Contents |
|-----------|----------|
| `external/` | Downloaded pretrained weights (Demucs cache is usually under `~/.cache/torch`; Basic Pitch ships in-package) |
| `imported/` | Converted project checkpoints (`.pt`, optional) |
| `finetuned/` | Your training runs |

Large binaries are **gitignored** (`*.pt`, `*.ckpt`, `*.safetensors`, `*.nemo`).

## Optional pretrained CPU backend pair

The CPU config explicitly selects Demucs + Basic Pitch, whose own pretrained
models are acquired by their libraries. This path is mechanically supported,
but optional dependency installation, model acquisition, audio quality, and
reference-hardware performance still require separate validation; this
repository ships neither those caches nor a trained project checkpoint.

```bash
python scripts/install_pretrained.py     # demucs + basic-pitch (ONNX, no TF)
python scripts/download_pretrained.py    # optional warm cache
python main.py --config configs/model_config.cpu.yaml --audio path/to/mix.wav
```

## Native Mamba path

```bash
# The pure-PyTorch fallback works without mamba-ssm on CPU.
python main.py --config configs/model_config.mamba.yaml \
  --audio path/to/mix.wav \
  --checkpoint /path/to/trained_checkpoint.pt
```

A validated project checkpoint is required for the Mamba path; random
initialization is not usable file/API output.
