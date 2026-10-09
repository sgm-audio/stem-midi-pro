# Mamba-3 per-track experiment setup

<!-- STATUS: research -->

This is a research script, not part of the root API. Its name describes the
experiment; `per_track_processor.py` currently imports `Mamba` from
`mamba_ssm`. It has no verified trained checkpoint or deployment path.

## Runtime requirements

The script imports PyTorch, Mamba-SSM, NumPy, and audio/data dependencies.
Install a PyTorch/Mamba-SSM combination supported by the target platform. The
upstream Mamba reference in `research/mamba-ssm-reference/mamba/README.md`
describes its platform/build requirements. This repository does not pin a
separate research environment, and CPU execution has not been verified here.

## Training command

Run from the repository root. The trainer accepts CLI flags; it does not read
`config.yaml` and has no `--config` option.

```bash
python research/mamba3_per_track/train.py \
  --data /path/to/musdb18hq/train \
  --save ./outputs/mamba3-research \
  --steps 50 \
  --batch 2 \
  --device cpu \
  --no-amp
```

The `--device` parser default is `cuda`, so select `--device cpu` explicitly
for an attempted CPU run. The above command is source-verified but not
runtime-verified in this environment.

### Available flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--data` | Research-local MUSDB path | Dataset directory. |
| `--save` | `./checkpoints` | Checkpoint/log output directory. |
| `--resume` | unset | Checkpoint path or alias such as `last`/`best`. |
| `--steps` | `100000` | Training step count. |
| `--batch` | `16` | Batch size. |
| `--lr` | `3e-4` | Learning rate. |
| `--device` | `cuda` | Requested device. |
| `--no-amp` | false | Disable AMP. |
| `--grad-accum` | `1` | Gradient accumulation steps. |
| `--grad-ckpt` | false | Enable gradient checkpointing. |
| `--val-split` | `0.1` | Validation split. |
| `--val-every` | `2000` | Validation interval. |
| `--ckpt-every` | `1000` | Checkpoint interval. |
| `--log-every` | `50` | Logging interval. |
| `--early-stop-patience` | `10` | Early-stop patience. |
| `--num-workers` | `8` | DataLoader workers. |
| `--cache-in-ram` | false | Cache dataset samples in memory. |
| `--d-model` | `256` | Model dimension. |
| `--d-state` | `32` | SSM state dimension. |
| `--n-layers` | `6` | Layer count. |

The sample dataset directory layout is described in
[`mamba3_datasets.py`](mamba3_datasets.py). Dataset licensing/provenance must
be verified independently before training or redistribution.
