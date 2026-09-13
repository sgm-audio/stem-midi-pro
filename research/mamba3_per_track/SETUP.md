# Setup & Training — Mamba-3 Per-Track Adaptive Filter Bank

## 1. System prep

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential cmake curl git

# NVIDIA driver check
nvidia-smi
# → confirm CUDA version ≥ 11.8, driver ≥ 520
```

## 2. Python + venv

```bash
# Install Python 3.10 if not present
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev

# Create project root
mkdir -p ~/mamba3 && cd ~/mamba3
git clone <your-repo-url> .
python3.10 -m venv venv
source venv/bin/activate
```

## 3. Install PyTorch (CUDA)

```bash
# Check nvidia-smi output → pick matching torch version
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

## 4. Install remaining deps

```bash
pip install librosa soundfile mido tqdm numpy pyyaml auraloss
pip install mamba-ssm  # compiles CUDA kernels, needs nvcc
```

## 5. Prepare MUSDB18-HQ data

```bash
# Extract to datasets/
mkdir -p datasets/musdb18hq/train
# Each track dir should look like:
#   datasets/musdb18hq/train/The_Singing_Guitar/
#     bass.wav     drums.wav    vocals.wav    other.wav
#     bass.crepe.npy  bass.pitched.npy
#     drums.crepe.npy  drums.pitched.npy
#     vocals.crepe.npy  vocals.pitched.npy
#     other.crepe.npy  other.pitched.npy
```

## 6. Train

```bash
cd ~/mamba3
source venv/bin/activate

# First run (100k steps, batch 16, A100/H100)
python train.py \
  --data ./datasets/musdb18hq/train \
  --save ./checkpoints \
  --steps 100000 \
  --batch 16 \
  --log-every 50 \
  --val-every 2000

# Monitor
watch -n 10 'tail -20 checkpoints/training.log 2>/dev/null || echo "no log yet"'
nvidia-smi --query-gpu=index,temperature.gpu,utilization.gpu,memory.used --format=csv
htop  # CPU worker load
```

## 7. Resume training

```bash
# By alias (auto-detects in --save dir):
python train.py --data ./datasets/musdb18hq/train --save ./checkpoints --resume last
python train.py --data ./datasets/musdb18hq/train --save ./checkpoints --resume best
python train.py --data ./datasets/musdb18hq/train --save ./checkpoints --resume latest
python train.py --data ./datasets/musdb18hq/train --save ./checkpoints --resume interrupt

# By exact filename:
python train.py --data ./datasets/musdb18hq/train --save ./checkpoints --resume checkpoint_042000.pt

# By full path:
python train.py --data ./datasets/musdb18hq/train --save ./checkpoints --resume /absolute/path/to/model.pt
```

## 8. Common flags

| Flag | Default | Notes |
|------|---------|-------|
| `--data` | `./datasets/musdb18hq/train` | Path to MUSDB18-HQ train dir |
| `--save` | `./checkpoints` | Where checkpoints + logs go |
| `--resume` | (none) | `last`/`best`/`latest`/`interrupt` or filename |
| `--steps` | 100000 | Total training steps |
| `--batch` | 16 | Per-GPU batch size |
| `--lr` | 3e-4 | Peak learning rate |
| `--device` | cuda | `cuda` or `cpu` |
| `--no-amp` | (off) | Disable mixed precision |
| `--grad-accum` | 1 | Accumulate N steps (effective batch = batch × accum) |
| `--grad-ckpt` | (off) | Enable gradient checkpointing (VRAM-limited only) |
| `--val-every` | 2000 | Validate every N steps |
| `--ckpt-every` | 1000 | Save checkpoint every N steps |
| `--num-workers` | 8 | DataLoader workers |
| `--cache-in-ram` | (off) | Preload all stems into RAM |
| `--d-model` | 256 | Mamba model dimension |
| `--d-state` | 32 | SSM state dimension |
| `--n-layers` | 6 | Number of Mamba layers |

## 9. Quick sanity check

```bash
# Dry run with synthetic data (no GPU needed)
python train.py --data /tmp/fake --save /tmp/test --steps 50 --batch 2 --device cpu --no-amp
```
