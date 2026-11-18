# Setup — Stem+MIDI Pro

Install Stem+MIDI Pro on **bare-metal** hardware. Target: Intel i5 (or similar x86-64), 4–8 GB RAM, **CPU only, no CUDA**, Python **3.13**.

> Looking for the Mamba-3 research experiments instead? See [research/mamba3_per_track/SETUP.md](research/mamba3_per_track/SETUP.md).

## 1. Install Python 3.13

- **Windows:** installer from python.org, check "Add to PATH".
- **macOS:** `brew install python@3.13`
- **Debian/Ubuntu:** use [deadsnakes](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa) (`sudo apt install python3.13 python3.13-venv`) or [uv](https://docs.astral.sh/uv/) / pyenv.

Verify: `python3.13 --version` → `Python 3.13.x`.

## 2. Create a virtual environment

```bash
python3.13 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- PyTorch installs CPU builds by default on Linux CPU-only machines; no CUDA toolkit is needed or used.
- `mamba-ssm` runs its pure-PyTorch reference implementation on CPU — functional but slower than the CUDA kernels. `causal-conv1d` is intentionally **not** installed (CUDA-only).
- For development also `pip install pytest black flake8 isort mypy`.

## 4. Verify the install

```bash
python demo.py          # smoke test with synthetic audio; no input file needed
```

## 5. Run

**CLI inference with a validated project checkpoint:**

```bash
python main.py --audio /path/to/song.wav \
  --config configs/model_config.yaml \
  --checkpoint /path/to/trained_checkpoint.pt
```

For the optional pretrained CPU backend pair, install the dependencies with
`scripts/install_pretrained.py` and use `configs/model_config.cpu.yaml`; that
config declares its own pretrained weight source and does not use random Mamba
weights. No trained project checkpoint is shipped here.

**API server:**

```bash
uvicorn api:app --reload
# Swagger UI: http://localhost:8000/docs
```

Useful environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_CONFIG_PATH` | `configs/model_config.yaml` | model config |
| `MODEL_CHECKPOINT_PATH` | unset | validated project checkpoint to load at startup; optional only for the explicit pretrained CPU backend config |
| `API_KEYS` | unset (auth off) | comma-separated API keys; clients send `Authorization: Bearer <key>` on `/process` |
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:5173` | CORS allowlist (`*` disables credentials) |
| `MAX_UPLOAD_BYTES` | `524288000` (500 MB) | upload cap |
| `PORT` | `8000` | bind port |

## 6. Performance expectations (CPU)

- Declared target: bare-metal Intel i5 with 4–8 GB RAM.
- No trained-checkpoint benchmark on the reference i5 is committed; do not treat historical seconds-per-minute estimates as a guarantee.
- One API request at a time. `torch.set_num_threads` can be tuned with `OMP_NUM_THREADS`.
- Native-Mamba file/API inference requires `MODEL_CHECKPOINT_PATH` or the CLI `--checkpoint`; the explicit pretrained CPU backend profile is the documented exception. No trained project checkpoint is shipped in this repository.

## Troubleshooting

| Problem | Fix |
|---|---|
| `mamba_ssm` import/build error | Ensure Python 3.13 and a recent PyTorch (CPU wheel) are installed; the package falls back to the pure-PyTorch path on CPU. |
| OOM on long files | Files are capped at 10 min / 500 MB; close other apps, or wait for the smaller CPU config (CFG-9.1.3). |
| `503 Model not loaded` from the API | Set `MODEL_CHECKPOINT_PATH` to a valid trained checkpoint, then wait for `/ready`; startup intentionally refuses random weights. |
| `Sample rate not supported` | Resample to 44.1 kHz or 48 kHz (e.g. with Audacity or `ffmpeg`). |

---
License: Apache-2.0 - see [LICENSE](LICENSE) and the [README](README.md#license).
