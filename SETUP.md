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

**CLI inference:**

```bash
python main.py --audio /path/to/song.wav --config configs/model_config.yaml
```

**API server:**

```bash
uvicorn api:app --reload
# Swagger UI: http://localhost:8000/docs
```

Useful environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_CONFIG_PATH` | `configs/model_config.yaml` | model config |
| `MODEL_CHECKPOINT_PATH` | unset | checkpoint to load at startup |
| `API_KEYS` | unset (auth off) | comma-separated API keys; clients send `Authorization: Bearer <key>` on `/process` |
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:5173` | CORS allowlist (`*` disables credentials) |
| `MAX_UPLOAD_BYTES` | `524288000` (500 MB) | upload cap |
| `PORT` | `8000` | bind port |

## 6. Performance expectations (CPU)

- **~30–60 s of processing per 1 minute of audio** on the reference i5.
- 4–8 GB RAM is sufficient for the default config; a smaller CPU preset config is planned (TODO CFG-9.1.3).
- One request at a time. `torch.set_num_threads` tuning (TODO PERF-7.4) will use all cores.

## Troubleshooting

| Problem | Fix |
|---|---|
| `mamba_ssm` import/build error | Ensure Python 3.13 and a recent PyTorch (CPU wheel) are installed; the package falls back to the pure-PyTorch path on CPU. |
| OOM on long files | Files are capped at 10 min / 500 MB; close other apps, or wait for the smaller CPU config (CFG-9.1.3). |
| `503 Model not loaded` from the API | Wait for startup to finish; the model loads at server start. |
| `Sample rate not supported` | Resample to 44.1 kHz or 48 kHz (e.g. with Audacity or `ffmpeg`). |

---
License: Apache-2.0 - see [LICENSE](LICENSE) and the [README](README.md#license).
