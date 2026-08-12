# Setup Guide — Stem+MIDI Pro

## Requirements

- **Python:** 3.13
- **Hardware:** Bare-metal Intel i5 (4–8 GB DDR4 RAM), CPU-only
- **OS:** Linux (Ubuntu 22.04+ recommended), macOS 14+, or Windows 11

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/stem-midi-pro/stem-midi-pro.git
cd stem-midi-pro

# 2. Create a virtual environment
python3.13 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .

# 4. Verify
python check_syntax.py
python verify_structure.py
```

## Quick Start

```bash
# Smoke test with synthetic audio
python demo.py

# Process a real audio file
python main.py --config configs/model_config.yaml --audio path/to/audio.wav

# Start the API server
make serve
# or: uvicorn api:app --host 0.0.0.0 --port 8000
```

## CPU Performance Notes

- The model uses Mamba-SSM's pure-PyTorch reference implementation (no CUDA required).
- Inference on a bare-metal i5 takes ~30–60 seconds per minute of audio.
- Use `configs/model_config.cpu.yaml` for a smaller model that fits in 4–8 GB RAM.
- INT8 dynamic quantization is available via `--quantize` flag (experimental).

## Docker

```bash
make docker-build
make docker-run
```

## Research Path (Mamba-3)

The Mamba-3 research code is in `/research/mamba3_per_track/`. See `research/README.md` for details.
This path requires CUDA and is not supported on the CPU-only production target.

## License

This software is licensed under the PolyForm Small Business License 1.0.0.
See [LICENSE](./LICENSE) and [LICENSE_NOTICE.md](./LICENSE_NOTICE.md) for details.
