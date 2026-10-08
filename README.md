# Stem+MIDI Pro

[![CodeQL](https://github.com/sgm-audio/stem-midi-pro/actions/workflows/codeql.yml/badge.svg)](https://github.com/sgm-audio/stem-midi-pro/actions/workflows/codeql.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Audio AI for guitar/bass stem separation and MIDI transcription, built on Mamba-SSM. Runs **CPU-only on bare-metal** — a bare-metal Intel i5 with 4–8 GB RAM is the reference target. Python **3.13**.

> **Status:** Active development toward production readiness. See [TODO.md](TODO.md) for the tracked work items and current state. Some features documented in [ARCHITECTURE.md](ARCHITECTURE.md) are still `[PLANNED]`; [SUMMARY.md](SUMMARY.md) marks exactly what works today.

## License

Stem+MIDI Pro is licensed under the **Apache License 2.0** (see [LICENSE](LICENSE)).

In plain English:

- **Free for everyone** — individuals, companies, and commercial use, including modification and redistribution.
- You must keep the license notice and attribute third-party components ([NOTICE](NOTICE)).
- Explicit patent grant from contributors; trademark rights are not granted.

> History: previously PolyForm Small Business 1.0.0; relicensed to Apache-2.0 on 2026-10-08 to align with upstream (see [docs/architecture-decision-records/0006-relicense-apache-2.0.md](docs/architecture-decision-records/0006-relicense-apache-2.0.md)).

Third-party attributions are in [NOTICE](NOTICE).

## What It Does

- **Stem separation** — splits a mixed recording into guitar and bass WAV stems using a Mamba-SSM separator (`models/mamba_separator.py`).
- **MIDI transcription** — converts the stems to MIDI with onset, pitch, velocity, expression (bend/vibrato/slide), and per-note confidence (`models/mamba_transcriber.py`).
- **Confidence metadata** — each MIDI note carries its confidence as CC#127 so your DAW can flag uncertain notes (`models/confidence_injector.py`).
- **Quality gates** — routes each result into a `studio` / `draft` / `complex` tier with a transparent processing report (`utils/quality_gates.py`).

Everything runs locally. No GPU, no cloud, no CUDA required.

## Requirements

- Python **3.13**
- 4–8 GB RAM, x86-64 CPU (bare-metal Intel i5 is the reference target)
- OS: Linux, macOS, or Windows
- No CUDA. The CUDA/TensoRT path was removed; see `docs/architecture-decision-records/0002-cpu-only-target.md`.

## Quickstart

```bash
# 1. Create an environment
python3.13 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Smoke test with synthetic audio (no input file needed)
python demo.py

# 4. Process a real audio file from the CLI
python main.py --audio /path/to/song.wav --config configs/model_config.yaml

# 5. Run the HTTP API
uvicorn api:app --reload
# Interactive API docs: http://localhost:8000/docs
```

See [SETUP.md](SETUP.md) for detailed installation, and [USER_GUIDE.md](USER_GUIDE.md) for end-user documentation.

## CPU Performance Expectations

This project targets **bare-metal CPU**, not GPUs. Honest numbers on the reference i5 target:

- **~30–60 seconds of processing per 1 minute of audio** (separation + transcription, combined).
- A 3-minute song takes roughly **1.5–3 minutes**; a 10-minute song (the maximum) takes **5–10 minutes**.
- One concurrent request on the reference hardware. Scale up CPU/RAM for more.

`mamba-ssm` runs its pure-PyTorch reference implementation on CPU — functional, but slower than the CUDA kernels the upstream project benchmarks with.

## Project Structure

```
main.py                  # StemMidiModel + CLI entry point
api.py                   # FastAPI service (uvicorn api:app)
demo.py                  # Synthetic-audio smoke test
train.py                 # Training script
configs/model_config.yaml
models/                  # separator, transcriber, confidence injector, losses
data/datasets.py         # Slakh2100 (YourMT3 layout) + MUSDB18-HQ loaders, synthetic fallback
utils/                   # quality gates, template engine
user_content/            # user-facing message templates
tests/                   # pytest suite
research/                # Mamba-3 per-track experiments (as-is, no support)
docs/                    # ADRs and operations docs
```

## Mission: Canadian Artists

The mission behind Stem+MIDI Pro is to serve the Canadian independent-music community:

1. **Training-data prioritization** for Canadian artists — curation weights and provenance live in `data/datasets.py` and [PRD_AND_ARD.md](PRD_AND_ARD.md).
2. **Indie-friendly licensing** — Apache-2.0: free for everyone, including small studios and commercial use.
3. **Local success stories** in user-facing content (`user_content/`).

The core datasets used for training (Slakh2100, MUSDB18-HQ) are open-license (CC-BY). The trained model weights and this codebase are under the Apache License 2.0.

## Documentation

| Doc | Contents |
|-----|----------|
| [SETUP.md](SETUP.md) | Installing on bare-metal i5 / Python 3.13 |
| [USER_GUIDE.md](USER_GUIDE.md) | End-user guide |
| [API_DOCUMENTATION.md](API_DOCUMENTATION.md) | HTTP API reference |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture (annotated with implementation status) |
| [SUMMARY.md](SUMMARY.md) | Feature-by-feature implementation status |
| [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) | Contributor guide |
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [TODO.md](TODO.md) | Tracked work items |
| [docs/architecture-decision-records/](docs/architecture-decision-records/) | ADRs 0001–0005 |
| [docs/operations/](docs/operations/) | Runbook, monitoring, upgrading, rollback |
| [research/README.md](research/README.md) | Mamba-3 research code (as-is) |

## Acknowledgements

Built with PyTorch, [Mamba-SSM](https://github.com/state-spaces/mamba), librosa, soundfile, mido, and FastAPI.

---

*Stem+MIDI Pro: stem separation + editable MIDI drafts. Not magic—just math that respects your craft.*
