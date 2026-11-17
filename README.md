# Stem+MIDI Pro

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Audio AI for guitar/bass stem separation and MIDI transcription, built on Mamba-SSM. Runs **CPU-only on bare-metal** — a bare-metal Intel i5 with 4-8 GB RAM is the reference target. Python **3.13** (target). The test suite: `pytest -q` → **148 passed, 2 skipped, 0 failed** (2026-11-17; the skips are optional `mamba_ssm` checkpoint tests).

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
- **MIDI transcription** — converts each stem to MIDI with onset, pitch, velocity, and per-note confidence (`models/mamba_transcriber.py`). Single-pitch-per-onset (no chords yet); the expression head (bend/vibrato/slide) exists but is untrained and is NOT encoded into MIDI.
- **Confidence metadata** — each MIDI note carries its confidence as a MIDI CC so your DAW can flag uncertain notes (`models/confidence_injector.py`). Default **CC#16**; CC#120-127 are Channel Mode messages per the MIDI spec, so legacy CC#127 is opt-in via `MIDI_CONFIDENCE_CC=127`. Because a channel controller is shared by every note on the channel, the exact per-note values are also written to a companion `marker` track (`utils/midi.py`).
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

This project targets **bare-metal CPU**, not GPUs. Measured on 2 threads with the pure-PyTorch Mamba
fallback and the production `configs/model_config.yaml` (`process_audio_file`: separation +
transcription, MIDI written; 2026-11-17, on top of `main`):

| Input length | Path | Wall time | Peak RSS |
|---|---|---|---|
| 5 min | chunked (`chunk_duration_sec: 2`, default) | 104.0 s | 1.60 GB |
| 5 min | one pass (`chunk_duration_sec: 300`) | 69.0 s | 2.16 GB |
| 10 min | chunked (`chunk_duration_sec: 2`) | 173.6 s | 2.22 GB |
| 10 min | one pass | not measured (peak grows with length) | — |
| 1 min | not re-measured on this box | — | — |

- That is **~0.25-0.3x realtime** for separation + transcription combined. Imports plus the idle model
  account for ~0.65 GB of each peak, so most of it is the runtime, not the audio.
- **Long inputs are separated in overlapping chunks (`audio.chunk_duration_sec`, default 2 s) and
  transcribed in frame segments (`inference.segment_seconds`, default 60 s), so peak memory tracks the
  chunk size rather than the file length.** A single pass is faster (no per-chunk re-priming) but its
  peak grows with the input: 2.16 GB at 5 minutes.
- One concurrent request on the reference hardware. Scale up CPU/RAM for more.
- **No tempo detection**: MIDI is written on a fixed grid at the tempo you pass (`tempo_us`, default
  120 BPM), so quantisation is only as good as that choice.
- **Monophonic transcription**: the pitch head is a 128-way argmax, so chords are not representable
  (see `AUDIT_REPORT.md` H-5). The MIDI writer *does* preserve simultaneity, so a polyphonic head can be
  dropped in later without touching it.

`mamba-ssm` (the upstream CUDA package) is NOT used on this path — CPU inference runs the locally implemented pure-PyTorch Mamba fallback in `models/_mamba_compat.py` (same block math, chunked scan, no fused kernels — functional but slower than upstream's CUDA kernels). `MAMBA_BACKEND` records which implementation is live and is surfaced in logs, `/model-info`, and the processing report; `Mamba.SCAN_CHUNK` (default 256 frames) bounds the scan's working set, and `STEM_MIDI_REQUIRE_MAMBA_SSM=1` makes a missing package a hard error instead of a fallback.

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

1. **Training-data prioritization** for Canadian artists — planned, not implemented. [PRD_AND_ARD.md](PRD_AND_ARD.md) describes it; `data/datasets.py` has no weighting code and the repository ships no dataset.
2. **Indie-friendly licensing** — the code is Apache-2.0: free for everyone, including small studios and commercial use.
3. **Local success stories** in user-facing content — planned; the templates in `user_content/` do not contain any yet.

The data loaders target Slakh2100-YourMT3-16k (**CC BY 4.0**, commercial use OK with attribution) and, optionally, MUSDB18-HQ (educational/non-commercial only — do **not** use for commercial training). Training on Slakh2100 yields weights you can ship; cite Manilow et al., WASPAA 2019. No trained weights are included in this repository. The code is Apache-2.0.

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

## Before you deploy

A trained checkpoint is required. Without `MODEL_CHECKPOINT_PATH` the API refuses to start serving
(`/ready` → 503) rather than returning random-weight stems and MIDI:

```bash
export MODEL_CHECKPOINT_PATH=checkpoints/stem_midi_v1.pt
export MODEL_CONFIG_PATH=configs/model_config.yaml
```

`ALLOW_UNTRAINED_MODEL=1` (or `inference.allow_untrained_model: true`) overrides this for smoke tests
only; `/ready` and `/model-info` will then report `"trained_weights": false`.
