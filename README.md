# Stem+MIDI Pro

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Audio AI for guitar/bass stem separation and MIDI transcription, built on Mamba-SSM and optional Demucs/Basic Pitch backends. The repository contains model code and tests, but **does not ship trained checkpoint weights**. Python **3.13** and CPU execution are targets; reference-i5 throughput and audio quality are not certified by this repository.

> **Status:** Active development; **not production-ready without a separately trained and validated checkpoint**. The CLI and API refuse to serve file inference from random initialization. See [TODO.md](TODO.md), [VERIFICATION_MATRIX.md](VERIFICATION_MATRIX.md), and [SUMMARY.md](SUMMARY.md) for current evidence and limitations.

## License

Stem+MIDI Pro is licensed under the **Apache License 2.0** (see [LICENSE](LICENSE)).

In plain English:

- **Free for everyone** — individuals, companies, and commercial use, including modification and redistribution.
- You must keep the license notice and attribute third-party components ([NOTICE](NOTICE)).
- Explicit patent grant from contributors; trademark rights are not granted.

> History: previously PolyForm Small Business 1.0.0; relicensed to Apache-2.0 on 2026-10-08 to align with upstream (see [docs/architecture-decision-records/0006-relicense-apache-2.0.md](docs/architecture-decision-records/0006-relicense-apache-2.0.md)).

Third-party attributions are in [NOTICE](NOTICE).

## What It Does

- **Configured stem separation** — the model honors the selected separator backend (`mamba` or optional `demucs`) and returns independent guitar, bass, and residual stems. The pure-PyTorch Mamba path is structural code; useful output requires trained weights.
- **Independent MIDI transcription** — guitar and bass are transcribed separately. MIDI preserves absolute audio-frame timing, duration, overlapping notes, and same-onset chords.
- **Expression mapping** — bend, vibrato, and slide fields map to pitch-wheel, CC1, and CC74 messages, written only when a caller asks for them (`include_expression=True`). Export is off by default on purpose: the head is untrained, so writing it into every file would fabricate phrasing. Every event that carries expression is labelled `source: expression_head (untrained; ARCH-11.4.4)`, and mapping is never evidence that a checkpoint learned expression.
- **Confidence metadata** — note confidence is canonical in JSON and is also encoded as standard MIDI Polyphonic Key Pressure when supported by a DAW. CC#127 is retained only as a deprecated JSON alias and is never emitted.
- **Quality gates** — reports expose a named centroid-separation proxy, not SI-SDR. Untrained weights and artifact flags force manual review rather than certification (`utils/quality_gates.py`).

Everything runs locally. No GPU or cloud service is required by the core path; optional Demucs/Basic Pitch dependencies must be installed when configured.

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

# 3. Structural smoke test with synthetic audio (random weights; not production output)
python demo.py

# 4. Process a real audio file from the CLI. A trained checkpoint is required.
python main.py --audio /path/to/song.wav \
  --config configs/model_config.yaml \
  --checkpoint /path/to/trained_checkpoint.pt

# 5. Run the HTTP API. Set MODEL_CHECKPOINT_PATH before starting it.
export MODEL_CHECKPOINT_PATH=/path/to/trained_checkpoint.pt
uvicorn api:app --reload
# Interactive API docs: http://localhost:8000/docs
```

See [SETUP.md](SETUP.md) for detailed installation, and [USER_GUIDE.md](USER_GUIDE.md) for end-user documentation.

## CPU Performance Expectations

This project targets **bare-metal CPU**, not GPUs. No reproducible benchmark on the stated Intel i5/4–8 GB target is committed yet; do not treat historical 30–60 seconds/minute estimates as certified performance. The current test environment is Python 3.11, while the project target is Python 3.13. A real benchmark must use a trained checkpoint, real audio, peak RSS, and the target hardware.

The pure-PyTorch Mamba fallback runs on CPU. The optional `configs/model_config.cpu.yaml` selects Demucs + Basic Pitch and fails with actionable dependency errors when those packages are absent.

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

The data loaders target Slakh2100-YourMT3-16k (**CC BY 4.0**, commercial use OK with attribution) and, optionally, MUSDB18-HQ (educational/non-commercial only — do **not** use for commercial training). Training on Slakh2100 may produce a checkpoint after independent evaluation; cite Manilow et al., WASPAA 2019. **No trained weights are included in this repository**, so no separation quality, confidence calibration, expression quality, or DAW usability certification is claimed. The code is Apache-2.0.

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
| [docs/architecture-decision-records/](docs/architecture-decision-records/) | ADRs 0001–0007 (including superseded history) |
| [docs/operations/](docs/operations/) | Runbook, monitoring, upgrading, rollback |
| [research/README.md](research/README.md) | Mamba-3 research code (as-is) |

## Acknowledgements

Built with PyTorch, [Mamba-SSM](https://github.com/state-spaces/mamba), librosa, soundfile, mido, and FastAPI.

---

*Stem+MIDI Pro: stem separation + editable MIDI drafts. Not magic—just math that respects your craft.*
