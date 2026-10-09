# Stem+MIDI Pro

[![CodeQL](https://github.com/sgm-audio/stem-midi-pro/actions/workflows/codeql.yml/badge.svg)](https://github.com/sgm-audio/stem-midi-pro/actions/workflows/codeql.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

<!-- STATUS: in progress -->

> **Prototype, not a production service.** This repository contains an
> experimental audio-model pipeline and a FastAPI wrapper. There is no trained
> model checkpoint, deployed web interface, validated quality benchmark, or
> production support commitment in this checkout. Treat generated audio and
> MIDI as unverified research output.

## What is in this repository

The current implementation is at the repository root:

- `api.py` — FastAPI routes for health, model information, audio processing,
  and user-content templates.
- `main.py` — NeMo model wrapper and audio-processing orchestration.
- `models/` — Mamba-based separation/transcription prototypes, confidence
  metadata, and training losses.
- `data/datasets.py` — synthetic samples and partial Slakh/MUSDB loaders.
- `train.py` — PyTorch Lightning training entry point.
- `configs/` — model configuration. `example_data_config.yaml` is read by the
  root trainer; the research trainer instead uses command-line flags.
- `tests/` — small unit tests; several model/audio tests are skipped because
  the complete model stack is not isolated from the heavy dependencies.
- `research/` — separate Mamba-3 experiments and a tracked upstream Mamba
  reference snapshot; neither is the v1 API runtime dependency.

`stem_midi_pro/` is a second, older copy of much of the project, including a
second identical Mamba source snapshot. The root-level files are the current
entry points. The nested copy has not been confirmed as intentionally
supported and is retained pending a maintainer decision.

## Known limitations

- Model inference, training, streaming, and Docker deployment have not been
  verified end to end in this checkout. See [Architecture](ARCHITECTURE.md) for
  the known implementation gaps.
- The model is randomly initialized unless a checkpoint is supplied. The
  configured `si_sdr` report value is currently a spectral-centroid proxy, not
  a measured SI-SDR score; confidence and phase outputs are not calibrated.
- The current pipeline transcribes the guitar stem only. The API currently
  packages that event stream as both guitar and bass MIDI; it is not an
  independent bass transcription.
- The API writes uploads to a temporary file during processing and attempts to
  unlink it afterward. This is not secure erasure or an in-memory-only
  guarantee.
- The project has no dependency lockfile or declared Python support range.
  `requirements.txt` includes PyTorch, NeMo, and Mamba-SSM; installation may
  require platform-specific CUDA/PyTorch setup. Its broad multipart dependency
  floor permits versions with later published advisories; see
  [SECURITY.md](SECURITY.md) and do not deploy before resolving and auditing a
  compatible dependency set.
- There is no normal test/lint CI workflow. The CodeQL workflow is present, but
  the latest GitHub runs were reported as failed; their logs were unavailable
  during this review.
- `route_by_quality()` still returns stale editor/payment/refund draft copy to
  Python callers. No corresponding service exists, and the routing object is
  not in the HTTP ZIP response; do not surface it as a live offer.

## Setup and checks

Install the dependencies in an environment compatible with the selected
PyTorch, NeMo, and Mamba-SSM builds. The commands below are repository entry
points, not a claim that the full model stack installs on every platform:

```bash
python -m pip install -r requirements.txt
python check_syntax.py
python verify_structure.py
pytest
```

`pytest.ini` limits default test discovery to `tests/`; the full suite still
requires the declared audio/model dependencies. `python demo.py` attempts a
synthetic forward path with an untrained model, but the known batch/shape
contract issue may stop it before completion. Any output is not a quality
evaluation.

## Run the API prototype

From the repository root, in an environment with the full dependencies:

```bash
export MODEL_CONFIG_PATH=configs/model_config.yaml
# Optional. If omitted, the prototype initializes random weights.
export MODEL_CHECKPOINT_PATH=/path/to/model.nemo
# Optional API key. When set, use the X-API-Key header on protected routes.
export API_KEY=replace-with-a-local-secret
uvicorn api:app --host 0.0.0.0 --port 8000
```

The current routes are `/health`, `/model-info`, `/process`, `/templates`, and
`/render-template`; there is no `/api/v1` prefix. Upload constraints and the
optional API-key behavior are described in [API documentation](API_DOCUMENTATION.md).
Do not expose this prototype publicly without reviewing its authentication,
request limits, concurrency behavior, dependency resolution, and model output.

## Documentation

- [API documentation](API_DOCUMENTATION.md)
- [Architecture and implementation status](ARCHITECTURE.md)
- [Development guide](DEVELOPMENT_GUIDE.md)
- [Prototype operator notes](USER_GUIDE.md)
- [Product and architecture requirements](PRD_AND_ARD.md) — aspirational,
  not a statement of implemented behavior
- [Research area](research/README.md)
- [Review work plan](TODO.md) — historical items are not all validated

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
