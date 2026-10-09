# Development guide

<!-- STATUS: in progress -->

This guide documents the current repository layout and available checks. The
project is a research prototype; model training and end-to-end inference are
not yet validated.

## Getting started

Work from the repository root. The active source files are at the top level;
there is also a duplicate legacy tree under `stem_midi_pro/` that is not the
current entrypoint.

The root `requirements.txt` contains PyTorch, NeMo, Mamba-SSM, audio, API, and
test dependencies. There is no lockfile or declared supported Python range.
Mamba-SSM and NeMo may require platform-specific PyTorch/CUDA setup; the file
is not a guaranteed portable CPU install recipe. There is no `.env.example`.

The API reads these optional/configuration variables:

- `MODEL_CONFIG_PATH` — YAML path; default `configs/model_config.yaml`.
- `MODEL_CHECKPOINT_PATH` — optional checkpoint path; if omitted, weights are
  random. If set, the file must exist.
- `API_KEY` — optional key required as `X-API-Key` on `/process`, `/templates`,
  and `/render-template`.
- `CORS_ORIGINS` — comma-separated origin allowlist; defaults to local Vite
  origins. `*` disables credentialed CORS.
- `MAX_UPLOAD_BYTES` — upload-file size limit; default 50 MiB.
- `PORT` — used by `python api.py`; direct `uvicorn` commands set the port
  themselves.

## Repository map

- `api.py`, `main.py`, `train.py`, `demo.py` — current root entrypoints.
- `models/`, `data/`, `utils/`, `configs/` — production-path prototype code.
- `tests/` — current unit tests; `pytest.ini` restricts default discovery here.
- `research/` — separate Mamba-3 experiment and upstream reference snapshot.
- `stem_midi_pro/` — duplicate historical tree, retained pending a maintainer
  decision.

## Checks

Run these from the repository root, after installing the relevant dependencies:

```bash
python check_syntax.py
python verify_structure.py
pytest
```

`check_syntax.py` checks first-party Python files and skips the two vendored
Mamba trees. `pytest.ini` prevents default discovery from recursing into the
research/vendor tests. The complete pytest suite still requires dependencies
such as PyTorch, NumPy, SoundFile, and Mido; model/audio tests are currently
skipped or otherwise not isolated from NeMo/Mamba.

There is no configured formatter, linter, type checker, coverage gate, or normal
CI test/build workflow. The only checked-in workflow is CodeQL. Do not treat
optional local `ruff`, `black`, or `mypy` runs as enforced project policy.

## Available commands

```bash
# Attempts a synthetic forward path; random weights, known batch/shape issue may fail
python demo.py

# Start the local API from the repository root
uvicorn api:app --host 0.0.0.0 --port 8000

# Root model CLI (requires the full model stack and an audio file)
python main.py --audio path/to/audio.wav

# Root Lightning trainer; end-to-end training is not yet validated
python train.py \
  --config configs/model_config.yaml \
  --data-config example_data_config.yaml \
  --output-dir ./outputs

# Research trainer flags are parsed directly; it does not accept --config
python research/mamba3_per_track/train.py --help
```

The top-level `example_data_config.yaml` selects synthetic samples explicitly.
The root trainer, model/data tensor shapes, and training loop still have
unresolved issues; see [Architecture](ARCHITECTURE.md) before relying on the
training example.

## Contribution workflow

`CONTRIBUTING.md` is not present. Use the repository's established style,
keep changes focused, update tests/docs for behavior changes, and submit a pull
request for maintainer review. The repository branch/history and release
version lineage need maintainer reconciliation; do not assume the `v0.1.0`
tag describes this tree.

## Model and API changes

- Keep tensor shapes explicit in model code and tests.
- Do not call the `si_sdr` report field SI-SDR: it currently contains a
  spectral-centroid heuristic.
- The current pipeline transcribes only guitar; do not describe bass MIDI as an
  independent transcription.
- Do not claim streaming state, browser editing, GDPR compliance, secure audio
  erasure, or measured latency unless those are implemented and validated.
- The API writes upload bytes to a temporary file before attempting cleanup.
- Update `API_DOCUMENTATION.md` and `ARCHITECTURE.md` when verified behavior
  changes.
