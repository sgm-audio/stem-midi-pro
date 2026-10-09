# Repository notes for coding agents

## Project state

Stem+MIDI Pro is a prototype, not a production-ready service. The active
entrypoints and model code are at the repository root. There is no trained
checkpoint in the current tree, and end-to-end inference/training have not been
verified. Do not repeat product claims about studio quality, compliance,
latency, or retention as established facts.

<!-- STATUS: research -->
`stem_midi_pro/` is a second, older project tree with duplicate model code and a
second copy of the vendored Mamba reference. No root code imports from it. Its
retention/removal is a maintainer decision; do not treat it as the canonical
implementation.

## Entry points

| Command | Purpose / caveat |
| --- | --- |
| `python check_syntax.py` | AST syntax check; skips the two vendored Mamba trees. |
| `python verify_structure.py` | Checks expected root files and research paths. |
| `pytest` | Runs tests under `tests/` (`pytest.ini`); requires project dependencies. |
| `python demo.py` | Attempts a synthetic forward path with random weights; known batch/shape defects may stop it before completion. |
| `python main.py --audio <file>` | CLI inference; requires the full model stack and a usable checkpoint for meaningful output. |
| `uvicorn api:app --host 0.0.0.0 --port 8000` | API; model is loaded during startup. |
| `python train.py --config configs/model_config.yaml --data-config example_data_config.yaml --output-dir ./outputs` | Lightning training entrypoint; dataset/model path has unresolved blockers. |
| `python research/mamba3_per_track/train.py --help` | Research CLI; arguments are flags, not the `--config` form shown in older docs. |

The full runtime stack includes PyTorch, PyTorch Lightning, NeMo, Mamba-SSM,
Librosa, SoundFile, and Torchaudio. There is no root lockfile, supported Python
range, or complete CPU-only installation path.

## Verified implementation gaps

- `models/mamba_transcriber.py` now stores its config, builds the mel basis
  once as a non-persistent buffer, and uses scalar `math.log10()` for mel
  boundaries. Its complete forward path remains unverified.
- `main.py` produces batch-nested MIDI event lists, while
  `ConfidenceInjector.inject_midi_metadata()` expects a flat event list and
  indexes confidence/energy tensors as if their first dimension were time.
  The normal `forward()` path therefore needs a batch/shape fix before it can
  complete reliably.
- `main.py:process_audio_streaming()` passes `state_cache=` to
  `MambaSeparator.forward()`, whose signature accepts only `audio`; real SSM
  state caching is not implemented.
- `utils/quality_gates.py:route_by_quality()` returns stale editor, paid
  human-review, and refund copy. No matching service exists. Do not expose that
  Python return as a live product offer without maintainer approval.
- The model reports a spectral-centroid heuristic through the `si_sdr` field;
  it is not SI-SDR. The phase head is not a measured phase-coherence metric.
- The pipeline transcribes only the guitar stem. `api.py` currently packages
  the same MIDI event stream as both guitar and bass.
- `data/datasets.py` contains partial Slakh/MUSDB loaders, not
  Canadian-artist-specific datasets. Explicit `dataset_type: synthetic` now
  selects generated samples even when the configured root exists. A missing
  root can still silently fall back to synthetic samples even when a real
  dataset type is selected; strict-failure versus fallback behavior needs an
  owner decision. Real-data loaders and training shapes remain unvalidated.
- `train.py` requests Lightning training but the model/loss/data contracts have
  not been validated end to end.

## Configuration and security

- `configs/model_config.yaml` sets `training.precision: fp8`, while
  `train.py` maps that value to Lightning precision 16. It is not FP8 training.
- API authentication is optional: `API_KEY` enables `X-API-Key` checks on
  `/process`, `/templates`, and `/render-template`; an unset key leaves those
  routes open. No rate limiting or concurrency cap is implemented.
- The API checks each uploaded file against `MAX_UPLOAD_BYTES` (default 50 MiB)
  and duration (600 seconds), then writes it to a temporary file and unlinks it
  after processing. FastAPI parses multipart data before endpoint checks, and
  there is no upstream total-request-body cap; do not present this as a
  pre-parser or total-body limit.
- `MODEL_CHECKPOINT_PATH` is optional. Without it, the model is randomly
  initialized and is not useful as a quality-tested transcription service.

## Repository conventions

- Apache-2.0 is the current root license; do not apply the PolyForm Small
  Business terms or SPDX identifiers proposed in the historical `TODO.md`.
- The only tracked workflow is `.github/workflows/codeql.yml`; there is no
  normal test/lint/build CI workflow.
- `research/README.md` is the index for experimental code. The vendored Mamba
  source is reference material, not a runtime dependency.
- The GitHub `v0.1.0` release description refers to product areas absent from
  this checkout. This local clone is shallow/grafted and contains no matching
  tag, so release lineage cannot be established here. Do not associate that
  release with this implementation without maintainer confirmation.
