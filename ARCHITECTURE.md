# Architecture and implementation status

<!-- STATUS: in progress -->

This document describes the code in the current checkout. It is not a production
architecture or evidence that end-to-end model inference has passed validation.
The repository contains no trained checkpoint, and the main audio path has
known runtime/shape defects.

## Repository layout

The active implementation is rooted at the repository top level:

```text
api.py                     FastAPI routes and ZIP packaging
main.py                    NeMo model wrapper and orchestration
models/                    Mamba separator, transcriber, losses, metadata
utils/                     quality routing and template rendering
data/datasets.py           synthetic and partial real-data loaders
configs/model_config.yaml  v1 model configuration
research/                  separate Mamba-3 experiment and upstream reference
stem_midi_pro/             older duplicate tree; not imported by root modules
```

The duplicate `stem_midi_pro/` subtree and its duplicate Mamba snapshot are
retained pending a maintainer decision. The root-level modules are the intended
entry points for this checkout.

## Current runtime shape

1. `api.py` creates a FastAPI application with `/health`, `/model-info`,
   `/process`, `/templates`, and `/render-template` routes.
2. On startup, it reads the YAML file at `MODEL_CONFIG_PATH` and constructs
   `StemMidiModel`. If `MODEL_CHECKPOINT_PATH` is unset, model weights are
   randomly initialized. If a checkpoint path is set, it must exist.
3. `/process` validates a filename suffix, duration, sample rate, and upload
   size; writes the upload to a temporary file; calls
   `StemMidiModel.process_audio_file()`; creates WAV/MIDI/report entries in a
   ZIP; and attempts to remove the temporary file.
4. `StemMidiModel.forward()` calls `MambaSeparator`, then
   `MambaTranscriber`, constructs MIDI metadata, calculates heuristic quality
   values, and calls `route_by_quality()`.

The sequence above is the intended call chain, not a claim that the current
model completes it successfully.

## Components

### API (`api.py`)

- Supported suffixes are `.wav`, `.flac`, and `.mp3`; the code validates the
  audio via SoundFile metadata.
- The default file-size limit is 50 MiB and the duration limit is 600 seconds.
- Only 44.1 kHz and 48 kHz are accepted by API validation.
- `API_KEY` is optional and uses the `X-API-Key` header on `/process`,
  `/templates`, and `/render-template`.
- There is no rate limiting or concurrency cap. Upload bytes are temporarily
  written to disk; unlinking is attempted after processing.

### Model (`main.py`, `models/`)

`StemMidiModel` is a NeMo `ModelPT` wrapper. The separator computes an STFT,
projects magnitudes through Mamba blocks, produces three masks, and applies an
inverse STFT. The transcriber builds a mel representation and predicts onset,
pitch, velocity, expression, and confidence outputs.

Important current limitations:

- `MambaTranscriber` now stores its config, registers a cached mel basis, and
  computes scalar mel boundaries with `math.log10()`. The full transcriber and
  end-to-end inference path still lack a passing runtime validation.
- `main.py` creates one MIDI-event list per batch item. The confidence injector
  currently expects a flat event list and 1-D confidence/energy arrays; the
  normal batched forward path does not meet that contract.
- The implemented confidence/phase values are not calibrated. The separator's
  spectral-centroid heuristic is exported under the `si_sdr` name, and the
  phase-head value is not a measured phase-coherence metric.
- The transcriber runs on the guitar stem only. `api.py` packages the same
  event stream as both guitar and bass MIDI; bass transcription is not
  implemented independently.
- The transcriber/loss/data shapes and training loop have not been validated as
  a complete training path.

### Data (`data/datasets.py`)

Synthetic audio is available for development. The Slakh loader reads mixture
and stem files when present but uses synthetic transcription targets. The
MUSDB loader treats the `other` stem as guitar and currently supplies a zero
bass target. These are adapters/prototypes, not Canadian-artist-specific data
loaders.

The top-level `example_data_config.yaml` selects synthetic samples explicitly,
including when `root_dir` already exists. The `AudioDataset` constructor also
retains its legacy fallback behavior for missing paths; this can silently train
on synthetic samples even when a real dataset type is selected. Whether such a
configuration should fail fast is a maintainer decision. Dataset tensor shapes
and real-data selection remain unvalidated.

### Quality gates (`utils/quality_gates.py`)

The report chooses Studio/Draft/Complex using configured confidence thresholds
and the `si_sdr` field. Artifact flags are recorded but do not currently force
the Complex tier. Because `si_sdr` is a centroid heuristic, the thresholds
must not be interpreted as measured separation quality.

`route_by_quality()` returns a separate Python routing dictionary whose draft
copy still mentions a web editor, paid human review, and refunds. No matching
UI/payment/refund service is implemented. This dictionary is returned by
`process_audio_file()` but is not included in the API ZIP; do not expose it as a
live offer without owner review.

### Research (`research/`)

`research/mamba3_per_track/` is a separate experiment with its own CLI and data
format. The CLI does not load `research/mamba3_per_track/config.yaml`; it reads
command-line flags. `research/mamba-ssm-reference/mamba/` is a tracked upstream
snapshot, not a runtime dependency. See [research/README.md](research/README.md).

## Streaming, training, and deployment status

<!-- STATUS: research -->

- True SSM state caching is not implemented. `MambaSeparator.forward()` accepts
  only `audio`, while `process_audio_streaming()` passes a `state_cache`
  keyword, so that path raises a signature error before processing.
- Chunk overlap handling, phase-coherent overlap-add, CUDA graph capture,
  pinned-memory I/O, WebSockets, and deterministic sub-5-ms latency are not
  implemented or benchmarked.
- `train.py` maps the configured `fp8` value to Lightning precision 16; this is
  not FP8. Dataset target shapes and loss shapes also need repair and tests.
- The Dockerfile and runtime image were not built during this review. The image
  uses a CUDA runtime base and installs a heavy, unpinned dependency set.

## Security and data handling

The API has an optional API key, basic size/duration validation, and a default
local CORS allowlist. It has no rate limiting or concurrency cap. The upload is
written to a temporary file before processing; cleanup attempts to unlink it,
which does not guarantee secure erasure. No GDPR/CCPA compliance assessment or
retention policy is present in the code.

## Planned product behavior

Features described in `PRD_AND_ARD.md`, `USER_GUIDE.md`, and `user_content/`
that refer to an online service, trained quality, a browser MIDI editor, tempo
or tuning detection, human review/refunds, persistent accounts, or measured
latency are requirements or draft copy only. They must not be presented as
implemented behavior until they exist and have been tested.
