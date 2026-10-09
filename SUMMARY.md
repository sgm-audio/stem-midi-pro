# Repository implementation summary

<!-- STATUS: in progress -->

This is a concise inventory of the current checkout, not a claim that all
planned product features are complete. The repository is an experimental
prototype with no trained checkpoint and no end-to-end validation result.

## Current files

- `api.py` — FastAPI application with health/model-info routes, an upload route,
  template routes, optional API-key checks, upload validation, temporary-file
  handling, MIDI serialization, and ZIP packaging.
- `main.py` — NeMo `ModelPT` wrapper for separation, transcription, metadata,
  and routing.
- `models/mamba_separator.py` — Mamba-based STFT/mask prototype.
- `models/mamba_transcriber.py` — Mamba-based onset/pitch/velocity/expression
  heads and confidence output.
- `models/confidence_injector.py` — per-event confidence/alignment metadata.
- `models/losses.py` — multi-resolution STFT, spectral-flatness, crest-factor,
  and onset/energy alignment losses; several advertised transcription losses
  are not implemented.
- `data/datasets.py` — synthetic samples and partial Slakh/MUSDB data adapters.
- `utils/quality_gates.py` — confidence/metric-based Studio/Draft/Complex
  routing.
- `utils/template_engine.py` and `user_content/` — local Markdown-template
  rendering and draft copy.
- `research/` — experimental Mamba-3 code and a vendored upstream Mamba
  reference snapshot.
- `tests/` — API, audio-loading, dataset, MIDI, quality-gate, template, and
  model-module tests. Some audio/model tests are skipped or dependency-gated;
  the full stack has not been run here.

## Important gaps

- No trained model checkpoint is included. The default API model has random
  weights.
- The transcriber now stores its config and caches the mel basis, but MIDI
  event batching still does not match the confidence-injector contract.
  End-to-end forward inference is therefore not verified.
- The streaming helper passes a `state_cache` argument unsupported by the
  separator; true SSM streaming is not implemented.
- The pipeline transcribes guitar only; the bass MIDI entry is generated from
  the same event list.
- `si_sdr` is a spectral-centroid heuristic, not a measured SI-SDR score.
- Synthetic/real dataset selection, training target shapes, and loss shapes
  need tests and repair.
- No normal test/lint/build CI workflow, dependency lockfile, or Python support
  declaration is present. The only workflow is CodeQL.
- Product copy in `USER_GUIDE.md`/`user_content/` has been reduced to prototype
  caveats; no web editor, human review/refund service, or latency benchmark is
  implemented. The separate Python `route_by_quality()` return still contains
  stale editor/payment/refund copy and needs an owner decision before exposure.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the component map and
[API_DOCUMENTATION.md](API_DOCUMENTATION.md) for route behavior.
