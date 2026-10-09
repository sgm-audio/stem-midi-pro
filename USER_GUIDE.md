# Prototype operator notes

<!-- STATUS: in progress -->

This repository does not include a hosted web interface or a customer-ready
service. There is no trained checkpoint in the current tree, and the output
quality is not validated. These notes are for local development only; do not
use them as a promise of production behavior, retention, privacy, or support.

## Local API

Follow the environment requirements in [README.md](README.md), then start the
API from the repository root:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The model is constructed at startup. Set `MODEL_CHECKPOINT_PATH` to an
existing compatible checkpoint if one is available. Without that variable,
the model uses random initialization and its output is not meaningful as a
transcription result.

The API exposes `/health`, `/model-info`, `/process`, `/templates`, and
`/render-template`. See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for the
current request/response contract and environment variables.

## Upload constraints

`POST /process` accepts `.wav`, `.flac`, and `.mp3` filename suffixes, with a
maximum duration of 600 seconds and a default per-file byte limit of 50 MiB.
FastAPI parses multipart data before this handler checks file bytes; the limit
is not a total request-body cap. Public deployments need an upstream body cap.
API validation accepts only 44.1 kHz and 48 kHz audio. Bit depth is logged as a
warning, not rejected. The actual audio/model path has not been validated end
to end.

Uploads are written to a temporary file during processing. The API attempts to
unlink that file after the request; this is not secure erasure, and this
prototype makes no 24-hour deletion or in-memory-only guarantee. Avoid
sensitive recordings in untrusted deployments.

## Output limitations

The ZIP response is intended to contain guitar and bass WAVs, two MIDI files,
and a JSON report. The model currently transcribes only the guitar stem and
uses the same MIDI event stream for the bass MIDI file. The `si_sdr` report
field is a spectral-centroid heuristic rather than SI-SDR, and the confidence
scores are not calibrated. Do not use the quality tier as a professional
quality guarantee.

No web MIDI editor, automatic tempo/tuning detection, human-review service,
refund process, user accounts, or published latency target is implemented in
this checkout.

## Development checks

```bash
python check_syntax.py
python verify_structure.py
pytest
```

The test suite and full model stack require the packages in `requirements.txt`.
Some audio/model tests remain skipped; see the source tests and
[Architecture](ARCHITECTURE.md) for known gaps.
