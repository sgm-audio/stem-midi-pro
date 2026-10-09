# API documentation

<!-- STATUS: in progress -->

This describes the checked-in FastAPI wrapper, not a hosted or production
service. The model stack and processing path have not been validated end to
end; see [Architecture](ARCHITECTURE.md). The API has no `/api/v1` prefix.

## Base URL and routes

When run locally on the default port, the base URL is `http://localhost:8000`.
The routes are:

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Returns 200 when the global model is loaded; otherwise 503. |
| `GET` | `/model-info` | Returns loaded model configuration; otherwise 503. |
| `POST` | `/process` | Accepts a multipart upload and returns a ZIP response. |
| `GET` | `/templates` | Lists Markdown files under `user_content/`. |
| `POST` | `/render-template` | Renders a named template. The current function signature uses `template_name` as a query parameter and accepts an optional `variables` dictionary. |

FastAPI's interactive documentation is available at `/docs` when the
application starts successfully.

## Authentication, CORS, and operational limits

Authentication is optional. If `API_KEY` is set, `/process`, `/templates`,
and `/render-template` require the `X-API-Key` header. If it is unset, those
routes are unauthenticated. `/health` and `/model-info` are unauthenticated.
There is no rate limiter or concurrency cap.

`CORS_ORIGINS` is a comma-separated list. Its default is
`http://localhost:3000,http://localhost:5173`. Setting it to `*` enables all
origins and disables credentialed CORS; a specific allowlist enables
credentials.

`POST /process` currently enforces:

- Filename suffix: `.wav`, `.flac`, or `.mp3`.
- Maximum duration: 600 seconds.
- Accepted sample rates: 44,100 Hz or 48,000 Hz. Other rates are rejected by
  validation; they are not resampled by the API validation step.
- Maximum uploaded file bytes: `MAX_UPLOAD_BYTES`, default 50 MiB. The handler
  counts bytes while reading the parsed `UploadFile`, and also checks
  `Content-Length` with 1 MiB allowance for multipart overhead. Because FastAPI
  parses the multipart body before invoking this endpoint, these checks are not
  a pre-parser/ASGI request-body limit. Enforce a total body cap at a trusted
  reverse proxy or server boundary before public deployment.
- Bit depth is inspected and logged as a warning when outside the preferred
  16/24-bit range; it is not a hard rejection criterion.

The API reads the upload, writes it to a named temporary file for processing,
and attempts to unlink that file in a `finally` block. This is not secure
media erasure and should not be described as in-memory-only processing or a
24-hour retention guarantee. Concurrent requests can each consume upload
memory and model resources.

## Process audio

### Request

```bash
curl -X POST "http://localhost:8000/process" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@/path/to/audio.wav" \
  -o output.zip
```

The header is required only when `API_KEY` is configured.

### Response

On success, the endpoint returns `application/zip` with:

- `guitar_stem.wav`
- `bass_stem.wav`
- `guitar.mid`
- `bass.mid`
- `processing_report.json`

The `bass.mid` file is currently generated from the same event stream as
`guitar.mid`; it is not a separate bass transcription. MIDI output uses a fixed
120 BPM tempo and a 1/16-note default duration because note durations are not
predicted.

The report JSON currently includes:

| Field | Meaning in the current code |
| --- | --- |
| `si_sdr` | A spectral-centroid separation heuristic, not SI-SDR. |
| `phase_coherence` | A learned phase-head score, not a measured coherence metric. |
| `avg_confidence` | Mean output of an uncalibrated confidence head. |
| `artifact_flags` | Basic heuristic flags. |
| `low_confidence_notes` | Currently derived from low-confidence frames, not reliably counted notes. |
| `quality_tier` | Studio/draft/complex classification based on confidence and the `si_sdr` field; artifact flags do not currently force the complex tier. |

These names and scores should not be used as validated quality measurements.

### Errors

The endpoint can return:

- `400` for unsupported suffix, invalid audio, duration, or sample rate.
- `401` when API-key authentication is enabled and the key is absent or wrong.
- `413` when the configured file-size limit is exceeded.
- `503` when the model is not loaded.
- `500` for an unhandled processing/packaging error.

Error details are returned in FastAPI's `{"detail": "..."}` format. The
current implementation includes exception text in some `500` responses; do
not assume those messages are sanitized for public deployment.

## Model information and startup configuration

The model is loaded during FastAPI startup. Configuration is supplied by:

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_CONFIG_PATH` | `configs/model_config.yaml` | YAML model configuration. |
| `MODEL_CHECKPOINT_PATH` | unset | Optional model checkpoint. If set, the file must exist. If unset, the model is randomly initialized. |
| `API_KEY` | unset | Optional key for protected routes. |
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:5173` | Comma-separated CORS origins; `*` disables credentials. |
| `MAX_UPLOAD_BYTES` | `52428800` | Maximum upload file bytes (50 MiB). |
| `PORT` | `8000` | Port used by `python api.py`; when launching `uvicorn` directly, pass its port on the command line. |

Run from the repository root:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The full PyTorch/NeMo/Mamba-SSM stack is required to load the model. There is
no production-ready container guarantee; the Dockerfile has not been built in
the review environment. The API dependency bounds are broad and unpinned; see
[SECURITY.md](SECURITY.md) for unresolved multipart/Starlette advisory ranges.

## Versioning status

`api.py` currently hard-codes OpenAPI version `1.0.0`, but the repository has
no package version source or checked-in changelog. The GitHub `v0.1.0` release
description covers product areas absent from this checkout. This local clone is
shallow/grafted and contains no matching tag, so release lineage cannot be
established here. Maintainers must reconcile release history before treating
either version as authoritative.

<!-- STATUS: research -->
Latency targets, professional-quality claims, refunds/human review, privacy or
regulatory compliance, and browser/editor behavior are product requirements,
not implemented or measured API guarantees.
