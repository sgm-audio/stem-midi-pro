# Stem+MIDI Pro API Documentation

## Overview

The Stem+MIDI Pro API is a FastAPI service for guitar/bass stem separation and MIDI transcription. It runs CPU-only; see [SETUP.md](SETUP.md) for deployment.

## Base URL

There is **no `/api/v1` prefix**. Endpoints are mounted at the server root:

```
http://<host>:8000/process
http://<host>:8000/health
...
```

Interactive docs (Swagger UI) are at `/docs`; OpenAPI JSON at `/openapi.json`.

## Authentication

Optional API-key auth protects `POST /process` only:

- When the `API_KEYS` environment variable is **unset/empty**, auth is disabled (development default).
- When set, it is a comma-separated list of valid keys; clients must send `Authorization: Bearer <key>`.
- Missing Bearer header → `401`; presented but unknown key → `403`.

`/live`, `/ready`, `/health`, `/metrics`, `/warmup`, `/templates`, `/render-template`, `/model-info`, `/docs`, `/redoc`, and `/openapi.json` are always open.

Implemented per TODO API-6.5 (multi-key Bearer auth with `secrets.compare_digest`).

## Rate Limiting & Concurrency

A hard concurrency cap of **1 in-flight processing request** is enforced (`asyncio.Semaphore(1)`); excess requests fail fast with `503 Service Unavailable` + `Retry-After: 1`. On the bare-metal i5 target assume **one request at a time, ~30–60 s per minute of audio**. Per-key token-bucket rate limiting is not implemented (TODO ARCH-11.10.2).

## Endpoints

### `GET /live`

Liveness probe — the process is up. Always returns `200 {"status": "alive"}`.

### `GET /ready`

Readiness probe — the model is loaded and serving. Returns `503 {"detail": "Model not ready"}` with `Retry-After: 1` while loading, otherwise `200 {"status": "ready", "model_loaded": true}`.

### `GET /health`

Deprecated alias of `/ready`, kept for back-compat (hidden from the OpenAPI schema).

### `GET /metrics`

Prometheus text exposition (`requests_total`, `request_duration_seconds`, `in_flight_requests`). Uses `prometheus_client` when installed, otherwise a hand-rolled equivalent.

### `POST /warmup`

Runs a 1-second synthetic inference to warm the model (also runs automatically at startup). `503` + `Retry-After` if the model is not loaded.

### `POST /process`

Upload an audio file; receive a ZIP with stems, MIDI, and a processing report.

**Request:** `multipart/form-data` with field `file`.

- Formats: `.wav`, `.flac`, `.mp3`
- Max duration: 600 s (10 min)
- Max upload size: 500 MB (`MAX_UPLOAD_BYTES` env override)
- Sample rates: 44.1 kHz or 48 kHz

**Query parameters:**

| Param | Type | Default | Description |
|---|---|---|---|
| `quantize` | bool | `false` | When `true`, run inference with INT8 dynamic quantization of Linear layers (PERF-7.7/7.8). Lower CPU matmul cost; same output shapes. |

**Success — `200 OK`**, `Content-Type: application/zip`. ZIP contents:

| File | Contents |
|---|---|
| `guitar_stem.wav` | Separated guitar stem |
| `bass_stem.wav` | Separated bass stem |
| `guitar.mid` | Guitar MIDI transcription (CC#127 = per-note confidence) |
| `bass.mid` | Bass MIDI transcription |
| `processing_report.json` | Metrics (see schema below) |

**Example:**

```bash
curl -X POST "http://localhost:8000/process" \
     -H "Authorization: Bearer <key>" \
     -F "file=@song.wav" \
     -o output.zip
```

```python
import requests
r = requests.post("http://localhost:8000/process",
                  headers={"Authorization": "Bearer <key>"},
                  files={"file": open("song.wav", "rb")})
r.raise_for_status()
open("output.zip", "wb").write(r.content)
```

### `GET /model-info`

Returns the loaded model configuration: `model_name`, `audio_config`, `separator_config`, `transcriber_config`, `training_config`, `quality_gates`. `503` while the model is loading.

### `GET /templates`

Lists the 7 user-facing templates available in `user_content/`.

### `POST /render-template`

Renders a template with variables. Parameters: `template_name` (basename only, must exist under `user_content/`) and optional `variables` dict. Returns `{"template": ..., "rendered": ...}`. `404` for unknown template names.

## Error Codes

All errors return `{"detail": "<message>"}`.

| Code | Meaning |
|---|---|
| `400` | Invalid file: bad extension, undecodable audio, duration > 600 s, unsupported sample rate, malformed `Content-Length` |
| `401` | Missing `Authorization: Bearer` header (when `API_KEYS` is configured) |
| `403` | Presented API key is invalid |
| `404` | Unknown endpoint or template name |
| `405` | Wrong HTTP method |
| `413` | Upload exceeds 500 MB (`MAX_UPLOAD_BYTES`) |
| `500` | Processing failed (see server logs) |
| `503` | Model not loaded (startup), out of memory, or concurrency cap reached — all with `Retry-After: 1` |

Every error response also carries a `request_id` field matching the `X-Request-ID` response header (API-6.9).

## Processing Report Schema

`processing_report.json` inside the ZIP:

| Field | Type | Description |
|---|---|---|
| `si_sdr` | float | Separation-quality estimate in dB (currently a spectral-centroid proxy — see TODO C-2.6) |
| `phase_coherence` | float | Stem phase coherence, 0–1 |
| `avg_confidence` | float | Mean transcription confidence, 0–1 |
| `artifact_flags` | list[str] | e.g. `["clipping"]`, empty when clean |
| `low_confidence_notes` | int | Notes below the confidence threshold |
| `quality_tier` | string | `"studio"` / `"draft"` / `"complex"` |

## Quality Tiers

1. **Studio** — `avg_confidence ≥ 0.85` **and** `si_sdr ≥ 20 dB`: direct download.
2. **Draft** — `0.70 ≤ avg_confidence < 0.85`: refine low-confidence notes in your DAW (CC#127 tags them).
3. **Complex** — `avg_confidence < 0.70` or artifact flags present: raw output plus review options.

## CORS

Configured via the `CORS_ORIGINS` env var (comma-separated list):

- Unset → defaults to `http://localhost:3000,http://localhost:5173` with credentials allowed.
- A specific list → those origins, credentials allowed.
- `*` (or unset/empty handled as wildcard) → `allow_origins=["*"]` with credentials **disabled**, per the CORS spec (browsers reject `*` + credentials).

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_CONFIG_PATH` | `configs/model_config.yaml` | Model config YAML |
| `MODEL_CHECKPOINT_PATH` | unset | Optional checkpoint to load at startup |
| `API_KEYS` | unset (auth off) | Comma-separated API keys; clients send `Authorization: Bearer <key>` |
| `CORS_ORIGINS` | localhost:3000,5173 | CORS allowlist |
| `MAX_UPLOAD_BYTES` | `524288000` (500 MB) | Upload size cap |
| `PORT` | `8000` | Bind port (when run via `python api.py`) |

## Performance Characteristics (CPU-only)

- Reference hardware: bare-metal Intel i5, 4–8 GB RAM.
- **~30–60 s of processing per 1 minute of audio**, end to end.
- One concurrent request; a 3-minute song takes roughly 1.5–3 minutes.
- Batch size 1. There is no GPU path.

## Security & Privacy

- Uploaded files go to a temp file that is deleted after the request (success or failure).
- No audio is persisted; no personal data is collected.
- Model runs in inference mode only.

---
*Stem+MIDI Pro: stem separation + editable MIDI drafts. Not magic—just math that respects your craft.*
