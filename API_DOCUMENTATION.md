# Stem+MIDI Pro API Documentation

## Overview

The Stem+MIDI Pro API is a FastAPI service for guitar/bass stem separation and MIDI transcription. It runs CPU-only; see [SETUP.md](SETUP.md) for deployment. A validated Mamba checkpoint is required unless the config explicitly selects the tested pretrained Demucs + Basic Pitch pair with `weights_source: pretrained_dependencies`. Missing/invalid weights keep the process live but make `/ready` and `/process` return `503` rather than serving random weights.

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

A hard concurrency cap of **1 in-flight processing request** is enforced (`asyncio.Semaphore(1)`); excess requests fail fast with `503 Service Unavailable` + `Retry-After: 1`. No reproducible reference-i5 throughput result is committed, so historical seconds-per-minute estimates are not a service guarantee. Per-key token-bucket rate limiting is not implemented (TODO ARCH-11.10.2).

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
| `quantize` | bool | `false` | When `true`, run inference with INT8 dynamic quantization of Linear layers (PERF-7.7/7.8). **Measured on this CPU target it is not faster**: 0.21 s fp32 vs 0.22 s INT8 for a 3 s clip (d_model 128, 4 layers, 2 threads) — the Python selective scan dominates, not the matmuls — and the stems differ from fp32 by max 5.9e-3. Same output shapes; kept for completeness, not as an optimisation. |

**Success — `200 OK`**, `Content-Type: application/zip`. ZIP contents:

| File | Contents |
|---|---|
| `guitar_stem.wav` | Separated guitar stem |
| `bass_stem.wav` | Separated bass stem |
| `guitar.mid` | Independent guitar MIDI transcription; confidence uses Polyphonic Key Pressure when present |
| `bass.mid` | Independent bass MIDI transcription; confidence uses Polyphonic Key Pressure when present |
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

r = requests.post(
    "http://localhost:8000/process",
    headers={"Authorization": "Bearer <key>"},
    files={"file": open("song.wav", "rb")},
)
r.raise_for_status()
open("output.zip", "wb").write(r.content)
```

### `GET /model-info`

Returns the loaded model configuration plus `backends` and `checkpoint_loaded`: `model_name`, `audio_config`, `separator_config`, `transcriber_config`, `training_config`, `quality_gates`. `503` while the model is loading or the checkpoint is missing.

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
| `separation_proxy` | float | Centroid-separation heuristic reported for observability; it is not used as a tier threshold, SI-SDR, or a dB certification |
| `si_sdr` | float | Deprecated compatibility alias for `separation_proxy`; do not interpret as SI-SDR |
| `metric_semantics` | string | `centroid_separation_proxy_not_si_sdr` |
| `phase_coherence` | float | Model-reported phase score, 0–1; not independent quality certification |
| `avg_confidence` | float | Mean confidence over emitted guitar and bass notes, 0–1 |
| `confidence_encoding` | string | `polyphonic_key_pressure`; JSON event metadata remains canonical |
| `artifact_flags` | list[str] | e.g. `["clipping"]`; clean output is `["none"]`, and random weights are `["untrained_weights"]` |
| `low_confidence_notes` | int | Emitted notes below the confidence threshold, not all low-confidence frames |
| `model_weights_loaded` | bool/null | Whether a validated checkpoint was loaded |
| `quality_tier` | string | Heuristic `"studio"` / `"draft"` / `"complex"`, not an accuracy certificate |
| `invalid` | bool | True when any reported metric was NaN/Inf, i.e. the render is unusable regardless of the other numbers |
| `backend` | string/null | Which separator implementation produced the render; always present so a parser never branches on its existence |

**Numeric values may be `null` rather than `NaN`.** A non-finite metric (which the pipeline really produces for silent, DC-offset and non-finite input) is delivered as `null` so the document stays valid RFC 8259 — bare `NaN` is not JSON and makes `JSON.parse()` reject the whole download.

## Confidence: raw head vs the number the tier uses

`processing_report.json` carries two confidences:

| Field | Meaning |
|---|---|
| `avg_confidence` | Mean of the emitted notes' confidence, exactly as the head produced it. Observability only — measured on the small CPU config the untrained head ranks a clipped square (0.52) and white noise (0.51) *above* clean music (0.30). |
| `calibrated_confidence` | `avg_confidence` scaled by a content factor (stem tonality × crest factor, `main._content_factor`). **This is the only number the quality tier reads.** Silence and distortion-calibrated material land near 0; clean material is unchanged. |

Both are `0.0` when no notes were emitted, and `notes_total` / `total_notes` say whether the render
carried anything at all.

## Quality Tiers

1. **Studio gate** — `avg_confidence ≥ 0.85`, no artifact flags, and `low_confidence_notes == 0`: direct download with a warning to verify; the reported proxy is observability only and this is not SI-SDR certification.
2. **Draft** — `avg_confidence ≥ 0.70` without artifacts or untrained weights, but below the studio conditions: refine low-confidence notes in your DAW.
3. **Complex** — invalid metrics, `avg_confidence < 0.70`, artifact flags, or untrained weights: raw output plus manual review options.

## MIDI export behaviour

| Setting (model config) | Default | Effect |
|---|---|---|
| `midi.tempo` | `auto` | `auto` estimates the beat grid from the upload (`utils/tempo.py`; falls back to 120 BPM when there is no measurable pulse), `fixed` always writes 120 BPM, a number is read as BPM. The estimate never changes absolute note times — onset ticks are converted with the same tempo that is written into `set_tempo`. |
| `midi.include_expression` | `false` | When `true`, exported files carry the expression head's bend/vibrato/slide as pitch-wheel + CC. The head is untrained, so each event tags `expression.source`; values are omitted by default rather than invented. |

`/model-info` and the processing report stay unchanged; the detected tempo is reported per stem
(`tempo_us`, `tempo_source`) and once in `summary.timing` (`source`, `bpm`, `confidence`).

## CORS

Configured via the `CORS_ORIGINS` env var (comma-separated list):

- Unset → defaults to `http://localhost:3000,http://localhost:5173` with credentials allowed.
- A specific list → those origins, credentials allowed.
- `*` (or unset/empty handled as wildcard) → `allow_origins=["*"]` with credentials **disabled**, per the CORS spec (browsers reject `*` + credentials).

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_CONFIG_PATH` | `configs/model_config.yaml` | Model config YAML |
| `MODEL_CHECKPOINT_PATH` | unset | Path to a validated Mamba checkpoint; optional only for an explicit pretrained-backend config |
| `API_KEYS` | unset (auth off) | Comma-separated API keys; clients send `Authorization: Bearer <key>` |
| `CORS_ORIGINS` | localhost:3000,5173 | CORS allowlist |
| `MAX_UPLOAD_BYTES` | `524288000` (500 MB) | Upload size cap |
| `PORT` | `8000` | Bind port (when run via `python api.py`) |

## Performance Characteristics (CPU-only)

- Declared target: bare-metal Intel i5, 4–8 GB RAM.
- No reference-i5 benchmark is currently committed. Any performance claim must use a trained checkpoint, real audio, peak RSS, and the target hardware.
- One concurrent request by design; batch size 1. There is no required GPU path.

## Security & Privacy

- Uploaded files go to a temp file that is deleted after the request (success or failure).
- No audio is persisted; no personal data is collected.
- Model runs in inference mode only.

---
*Stem+MIDI Pro: stem separation + editable MIDI drafts. Not magic—just math that respects your craft.*
