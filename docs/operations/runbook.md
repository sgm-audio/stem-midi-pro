# Operations Runbook — Stem+MIDI Pro

Common issues and fixes for a small CPU deployment (bare-metal i5, systemd or direct uvicorn; see [SETUP.md](../../SETUP.md)).

## Service won't start

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: Model config not found` | wrong CWD or `MODEL_CONFIG_PATH` | run from repo root or set `MODEL_CONFIG_PATH=configs/model_config.yaml` |
| Import error on `mamba_ssm` | broken install / wrong Python | confirm Python 3.13, reinstall `pip install -r requirements.txt` |
| Port already in use | another uvicorn running | `lsof -i :8000`, kill, or set `PORT` |
| Startup hangs before "Model loaded" | model build is slow on CPU | wait; first init can take tens of seconds. Watch RAM. |

## Requests failing

| Symptom | Likely cause | Fix |
|---|---|---|
| `503 Model not loaded` | startup failed or no validated checkpoint/backend weights are configured | set `MODEL_CHECKPOINT_PATH` (or use the explicit pretrained CPU backend config), then retry after `/ready` returns 200 |
| `401/403 Invalid or missing API key` | `API_KEYS` is enabled and the Bearer token is missing or wrong | send `Authorization: Bearer <key>` or unset `API_KEYS` |
| `413 Upload exceeds maximum size` | file exceeds the configured `MAX_UPLOAD_BYTES` (default 500 MB) | trim/compress, or lower/raise the limit deliberately (watch RAM) |
| `400 Sample rate not supported` | not 44.1/48 kHz | resample the file |
| `500 Processing failed` | model bug (see TODO Section 2) | check logs; if reproducible, file an issue with the input characteristics |
| Request takes minutes | CPU inference is synchronous and the target hardware is not benchmark-certified | record a baseline with the deployed checkpoint; do not rely on historical seconds-per-minute estimates |

## Resource problems

- **OOM kill / swap thrash:** close other apps; 10-min stereo uploads peak higher — reduce max duration via config or `MAX_UPLOAD_BYTES`. A smaller CPU model config is planned (TODO CFG-9.1.3).
- **CPU pegged at 100%:** expected during processing — the model is single-request CPU-bound.

## Logs

- Development: uvicorn stdout.
- Production (systemd): `journalctl -u stem-midi-pro -f` (unit file is TODO BLD-9.2.9).
- When installed, `structlog` emits JSON logs; otherwise set `LOG_FORMAT=json` for the stdlib fallback. OpenTelemetry tracing remains planned (TODO API-6.19).

## First-response checklist

1. `curl localhost:8000/live` → 200; then `curl localhost:8000/ready` → 200 only when validated weights are loaded.
2. Check process RAM (`ps` / Task Manager) — near the declared memory budget?
3. Check the last request in the logs for a `500` detail message.
4. Restart the service; model reload is the cheapest reset.
