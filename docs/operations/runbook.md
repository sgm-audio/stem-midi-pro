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
| `503 Model not loaded` | hit before startup finished | retry after startup completes |
| `401 Invalid or missing X-API-Key` | auth enabled, wrong/missing key | set header or unset `API_KEY` |
| `413 Upload exceeds maximum size` | file > 50 MiB | trim/compress, or raise `MAX_UPLOAD_BYTES` (watch RAM) |
| `400 Sample rate not supported` | not 44.1/48 kHz | resample the file |
| `500 Processing failed` | model bug (see TODO Section 2) | check logs; if reproducible, file an issue with the input characteristics |
| Request takes minutes | normal on CPU | ~30–60 s per minute of audio; inform users |

## Resource problems

- **OOM kill / swap thrash:** close other apps; 10-min stereo uploads peak higher — reduce max duration via config or `MAX_UPLOAD_BYTES`. A smaller CPU model config is planned (TODO CFG-9.1.3).
- **CPU pegged at 100%:** expected during processing — the model is single-request CPU-bound.

## Logs

- Development: uvicorn stdout.
- Production (systemd): `journalctl -u stem-midi-pro -f` (unit file is TODO BLD-9.2.9).
- JSON structured logging (`structlog`, `LOG_FORMAT=json`) is planned (TODO API-6.10).

## First-response checklist

1. `curl localhost:8000/health` → 200?
2. Check process RAM (`ps` / Task Manager) — near 8 GB limit?
3. Check the last request in the logs for a `500` detail message.
4. Restart the service; model reload is the cheapest reset.
