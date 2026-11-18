# Monitoring — Stem+MIDI Pro

## Current state

The service exposes `GET /live` (always 200), `GET /ready` (200 when a validated model is loaded, otherwise 503), `GET /health` (deprecated alias of `/ready`), and `GET /metrics` (Prometheus text exposition: `requests_total`, `request_duration_seconds`, `in_flight_requests`). When installed, structlog emits JSON logs; the stdlib fallback supports `LOG_FORMAT=json`. OpenTelemetry tracing remains planned (TODO API-6.19, ARCH-11.14).

## What to watch now (minimal viable)

Even without `/metrics`, a small deployment should monitor:

| Signal | How | Alert when |
|---|---|---|
| Process alive | `curl /health` every 30–60 s (uptime monitor works) | any non-200 |
| Memory | OS-level (`ps`, systemd cgroup) | sustained > 6 GB on an 8 GB box |
| Disk | OS-level | < 10% free (temp files) |
| Response time | time a `POST /process` with a representative test file and the deployed checkpoint | > ~2× a locally recorded baseline |
| Log errors | scrape uvicorn/journald output for `500` | any repeated 5xx |

## Metrics and alerts (API-6.16)

Use `/metrics` to alert on:

- `requests_total{route,status}` — rate of 5xx > 5% over 5 m
- Application logs — repeated OOM or processing failures
- `request_duration_seconds{route}` — p95 on `/process` vs. the locally recorded CPU baseline
- `in_flight_requests` — should never exceed 1 on the i5 target
- Quality-tier distribution from retained processing-report summaries (not exposed by `/metrics`) — a sudden drift toward `complex` suggests an input or model regression (ARCH-11.14.3)

Grafana dashboard skeletons: TODO BLD-9.2.10 (`deploy/observability/`).

## Privacy note

Metrics must never include audio content or filenames beyond what operationally necessary; aggregate only.
