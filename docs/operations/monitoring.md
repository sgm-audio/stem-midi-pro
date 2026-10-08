# Monitoring — Stem+MIDI Pro

## Current state

Today the service exposes only `GET /health` (200/503). The Prometheus `/metrics` endpoint, structlog JSON logs, and OpenTelemetry tracing are planned — TODO API-6.16, API-6.10, API-6.19, ARCH-11.14.

## What to watch now (minimal viable)

Even without `/metrics`, a small deployment should monitor:

| Signal | How | Alert when |
|---|---|---|
| Process alive | `curl /health` every 30–60 s (uptime monitor works) | any non-200 |
| Memory | OS-level (`ps`, systemd cgroup) | sustained > 6 GB on an 8 GB box |
| Disk | OS-level | < 10% free (temp files) |
| Response time | time a `POST /process` with a 1-min test file | > ~2× baseline (baseline ≈ 30–60 s) |
| Log errors | scrape uvicorn/journald output for `500` | any repeated 5xx |

## Planned metrics (API-6.16)

When `/metrics` lands, alert on:

- `requests_total{route,status}` — rate of 5xx > 5% over 5 m
- `errors_total{route,kind}` — any OOM kind
- `request_duration_seconds{route}` — p95 on `/process` vs. the CPU baseline
- `in_flight_requests` — should never exceed 1 on the i5 target
- Quality-tier distribution (studio/draft/complex) over time — a sudden drift toward `complex` suggests an input or model regression (ARCH-11.14.3)

Grafana dashboard skeletons: TODO BLD-9.2.10 (`deploy/observability/`).

## Privacy note

Metrics must never include audio content or filenames beyond what operationally necessary; aggregate only.
