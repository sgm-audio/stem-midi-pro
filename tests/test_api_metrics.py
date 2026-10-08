# SPDX-License-Identifier: Apache-2.0
"""Prometheus /metrics endpoint tests (T-8.3.12).

Expected behavior (per TODO API-6.16):
- GET /metrics → 200 with Prometheus text exposition format
- requests_total counter increments after a request

Xfail-guarded: API-6.16 is not yet implemented.
"""

import pytest

fastapi = pytest.importorskip("fastapi", reason="requires fastapi")
pytest.importorskip("httpx", reason="TestClient requires httpx")


def _client():
    pytest.importorskip("torch")
    try:
        import api
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"Cannot import api.py (see T-8.2.1): {exc}")

    from fastapi.testclient import TestClient

    return TestClient(api.app, raise_server_exceptions=True)


def _scrape_value(body: str, metric: str) -> float:
    """Extract `metric <value>` (with no labels) or sum labelled samples."""
    total = 0.0
    found = False
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith(metric):
            found = True
            total += float(line.rsplit(" ", 1)[-1])
    return total if found else -1.0


def test_requests_total_increments():
    """requests_total counter must go up after hitting an endpoint."""
    client = _client()

    before = _scrape_value(client.get("/metrics").text, "requests_total")
    client.get("/health")  # any request; auth-open per API-6.5
    after = _scrape_value(client.get("/metrics").text, "requests_total")

    assert after >= 0, "requests_total counter missing from /metrics"
    if before >= 0:
        assert after > before
