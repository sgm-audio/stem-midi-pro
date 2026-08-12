# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Tests for /metrics (Prometheus) endpoint."""
from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from starlette.testclient import TestClient


@pytest.fixture
def client():
    from api import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    app.router.lifespan_context = noop_lifespan
    return TestClient(app)


def test_metrics_returns_prometheus_text(client):
    """/metrics should return Prometheus exposition format with our metrics."""
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers.get("content-type", "").startswith("text/plain")
    body = r.text
    # The two counters and one gauge we register in api.py module scope
    assert "stem_midi_requests_total" in body, "requests_total missing"
    assert "stem_midi_in_flight_requests" in body, "in_flight gauge missing"
    # Prometheus exposition includes HELP/TYPE comments
    assert "# HELP" in body
    assert "# TYPE" in body


def test_metrics_has_help_line(client):
    """Each registered metric should have a # HELP comment."""
    r = client.get("/metrics")
    assert "# HELP stem_midi_requests_total" in r.text
    assert "# HELP stem_midi_errors_total" in r.text
    assert "# HELP stem_midi_request_duration_seconds" in r.text
    assert "# HELP stem_midi_in_flight_requests" in r.text


def test_requests_total_does_not_decrease(client):
    """requests_total counter should be monotonically non-decreasing."""
    r0 = client.get("/metrics")
    baseline = _sum_counter(r0.text, "stem_midi_requests_total")
    client.get("/metrics")
    client.get("/metrics")
    r1 = client.get("/metrics")
    after = _sum_counter(r1.text, "stem_midi_requests_total")
    assert after >= baseline


def _sum_counter(text: str, metric_name: str) -> float:
    """Sum all sample values for the given counter across labels."""
    total = 0.0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(metric_name + "{") or line == metric_name:
            try:
                value = float(line.split()[-1])
                total += value
            except (ValueError, IndexError):
                continue
    return total
