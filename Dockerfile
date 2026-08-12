# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
# Multi-stage Dockerfile for Stem+MIDI Pro (CPU-only, bare-metal i5)

# Stage 1: Builder
FROM python:3.13-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Stage 2: Runtime
FROM python:3.13-slim

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 appuser

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application
COPY --chown=appuser:appuser . .

# Create directories
RUN mkdir -p /app/data /app/outputs /app/configs /app/user_content && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Health check: liveness probe
HEALTHCHECK --interval=30s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/live')" || exit 1

# Run with graceful shutdown
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "30"]
