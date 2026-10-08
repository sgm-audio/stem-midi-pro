# syntax=docker/dockerfile:1

# ---------- Stage 1: builder ----------
FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Build tools needed to compile any sdists (e.g. mamba-ssm source build)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install project (CPU target) into /install prefix
COPY pyproject.toml README.md ./
COPY models/ models/
COPY utils/ utils/
COPY data/ data/
COPY requirements.txt ./
RUN pip install --prefix=/install -r requirements.txt \
    && pip install --prefix=/install --no-deps -e .

# ---------- Stage 2: runtime ----------
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:${PATH}"

# Runtime system deps: libsndfile (soundfile/librosa), curl (healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application code
COPY models/ models/
COPY utils/ utils/
COPY data/ data/
COPY configs/ configs/
COPY user_content/ user_content/
COPY api.py main.py audio_io.py ./

# Non-root user
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD curl -fs http://localhost:8000/live || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "30"]
