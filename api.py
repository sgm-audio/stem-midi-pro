# SPDX-License-Identifier: Apache-2.0
"""
FastAPI API for Stem+MIDI Pro service.

Hardened per TODO.md Section 6:
- API-6.1/6.8  lifespan startup/shutdown (replaces @app.on_event)
- API-6.2      CORS: "*" => no credentials; explicit list => credentials
- API-6.3      500 MB upload cap (413)
- API-6.4      asyncio.Semaphore(1) concurrency cap (503 + Retry-After)
- API-6.5      Bearer API-key auth on /process only (env API_KEYS)
- API-6.6/6.15 single sf.SoundFile validate+open, native-rate mono load (audio_io)
- API-6.7      MemoryError / torch.cuda.OutOfMemoryError -> 503 + Retry-After
- API-6.9      X-Request-ID middleware (uuid4), echoed + in error responses
- API-6.10     structlog JSON logging if installed; else stdlib + field filter
- API-6.11/12  build_midi_from_events lives in utils/midi.py, uses duration_frames
- API-6.14     /render-template POST with JSON body + template allowlist
- API-6.16     /metrics (prometheus_client if installed; else hand-rolled text)
- API-6.17     /live and /ready (503 + Retry-After while model loading)
- API-6.18     security headers (nosniff, HSTS when behind TLS)
- API-6.20     /warmup + auto-warm 1s synthetic inference in lifespan
- ARCH-11.10.3 temp-file cleanup on all exception paths (try/finally)

Skipped by design:
- API-6.13 streaming redesign (main.py owner)
- API-6.19 OpenTelemetry tracing (planned; dep intentionally not added)
- ARCH-11.10.2 rate limiting (separate work item)
"""

import asyncio
import contextvars
import io
import json
import logging
import os
import secrets
import tempfile
import time
import uuid
import zipfile
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from audio_io import AudioValidationError, open_and_validate
from utils.midi import build_midi_from_events
from utils.template_engine import list_templates, render_template

# ---------------------------------------------------------------------------
# Logging (API-6.10): structlog JSON if installed; stdlib + field filter else.
# ---------------------------------------------------------------------------

_request_ctx: "contextvars.ContextVar[dict[str, Any] | None]" = contextvars.ContextVar(
    "request_ctx", default=None
)

try:  # optional dep — do NOT require it
    import structlog  # type: ignore

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )
    try:
        import structlog.contextvars as _scv

        def _bind_ctx(fields: dict[str, Any]) -> None:
            _scv.bind_contextvars(**fields)

        def _clear_ctx() -> None:
            _scv.clear_contextvars()

    except ImportError:

        def _bind_ctx(fields: dict[str, Any]) -> None:
            _request_ctx.set(fields)

        def _clear_ctx() -> None:
            _request_ctx.set({})

    _log = structlog.get_logger("stem_midi_api")
    _HAVE_STRUCTLOG = True
except ImportError:
    _HAVE_STRUCTLOG = False
    _log = logging.getLogger("stem_midi_api")

    class _RequestFieldFilter(logging.Filter):
        """Inject request_id/route/latency_ms/status_code into stdlib records."""

        def filter(self, record: logging.LogRecord) -> bool:
            ctx = _request_ctx.get() or {}
            record.request_id = ctx.get("request_id", "-")
            record.route = ctx.get("route", "-")
            record.latency_ms = ctx.get("latency_ms", "-")
            record.status_code = ctx.get("status_code", "-")
            return True

    def _bind_ctx(fields: dict[str, Any]) -> None:
        ctx = dict(_request_ctx.get() or {})
        ctx.update(fields)
        _request_ctx.set(ctx)

    def _clear_ctx() -> None:
        _request_ctx.set({})


def _configure_logging() -> None:
    if _HAVE_STRUCTLOG:
        # structlog renders to stdlib root; keep a simple sink
        logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
        return
    handler = logging.StreamHandler()
    handler.addFilter(_RequestFieldFilter())
    if os.getenv("LOG_FORMAT", "").lower() == "json":
        handler.setFormatter(
            logging.Formatter(
                '{"message": "%(message)s", "request_id": "%(request_id)s", '
                '"route": "%(route)s", "latency_ms": "%(latency_ms)s", '
                '"status_code": "%(status_code)s", "level": "%(levelname)s"}'
            )
        )
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s "
                "[request_id=%(request_id)s route=%(route)s "
                "status=%(status_code)s latency_ms=%(latency_ms)s] %(message)s"
            )
        )
    root = logging.getLogger("stem_midi_api")
    root.handlers[:] = [handler]
    root.setLevel(os.getenv("LOG_LEVEL", "INFO"))


def log_event(level: str, event: str, **fields: Any) -> None:
    """Emit a structured log line regardless of backend."""
    if _HAVE_STRUCTLOG:
        getattr(_log, level)(event, **fields)
    else:
        _CTX_KEYS = ("request_id", "route", "latency_ms", "status_code")
        _bind_ctx({k: v for k, v in fields.items() if k in _CTX_KEYS})
        extra = {k: v for k, v in fields.items() if k not in _CTX_KEYS}
        msg = f"{event} {json.dumps(extra, default=str)}" if extra else event
        getattr(_log, level)(msg)


# ---------------------------------------------------------------------------
# Metrics (API-6.16): prometheus_client if installed; hand-rolled fallback.
# ---------------------------------------------------------------------------

try:  # optional dep — do NOT require it
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _HAVE_PROM = True
    # Dedicated registry: avoids "Duplicated timeseries" when this module is
    # reloaded (e.g. tests rebuilding the app after env changes).
    _registry = CollectorRegistry()
    _m_requests = Counter(
        "requests_total", "HTTP requests", ["route", "status"], registry=_registry
    )
    _m_duration = Histogram(
        "request_duration_seconds", "Request latency", ["route"], registry=_registry
    )
    _m_in_flight = Gauge("in_flight_requests", "Requests currently in flight", registry=_registry)

    def metrics_track(route: str, status: int, duration: float) -> None:
        _m_requests.labels(route=route, status=str(status)).inc()
        _m_duration.labels(route=route).observe(duration)

    def metrics_render() -> tuple[bytes, str]:
        return generate_latest(_registry), CONTENT_TYPE_LATEST

except ImportError:
    _HAVE_PROM = False
    _counters: dict[str, int] = {}
    _dur_sum: dict[str, float] = {}
    _dur_count: dict[str, int] = {}
    _in_flight_value = 0

    def metrics_track(route: str, status: int, duration: float) -> None:
        key = f"{route}|{status}"
        _counters[key] = _counters.get(key, 0) + 1
        _dur_sum[route] = _dur_sum.get(route, 0.0) + duration
        _dur_count[route] = _dur_count.get(route, 0) + 1

    def metrics_render() -> tuple[bytes, str]:
        lines = ["# TYPE requests_total counter"]
        for key, val in sorted(_counters.items()):
            route, status = key.split("|", 1)
            lines.append(f'requests_total{{route="{route}",status="{status}"}} {val}')
        lines.append("# TYPE request_duration_seconds summary")
        for route in sorted(_dur_count):
            lines.append(f'request_duration_seconds_count{{route="{route}"}} {_dur_count[route]}')
            lines.append(f'request_duration_seconds_sum{{route="{route}"}} {_dur_sum[route]:.6f}')
        lines.append("# TYPE in_flight_requests gauge")
        lines.append(f"in_flight_requests {_in_flight_value}")
        return ("\n".join(lines) + "\n").encode(), "text/plain; version=0.0.4; charset=utf-8"


def _inc_in_flight(delta: int) -> None:
    if _HAVE_PROM:
        _m_in_flight.inc(delta)
    else:
        global _in_flight_value
        _in_flight_value += delta


# ---------------------------------------------------------------------------
# App + lifespan (API-6.1, 6.8, 6.20)
# ---------------------------------------------------------------------------

model = None  # StemMidiModel, lazily imported in load_model() (T-8.2.1)
model_config: dict | None = None
_model_ready = False
_processing_sem: asyncio.Semaphore | None = None  # API-6.4 concurrency cap

SUPPORTED_FORMATS = {".wav", ".flac", ".mp3"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(500 * 1024 * 1024)))  # API-6.3: 500 MB
RETRY_AFTER_SECONDS = "1"


# API-6.5: comma-separated Bearer keys; empty => open. Read per-request so the
# environment can be changed (and tests can monkeypatch) without re-importing.
def _api_keys() -> set:
    return {k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()}


ALLOWED_TEMPLATES = frozenset(list_templates())  # API-6.14 allowlist (7 files)


def load_model():
    """Load StemMidiModel from config/checkpoint. Imports lazily (T-8.2.1)."""
    global model_config

    config_path = os.getenv("MODEL_CONFIG_PATH", "configs/model_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found at {config_path}")

    import yaml

    with open(config_path) as f:
        model_config = yaml.safe_load(f)

    from main import StemMidiModel  # lazy: keeps `import api` torch-free

    checkpoint_path = os.getenv("MODEL_CHECKPOINT_PATH")
    if checkpoint_path and os.path.exists(checkpoint_path):
        log_event("info", "loading_checkpoint", checkpoint=checkpoint_path)
        from main import load_from_checkpoint

        m = load_from_checkpoint(checkpoint_path, config_path)
    else:
        m = StemMidiModel(model_config)

    m.eval()
    return m


async def warmup() -> None:
    """Run a 1-second synthetic inference so the first real request isn't slow (API-6.20)."""
    global model
    if model is None:
        log_event("warning", "warmup_skipped", reason="model_not_loaded")
        return
    try:
        import torch  # lazy

        sr = int((model_config or {}).get("audio", {}).get("sample_rate", 44100))
        synth = np.random.randn(sr).astype(np.float32) * 0.01  # 1s quiet noise

        def _run() -> None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                path = tmp.name
            try:
                sf.write(path, synth, sr, format="WAV")
                with torch.inference_mode():
                    model.process_audio_file(path)
            finally:
                with suppress(OSError):
                    os.unlink(path)

        await asyncio.to_thread(_run)
        log_event("info", "warmup_complete")
    except Exception as e:  # warmup must never crash startup
        log_event("warning", "warmup_failed", error=str(e))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup (API-6.1): load model + auto-warm. Shutdown (API-6.8): drain + cleanup."""
    global model, _model_ready, _processing_sem
    _configure_logging()
    _processing_sem = asyncio.Semaphore(1)

    log_event("info", "model_loading")
    try:
        model = load_model()
        _model_ready = True
        log_event("info", "model_loaded")
    except Exception as e:
        log_event("error", "model_load_failed", error=str(e))
        # Keep the process up so /live answers 200 and /ready answers 503.

    if model is not None:
        await warmup()

    yield

    # Graceful shutdown (API-6.8): wait for in-flight processing to drain.
    if _processing_sem is not None:
        async with _processing_sem:  # blocks until the in-flight request releases
            pass
    _model_ready = False
    model = None
    try:  # clear CUDA cache if a GPU build ever runs this (API-6.8)
        import torch

        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    log_event("info", "shutdown_complete")


app = FastAPI(
    title="Stem+MIDI Pro API",
    description="Professional audio AI service for stem separation and MIDI transcription",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS (API-6.2)
# ---------------------------------------------------------------------------

_raw_cors = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
_raw_origins = [o.strip() for o in _raw_cors if o.strip()]

if not _raw_origins or _raw_origins == ["*"]:
    _cors_origins = ["*"]
    _cors_credentials = False
elif "*" in _raw_origins:
    # Mismatched config: wildcard mixed with explicit origins is ambiguous
    # about credentials — fail fast instead of silently choosing (API-6.2).
    raise RuntimeError(
        "Invalid CORS_ORIGINS: '*' cannot be combined with explicit origins. "
        "Use '*' alone (no credentials) or a comma-separated origin list (with credentials)."
    )
else:
    _cors_origins = _raw_origins
    _cors_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# OOM exception types, resolved lazily so torch stays optional (API-6.7)
# ---------------------------------------------------------------------------


def _oom_exceptions() -> tuple:
    excs: list = [MemoryError]
    try:
        import torch

        if hasattr(torch, "cuda") and hasattr(torch.cuda, "OutOfMemoryError"):
            excs.append(torch.cuda.OutOfMemoryError)
    except ImportError:
        pass
    return tuple(excs)


# ---------------------------------------------------------------------------
# Auth (API-6.5): Bearer key, /process only. Docs/health/metrics stay open.
# ---------------------------------------------------------------------------


async def require_api_key(authorization: str | None = Header(default=None)) -> None:
    """401 when no/invalid Bearer token presented and API_KEYS is configured."""
    keys = _api_keys()
    if not keys:
        return
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization: Bearer <key>")
    token = authorization[7:].strip()
    if not any(secrets.compare_digest(token, k) for k in keys):
        raise HTTPException(status_code=403, detail="Invalid API key")


# ---------------------------------------------------------------------------
# Middleware: request-ID (API-6.9), security headers (API-6.18), metrics,
# per-request structured log line (API-6.10).
# ---------------------------------------------------------------------------


@app.middleware("http")
async def max_body_middleware(request: Request, call_next):
    """Belt-and-braces 500MB cap on the whole request body (API-6.3)."""
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_UPLOAD_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Request body exceeds {MAX_UPLOAD_BYTES} bytes"},
                )
        except ValueError:
            return JSONResponse(
                status_code=400, content={"detail": "Invalid Content-Length header"}
            )
    return await call_next(request)


# NOTE: defined after max_body_middleware so Starlette applies it OUTERMOST —
# every response (including early 413/400 above) gets X-Request-ID + security
# headers and gets counted in metrics.
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id
    route = request.url.path
    _bind_ctx({"request_id": request_id, "route": route})
    _inc_in_flight(1)
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        status = 500
        _inc_in_flight(-1)
        duration = time.perf_counter() - start
        metrics_track(route, status, duration)
        log_event(
            "error",
            "unhandled_exception",
            status_code=status,
            latency_ms=round(duration * 1000, 2),
        )
        raise
    duration = time.perf_counter() - start
    _inc_in_flight(-1)
    metrics_track(route, status, duration)
    _bind_ctx({"status_code": status, "latency_ms": round(duration * 1000, 2)})
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"  # API-6.18
    # HSTS only when actually behind TLS (note: the API itself speaks plain
    # HTTP on the bare-metal box; a TLS-terminating proxy sets fwd headers).
    if request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    log_event(
        "info",
        "request_complete",
        status_code=status,
        latency_ms=round(duration * 1000, 2),
        route=route,
        request_id=request_id,
    )
    return response


# ---------------------------------------------------------------------------
# Error responses carry request_id (API-6.9)
# ---------------------------------------------------------------------------


def _request_id_of(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "request_id": _request_id_of(request)},
        headers=exc.headers or None,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "request_id": _request_id_of(request)},
    )


# ---------------------------------------------------------------------------
# Health endpoints (API-6.17)
# ---------------------------------------------------------------------------


@app.get("/live")
async def live():
    """Liveness: process is up. Always 200."""
    return {"status": "alive"}


@app.get("/ready")
async def ready():
    """Readiness: model loaded and serving. 503 + Retry-After while loading."""
    if not _model_ready or model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not ready",
            headers={"Retry-After": RETRY_AFTER_SECONDS},
        )
    return {"status": "ready", "model_loaded": True}


@app.get("/health", include_in_schema=False)
async def health():
    """Deprecated alias of /ready (kept for back-compat)."""
    return await ready()


@app.get("/metrics")
async def metrics():
    """Prometheus exposition (API-6.16)."""
    body, content_type = metrics_render()
    return PlainTextResponse(body.decode(), media_type=content_type)


@app.post("/warmup")
async def warmup_endpoint():
    """Manually trigger the 1s synthetic warm-up inference (API-6.20)."""
    if not _model_ready or model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
            headers={"Retry-After": RETRY_AFTER_SECONDS},
        )
    await warmup()
    return {"status": "warmed"}


# ---------------------------------------------------------------------------
# Response packaging
# ---------------------------------------------------------------------------


def validate_audio_file(file_path: str) -> dict[str, Any]:
    """Validate an audio file and return its metadata dict.

    Thin wrapper over ``audio_io.open_and_validate`` for the API surface:
    raises ``HTTPException(400)`` instead of ``AudioValidationError`` so
    request handlers and tests get a proper HTTP error.
    """
    try:
        with open_and_validate(file_path) as (_snd, info):
            return info
    except AudioValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def create_response_zip(outputs: dict) -> io.BytesIO:
    """Create a ZIP in memory: stems (wav) + MIDI + processing report."""
    zip_buffer = io.BytesIO()
    sr = (model_config or {}).get("audio", {}).get("sample_rate", 44100)

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for stem_name in ("guitar", "bass"):
            buf = io.BytesIO()
            sf.write(buf, outputs["stems"][stem_name], sr, format="WAV")
            zip_file.writestr(f"{stem_name}_stem.wav", buf.getvalue())

        midi_data = outputs["midi"]
        zip_file.writestr("guitar.mid", build_midi_from_events(midi_data, "guitar"))
        zip_file.writestr("bass.mid", build_midi_from_events(midi_data, "bass"))

        report = outputs["report"]
        zip_file.writestr(
            "processing_report.json",
            json.dumps(
                {
                    "si_sdr": float(report.si_sdr),
                    "phase_coherence": float(report.phase_coherence),
                    "avg_confidence": float(report.avg_confidence),
                    "artifact_flags": report.artifact_flags,
                    "low_confidence_notes": report.low_confidence_notes,
                    "quality_tier": report.quality_tier.value,
                },
                indent=2,
            ),
        )

    zip_buffer.seek(0)
    return zip_buffer


# ---------------------------------------------------------------------------
# /process
# ---------------------------------------------------------------------------


async def _read_upload(request: Request, file: UploadFile) -> tuple[str, bytes]:
    """Validate headers/extension and stream the upload with a size cap."""
    multipart_overhead = 1024 * 1024
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_UPLOAD_BYTES + multipart_overhead:
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds maximum size of {MAX_UPLOAD_BYTES} bytes",
                )
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header") from None

    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format {file_ext}. Supported: {SUPPORTED_FORMATS}",
        )

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds maximum size of {MAX_UPLOAD_BYTES} bytes",
            )
        chunks.append(chunk)
    return file_ext, b"".join(chunks)


async def _do_process(tmp_file_path: str, quantize: bool = False) -> io.BytesIO:
    """Validate (single sf open, native-rate mono load) + run the model."""
    # API-6.6/6.15: one sf.SoundFile open covers validation AND the audio
    # load; model.process_audio_file re-loads internally (main.py owner).
    with open_and_validate(tmp_file_path) as (_snd, info):
        log_event(
            "info",
            "audio_validated",
            duration=round(info["duration"], 2),
            sample_rate=info["sample_rate"],
            channels=info["channels"],
        )

    outputs = await asyncio.to_thread(model.process_audio_file, tmp_file_path, quantize)
    return create_response_zip(outputs)


@app.post("/process")
async def process_audio(
    request: Request,
    file: UploadFile = File(...),  # noqa: B008 - FastAPI dependency idiom
    quantize: bool = False,
    _: None = Depends(require_api_key),
):
    """
    Process an audio file to extract stems and generate MIDI.

    Returns a ZIP with guitar/bass stems, MIDI files, and a processing report.
    """
    # Readiness is surfaced via /ready; here a present model (e.g. injected in
    # tests or by a lazy loader) is enough to serve.
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
            headers={"Retry-After": RETRY_AFTER_SECONDS},
        )

    # API-6.4: one concurrent processing job.
    sem = _processing_sem
    if sem is not None and sem.locked():
        raise HTTPException(
            status_code=503,
            detail="Server busy processing another request; retry shortly",
            headers={"Retry-After": RETRY_AFTER_SECONDS},
        )

    file_ext = content = None
    try:
        file_ext, content = await _read_upload(request, file)
    finally:
        await file.close()

    tmp_file_path: str | None = None
    acquired = False
    try:
        if sem is not None:
            await sem.acquire()
            acquired = True
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(content)
                tmp_file_path = tmp_file.name

            log_event("info", "processing_started", filename=file.filename)
            zip_buffer = await _do_process(tmp_file_path, quantize)
        finally:
            if acquired:
                sem.release()

        download_filename = f"{Path(file.filename or 'audio').stem}_stem-midi-package.zip"
        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={download_filename}"},
        )

    except HTTPException:
        raise
    except _oom_exceptions() as e:  # API-6.7
        log_event("error", "out_of_memory", error=str(e))
        raise HTTPException(
            status_code=503,
            detail="Out of memory; retry after resources free up",
            headers={"Retry-After": RETRY_AFTER_SECONDS},
        ) from e
    except AudioValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        log_event("error", "processing_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Processing failed: {e!s}") from e
    finally:
        # ARCH-11.10.3: temp file removed on every path, success or failure.
        if tmp_file_path:
            with suppress(OSError):
                os.unlink(tmp_file_path)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


@app.get("/templates")
async def get_templates():
    """List available user-facing content templates (open)."""
    return {"templates": list_templates()}


@app.post("/render-template")
async def render_template_endpoint(request: Request):
    """
    Render a template (API-6.14). JSON body:
    {"template_name": "completion_delivery.md", "variables": {"key": "value"}}

    Only the allowlisted templates in user_content/ are accepted (400 else).
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body must be JSON") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    template_name = body.get("template_name")
    variables = body.get("variables") or {}
    if not isinstance(template_name, str) or not template_name:
        raise HTTPException(status_code=400, detail="'template_name' (string) is required")
    if template_name not in ALLOWED_TEMPLATES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Template '{template_name}' is not allowed. "
                f"Allowed: {sorted(ALLOWED_TEMPLATES)}"
            ),
        )
    if not isinstance(variables, dict):
        raise HTTPException(status_code=400, detail="'variables' must be an object")

    try:
        rendered = render_template(template_name, variables)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Template '{template_name}' not found."
        ) from None
    return {"template": template_name, "rendered": rendered}


@app.get("/model-info")
async def get_model_info():
    """Info about the loaded model."""
    if not _model_ready or model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
            headers={"Retry-After": RETRY_AFTER_SECONDS},
        )
    cfg = model_config or {}
    return {
        "model_name": cfg.get("name", "unknown"),
        "audio_config": cfg.get("audio", {}),
        "separator_config": cfg.get("separator", {}),
        "transcriber_config": cfg.get("transcriber", {}),
        "training_config": cfg.get("training", {}),
        "quality_gates": cfg.get("quality_gates", {}),
    }


def main() -> None:
    """Console-script entry point (pyproject ``stem-midi-api``)."""
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )


if __name__ == "__main__":
    main()
