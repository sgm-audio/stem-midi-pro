# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""
FastAPI API for Stem+MIDI Pro service.
Provides endpoints for audio processing, health checks, and file downloads.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any

import mido
import numpy as np
import soundfile as sf
import structlog
import torch
import zipfile
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from mido import MidiFile, MidiTrack, Message, MetaMessage
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUESTS_TOTAL = Counter(
    "stem_midi_requests_total", "Total requests", ["route", "status"]
)
ERRORS_TOTAL = Counter(
    "stem_midi_errors_total", "Total errors", ["route", "kind"]
)
REQUEST_DURATION = Histogram(
    "stem_midi_request_duration_seconds", "Request duration", ["route"]
)
IN_FLIGHT = Gauge("stem_midi_in_flight_requests", "In-flight requests")

# ---------------------------------------------------------------------------
# Lazy model import (avoids import-time CUDA dep)
# ---------------------------------------------------------------------------
StemMidiModel = None  # type: ignore
load_from_checkpoint = None  # type: ignore


def _import_model():
    global StemMidiModel, load_from_checkpoint
    if StemMidiModel is None:
        from main import StemMidiModel as _Model
        from main import load_from_checkpoint as _load_ckpt

        StemMidiModel = _Model
        load_from_checkpoint = _load_ckpt


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SUPPORTED_FORMATS = {".wav", ".flac", ".mp3"}
MAX_DURATION_SECONDS = 600
SUPPORTED_SAMPLE_RATES = {44100, 48000}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB
ALLOWED_TEMPLATES = {
    "upload_confirmation.md",
    "progress_updates.md",
    "completion_delivery.md",
    "rights_usage_prompt.md",
    "feedback_refinement.md",
    "implicit_feedback.md",
    "landing_page.md",
}

# Global state
model: Optional[object] = None
model_config: Optional[Dict] = None
inference_semaphore = asyncio.Semaphore(1)
API_KEYS: set = set()


def _load_api_keys():
    keys_env = os.getenv("API_KEYS", "")
    if keys_env:
        return set(k.strip() for k in keys_env.split(",") if k.strip())
    return set()


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_config, API_KEYS

    API_KEYS = _load_api_keys()
    logger.info("Loading Stem+MIDI Pro model...")
    _import_model()

    config_path = os.getenv("MODEL_CONFIG_PATH", "configs/model_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found at {config_path}")

    import yaml
    with open(config_path, "r") as f:
        model_config = yaml.safe_load(f)

    model = StemMidiModel(model_config)
    checkpoint_path = os.getenv("MODEL_CHECKPOINT_PATH")
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info("Loading checkpoint", path=checkpoint_path)
        # Module helper — StemMidiModel has no instance load_from_checkpoint
        model = load_from_checkpoint(checkpoint_path, config_path)
    model.eval()

    # Warmup: run a 1-sec synthetic inference so first real request isn't slow
    logger.info("Running warmup inference...")
    try:
        warmup_audio = torch.randn(1, 1, model_config["audio"]["sample_rate"])
        with torch.inference_mode():
            model.forward(warmup_audio)
        logger.info("Warmup complete")
    except Exception as e:
        logger.warning("Warmup failed (non-fatal)", error=str(e))

    logger.info("Model loaded successfully")
    yield

    # Teardown: drain in-flight requests
    logger.info("Shutting down, draining requests...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Stem+MIDI Pro API",
    description="Professional audio AI service for stem separation and MIDI transcription",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware stack
app.add_middleware(RequestIDMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

# CORS
origins_env = os.getenv("CORS_ORIGINS", "")
if not origins_env or origins_env == "*":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    origins = [o.strip() for o in origins_env.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------
async def verify_api_key(request: Request):
    if not API_KEYS:
        return  # No keys configured — allow all
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing API key")
    token = auth[len("Bearer "):]
    if token not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
@app.get("/live")
async def live():
    """Liveness probe — always 200 if process is up."""
    return {"status": "alive"}


@app.get("/ready")
async def ready():
    """Readiness probe — 503 while model not loaded."""
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "detail": "Model not loaded"},
            headers={"Retry-After": "5"},
        )
    return {"status": "ready"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")


@app.post("/warmup")
async def warmup():
    """Run a synthetic inference to warm up the model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        warmup_audio = torch.randn(1, 1, model_config["audio"]["sample_rate"])
        with torch.inference_mode():
            model.forward(warmup_audio)
        return {"status": "warmed_up"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warmup failed: {e}")


@app.get("/templates")
async def get_templates():
    """List available user-facing content templates."""
    from utils.template_engine import list_templates
    return {"templates": list_templates()}


@app.post("/render-template")
async def render_template_endpoint(request: Request):
    """
    Render a user-facing content template with the given variables.
    Body: {"template_name": "...", "variables": {...}}
    """
    from utils.template_engine import render_template

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    template_name = body.get("template_name", "")
    if template_name not in ALLOWED_TEMPLATES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown template '{template_name}'. Allowed: {sorted(ALLOWED_TEMPLATES)}",
        )

    variables = body.get("variables", {})
    try:
        rendered = render_template(template_name, variables)
        return {"template": template_name, "rendered": rendered}
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Template '{template_name}' not found."
        )


@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_name": model_config.get("name", "unknown"),
        "audio_config": model_config.get("audio", {}),
        "separator_config": model_config.get("separator", {}),
        "transcriber_config": model_config.get("transcriber", {}),
        "quality_gates": model_config.get("quality_gates", {}),
    }


# ---------------------------------------------------------------------------
# Audio validation
# ---------------------------------------------------------------------------
def validate_audio_file(file_path: str) -> Dict[str, Any]:
    """Validate audio file properties. Uses single sf.SoundFile context."""
    try:
        with sf.SoundFile(file_path) as f:
            info = {
                "duration": len(f) / f.samplerate,
                "sample_rate": f.samplerate,
                "channels": f.channels,
                "format": f.format,
                "subtype": f.subtype,
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {str(e)}")

    if info["duration"] > MAX_DURATION_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Audio duration {info['duration']:.2f}s exceeds maximum {MAX_DURATION_SECONDS}s",
        )
    if info["sample_rate"] not in SUPPORTED_SAMPLE_RATES:
        raise HTTPException(
            status_code=400,
            detail=f"Sample rate {info['sample_rate']}Hz not supported. Supported: {SUPPORTED_SAMPLE_RATES}",
        )
    return info


# ---------------------------------------------------------------------------
# MIDI building (moved from create_placeholder_midi)
# ---------------------------------------------------------------------------
def build_midi_from_events(midi_data: Dict, stem_type: str) -> bytes:
    """Convert MIDI metadata events to a real MIDI file using mido."""
    mid = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(MetaMessage("set_tempo", tempo=500000))
    track.append(MetaMessage("time_signature", numerator=4, denominator=4))
    track.append(
        MetaMessage("track_name", name=f"{stem_type} transcription")
    )
    track.append(
        MetaMessage(
            "instrument_name",
            name="Electric Guitar" if stem_type == "guitar" else "Electric Bass",
        )
    )
    track.append(
        MetaMessage("marker", text=f"{stem_type.capitalize()} - Stem+MIDI Pro")
    )

    hop_length = 512
    sample_rate = 44100
    ticks_per_beat = 480
    ticks_per_second = ticks_per_beat * (60 / 120)

    for ev in midi_data.get("midi_events", []):
        tick = int(ev["onset_frame"] * hop_length / sample_rate * ticks_per_second)
        note = ev["note"]
        velocity = min(127, max(1, ev["velocity"]))
        track.append(Message("note_on", note=note, velocity=velocity, time=tick))

        # Duration from confidence injector output, fallback to 1/16 note
        dur_frames = ev.get("duration_frames", int(0.25 * sample_rate / hop_length))
        dur_ticks = max(1, int(dur_frames * hop_length / sample_rate * ticks_per_second))
        track.append(Message("note_off", note=note, velocity=0, time=dur_ticks))

        cc_val = ev.get("cc_127_value", int(ev["confidence"] * 127))
        track.append(Message("control_change", control=127, value=cc_val, time=0))

    track.append(MetaMessage("end_of_track"))
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Response packaging
# ---------------------------------------------------------------------------
def create_response_zip(outputs: Dict) -> io.BytesIO:
    """Create a ZIP file in memory containing the processing results."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Guitar stem
        guitar_stem = outputs["stems"]["guitar"]
        guitar_buf = io.BytesIO()
        sf.write(guitar_buf, guitar_stem, model_config["audio"]["sample_rate"], format="WAV")
        zip_file.writestr("guitar_stem.wav", guitar_buf.getvalue())

        # Bass stem
        bass_stem = outputs["stems"]["bass"]
        bass_buf = io.BytesIO()
        sf.write(bass_buf, bass_stem, model_config["audio"]["sample_rate"], format="WAV")
        zip_file.writestr("bass_stem.wav", bass_buf.getvalue())

        # MIDI files (per-stem when available)
        midi_guitar = outputs.get("midi_guitar") or outputs.get("midi") or {}
        midi_bass = outputs.get("midi_bass") or {"midi_events": [], "summary": {}}
        zip_file.writestr("guitar.mid", build_midi_from_events(midi_guitar, "guitar"))
        zip_file.writestr("bass.mid", build_midi_from_events(midi_bass, "bass"))

        # Processing report
        report = {
            "si_sdr": float(outputs["report"].si_sdr),
            "phase_coherence": float(outputs["report"].phase_coherence),
            "avg_confidence": float(outputs["report"].avg_confidence),
            "artifact_flags": outputs["report"].artifact_flags,
            "low_confidence_notes": outputs["report"].low_confidence_notes,
            "quality_tier": outputs["report"].quality_tier.value,
        }
        zip_file.writestr("processing_report.json", json.dumps(report, indent=2))

    zip_buffer.seek(0)
    return zip_buffer


# ---------------------------------------------------------------------------
# Main processing endpoint
# ---------------------------------------------------------------------------
@app.post("/process")
async def process_audio(
    request: Request,
    file: UploadFile = File(...),
):
    """
    Process an audio file to extract stems and generate MIDI.
    Returns ZIP: guitar_stem.wav, bass_stem.wav, guitar.mid, bass.mid, processing_report.json
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    route = "/process"

    # Auth
    await verify_api_key(request)

    # Size limit
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_BYTES:
        REQUESTS_TOTAL.labels(route=route, status="413").inc()
        raise HTTPException(status_code=413, detail="Upload exceeds 500 MB limit")

    if model is None:
        REQUESTS_TOTAL.labels(route=route, status="503").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Concurrency cap
    if inference_semaphore.locked():
        REQUESTS_TOTAL.labels(route=route, status="503").inc()
        return JSONResponse(
            status_code=503,
            content={"detail": "Server at capacity"},
            headers={"Retry-After": "1"},
        )

    # File extension check
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        REQUESTS_TOTAL.labels(route=route, status="400").inc()
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format {file_ext}. Supported: {SUPPORTED_FORMATS}",
        )

    # Save to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        try:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        finally:
            await file.close()

    try:
        validate_audio_file(tmp_file_path)

        async with inference_semaphore:
            IN_FLIGHT.inc()
            logger.info("Processing", request_id=request_id, file=file.filename)
            try:
                outputs = model.process_audio_file(tmp_file_path)
            except MemoryError:
                ERRORS_TOTAL.labels(route=route, kind="oom").inc()
                raise HTTPException(
                    status_code=503,
                    detail="Out of memory",
                    headers={"Retry-After": "30"},
                )
            except torch.cuda.OutOfMemoryError:
                ERRORS_TOTAL.labels(route=route, kind="cuda_oom").inc()
                raise HTTPException(
                    status_code=503,
                    detail="GPU out of memory",
                    headers={"Retry-After": "30"},
                )
            finally:
                IN_FLIGHT.dec()

        zip_buffer = create_response_zip(outputs)
        base_filename = Path(file.filename).stem
        download_filename = f"{base_filename}_stem-midi-package.zip"

        REQUESTS_TOTAL.labels(route=route, status="200").inc()
        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={download_filename}",
                "X-Request-ID": request_id,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        ERRORS_TOTAL.labels(route=route, kind="internal").inc()
        logger.error("Processing failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        try:
            os.unlink(tmp_file_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )