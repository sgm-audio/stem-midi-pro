"""
FastAPI API for Stem+MIDI Pro service
Provides endpoints for audio processing, health checks, and file downloads.
"""

import io
import json
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import soundfile as sf
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from mido import Message, MetaMessage, MidiFile, MidiTrack
import uvicorn

from utils.template_engine import list_templates, render_template

if TYPE_CHECKING:
    from main import StemMidiModel

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Stem+MIDI Pro API",
    description="Professional audio AI service for stem separation and MIDI transcription",
    version="1.0.0"
)

# CORS: browsers reject Access-Control-Allow-Origin: * with credentials.
# Default to localhost origins + credentials; CORS_ORIGINS=* disables credentials.
_raw_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",") if o.strip()]
if not _raw_origins or _raw_origins == ["*"] or "*" in _raw_origins:
    _cors_origins = ["*"]
    _cors_credentials = False
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

# Global model instance
model: Optional["StemMidiModel"] = None
model_config: Optional[Dict] = None

# Supported audio formats and constraints
SUPPORTED_FORMATS = {".wav", ".flac", ".mp3"}
MAX_DURATION_SECONDS = 600  # 10 minutes
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))  # 50 MiB
API_KEY = os.getenv("API_KEY", "").strip()  # optional; if set, require X-API-Key
SUPPORTED_SAMPLE_RATES = {44100, 48000}
SUPPORTED_BIT_DEPTHS = {16, 24}  # Note: soundfile doesn't directly give bit depth, we'll infer


async def require_api_key_if_configured(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    """When API_KEY is set, require a matching X-API-Key header."""
    if not API_KEY:
        return
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def load_model() -> "StemMidiModel":
    """Load the StemMidiModel from configuration and an optional checkpoint."""
    global model_config

    config_path = os.getenv("MODEL_CONFIG_PATH", "configs/model_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found at {config_path}")

    checkpoint_path = os.getenv("MODEL_CHECKPOINT_PATH")
    if checkpoint_path and not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    import yaml

    with open(config_path, "r", encoding="utf-8") as config_file:
        model_config = yaml.safe_load(config_file)

    # Import the heavyweight model stack only when startup actually loads a model.
    from main import StemMidiModel

    loaded_model = StemMidiModel(model_config)
    if checkpoint_path:
        logger.info("Loading checkpoint from %s", checkpoint_path)
        loaded_model = loaded_model.load_from_checkpoint(checkpoint_path)
    else:
        logger.warning(
            "MODEL_CHECKPOINT_PATH is unset; API outputs will use random, untrained weights"
        )

    loaded_model.eval()
    return loaded_model


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model
    logger.info("Loading Stem+MIDI Pro model...")
    model = load_model()
    logger.info("Model loaded successfully!")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


def validate_audio_file(file_path: str) -> Dict[str, Any]:
    """
    Validate audio file properties.
    Returns validation info or raises HTTPException.
    """
    try:
        # Get audio file info
        info = sf.info(file_path)
        
        # Check duration
        if info.duration > MAX_DURATION_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=f"Audio duration {info.duration:.2f}s exceeds maximum {MAX_DURATION_SECONDS}s"
            )
        
        # Check sample rate
        if info.samplerate not in SUPPORTED_SAMPLE_RATES:
            raise HTTPException(
                status_code=400,
                detail=f"Sample rate {info.samplerate}Hz not supported. Supported: {SUPPORTED_SAMPLE_RATES}"
            )
        
        # Check format (by extension, but we already have the file)
        # Note: soundfile handles format detection
        
        # Bit depth inference (approximate)
        # soundfile doesn't directly give bit depth, but we can check subtype
        subtype = info.subtype
        bit_depth = None
        if 'FLOAT' in subtype or 'DOUBLE' in subtype:
            bit_depth = 32  # Float is typically 32-bit
        elif 'PCM_16' in subtype:
            bit_depth = 16
        elif 'PCM_24' in subtype:
            bit_depth = 24
        elif 'PCM_32' in subtype:
            bit_depth = 32
        
        if bit_depth and bit_depth not in SUPPORTED_BIT_DEPTHS:
            # Warning but not failure for now
            logger.warning(f"Bit depth {bit_depth} not in preferred list {SUPPORTED_BIT_DEPTHS}")
        
        return {
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "subtype": info.subtype,
            "bit_depth": bit_depth
        }
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio file: {str(e)}"
        )


def create_response_zip(outputs: Dict) -> io.BytesIO:
    """
    Create a ZIP file in memory containing the processing results.
    """
    # Create in-memory bytes buffer
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add guitar stem
        guitar_stem = outputs["stems"]["guitar"]
        guitar_buffer = io.BytesIO()
        sf.write(guitar_buffer, guitar_stem, model_config['audio']['sample_rate'], format='WAV')
        zip_file.writestr("guitar_stem.wav", guitar_buffer.getvalue())
        
        # Add bass stem
        bass_stem = outputs["stems"]["bass"]
        bass_buffer = io.BytesIO()
        sf.write(bass_buffer, bass_stem, model_config['audio']['sample_rate'], format='WAV')
        zip_file.writestr("bass_stem.wav", bass_buffer.getvalue())
        
        # Serialize the available event metadata. Note that the current pipeline
        # supplies the guitar event stream for both tracks and does not predict
        # note durations; create_placeholder_midi() uses fixed defaults.
        midi_data = outputs["midi"]
        
        # Guitar MIDI
        guitar_midi = create_placeholder_midi(midi_data, "guitar")
        zip_file.writestr("guitar.mid", guitar_midi)
        
        # Bass MIDI
        bass_midi = create_placeholder_midi(midi_data, "bass")
        zip_file.writestr("bass.mid", bass_midi)
        
        # Add processing report
        report = {
            "si_sdr": float(outputs["report"].si_sdr),
            "phase_coherence": float(outputs["report"].phase_coherence),
            "avg_confidence": float(outputs["report"].avg_confidence),
            "artifact_flags": outputs["report"].artifact_flags,
            "low_confidence_notes": outputs["report"].low_confidence_notes,
            "quality_tier": outputs["report"].quality_tier.value
        }
        zip_file.writestr("processing_report.json", json.dumps(report, indent=2))
    
    zip_buffer.seek(0)
    return zip_buffer


def create_placeholder_midi(midi_data: Dict, stem_type: str) -> bytes:
    """Convert MIDI metadata events to a MIDI file using a fixed 120 BPM tempo.

    Event onset frames are converted to absolute ticks, then serialized as MIDI
    delta times. The model does not currently predict note durations, so notes
    use a 1/16-note default duration.
    """
    ticks_per_beat = 480
    tempo = 500_000  # 120 BPM, in microseconds per quarter note
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(MetaMessage("set_tempo", tempo=tempo))
    track.append(MetaMessage("time_signature", numerator=4, denominator=4))
    track.append(MetaMessage("track_name", name=f"{stem_type} transcription"))
    instrument = "Electric Guitar" if stem_type == "guitar" else "Electric Bass"
    track.append(MetaMessage("instrument_name", name=instrument))
    track.append(MetaMessage("marker", text=f"{stem_type.capitalize()} — Stem+MIDI Pro"))

    audio_config = (model_config or {}).get("audio", {})
    hop_length = int(audio_config.get("hop_length", 512))
    sample_rate = int(audio_config.get("sample_rate", 44100))
    ticks_per_second = ticks_per_beat * 1_000_000 / tempo
    default_duration_ticks = max(1, ticks_per_beat // 4)

    # (absolute tick, ordering at that tick, insertion order, MIDI message)
    timeline = []
    for index, event in enumerate(midi_data.get("midi_events", [])):
        onset_frame = max(0, int(event["onset_frame"]))
        onset_tick = round(onset_frame * hop_length / sample_rate * ticks_per_second)
        note = int(event["note"])
        velocity = min(127, max(1, int(event["velocity"])))
        confidence = float(event.get("confidence", 0.0))
        cc_value = min(127, max(0, int(event.get("cc_127_value", confidence * 127))))

        timeline.append((onset_tick, 0, index, Message(
            "control_change", control=127, value=cc_value, time=0
        )))
        timeline.append((onset_tick, 1, index, Message(
            "note_on", note=note, velocity=velocity, time=0
        )))
        timeline.append((onset_tick + default_duration_ticks, -1, index, Message(
            "note_off", note=note, velocity=0, time=0
        )))

    previous_tick = 0
    for absolute_tick, _order, _index, message in sorted(timeline):
        message.time = absolute_tick - previous_tick
        track.append(message)
        previous_tick = absolute_tick

    track.append(MetaMessage("end_of_track"))

    buffer = io.BytesIO()
    mid.save(file=buffer)
    return buffer.getvalue()


@app.post("/process")
async def process_audio(
    request: Request,
    file: UploadFile = File(...),
    _: None = Depends(require_api_key_if_configured),
):
    """
    Process an audio file to extract stems and generate MIDI.
    
    Returns a ZIP file containing:
    - guitar_stem.wav
    - bass_stem.wav
    - guitar.mid
    - bass.mid
    - processing_report.json
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate + stream under try/finally so UploadFile is always closed
    # (Content-Length reject, bad extension, or size cap).
    try:
        # Early reject clearly oversized requests. Content-Length is the whole
        # multipart body; allow a small overhead so near-limit files are not 413'd
        # before the streamed file-byte cap below.
        _multipart_overhead = 1024 * 1024  # 1 MiB for multipart boundaries/headers
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > MAX_UPLOAD_BYTES + _multipart_overhead:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds maximum size of {MAX_UPLOAD_BYTES} bytes",
                    )
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid Content-Length header")

        file_ext = Path(file.filename or "").suffix.lower()
        if file_ext not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format {file_ext}. Supported: {SUPPORTED_FORMATS}"
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
        content = b"".join(chunks)
    finally:
        await file.close()

    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Validate audio properties
        _ = validate_audio_file(tmp_file_path)
        
        # Process the audio file
        logger.info(f"Processing {file.filename}...")
        outputs = model.process_audio_file(tmp_file_path)
        
        # Create ZIP response
        zip_buffer = create_response_zip(outputs)
        
        # Prepare filename for download
        base_filename = Path(file.filename or "audio").stem
        download_filename = f"{base_filename}_stem-midi-package.zip"
        
        # Return ZIP file
        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={download_filename}"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_file_path)
        except Exception:
            pass


@app.get("/templates")
async def get_templates(_: None = Depends(require_api_key_if_configured)):
    """List available user-facing content templates."""
    return {"templates": list_templates()}

@app.post("/render-template")
async def render_template_endpoint(
    template_name: str,
    variables: Optional[Dict[str, str]] = None,
    _: None = Depends(require_api_key_if_configured),
):
    """
    Render a user-facing content template with the given variables.

    Args:
        template_name: Name of the template file (e.g., 'completion_delivery.md')
        variables: Dict of variable values to fill in {{ placeholders }}
    """
    try:
        rendered = render_template(template_name, variables or {})
        return {"template": template_name, "rendered": rendered}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found.")

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
        "training_config": model_config.get("training", {}),
        "quality_gates": model_config.get("quality_gates", {})
    }


# For running directly with uvicorn (for development)
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
