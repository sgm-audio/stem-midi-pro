"""
FastAPI API for Stem+MIDI Pro service
Provides endpoints for audio processing, health checks, and file downloads.
"""

import os
import io
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any
import uuid

import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import our model and utilities
from main import StemMidiModel
from utils.quality_gates import ProcessingReport, QualityTier

# Initialize FastAPI app
app = FastAPI(
    title="Stem+MIDI Pro API",
    description="Professional audio AI service for stem separation and MIDI transcription",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model: Optional[StemMidiModel] = None
model_config: Optional[Dict] = None

# Supported audio formats and constraints
SUPPORTED_FORMATS = {".wav", ".flac", ".mp3"}
MAX_DURATION_SECONDS = 600  # 10 minutes
SUPPORTED_SAMPLE_RATES = {44100, 48000}
SUPPORTED_BIT_DEPTHS = {16, 24}  # Note: soundfile doesn't directly give bit depth, we'll infer


def load_model() -> StemMidiModel:
    """Load the StemMidiModel from configuration and checkpoint."""
    global model_config
    
    # Load configuration
    config_path = os.getenv("MODEL_CONFIG_PATH", "configs/model_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found at {config_path}")
    
    import yaml
    with open(config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Initialize model
    model = StemMidiModel(model_config)
    
    # Load checkpoint if provided
    checkpoint_path = os.getenv("MODEL_CHECKPOINT_PATH")
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model = model.load_from_checkpoint(checkpoint_path)
    else:
        print("Warning: No checkpoint provided, using randomly initialized model")
    
    model.eval()
    return model


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model
    print("Loading Stem+MIDI Pro model...")
    model = load_model()
    print("Model loaded successfully!")


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
            print(f"Warning: Bit depth {bit_depth} not in preferred list {SUPPORTED_BIT_DEPTHS}")
        
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
        
        # Add MIDI files (we need to convert metadata to actual MIDI)
        # For now, we'll create placeholder MIDI files
        # In a full implementation, we would use midi_utils to convert to MIDI
        midi_data = outputs["midi"]
        
        # Guitar MIDI
        guitar_midi = create_placeholder_midi(midi_data, "guitar")
        zip_file.writestr("guitar.mid", guitar_midi)
        
        # Bass MIDI
        bass_midi = create_placeholder_midi(midi_data, "bass")
        zip_file.writestr("bass.mid", bass_midi)
        
        # Add processing report
        import json
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
    """
    Create a placeholder MIDI file from metadata.
    In a full implementation, this would use a proper MIDI library.
    For now, we return a minimal valid MIDI file.
    """
    # This is a minimal MIDI file (header only, no tracks)
    # A real implementation would use pretty_midi or mido
    # For demo purposes, we return a fixed small MIDI file
    # Header: MThd + 6 bytes (format=0, nTracks=1, division=480)
    # Track: MTrk + ... + MetaEvent(end of track)
    # We'll create a simple one with just the header and end of track
    
    # Format 0, 1 track, 480 ticks per quarter note
    header = b'MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00\xE0'
    # Track header
    track_header = b'MTrk\x00\x00\x00\x04'
    # End of track meta event: FF 2F 00
    end_of_track = b'\xFF\x2F\x00'
    track = track_header + end_of_track
    
    return header + track


@app.post("/process")
async def process_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
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
    
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format {file_ext}. Supported: {SUPPORTED_FORMATS}"
        )
    
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        try:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        finally:
            await file.close()
    
    try:
        # Validate audio properties
        validation_info = validate_audio_file(tmp_file_path)
        
        # Process the audio file
        print(f"Processing {file.filename}...")
        outputs = model.process_audio_file(tmp_file_path)
        
        # Create ZIP response
        zip_buffer = create_response_zip(outputs)
        
        # Prepare filename for download
        base_filename = Path(file.filename).stem
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
        except:
            pass


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