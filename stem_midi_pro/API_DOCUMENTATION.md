# Stem+MIDI Pro API Documentation

## Overview

The Stem+MIDI Pro API provides a RESTful interface for accessing the audio processing capabilities of the Stem+MIDI Pro service. Built with FastAPI, it offers automatic API documentation, high performance, and easy integration.

## Base URL

```
/api/v1
```

All endpoints are prefixed with `/api/v1` in a production deployment. For simplicity in this documentation, we'll show the base paths.

## Authentication

Currently, the API does not implement authentication. In a production environment, you should add appropriate authentication middleware (e.g., API keys, JWT tokens) based on your security requirements.

## Rate Limiting

Rate limiting is not implemented in the base API. Consider adding it via middleware or using a reverse proxy (NGINX, Traefik) in production.

## Endpoints

### Health Check

```
GET /health
```

Check if the service is healthy and the model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**Responses:**
- `200 OK`: Service is healthy
- `503 Service Unavailable`: Model not loaded

### Process Audio

```
POST /process
```

Upload an audio file to extract stems and generate MIDI transcription.

**Request:**
- `file`: Audio file (multipart/form-data)
  - Supported formats: `.wav`, `.flac`, `.mp3`
  - Maximum duration: 10 minutes
  - Supported sample rates: 44.1 kHz, 48 kHz
  - Supported bit depths: 16-bit, 24-bit (flexible on input)

**Response:**
- `200 OK`: Returns a ZIP file containing:
  - `guitar_stem.wav` - Separated guitar stem
  - `bass_stem.wav` - Separated bass stem
  - `guitar.mid` - Guitar MIDI transcription
  - `bass.mid` - Bass MIDI transcription
  - `processing_report.json` - Quality metrics and metadata
- `400 Bad Request`: Invalid file format, duration, or sample rate
- `503 Service Unavailable`: Model not loaded
- `500 Internal Server Error`: Processing failed

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/process" \
     -F "file=@/path/to/audio.wav" \
     -o output.zip
```

**Example using Python requests:**
```python
import requests

response = requests.post(
    "http://localhost:8000/process",
    files={"file": open("audio.wav", "rb")}
)

if response.status_code == 200:
    with open("output.zip", "wb") as f:
        f.write(response.content)
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

### Get Model Information

```
GET /model-info
```

Retrieve information about the currently loaded model.

**Response:**
```json
{
  "model_name": "stem_midi_mamba_v1",
  "audio_config": {
    "sample_rate": 44100,
    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 80,
    "chunk_duration_sec": 2.0,
    "overlap_ratio": 0.5
  },
  "separator_config": {
    "d_model": 768,
    "n_layer": 12,
    "d_state": 16,
    "d_conv": 4,
    "expand": 2
  },
  "transcriber_config": {
    "d_model": 512,
    "n_layer": 8,
    "d_state": 12,
    "onset_head_dim": 64,
    "pitch_vocab_size": 128,
    "velocity_bins": 128,
    "expression_heads": ["bend", "vibrato", "slide"],
    "onset_threshold": 0.5
  },
  "training_config": {
    "precision": "fp8",
    "gradient_checkpointing": true
  },
  "quality_gates": {
    "studio_confidence_threshold": 0.85,
    "draft_confidence_threshold": 0.70,
    "min_confidence": 0.6,
    "min_si_sdr": 20.0
  }
}
```

**Responses:**
- `200 OK`: Model information returned
- `503 Service Unavailable`: Model not loaded

## Data Models

### Processing Report

The `processing_report.json` file in the response ZIP contains:

| Field | Type | Description |
|-------|------|-------------|
| `si_sdr` | float | Signal-to-Distortion Ratio estimate (dB) |
| `phase_coherence` | float | Phase coherence score between stems (0-1) |
| `avg_confidence` | float | Average confidence of MIDI transcription (0-1) |
| `artifact_flags` | list[str] | Detected artifacts (e.g., ["clipping", "noise_floor"]) |
| `low_confidence_notes` | int | Number of MIDI notes below confidence threshold |
| `quality_tier` | string | Overall quality assessment ("studio", "draft", or "complex") |

### Quality Tiers

The service uses a three-tier quality system:

1. **Studio Quality** (`confidence ≥ 0.85` AND `SI-SDR ≥ 20dB`)
   - Ready for professional use
   - Direct download without editing

2. **Draft Quality** (`0.70 ≤ confidence < 0.85`)
   - Good starting point requiring minor edits
   - Prompts user to use the MIDI editor

3. **Complex Material** (`confidence < 0.70` OR artifact flags present)
   - Challenging material requiring special handling
   - Offers options: raw download, human review, or refund

## Error Responses

All error responses follow this format:
```json
{
  "detail": "Error message describing the issue"
}
```

Common HTTP status codes:
- `400 Bad Request`: Invalid input (file format, duration, etc.)
- `404 Not Found`: Endpoint does not exist
- `405 Method Not Allowed`: Wrong HTTP method
- `413 Payload Too Large`: File exceeds size limits
- `500 Internal Server Error`: Unexpected server error
- `503 Service Unavailable`: Service temporarily unavailable (e.g., model loading)

## Performance Characteristics

### Latency Targets
- **Hop latency**: <5ms (for streaming applications)
- **Total separation**: <2 seconds (for 3-minute track)
- **MIDI transcription**: <4 seconds (for 3-minute track)
- **Total processing**: <6 seconds (end-to-end for 3-minute track)

### Throughput
- **Batch size**: 1 (optimized for low-latency streaming)
- **Concurrent requests**: Depends on GPU memory and instance type
- **Recommended instance**: NVIDIA A100 or H100 for production

## Security Considerations

### File Handling
- All files are processed in-memory only
- Temporary files are securely deleted after processing
- Original files are not stored permanently
- Output files are available only for the duration of the request

### Data Privacy
- Audio files are not persisted beyond processing
- No personal data is collected or stored
- Processing occurs in ephemeral containers
- GDPR-compliant by design (data minimization)

## Deployment Notes

### Environment Variables
The API can be configured using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_CONFIG_PATH` | Path to model configuration YAML | `configs/model_config.yaml` |
| `MODEL_CHECKPOINT_PATH` | Path to model checkpoint (optional) | None |
| `PORT` | Port to bind the API server | `8000` |

### Docker Deployment
The API is designed to run in the provided Docker container:

```bash
docker build -t stem-midi-pro .
docker run -p 8000:8000 stem-midi-pro
```

### Kubernetes Deployment
For production Kubernetes deployments, consider:
- Resource requests/limits based on GPU memory
- Liveness and readiness probes using the `/health` endpoint
- Horizontal pod autoscaling based on CPU/GPU utilization
- Persistent volumes for model checkpoints (if needed)

## Client Libraries

While the API can be called directly with any HTTP client, here are examples for common languages:

### JavaScript (Fetch API)
```javascript
async function processAudio(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('/process', {
    method: 'POST',
    body: formData
  });
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  const blob = await response.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'output.zip';
  document.body.appendChild(a);
  a.click();
  a.remove();
}
```

### Python (Requests)
See the example in the POST /process section above.

### cURL
See the example in the POST /process section above.

## Versioning

The API follows semantic versioning. Backward-incompatible changes will increment the major version number.

Current version: `1.0.0`

## Changelog

### v1.0.0
- Initial release
- Stem separation using Mamba-SSM
- MIDI transcription with expression detection
- Confidence-based quality routing
- RESTful API with FastAPI
- Docker containerization
- Canadian artist dataset support

## Support

For issues, questions, or feature requests, please refer to the project documentation or contact the maintainers.

---
*API Documentation Generated: $(date)*
*Stem+MIDI Pro: Studio-grade stem separation + editable MIDI drafts. Not magic—just math that respects your craft.*