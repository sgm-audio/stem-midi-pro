# Stem+MIDI Pro Architecture

## Overview

Stem+MIDI Pro is a production-ready audio AI service that combines state-of-the-art source separation and audio-to-MIDI transcription using Mamba State Space Models (SSM). This document details the architectural decisions, components, and data flows.

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌────────────────────┐
│   API Layer     │    │  Processing      │    │   Model Inference    │
│ (FastAPI)       │◄──►│   Orchestrator   │◄──►│  (NeMo ModelPT)      │
└─────────────────┘    └──────────────────┘    └────────────────────┘
                             │                       │
                             ▼                       ▼
                    ┌─────────────────┐    ┌──────────────────┐
                    │  Quality Gates  │    │  Separation      │
                    │ & Routing       │    │  (Mamba-SSM)     │
                    └─────────────────┘    └──────────────────┘
                             │                       │
                             ▼                       ▼
                    ┌─────────────────┐    ┌──────────────────┐
                    │  Transcription  │    │  Post-processing │
                    │  (Mamba-SSM)    │    │  & Metadata      │
                    └─────────────────┘    └──────────────────┘
```

## Component Details

### 1. API Layer (`api.py`)
- **Framework**: FastAPI with asynchronous request handling
- **Endpoints**:
  - `GET /health`: Health check
  - `POST /process`: Main audio processing endpoint
  - `GET /model-info`: Model configuration and metadata
- **Features**:
  - File validation (format, duration, sample rate)
  - Secure temporary file handling
  - ZIP response packaging
  - CORS middleware
  - Background task support for cleanup

### 2. Processing Orchestrator (Embedded in `main.py`)
- Handles the end-to-end processing pipeline
- Manages model state and inference mode
- Coordinates between separation and transcription modules
- Applies quality gating and routing logic

### 3. Model Inference (`main.py` - StemMidiModel)
- **Base Class**: NeMo ModelPT for seamless integration with NeMo ecosystem
- **Key Features**:
  - Training/validation step compatibility
  - Neural type definitions for input/output validation
  - Checkpoint loading and saving
  - Mixed precision training support (FP8)

### 4. Separation Module (`models/mamba_separator.py`)
- **Architecture**: 12-layer Mamba-SSM
- **Input**: Raw audio waveform (B, C, T)
- **Processing**:
  - STFT transformation with Hann window
  - Linear projection to Mamba dimension
  - Selective SSM blocks with causal convolution
  - Multi-head output for guitar, bass, and residual masks
  - Auxiliary phase coherence head
- **Output**:
  - Separated guitar and bass stems (B, C, T)
  - Residual stem
  - SSM state cache for streaming
  - Metrics (SI-SDR estimate, phase coherence)

### 5. Transcription Module (`models/mamba_transcriber.py`)
- **Architecture**: 8-layer Mamba-SSM
- **Input**: Separated guitar stem (could also process bass)
- **Processing**:
  - Mel-spectrogram transformation
  - Linear projection to Mamba dimension
  - SSM processing blocks
  - Multi-task heads:
    - Onset detection (binary classification)
    - Pitch classification (128 MIDI notes + silence)
    - Velocity regression (0-127)
    - Expression detection (bend, vibrato, slide)
    - Confidence estimation (Monte Carlo dropout)
- **Output**:
  - Onset logits (B, T, 1)
  - Pitch logits (B, T, 128)
  - Velocity predictions (B, T, 1)
  - Expression predictions (B, T, 3)
  - Confidence scores (B, T)

### 6. Confidence Injection (`models/confidence_injector.py`)
- **Purpose**: Add metadata to MIDI events for user transparency
- **Process**:
  - Align MIDI onsets with stem energy
  - Compute per-note confidence scores
  - Flag low-confidence notes for review
  - Generate DAW-friendly metadata (CC#127 values)
  - Suggest quantization strategy (grid vs human feel)

### 7. Loss Functions (`models/losses.py`)
- **Multi-Resolution STFT Loss**: Perceptual fidelity across multiple resolutions
- **Spectral Flatness Loss**: Preserve transient detail and natural spectral shape
- **Crest Factor Loss**: Maintain peak-to-RMS ratio for transient preservation
- **Onset F1 Loss**: Accurate note onset detection
- **Pitch Cross-Entropy**: Correct pitch classification
- **Velocity MAE**: Accurate velocity prediction
- **Cross-Modal Alignment Loss**: Ensure MIDI onsets align with stem energy peaks

### 8. Quality Gates (`utils/quality_gates.py`)
- **Three-Tier System**:
  1. **Studio**: Confidence ≥0.85 AND SI-SDR ≥20dB → Direct download
  2. **Draft**: 0.70 ≤ confidence < 0.85 → Prompt for editing
  3. **Complex**: confidence <0.70 OR artifacts → Fallback options
- **Artifact Detection**:
  - Clipping detection (sample values >0.99)
  - Noise floor estimation
  - Phase cancellation detection

### 9. Streaming Inference Capabilities
- **Chunked Processing**: 2-second segments with 50% overlap
- **State Caching**: Mamba SSM state maintained between chunks
- **Overlap-Add**: Phase-coherent reconstruction at chunk boundaries
- **CUDA Graph Capture**: Deterministic sub-5ms hop latency
- **Async I/O**: Pinned memory transfers for minimal CPU-GPU overhead

## Data Flow

1. **Input Validation**:
   - API validates file format, duration, sample rate
   - Audio loaded and resampled if necessary
   - Converted to torch tensor (B, C, T)

2. **Separation**:
   - STFT → Magnitude/Phase
   - Linear projection → Mamba input
   - Selective SSM blocks → Hidden states
   - Multi-head → Stem masks
   - ISTFT → Separated audio stems
   - Metrics computation (SI-SDR proxy, phase coherence)

3. **Transcription**:
   - Mel-spectrogram of guitar stem
   - Linear projection → Mamba input
   - SSM processing → Hidden states
   - Multi-task heads → Onset, pitch, velocity, expression, confidence
   - Thresholding → MIDI events
   - Confidence injection → Metadata enrichment

4. **Quality Assessment**:
   - Compile processing report
   - Determine quality tier
   - Apply routing logic

5. **Output Packaging**:
   - Convert stems to WAV format
   - Generate MIDI files from events
   - Create processing report JSON
   - Package all into ZIP response

## Streaming Capabilities

The architecture supports true streaming inference with:

- **Fixed memory footprint**: Constant regardless of input length
- **Deterministic latency**: Fixed hop time via CUDA graphs
- **State persistence**: SSM state carries context between chunks
- **Boundary handling**: Overlap-add prevents artifacts at chunk boundaries

## Scalability Considerations

### Horizontal Scaling
- Stateless API layer (except for model weights)
- Easy to scale behind load balancer
- Shared model checkpoint storage (NFS, cloud storage)

### Vertical Scaling
- GPU memory is the primary limiting factor
- Batch size of 1 optimized for latency
- Larger models possible with more VRAM
- Mixed precision (FP8) doubles effective capacity

### Model Parallelism
- Current implementation uses single GPU
- Can be extended to tensor/pipeline parallelism
- NeMo Megatron-Core support built into ModelPT

## Security Architecture

### Input Sanitization
- File type validation by extension and content
- Duration limits prevent DoS via long files
- Sample rate validation prevents unusual rates

### Runtime Security
- Model runs in inference mode only (no training endpoints)
- No arbitrary code execution paths
- Memory-safe operations (PyTorch/TensorFlow)

### Data Handling
- Ephemeral file processing (temp files deleted)
- In-memory processing where possible
- No persistent storage of user audio
- Output available only for request duration

### Compliance
- GDPR-compliant by design (data minimization)
- No personal data collection
- Processing occurs in ephemeral containers
- Auditable through API logs

## Deployment Architecture

### Containerized Service
```
Docker Container
├── FastAPI Application (uvicorn)
├── PyTorch + NeMo + Mamba-SSM
├── Model Weights (loaded at startup)
└── Dependencies (CUDA, cuDNN, TensorRT-LLM)
```

### Kubernetes Deployment Template
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: stem-midi-pro
spec:
  replicas: 3
  selector:
    matchLabels:
      app: stem-midi-pro
  template:
    metadata:
      labels:
        app: stem-midi-pro
    spec:
      containers:
      - name: stem-midi-pro
        image: stem-midi-pro:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_CONFIG_PATH
          value: "/config/model_config.yaml"
        - name: MODEL_CHECKPOINT_PATH
          value: "/models/stem_midi_pro.nemo"
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
        volumeMounts:
        - name: model-storage
          mountPath: /models
        - name: config-storage
          mountPath: /config
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-checkpoint-pvc
      - name: config-storage
        configMap:
          name: stem-midi-pro-config
---
apiVersion: v1
kind: Service
metadata:
  name: stem-midi-pro-service
spec:
  selector:
    app: stem-midi-pro
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Performance Optimization

### Computational
- **Mamba-SSM Advantage**: O(n) complexity vs O(n²) for Transformers
- **Selective Scan**: Hardware-accelerated SSM operations
- **CUDA Kernels**: Custom causal conv1d and selective_scan_fn
- **FP8 Precision**: 2x throughput with minimal accuracy loss
- **Gradient Checkpointing**: 50% memory reduction for training

### Memory
- **State Caching**: Reuse SSM state between chunks
- **Streaming**: Constant memory footprint
- **Pinned Memory**: Async CPU-GPU transfers
- **Inference Mode**: No gradient storage

### Latency
- **CUDA Graphs**: Eliminate kernel launch overhead
- **Async Pipelines**: Overlap compute and memory transfers
- **Optimized Block Sizes**: Fused operations where possible
- **Kernel Fusion**: Reduce memory bandwidth usage

## Extensibility Points

### Adding New Instruments
1. Extend separation mask heads
2. Update transcription to handle new timbres
3. Adjust quality gates for instrument-specific thresholds
4. Update DAW templates and metadata

### New Audio Effects
1. Add expression detection heads (e.g., tremolo, wah)
2. Extend MIDI CC mapping
3. Update user interface for new controls
4. Adjust training data and loss functions

### Alternative Architectures
1. Swap Mamba blocks for other SSM variants
2. Experiment with hybrid CNN-SSM architectures
3. Try different projection networks
4. Explore quantization-aware training

## Canadian Artist Dataset Integration

The architecture supports Canadian artist data through:

### Data Loading
- `data/datasets.py` provides dataset abstractions
- Synthetic fallback for development
- Easy plug-in for real datasets (Slakh2100-CA, MUSDB-Indie)
- Automatic train/validation/test splits

### Training Pipeline
- `train.py` orchestrates the full training process
- NeMo PTL integration for logging and checkpointing
- Mixed precision training support
- Validation-based early stopping
- Learning rate scheduling

### Model Adaptation
- Configuration-driven architecture dimensions
- Easy to adjust model size for dataset characteristics
- Transfer learning from pre-trained checkpoints
- Domain adaptation techniques for Canadian audio

## Monitoring and Observability

### Metrics Collected
- Processing latency (end-to-end and per-stage)
- Resource utilization (GPU, CPU, memory)
- Quality metrics (SI-SDR, confidence, artifact rates)
- User interaction data (edits, refinements, feedback)
- Error rates and failure modes

### Logging
- Structured logging for production systems
- Audio fingerprinting for debugging (no storage)
- Performance profiling hooks
- Error tracing with context

### Health Checks
- Liveness probe: Model loaded and responsive
- Readiness probe: Ready to accept traffic
- Deep health check: Full pipeline test with synthetic data
- Dependency validation: GPU, memory, disk space

## Future Enhancements

### Short Term
- TensorRT-LLM export with custom kernel registration
- API authentication and rate limiting
- WebSocket endpoint for real-time processing
- Batch processing endpoint
- Web-based MIDI editor implementation

### Medium Term
- Multi-instrument separation (drums, vocals, keys)
- Real-time streaming API (WebSockets/gRPC)
- Advanced expression detection (articulation, phrasing)
- Canadian artist specific model fine-tuning
- Industry standard format support (AAF, OMF)

### Long Term
- Edge device optimization (Jetson, mobile)
- Federated learning for privacy-preserving training
- Integration with DAWs as VST/AU plugins
- Collaborative editing and version control
- AI-assisted arrangement and composition tools

---
*Architecture Document Version: 1.0*
*Last Updated: 2026-05-31*