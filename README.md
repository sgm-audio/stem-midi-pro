# Stem+MIDI Pro

Audio AI prototype for guitar/bass stem separation and MIDI transcription using Mamba-SSM.

## Overview

Stem+MIDI Pro is an audio AI prototype that provides:
- High-fidelity guitar/bass stem separation using Mamba-SSM architecture
- Polyphonic audio-to-MIDI transcription with expression detection
- Confidence-aware routing and quality gating
- DAW-ready output with tempo sync, tuning detection, and metadata tagging
- Canadian artist dataset alignment for training data

## Features

### Audio Processing
- **Demucs separator** (Phase 2 Path A): High-fidelity guitar/bass stem separation using pretrained Demucs, fine-tunable on MUSDB18HQ with guitar/bass emphasis
- **Mamba-SSM separator** (Phase 2 Path B): Optional Mamba-SSM architecture for research/advanced use
- **Transcriber**: Basic Pitch (production) or Mamba-SSM (research) for polyphonic audio-to-MIDI transcription
- **Expression detection** (bends, vibrato, slides) via dedicated neural heads
- **Cross-modal alignment** ensuring MIDI notes match stem energy peaks
- **Confidence-aware** routing and quality gating

### Quality & Trust
- **Confidence scoring** per note with CC#127 MIDI tagging for DAW visualization
- **Quality-based routing** (studio/draft/complex tiers) with clear user communication
- **Fallback options** including human-reviewed refinement for challenging material
- **Transparent reporting** with SI-SDR, phase coherence, and artifact detection

### Technical Excellence
- **Linear-time complexity** Mamba-SSM architecture (8x less VRAM than Transformers)
- **FP8 precision** support with dynamic scaling
- **Streaming inference** with state caching and CUDA graph capture
- **Sub-5ms hop latency** targeting on NVIDIA H100/A100 GPUs

### Mission Alignment
- **Canadian artist focus** with specialized dataset loaders
- **Indie pricing tiers** and local success story highlights
- **Training data prioritization** from Canadian artists

## Project Structure

```
stem_midi_pro/
├── main.py                 # NeMo ModelPT wrapper (entry point)
├── demo.py                 # Usage demonstration
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container build file
├── verify_structure.py     # File validation script
├── README.md               # This file
│
├── configs/
│   └── model_config.yaml   # Model architecture and training config
│
├── data/
│   ├── datasets.py         # Slakh2100 / MUSDB18-HQ loaders
│   ├── example_data_config.yaml # Example data configuration
│   └── train.py            # Training script
│
├── models/
│   ├── mamba_separator.py  # Mamba-SSM source separation block
│   ├── mamba_transcriber.py # Polyphonic audio-to-MIDI head
│   ├── confidence_injector.py # Metadata tagging pipeline
│   └── losses.py           # Perceptual loss functions (MR-STFT, etc.)
│
├── utils/
│   └── quality_gates.py    # Confidence-based routing logic
│
└── user_content/           # All user-facing prompt templates
    ├── upload_confirmation.md
    ├── progress_updates.md
    ├── completion_delivery.md
    ├── rights_usage_prompt.md
    ├── feedback_refinement.md
    ├── implicit_feedback.md
    └── landing_page.md
```

## Installation

```bash
# Clone repository
git clone <repository-url>
# <repository-root> is the project root — no subdirectory to cd into

# Install dependencies
pip install -r requirements.txt

# For GPU support with TensorRT-LLM (optional)
# Install NVIDIA drivers, CUDA Toolkit, and TensorRT separately
```

## Usage

### Quick Demo (with synthetic data)
```bash
python demo.py
```

### Training with Canadian Artist Data
```bash
# Prepare data configuration (see example_data_config.yaml)
python train.py \
  --config configs/model_config.yaml \
  --data-config example_data_config.yaml \
  --output-dir ./outputs \
  --max-epochs 50 \
  --gpus 1
```

### Docker Deployment
```bash
# Build Docker container
docker build -t stem-midi-pro .

# Run service (example)
docker run -p 8000:8000 stem-midi-pro
```

## Technical Specifications

### Separator Options
- **Demucs** (Phase 2 Path A, production default): Pretrained separator, fine-tunable on MUSDB18HQ; CPU-compatible; no Mamba-SSM required
- **Mamba-SSM** (Phase 2 Path B, research): 12-layer Mamba-SSM (d_model=768) with selective scan; optional dependency

### Transcriber Options
- **Basic Pitch** (production): ONNX-based, no TensorFlow pins on Py3.13
- **Mamba-SSM** (research): 8-layer Mamba-SSM (d_model=512) with multi-task heads

### Precision
- **CPU-only**: FP32 (training), FP32 (inference)
- **GPU**: FP16/FP8 with TensorRT-LLM (aspirational, not required)

### Loss Functions
- Multi-Resolution STFT (3 resolutions for Path A)
- Spectral Flatness preservation
- Crest Factor preservation (transient detail)
- Onset F1 + Pitch Cross-Entropy + Velocity MAE (Phase 2 foundation)
- Cross-modal alignment (MIDI ↔ stem energy)

### Quality Gates
- **Studio Tier**: Confidence ≥0.85, SI-SDR ≥20dB → Direct download
- **Draft Tier**: 0.70 ≤ confidence < 0.85 → Editor launch prompt
- **Complex Tier**: <0.70 confidence → Human review/refund options

## Canadian Artist Alignment

As specified in the mission, this implementation prioritizes:
1. **Training data** from Canadian artists (Slakh2100-CA, MUSDB-Indie)
2. **Local success stories** in marketing and case studies
3. **Indie pricing tiers** for Canadian creators
4. **Dataset loader utilities** in `data/datasets.py`

## Next Steps

1. **Prepare real datasets**: Place Canadian artist audio data in the specified directory structure (or use `scripts/prepare_musdb.py` to download MUSDB18HQ)
2. **Phase 2 Path A — Fine-tune Demucs separator**:
   - Run: `python scripts/fine_tune_demucs.py --musdb-path ./data/musdb18hq --epochs 50`
   - Export: `python scripts/export_fine_tuned.py --checkpoint ./outputs/checkpoints/best.pt`
   - Inference: Use `model_config.cpu.yaml` with `restore_from_path` pointing to exported checkpoint
3. **Modify data configuration**: Update `example_data_config.yaml` with actual paths
4. **Start training**: Run the training script with appropriate hardware
5. **Export for deployment**: Use exported checkpoint with `model_config.cpu.yaml`
6. **Deploy**: Containerize with `Dockerfile.cpu` for self-host or `Dockerfile.gpu` for cloud GPU

## Acknowledgements

Built with:
- Demucs stem separation (Phase 2 Path A production default)
- Basic Pitch audio-to-MIDI transcription (ONNX, Py3.13-compatible)
- Mamba-SSM optional for research backends (Phase 2 Path B)
- PyTorch (CPU-only target)
- Librosa/Ruby for audio processing

---
*Stem+MIDI Pro: Prototype stem separation + editable MIDI drafts. Not magic—just math that respects your craft.*