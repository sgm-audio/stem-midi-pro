# Stem+MIDI Pro — Product Requirements & Architecture Requirements Document

> **Current-status note (2026-11):** This is a draft requirements/history document, not evidence that the current repository ships trained quality. See [VERIFICATION_MATRIX.md](VERIFICATION_MATRIX.md) for the current contract: native Mamba file inference requires a validated project checkpoint, while the explicit pretrained-backend profile supplies its own dependency weights; `separation_proxy` is not SI-SDR, confidence is JSON/Polyphonic Key Pressure rather than CC#127, and no reference-hardware or held-out quality claim is certified.
>
> **Reading the requirements below:** Part I and the original ARD preserve aspirational targets and historical design context for traceability. They are not current service guarantees. The current CPU/checkpoint/backend contract is documented in `README.md`, `SETUP.md`, `API_DOCUMENTATION.md`, `ARCHITECTURE.md`, and ADR 0007.

> **Version:** 1.0-draft  
> **Status:** Internal — for review and revision  
> **Generated from:** codebase audit of the Stem+MIDI Pro prototype repository

---

## Table of Contents

1. [Product Requirements Document](#part-i-product-requirements-document-prd)
   - 1.1 Vision & Mission
   - 1.2 Target Users & Personas
   - 1.3 Core Features
   - 1.4 Quality Tier System
   - 1.5 User Experience Requirements
   - 1.6 Performance Requirements
   - 1.7 Success Metrics / KPIs
   - 1.8 Competitive Landscape
   - 1.9 Constraints & Assumptions
2. [Architecture Requirements Document](#part-ii-architecture-requirements-document-ard)
   - 2.1 System Architecture Overview
   - 2.2 Component Specifications
   - 2.3 Data Flow
   - 2.4 API Design
   - 2.5 Infrastructure Requirements
   - 2.6 Security Requirements
   - 2.7 Monitoring & Observability
   - 2.8 Development Workflow
   - 2.9 Testing Strategy
   - 2.10 Build, CI/CD & Release
3. [Audit Findings: Stubs, Bugs, Gaps & Contradictions](#part-iii-audit-findings)
   - 3.1 Runtime-Crashing Bugs
   - 3.2 Functional Gaps (Stubs & Placeholders)
   - 3.3 Architecture & Data-Flow Contradictions
   - 3.4 Doc–Code Contradictions
   - 3.5 Developer Pain Points

---

# Part I: Product Requirements Document (PRD)

## 1.1 Vision & Mission

**Vision**  
Make studio-quality stem separation and editable MIDI transcription accessible to every musician, producer, and educator — with a focus on Canadian artists and indie creators.

**Mission**  
Deliver an AI service that separates mixed audio into guitar/bass stems and transcribes them into expressive, editable MIDI — with transparent quality reporting, confidence-aware routing, and DAW-ready output.

**Canadian Artist Alignment**  
- Prioritise training data from Canadian artists (Slakh2100-CA, MUSDB-Indie).
- Showcase Canadian success stories in marketing.
- Offer indie-friendly pricing tiers.

## 1.2 Target Users & Personas

| Persona | Needs | Pain Points |
|---------|-------|-------------|
| **Guitarist / Bassist** | Learn songs by ear; get notation/ tab from recordings | Manual transcription is slow; existing tools miss bends/vibrato |
| **Producer / Mix Engineer** | Isolate guitar/bass for remixing or sample extraction | Stem separation quality is inconsistent; artifacts ruin takes |
| **Music Educator** | Create teaching materials from existing songs | Needs clean stems + accurate MIDI for student arrangements |
| **Content Creator** | Extract backing parts for covers or reaction videos | Wants fast, one-click processing with DAW-ready output |
| **Indie Canadian Artist** | Repurpose old recordings without re-recording | Budget-conscious; needs transparent quality feedback |

## 1.3 Core Features

### F1: Stem Separation
- Separate mixed audio into **guitar**, **bass**, and **residual** stems.
- Phase-coherent processing — stems sum to original when recombined.
- Support mono and stereo input (downmix to mono for processing).

### F2: Audio-to-MIDI Transcription
- Transcribe guitar stem into MIDI notes with pitch, velocity, and timing.
- Detect musical expressions: **bend**, **vibrato**, **slide**.
- Output standard MIDI files compatible with all major DAWs.

### F3: Confidence-Aware Quality Routing
- Per-note confidence scores in canonical JSON, with optional note-specific MIDI Polyphonic Key Pressure (the former CC#127 requirement is superseded by ADR 0007).
- Three-tier routing: Studio (direct download), Draft (editor prompt), Complex (human review).
- Per-note "needs review" flagging for low-confidence events.

### F4: DAW-Ready Output Package
- WAV stems (guitar, bass).
- MIDI files with expression CC data + confidence metadata.
- JSON processing report (current runtime: `separation_proxy`, phase coherence, artifact flags, model status, and quality tier; reference-based SI-SDR remains an evaluation requirement).
- Tempo and tuning metadata for DAW alignment.

### F5: API & Integration
- REST API for programmatic access (upload → process → download ZIP).
- Real-time health checks and model metadata endpoint.
- CORS support for web frontend integration.

### F6: Training with Canadian Artist Data
- Dataset loaders for Slakh2100-CA and MUSDB-Indie.
- Synthetic data fallback for development.
- Data augmentation (time stretch, pitch shift, noise, gain).

## 1.4 Quality Tier System (original target, not current certification)

The original product target used a reference-based separation metric. The current
runtime uses a centroid-separation proxy instead, and no tier certifies audio or
transcription quality without trained/evaluated weights.

| Tier | Confidence | Reference metric target | User Action | UI Treatment |
|------|-----------|--------|-------------|--------------|
| **Studio** | ≥85% | ≥20 dB | Direct download | Green badge, no editor prompt |
| **Draft** | 70–85% | any | Editor launch | Yellow badge, highlight low-confidence notes |
| **Complex** | <70% | any, or artifacts | Choose: raw download / human review +$4.99 / refund | Red badge, multi-option dialog |

**Artifact Detection:**
- Clipping (sample value >0.99)
- Noise floor (RMS energy >0.1)
- Phase cancellation (cross-correlation >0.9)

## 1.5 User Experience Requirements

1. **One-click processing**: Upload → wait → download. No parameter tuning required.
2. **Transparent feedback**: Show quality tier, confidence scores, and what each means.
3. **Non-blocking fallbacks**: Complex material never results in a dead-end; always offer an action.
4. **DAW compatibility**: Output files must import without conversion into Ableton Live, Logic Pro, REAPER, Pro Tools, FL Studio.
5. **Mobile-friendly web upload** (future).
6. **Processing estimate before start**: Show expected wait time based on file length.

## 1.6 Performance Requirements (original targets; not benchmark-certified)

No target-hardware benchmark is committed. The historical values below are
requirements to validate or revise, not current performance claims.

| Metric | Target | Measurement |
|--------|--------|-------------|
| End-to-end latency (3-min track) | <6 s | Wall-clock from upload to ZIP |
| Hop latency (streaming) | <5 ms | CUDA kernel execution time |
| SI-SDR (separation quality) | ≥20 dB | Per-file metric |
| Onset F1 (transcription) | ≥0.85 | Against ground-truth MIDI |
| Max file duration | 10 min | Enforced at API layer |
| Concurrent requests | 10+ per GPU | Load test with A100 |

## 1.7 Success Metrics / KPIs

| KPI | Target | How Measured |
|-----|--------|-------------|
| Studio-tier rate | ≥60% of processed files | Per-file quality tier |
| User re-submission rate | <15% | Same-file re-upload within 24 h |
| Human review take rate | <5% (indicates low complex-tier rate) | "human review" option selected |
| API uptime | 99.9% | Health check pings |
| Average processing satisfaction | ≥4.5 / 5 | Post-download survey |

## 1.8 Competitive Landscape

| Competitor | Strengths | Weaknesses vs. Stem+MIDI Pro |
|-----------|-----------|------------------------------|
| **Demucs** (Meta) | Open-source, 4-stem separation, strong benchmarks | No MIDI transcription; no confidence routing; GPU-only |
| **Spleeter** (Deezer) | Fast, lightweight, good vocal separation | Guitar/bass quality is poor; no MIDI output |
| **Moises.ai** | Full-featured product, cloud-based, multi-stem, MIDI | Expensive; closed-source; no Canadian-artist focus |
| **Melodyne** (Celemony) | Best-in-class polyphonic pitch editing | Not stem separation; very expensive; desktop-only |
| **WAV Tool** (Acon Digital) | Affordable, good stem separation | No MIDI; no transparency on quality |

**Differentiators:**
- Confidence-aware routing (no other tool tells you when to trust the output).
- Mamba-SSM linear-time architecture (lower VRAM than Transformer-based competitors).
- Canadian artist dataset alignment.

## 1.9 Constraints & Assumptions (current status)

1. **CPU-only target:** the repository uses a pure-PyTorch Mamba fallback; the optional `mamba_ssm` package may require CUDA tooling to build and is not a hard dependency.
2. **NeMo ModelPT and PyTorch Lightning are removed** from the current path; `StemMidiModel` and `train.py` are plain PyTorch.
3. **No trained checkpoint or held-out quality evidence is shipped.** Dataset loaders and synthetic fallback support development, not a shippable quality claim.
4. **FP8/CUDA graph requirements are removed** from the current CPU path.
5. **Guitar + bass only** — drums, vocals, and keys are out of scope for v1.
6. **Single-process inference** is the current service contract; no target-hardware throughput is certified.

---

# Part II: Architecture Requirements Document (ARD)

> The diagrams and infrastructure table in this draft retain the original
> NeMo/GPU architecture for historical traceability. They are superseded for
> the current path by the plain-PyTorch CPU implementation in `ARCHITECTURE.md`
> and ADRs 0002, 0003, and 0007.

## 2.1 System Architecture Overview

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────────────┐
│   Client    │────▶│  FastAPI     │────▶│  StemMidiModel (NeMo)    │
│  (web/curl) │     │  (api.py)    │     │  ┌───────────────────┐  │
└─────────────┘     └──────────────┘     │  │ MambaSeparator    │  │
                           │             │  │ 12-layer SSM     │  │
                           ▼             │  └────────┬──────────┘  │
                    ┌──────────────┐     │           ▼             │
                    │  Response    │     │  ┌───────────────────┐  │
                    │  ZIP (WAV +  │     │  │ MambaTranscriber │  │
                    │  MIDI + JSON)│     │  │ 8-layer SSM      │  │
                    └──────────────┘     │  └────────┬──────────┘  │
                                          │           ▼             │
                                          │  ┌───────────────────┐  │
                                          │  │ ConfidenceInjector│  │
                                          │  └────────┬──────────┘  │
                                          │           ▼             │
                                          │  ┌───────────────────┐  │
                                          │  │ quality_gates     │  │
                                          │  └───────────────────┘  │
                                          └──────────────────────────┘
```

## 2.2 Component Specifications

### C1: API Layer (`api.py`)
- **Framework**: FastAPI with async request handling.
- **Endpoints**: `GET /health`, `POST /process`, `GET /model-info`.
- **Validation**: File extension, duration (max 10 min), sample rate (44.1/48 kHz), bit depth.
- **Output**: Response-streaming ZIP with stems, MIDI, and JSON report.
- **Lifecycle**: Load model on startup; clean up temp files in `finally` block.

### C2: Separation Module (`MambaSeparator`)
- **Architecture**: 12-layer Mamba-SSM (`d_model=768`, `d_state=16`, `d_conv=4`, `expand=2`).
- **Input**: Raw audio `(B, 1, T)`.
- **Processing**: STFT → linear projection → Mamba backbone → mask heads → ISTFT.
- **Output Heads**: Guitar mask, bass mask, residual mask, phase coherence score.
- **Streaming**: SSM state cache for chunked overlap-add inference.

### C3: Transcription Module (`MambaTranscriber`)
- **Architecture**: 8-layer Mamba-SSM (`d_model=512`, `d_state=12`).
- **Input**: Guitar stem audio `(B, 1, T)`.
- **Processing**: Mel spectrogram → linear proj → Mamba backbone → multi-task heads.
- **Output Heads**: Onset (binary), pitch (128 classes), velocity (regression), expression (3-class), confidence (MC dropout).

### C4: Confidence Injection (`ConfidenceInjector`)
- Cross-modal validation: align MIDI onsets with stem energy peaks.
- Per-note: confidence score, stem energy, alignment score, `needs_review` flag.
- DAW metadata: canonical JSON confidence, optional Polyphonic Key Pressure, and quantization suggestion (`grid` / `human`).

### C5: Loss Functions (`PerceptualAudioLoss`)
| Loss | Weight | Purpose |
|------|--------|---------|
| MR-STFT (6 resolutions) | 1.0 | Spectral fidelity |
| Spectral flatness | 0.1 | Transient preservation |
| Crest factor | 0.05 | Dynamic range |
| Onset F1 (future) | 1.0 | Accurate note start detection |
| Pitch cross-entropy (future) | 0.8 | Correct note classification |
| Velocity MAE (future) | 0.3 | Accurate dynamics |
| Cross-modal alignment | 0.5 | MIDI ↔ energy alignment |

### C6: Quality Gates (`route_by_quality`)
- Deterministic routing based on `ProcessingReport` fields.
- Three-tier output (studio/draft/complex) with user-facing action + badge + message.

## 2.3 Data Flow

```
1. Upload → Validate file (ext, duration, sample rate, bit depth)
2. Save to temp file
3. Load audio → tensor (B, 1, T)       [STUB: returns random noise]
4. MambaSeparator.forward(audio)
   → guitar_stem, bass_stem, residual, state_cache, metrics
5. MambaTranscriber.forward(guitar_stem)
   → onset_logits, pitch_logits, velocity, expression, confidence
6. _logits_to_midi(onset, pitch, velocity, confidence)
   → midi_events list
7. ConfidenceInjector.inject_midi_metadata(midi_events, confidence, stem_energy)
   → enhanced_midi dict
8. ProcessingReport(...) + route_by_quality(report)
   → routing_decision
9. Package into ZIP:
   - guitar_stem.wav, bass_stem.wav
   - guitar.mid, bass.mid                     [STUB: empty MIDI]
   - processing_report.json
10. Stream ZIP response
11. Clean up temp file
```

## 2.4 API Design

### `POST /process`
```
Request:  multipart/form-data with "file" field
Response: application/zip containing:
  - guitar_stem.wav
  - bass_stem.wav
  - guitar.mid
  - bass.mid
  - processing_report.json
Errors:   400 (bad file), 503 (model not loaded), 500 (processing error)
```

### `GET /live`, `GET /ready`, and `GET /health`
```
/live:  {"status": "alive"} (always 200)
/ready: {"status": "ready", "model_loaded": true} or 503 while unavailable
/health: deprecated alias of /ready
```

### `GET /model-info`
```
Response: model config plus selected backends and checkpoint_loaded
Errors:   503 if model not ready
```

## 2.5 Infrastructure Requirements (historical draft)

| Resource | Requirement | Notes |
|----------|-------------|-------|
| GPU | NVIDIA A100 40GB+ or H100 | Mamba-SSM CUDA kernels |
| CUDA | 12.1+ | Matches Docker base image |
| Python | 3.10+ | 3.13 in dev environment |
| RAM | 32 GB+ (16 GB min) | NeMo + model weights |
| Storage | 10 GB (app) + variable (datasets) | Model checkpoints, datasets |
| Networking | Port 8000 (API) | Configurable via `PORT` env |

**Container** (Dockerfile): `nvidia/cuda:12.1.1-runtime-ubuntu22.04`  
**Orchestration** (future): Kubernetes with GPU node pool and PersistentVolume for checkpoints.

## 2.6 Security Requirements

1. **No persistent storage of user audio**: process in-memory, delete temp files immediately.
2. **File validation**: reject by extension, duration, sample rate, content inspection.
3. **Minimal attack surface**: inference-only model (no training endpoints).
4. **GDPR by design**: no personal data collection, ephemeral containers, auditable logs.
5. **CORS**: use the `CORS_ORIGINS` allowlist; wildcard mode disables credentials.
6. **Auth:** optional comma-separated `API_KEYS` protect `/process` with `Authorization: Bearer <key>`; health/docs endpoints remain open.

## 2.7 Monitoring & Observability

| Category | Metrics |
|----------|---------|
| **Latency** | Per-stage timing (upload, separation, transcription, packaging) |
| **Quality** | separation proxy (not SI-SDR), phase score, avg confidence, artifact flags, tier distribution; reference SI-SDR remains an evaluation metric |
| **Resource** | GPU util, VRAM, CPU, memory, request queue depth |
| **Errors** | 4xx/5xx rate, model load failures, validation failures |
| **Business** | Files processed, studio/draft/complex ratio, human review take rate |

## 2.8 Development Workflow

### Commands
```bash
# Syntax check (fast)
python check_syntax.py

# Smoke test with synthetic data (random weights; structural only)
python demo.py

# CLI inference with a validated project checkpoint
python main.py --audio path/to/file.wav \
  --config configs/model_config.yaml \
  --checkpoint path/to/trained_checkpoint.pt

# Dev API server (set MODEL_CHECKPOINT_PATH first, unless using the explicit
# pretrained CPU backend config)
uvicorn api:app --reload          # or: python api.py

# Training (synthetic data by default)
python train.py --config configs/model_config.yaml --data-config example_data_config.yaml --output-dir ./outputs
```

### Required Order
1. `python check_syntax.py` — AST-level syntax check.
2. `python demo.py` — verify the random-weight structural smoke path.
3. `python -m pytest -q -m "not cuda"` — run the contract/regression suite.
4. Exercise the API only with a validated checkpoint or the explicit pretrained CPU backend pair.

## 2.9 Testing Strategy

| Test Type | Tool | Scope | Priority |
|-----------|------|-------|----------|
| Unit tests | pytest | Individual model components, loss functions, quality gates | P0 |
| Integration | pytest + httpx | API endpoints, ZIP packaging, file validation | P1 |
| Model regression | pytest | Forward pass shape checks, output ranges | P1 |
| Audio pipeline | custom | Real audio file → separation → transcription → MIDI | P2 |

**Test prerequisites:**
- Mock `torch` tensors to avoid CUDA requirement for unit tests.
- Use `tmp_path` fixture for temp file testing.
- Test quality gates with synthetic metrics.

**Current/target coverage:**
- Existing component, loss, quality-gate, API, integration, MIDI, and checkpoint tests live under `tests/`.
- `tests/test_remediation_contracts.py` covers the checkpoint, backend, independent-stream, timing, expression, confidence, and report contracts.
- Additional real-audio, target-hardware, optional-backend, and DAW tests remain evidence work.

## 2.10 Build, CI/CD & Release

### Versioning
- Semantic versioning (`MAJOR.MINOR.PATCH`).
- Pre-release suffix for alpha/beta (`1.0.0-alpha.1`).

### CI Pipeline (future)
```yaml
jobs:
  lint:     flake8 . && black --check . && isort --check .
  typecheck: mypy .
  syntax:   python check_syntax.py
  test:     pytest --cov=. --cov-report=term-missing
  build:    docker build -t stem-midi-pro .
```

### Release Process
1. Bump version in `api.py` (`version="..."`).
2. Update `CHANGELOG.md`.
3. Tag commit (`vX.Y.Z`).
4. Build and push Docker image.
5. Deploy to staging → smoke test → deploy to production.

---

# Part III: Audit Findings

All issues discovered during codebase audit at commit `51d28b7`. **Many items below are historical** — the current tree (`5e18e76`) has since removed NeMo/PyTorch-Lightning, renamed `data/canadian_datasets.py` → `data/datasets.py`, and fixed the crashes listed in 3.1–3.2. Items still open are cross-referenced in `TODO.md`. Do not treat this section as the current bug list; see `TODO.md` Section 2 for live status.

## 3.1 Runtime-Crashing Bugs

| ID | File | Line | Severity | Description |
|----|------|------|----------|-------------|
| B1 | `models/confidence_injector.py` | 1 | **CRITICAL** | **Missing `import torch`**. The file uses `torch.Tensor`, `torch.exp`, `torch.tensor`, `torch.mean` but never imports `torch`. Importing this module will raise `NameError`. |
| B2 | `models/mamba_separator.py` | 117 | **CRITICAL** | **Shape mismatch in mask×phase multiplication**. `mask_mag.unsqueeze(1)` has shape `(B, 1, T_frames, F)` but `torch.exp(1j * phase)` has shape `(B, F, T_frames)`. These are not broadcast-compatible. `forward()` will crash. |
| B3 | `models/mamba_separator.py` | 123 | **HIGH** | **Wrong ISTFT length computation**. `length=T * self.hop_length` computes `audio_samples * hop_length` which is orders of magnitude too large. `T` is the audio sample count, not the frame count. The ISTFT will fail or produce incorrect output. |
| B4 | `main.py` | 84–85 | **HIGH** | **Dataset batch structure mismatch**. `CanadianAudioDataset.__getitem__` returns `{'mixture': ..., 'targets': {'guitar': ..., 'bass': ...}}` but `training_step` expects `batch['target_guitar']`, `batch['target_bass']`, `batch['target_onsets']`, `batch['target_pitch']` at the top level. Training will crash with `KeyError`. |
| B5 | `main.py` | 97–103 | **MEDIUM** | **`sep_metrics` index error**. `sep_metrics` is `[si_sdr_est, phase_score]` of shape `(B, 2)`. Line 98 indexes `[0, 0]` (first batch, first col = SI-SDR), but line 99 indexes `[1, 0]` (second row, first col) instead of `[0, 1]` (first row, second col = phase). This swaps the metrics. |
| B6 | `main.py` | 112 | **MEDIUM** | **NeMo type violation**. Returning Python dicts (`midi_metadata`, `processing_report`, `routing_decision`) annotated as `ChannelType()`. NeMo expects tensor outputs matching declared types. Will fail `@typecheck()` validation. |

## 3.2 Functional Gaps (Stubs & Placeholders)

| ID | File | Line | Description |
|----|------|------|-------------|
| G1 | `main.py` | 248–253 | `_load_audio()` ignores its `path` argument and returns random noise. No `librosa` or `soundfile` call. The `--audio` CLI flag is effectively ignored. |
| G2 | `api.py` | 208–229 | `create_placeholder_midi()` ignores both parameters (`midi_data`, `stem_type`). Returns a minimal MIDI file (header + end-of-track only, zero notes). |
| G3 | `data/canadian_datasets.py` | 303–317 | `Slakh2100CADataset` and `MUSDBIndieDataset` both call `super()._get_synthetic_item()`. No real dataset loading is implemented. |
| G4 | `data/canadian_datasets.py` | 141–156 | `CanadianAudioDataset._get_real_item()` falls back to `_get_synthetic_item()`. |
| G5 | `configs/model_config.yaml` | 37 | `precision: "fp8"` but `train.py:79` maps it to `precision=16` in Lightning. Real FP8 requires TransformerEngine kernels which are not installed. |
| G6 | `main.py` | 82 | `self.separator(audio)` is called without `state_cache` parameter, so streaming state caching is never exercised. |
| G7 | `api.py` | 269 | `validation_info = validate_audio_file(...)` result is never read. |
| G8 | `main.py` | 43 | `register_artifact("target_stems", None)` — NeMo artifacts are meant for checkpoint serialization; these are never populated. |
| G9 | `api.py` | 234 | `background_tasks: BackgroundTasks` parameter is injected by FastAPI but never used. Temp file cleanup happens in-line via `finally` block, not as a background task. |
| G10 | `requirements.txt` | — | `httpx` is required for API testing (per ARD 2.9 recommendation) but not listed. |

## 3.3 Architecture & Data-Flow Contradictions

| ID | Description |
|----|-------------|
| C1 | **Training loop can't work**: `train.py` uses `training_step` which accesses `batch['target_guitar']`, `batch['target_bass']`, etc. but `CanadianAudioDataset` returns `{'mixture': ..., 'targets': {'guitar': ..., 'bass': ...}}`. The key structure is completely different. Also, `target_onsets` and `target_pitch` are never returned by any dataset method. |
| C2 | **Loss function expects unimplemented config keys**: `PerceptualAudioLoss` uses `cfg['loss']['mr_stft_weight']` etc. which exist in `model_config.yaml`, but it never uses `onset_f1_weight`, `pitch_ce_weight`, or `velocity_mae_weight` — the forward only uses the cross-modal alignment loss. The onset/pitch/velocity losses are referenced in docs but not implemented. |
| C3 | **Docker CMD shows `--help`**: `CMD ["python3", "main.py", "--help"]` is the default container command. The container will print help and exit unless the user overrides the command. Should default to `CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]`. |
| C4 | **`mamba_separator.py` imports `selective_scan_fn` but never uses it**: `from mamba_ssm import Mamba, selective_scan_fn` imports `selective_scan_fn` but the module only uses `Mamba`. |
| C5 | **`mamba_transcriber.py` always creates new Hann window**: Line 107 creates `torch.hann_window(n_fft, device=audio.device)` on every `forward()` call, allocating a new buffer each time. Should be registered as a persistent buffer once in `__init__`. |
| C6 | **`mamba_transcriber.py` hardcodes STFT params**: `_mel_spectrogram()` uses `n_fft=2048` and `hop_length=512` (lines 99-100) instead of reading `self.cfg['audio']['n_fft']` and `self.cfg['audio']['hop_length']`. Changing the config won't affect the transcriber's frequency resolution. |
| C7 | **Time-stretch in augmentation truncates audio**: `canadian_datasets.py:163-167` applies `librosa.effects.time_stretch` which changes array length, but never trims/pads back to the original `segment_length * sample_rate`. The returned tensors may have inconsistent lengths. |
| C8 | **`models/losses.py:8` has mutable default argument**: `resolutions=[(2048, 512), ...]` is shared across all instances. If mutated (it isn't currently), it would cause state leakage. |

## 3.4 Doc–Code Contradictions

| ID | Document | Claim | Reality |
|----|----------|-------|---------|
| D1 | `README.md:3` | "Production-ready Terraform module for deploying ... on GCP" | No Terraform files exist. Project is a Python prototype with no deployment scripts. |
| D2 | `README.md:3` | "Production-ready" (headline) | Core audio loading, MIDI generation, and dataset loading are stubs. No test suite. No CI. No deployment config. |
| D3 | `ARCHITECTURE.md:120-125` | Detailed streaming inference with state caching, overlap-add, CUDA graphs | `separator(audio)` never passes `state_cache`. CUDA graph capture is never called. Overlap-add logic is not implemented. |
| D4 | `DEVELOPMENT_GUIDE.md:136-144` | `pytest` examples with `--cov=stem_midi_pro` | No `tests/` directory exists. `pytest` will find zero tests. |
| D5 | `ARCHITECTURE.md:329-345` | "Data Loading" section with real dataset integration | All dataset methods return synthetic data. |
| D6 | `SUMMARY.md:192` | Refers to `CHANGELOG.md`, `CONTRIBUTING.md`, `docs/` directory | None of these exist. |
| D7 | Various docs | `$(date)` placeholder in footers | Never replaced with actual dates. |
| D8 | `README.md:101` | `python data/train.py` | File is at `./train.py`, not `data/train.py`. Command will fail with `FileNotFoundError`. |
| D9 | `DEVELOPMENT_GUIDE.md:327,330,333,336` | `flake8 stem_midi_pro`, `black stem_midi_pro`, etc. | No `stem_midi_pro` directory exists — files are at repo root. Should be `flake8 .`, `black .`, etc. |
| D10 | `DEVELOPMENT_GUIDE.md:141` | `pytest --cov=stem_midi_pro` | Wrong package path. Should be `--cov=.` or the actual package name if one is created. |
| D11 | `README.md:82` | `cd stem_midi_pro` after clone | Assumes a `stem_midi_pro` subdirectory. The repo root _is_ the project — no `cd` needed. |
| D12 | `README.md:42-74` | Project structure tree shows all paths under `stem_midi_pro/` | Files are at repo root, not in a `stem_midi_pro/` subdirectory. The tree is misleading. |

## 3.5 Developer Pain Points

| ID | Pain Point | Impact |
|----|------------|--------|
| P1 | **Historical Mamba-SSM CUDA build constraint.** The optional `mamba_ssm` package may compile CUDA kernels, but the current production path has a pure-PyTorch CPU fallback and does not require it. | Optional backend installation still needs platform-specific validation; CPU structural tests do not require it. |
| P2 | **Historical NeMo dependency concern.** NeMo and PyTorch Lightning were removed from the current path. | Current contributors use the plain PyTorch module/loop; this historical pain point is resolved. |
| P3 | **`verify_structure.py` hardcodes `stem_midi_pro/` prefix** but files are at repo root. Running it from repo root always reports all files as missing. | Confusing for new developers trying to verify setup. |
| P4 | **No trained checkpoint exists.** Random-initialized forwards are available for tests/training, but file/API inference now refuses them. | Quality cannot be evaluated or certified without an external trained checkpoint and held-out data. |
| P5 | **Resolved:** `api.py` imports the model lazily and does not instantiate it at module import. | Utilities can be imported independently; model loading happens during lifespan or an explicit call. |
| P6 | **Resolved:** the core requirements no longer include unused TensorRT packages. | CPU installation avoids the former NVIDIA-only dependency trap; optional pretrained backends remain separately installed. |
| P7 | **Loss functions advertised but not fully wired**: onset F1, pitch cross-entropy, velocity MAE, and duration IoU implementations exist, but only stem/onset targets are currently passed by the orchestrator. | Training gradients remain incomplete for the un-wired multitask terms. |

---

*Document generated from codebase audit. PRD sections define target state; ARD sections define required architecture shifts; Audit sections document current-state gaps to be resolved.*
