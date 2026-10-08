# Stem+MIDI Pro — Feature Summary

> **Legend:** `[IMPLEMENTED]` works in the code today · `[PARTIAL]` partially works or has known bugs · `[PLANNED]` designed but not built (tracked in [TODO.md](TODO.md)).
> Audit basis: current source tree, 2026-06. Target platform: **CPU-only, bare-metal i5, Python 3.13**. License: Apache-2.0 ([LICENSE](LICENSE)).

## Core Pipeline

| Feature | Status | Notes |
|---|---|---|
| Guitar/bass stem separation (Mamba-SSM, `models/mamba_separator.py`) | [PARTIAL] | Runs, but has known defects tracked as C-2.1, C-2.5, C-2.6, C-2.8 in TODO.md (invalid Mamba kwargs, broken state cache, SI-SDR is a centroid proxy, inverted phase-cancellation check). |
| MIDI transcription (onset/pitch/velocity, `models/mamba_transcriber.py`) | [PARTIAL] | Multi-task heads exist; `self.cfg` bug (C-2.2) and per-forward mel-basis rebuild (C-2.9) open. |
| Expression detection (bend, vibrato, slide) | [PARTIAL] | Head exists; not trained (ARCH-11.4.4). No tuning, tempo, palm-mute, harmonics, or slap/pop detection. |
| Per-note confidence → MIDI CC#127 (`models/confidence_injector.py`) | [IMPLEMENTED] | CC#127 tagging works; per-note duration is a fixed 1/16 note (API-6.12). |
| Quality gates studio/draft/complex (`utils/quality_gates.py`) | [IMPLEMENTED] | Thresholds from `configs/model_config.yaml`; any artifact flag (other than `none`) forces `complex` regardless of confidence. |
| Audio loading (`main.py:_load_audio`) | [IMPLEMENTED] | Real `soundfile` load with librosa fallback, stereo→mono downmix, float32. |
| Processing report (`si_sdr`, `phase_coherence`, `avg_confidence`, `artifact_flags`, `low_confidence_notes`, `quality_tier`) | [IMPLEMENTED] | Emitted as `processing_report.json` in the output ZIP. |
| CLI (`python main.py --audio <file>`) | [IMPLEMENTED] | |
| Demo (`python demo.py`) | [IMPLEMENTED] | Synthetic-audio smoke test. |

## API (`api.py`)

| Feature | Status | Notes |
|---|---|---|
| `POST /process` → ZIP (stems, MIDI, report) | [IMPLEMENTED] | Streams upload with 500 MB cap (413), validates format/duration/sample rate. |
| API-key auth (`Authorization: Bearer`) | [IMPLEMENTED] | Enabled when `API_KEYS` env var is set (comma-separated); `/process` only; 401/403 responses. |
| Upload size limit (500 MB, 413) | [IMPLEMENTED] | |
| CORS allowlist | [IMPLEMENTED] | `CORS_ORIGINS` env; credentials disabled when `*`. |
| `/health` | [IMPLEMENTED] | Combined liveness/readiness; split into `/live` + `/ready` is planned (API-6.17). |
| `/metrics` (Prometheus) | [PLANNED] | API-6.16. |
| `/model-info` | [IMPLEMENTED] | |
| `/templates`, `/render-template` | [IMPLEMENTED] | Renders the 7 templates in `user_content/`; template names are basename-sanitized. |
| Concurrency cap (503 + Retry-After) | [PLANNED] | API-6.4. |

## Training & Data

| Feature | Status | Notes |
|---|---|---|
| `train.py` training loop | [PARTIAL] | Currently PyTorch Lightning; planned move to a plain PyTorch loop (RF-3.1.9, Section 12). |
| `data/datasets.py` — `AudioDataset`, `Slakh2100YourMT3Dataset`, `MUSDB18HQDataset`, `get_data_loaders` | [PARTIAL] | Slakh/MUSDB loaders exist with known bugs (C-2.3, C-2.4, C-2.7); synthetic fallback when data path is missing. The old `Slakh2100CADataset` / `MUSDBIndieDataset` names are **retired** — see R-1.3 in TODO.md. |
| Losses (MR-STFT, crest, flatness) | [IMPLEMENTED] | Via `auraloss` in `models/losses.py`. |
| Onset F1 / pitch CE / velocity MAE / duration IoU losses | [PLANNED] | ARCH-11.6; stub TODO in `models/losses.py:34`. |

## Deployment & Ops

| Feature | Status | Notes |
|---|---|---|
| Bare-metal / venv deployment | [IMPLEMENTED] | See [SETUP.md](SETUP.md). |
| Docker (multi-stage, python:3.13-slim) | [PLANNED] | BLD-9.2.4; a legacy CUDA-based Dockerfile exists but is being replaced. |
| systemd unit, compose file | [PLANNED] | BLD-9.2.8–9. |
| Observability (Prometheus, Grafana) | [PLANNED] | BLD-9.2.10; see `docs/operations/monitoring.md`. |
| pytest suite (`tests/`) | [PARTIAL] | Test files exist; several are marked skip pending Section 8 work. |

## Explicitly Not Done (removed or never implemented)

- TensorRT-LLM export — **[REMOVED]**, CUDA-only.
- FP8 precision — **[REMOVED]**, `precision: fp8` in the config is a pass-through placeholder for Lightning; CPU target uses fp32.
- CUDA graph capture / sub-5ms hop latency — **[REMOVED]**, GPU-only claims. CPU reality: ~30–60 s per minute of audio.
- Tuning detection, tempo detection, palm-mute, harmonics, slap/pop detection — **[REMOVED]** from docs; no such code exists.
- True streaming SSM state passing — **[PLANNED]** (ARCH-11.8.1); current chunked path concatenates activations (C-2.5).
- NeMo / PyTorch Lightning dependence — **[PLANNED removal]** (RF-3.1.7–9).

## Research Code

`research/mamba3_per_track/` contains the Mamba-3 per-track experiments — separate from the production path, provided as-is. See `research/README.md`.

---
*Last audited: 2026-06. See [TODO.md](TODO.md) for live status.*
