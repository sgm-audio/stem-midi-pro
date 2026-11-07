# Stem+MIDI Pro — Feature Summary

> **Legend:** `[IMPLEMENTED]` works in the code today · `[PARTIAL]` partially works or has known bugs · `[PLANNED]` designed but not built (tracked in [TODO.md](TODO.md)).
> Audit basis: current source tree. Target platform: **CPU-only, bare-metal i5, Python 3.13**. License: Apache-2.0 ([LICENSE](LICENSE)).

## Core Pipeline

| Feature | Status | Notes |
|---|---|---|
| Guitar/bass stem separation (Mamba-SSM, `models/mamba_separator.py`) | [IMPLEMENTED] | Runs on CPU via pure-PyTorch Mamba fallback (`models/_mamba_compat.py`). Known defects C-2.1 (invalid Mamba kwargs), C-2.5 (state cache removed/documented), C-2.6 (SI-SDR is a centroid proxy), C-2.8 (inverted phase-cancellation) were fixed 2026-10-07. Metric reported as `si_sdr` is a spectral-centroid proxy, not true SI-SDR — see TODO C-2.6. |
| MIDI transcription (onset/pitch/velocity, `models/mamba_transcriber.py`) | [IMPLEMENTED] | Multi-task heads exist. Bugs C-2.2 (`self.cfg` not assigned) and C-2.9 (per-forward mel-basis rebuild) fixed 2026-10-07. `self.cfg` is assigned; mel basis is a registered buffer built once in `__init__`. |
| Expression detection (bend, vibrato, slide) | [PARTIAL] | Head exists; not trained (ARCH-11.4.4). No tuning, tempo, palm-mute, harmonics, or slap/pop detection. |
| Per-note confidence → MIDI CC#127 (`models/confidence_injector.py`) | [IMPLEMENTED] | CC#127 tagging works. Per-note duration now honors `duration_frames` (API-6.12 fixed 2026-10-07); falls back to a configurable default when absent. |
| Quality gates studio/draft/complex (`utils/quality_gates.py`) | [IMPLEMENTED] | Thresholds from `configs/model_config.yaml`; any artifact flag (other than `none`) forces `complex` regardless of confidence (T-8.3.9). |
| Audio loading (`main.py:_load_audio`) | [IMPLEMENTED] | Real `soundfile` load with librosa fallback, stereo→mono downmix, float32. |
| Processing report (`si_sdr`, `phase_coherence`, `avg_confidence`, `artifact_flags`, `low_confidence_notes`, `quality_tier`) | [IMPLEMENTED] | Emitted as `processing_report.json` in the output ZIP. |
| CLI (`python main.py --audio <file>`) | [IMPLEMENTED] | |
| Demo (`python demo.py`) | [IMPLEMENTED] | Synthetic-audio smoke test. |

## API (`api.py`)

| Feature | Status | Notes |
|---|---|---|
| `POST /process` → ZIP (stems, MIDI, report) | [IMPLEMENTED] | Streams upload with 500 MB cap (413), validates format/duration/sample rate. |
| API-key auth (`Authorization: Bearer`) | [IMPLEMENTED] | Enabled when `API_KEYS` env var is set (comma-separated); `/process` only; 401/403 responses (API-6.5). |
| Upload size limit (500 MB, 413) | [IMPLEMENTED] | |
| CORS allowlist | [IMPLEMENTED] | `CORS_ORIGINS` env; credentials disabled when `*` (API-6.2). |
| `/live` | [IMPLEMENTED] | Liveness probe — always 200 (API-6.17). |
| `/ready` | [IMPLEMENTED] | Readiness probe — 503 + `Retry-After` while model loads (API-6.17). |
| `/health` | [IMPLEMENTED] | Deprecated alias of `/ready`, hidden from OpenAPI schema. |
| `/metrics` (Prometheus) | [IMPLEMENTED] | Prometheus text exposition (`requests_total`, `request_duration_seconds`, `in_flight_requests`); `prometheus_client` if installed, else hand-rolled (API-6.16). |
| `/model-info` | [IMPLEMENTED] | |
| `/templates`, `/render-template` | [IMPLEMENTED] | Renders the 7 templates in `user_content/`; template names are basename-sanitized (API-6.14). |
| Concurrency cap (503 + Retry-After) | [IMPLEMENTED] | `asyncio.Semaphore(1)` around `process_audio_file` (API-6.4). |
| `/warmup` | [IMPLEMENTED] | 1-second synthetic inference at startup and on demand (API-6.20). |

## Training & Data

| Feature | Status | Notes |
|---|---|---|
| `train.py` training loop | [IMPLEMENTED] | Plain PyTorch loop (Lightning removed — RF-3.1.9). Epoch train/val, `--grad-accum`, grad clipping, best/last checkpoints, early stopping, `--resume`, explicit best-checkpoint load for final test. |
| `data/datasets.py` — `AudioDataset`, `Slakh2100YourMT3Dataset`, `MUSDB18HQDataset`, `get_data_loaders` | [IMPLEMENTED] | Synthetic fallback when data path is missing. Bugs C-2.3 (file_list set twice), C-2.4 (Slakh `_build_file_list` on missing root), C-2.7 (MUSDB `sum()` 0-d accumulator) fixed 2026-10-07. The old `Slakh2100CADataset` / `MUSDBIndieDataset` names are **retired**. |
| Losses (MR-STFT, spectral flatness, crest factor) | [IMPLEMENTED] | Native implementations in `models/losses.py` (not `auraloss` — `auraloss` is used only under `research/`). |
| Onset F1 / pitch CE / velocity MAE / duration IoU losses | [IMPLEMENTED] | `onset_f1_loss_fn`, `pitch_ce_loss_fn`, `velocity_mae_loss_fn`, `duration_iou_loss_fn` in `models/losses.py` (ARCH-11.6). |

## Deployment & Ops

| Feature | Status | Notes |
|---|---|---|
| Bare-metal / venv deployment | [IMPLEMENTED] | See [SETUP.md](SETUP.md). |
| Docker (multi-stage, python:3.13-slim) | [IMPLEMENTED] | `Dockerfile.cpu` (CPU-only) and `Dockerfile.gpu` (CUDA fine-tune) exist. |
| systemd unit, compose file | [PLANNED] | BLD-9.2.8–9. |
| Observability (Prometheus, Grafana) | [PARTIAL] | `/metrics` implemented (API-6.16); Grafana dashboards planned (BLD-9.2.10). |
| pytest suite (`tests/`) | [IMPLEMENTED] | Test files in `tests/`; run `pytest -m "not cuda"` for CPU-only. |

## Explicitly Not Done (removed or never implemented)

- TensorRT-LLM export — **[REMOVED]**, CUDA-only.
- FP8 precision — **[REMOVED]**; CPU target uses fp32.
- CUDA graph capture / sub-5ms hop latency — **[REMOVED]**, GPU-only claims. CPU reality: ~30–60 s per minute of audio.
- Tuning detection, tempo detection, palm-mute, harmonics, slap/pop detection — **[REMOVED]** from docs; no such code exists.
- True streaming SSM state passing — **[PLANNED]** (ARCH-11.8.1); current chunked path restarts state per chunk (C-2.5).
- NeMo / PyTorch Lightning dependence — **[REMOVED]** (RF-3.1.7–9); `main.py.StemMidiModel` is a plain `nn.Module`, `train.py` is a custom loop.

## Research Code

`research/mamba3_per_track/` contains the Mamba-3 per-track experiments — separate from the production path, provided as-is. See `research/README.md`.

---
*Last audited: 2026-11. See [TODO.md](TODO.md) for live status.*
