# Stem+MIDI Pro — Feature Summary

> **Legend:** `[IMPLEMENTED]` works mechanically in the code today · `[PARTIAL]` partially works or lacks trained/evaluation evidence · `[PLANNED]` designed but not built (tracked in [TODO.md](TODO.md)).
> Audit basis: current source tree. Target platform: **CPU-only, bare-metal i5, Python 3.13**. No trained checkpoint is shipped. License: Apache-2.0 ([LICENSE](LICENSE)).
> **Production boundary:** native-Mamba file/API inference requires a validated project checkpoint; the explicit pretrained CPU profile may use its separately loaded Demucs + Basic Pitch weights. Random-initialized forwards are available only for training and structural tests, are marked `untrained_weights`, and cannot pass the quality gate.

## Core Pipeline

| Feature | Status | Notes |
|---|---|---|
| Guitar/bass stem separation and backend selection | [PARTIAL] | `StemMidiModel` honors the configured separator factory (`mamba`, optional `demucs`, or test fake) and emits separate guitar/bass/residual tensors. The Mamba fallback is mechanically runnable, but no trained checkpoint or separation-quality evidence is shipped. |
| MIDI transcription (onset/pitch/velocity) | [PARTIAL] | Native Mamba and configured transcriber backends run independently for guitar and bass. Event decoding preserves absolute onset frames, durations, chords, and overlaps; musical accuracy still requires trained/evaluated weights. |
| Expression detection/mapping (bend, vibrato, slide) | [PARTIAL] | Serialized expression fields map to pitch-wheel, CC1, and CC74. The available expression head is not trained or calibrated here, so no expression-quality claim is made. |
| Per-note confidence metadata | [IMPLEMENTED] | Confidence is counted over emitted notes, stored canonically in JSON, and encoded as MIDI Polyphonic Key Pressure when supported. `cc_127_value` is a deprecated JSON alias; CC#127 is never emitted. |
| Quality gates and routing | [IMPLEMENTED] | Thresholds use an explicitly named centroid-separation proxy, not SI-SDR. Artifact flags and `untrained_weights` force `complex`/manual DAW review. |
| Audio loading (`main.py:_load_audio`) | [IMPLEMENTED] | Real `soundfile` load with librosa fallback, stereo→mono downmix, float32. |
| Processing report | [PARTIAL] | ZIP JSON includes `separation_proxy`, explicit non-SI-SDR semantics, confidence counts, model-weight status, artifact flags, and quality tier. It is not a reference-based quality certificate. |
| CLI (`python main.py --audio <file>`) | [PARTIAL] | Native Mamba requires `--checkpoint` or `MODEL_CHECKPOINT_PATH`; the explicit pretrained CPU backend profile is accepted only when its own dependencies attest to loaded weights. Missing weights fail instead of serving random output. |
| Demo (`python demo.py`) | [PARTIAL] | Synthetic random-weight structural smoke test only; it is not production audio evidence. |

## API (`api.py`)

| Feature | Status | Notes |
|---|---|---|
| `POST /process` → ZIP (stems, MIDI, report) | [PARTIAL] | Streams upload with 500 MB cap (413), validates format/duration/sample rate, and packages independent MIDI streams. Startup is not ready until a trained checkpoint is configured. |
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
| Onset F1 / pitch CE / velocity MAE / duration IoU losses | [PARTIAL] | The loss functions exist in `models/losses.py` (ARCH-11.6), but the current orchestrator wires only stem and onset targets; pitch, velocity, and duration training integration remains open. |

## Deployment & Ops

| Feature | Status | Notes |
|---|---|---|
| Bare-metal / venv deployment | [PARTIAL] | Core CPU installation exists; a trained checkpoint and optional backend dependencies must be supplied separately. See [SETUP.md](SETUP.md). |
| Docker (multi-stage, python:3.13-slim) | [IMPLEMENTED] | `Dockerfile.cpu` (CPU-only) and `Dockerfile.gpu` (CUDA fine-tune) exist. |
| systemd unit, compose file | [PLANNED] | BLD-9.2.8–9. |
| Observability (Prometheus, Grafana) | [PARTIAL] | `/metrics` implemented (API-6.16); Grafana dashboards planned (BLD-9.2.10). |
| pytest suite (`tests/`) | [IMPLEMENTED] | Test files in `tests/`; run `pytest -m "not cuda"` for CPU-only. |

## Explicitly Not Done (removed or never implemented)

- TensorRT-LLM export — **[REMOVED]**, CUDA-only.
- FP8 precision — **[REMOVED]**; CPU target uses fp32.
- CUDA graph capture / sub-5ms hop latency — **[REMOVED]**, GPU-only claims. No reference-i5 timing is certified; historical seconds-per-minute estimates are not a service guarantee.
- Tuning detection, tempo detection, palm-mute, harmonics, slap/pop detection — **[REMOVED]** from docs; no such code exists.
- True streaming SSM state passing — **[PLANNED]** (ARCH-11.8.1); current chunked path restarts state per chunk, but performs waveform overlap-add and translates local event positions to global frames.
- Reference-i5 performance and separation/transcription quality — **[UNVERIFIABLE]** until a trained checkpoint, target hardware benchmark, held-out audio, and independent metrics are supplied.
- NeMo / PyTorch Lightning dependence — **[REMOVED]** (RF-3.1.7–9); `main.py.StemMidiModel` is a plain `nn.Module`, `train.py` is a custom loop.

## Research Code

`research/mamba3_per_track/` contains the Mamba-3 per-track experiments — separate from the production path, provided as-is. See `research/README.md`.

---
*Last audited: 2026-11. See [TODO.md](TODO.md) for live status.*
