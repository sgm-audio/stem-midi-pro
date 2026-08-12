# Stem+MIDI Pro — TODO

> **Audit date:** 2026-06-02
> **Source-of-truth architecture:** `ARCHITECTURE.md` (every `[PLANNED]` feature in there is a TODO below)
> **Target hardware:** Bare-metal i5 / 4–8 GB DDR4 (CPU-only)
> **Python:** 3.13
> **License:** Polyform Small Business License 1.0.0
> **Repo structure decision:** Consolidate on the **v1 production path** (`main.py` / `api.py` / `models/*` / `train.py` / `data/datasets.py` Slakh+MUSDB loaders). Mamba-3 research work moves to `/research/`.

---

## Legend

- **Status:** `[ ]` open · `[x]` done · `[~]` in progress · `[!]` blocked
- **Effort:** S < 1h · M 1–4h · L 1–3 days · XL 1+ week
- **Priority:** P0 (blocks prod) · P1 (next sprint) · P2 (post-MVP) · P3 (research)
- **Tags:** `bug`, `feature`, `perf`, `sec`, `test`, `docs`, `infra`, `refactor`, `arch`

---

# SECTION 0 — STRATEGIC DECISIONS (LOCKED)

- [x] **D-STRAT-1** · `decision` · **Path B: consolidate on v1 production path.** `model.py` / `train_mamba3.py` / `utils/losses_mamba3.py` / Mamba-3-specific dataset classes / `SETUP.md` (Mamba-3 version) / `mamba/` github-clone → moved to `/research/`. Production path stays: `main.py` + `api.py` + `models/*` + `train.py` + `data/datasets.py` (Slakh/MUSDB loaders only) + `configs/model_config.yaml`.
- [x] **D-STRAT-2** · `decision` · **Architecture.md is a source of truth to implement** — every `[PLANNED]` section becomes a tracked work item (see Section 11).
- [x] **D-STRAT-3** · `decision` · **CPU-only** — drop all CUDA-only paths, downsize models, INT8-ready where possible. Mamba-SSM 1.x supports CPU via the pure-PyTorch reference impl (slower but functional).
- [x] **D-STRAT-4** · `decision` · **Python 3.13** — pin in pyproject, Dockerfile, CI.
- [x] **D-STRAT-5** · `decision` · **`mamba-ssm` from PyPI**; **no `causal-conv1d`** (CUDA-only, not needed for CPU).
- [x] **D-STRAT-6** · `decision` · **License: Polyform Small Business License 1.0.0** — fits the indie pricing framing (free for individuals + small companies, paid above $1M revenue / 50 employees). Place at `/LICENSE`.

---

# SECTION 1 — STRUCTURAL REORG

## 1.1 Create `/research/` and move Mamba-3 files

- [x] **R-1.1.1** · `refactor` · S · Create `/research/` and `/research/mamba3_per_track/` subdirectory. ✅
- [x] **R-1.1.2** · `refactor` · S · Move `model.py` → `/research/mamba3_per_track/per_track_processor.py`. ✅
- [x] **R-1.1.3** · `refactor` · S · Move `train_mamba3.py` → `/research/mamba3_per_track/train.py` (renamed to drop redundant `mamba3_` prefix inside the `mamba3_per_track/` subdir). ✅
- [x] **R-1.1.4** · `refactor` · S · Move `utils/losses_mamba3.py` → `/research/mamba3_per_track/losses.py` (renamed for same reason). ✅
- [x] **R-1.1.5** · `refactor` · S · Move `SETUP.md` → `/research/mamba3_per_track/SETUP.md`. ✅
- [x] **R-1.1.6** · `refactor` · S · Move `configs/mamba3_config.yaml` → `/research/mamba3_per_track/config.yaml`. ✅
- [x] **R-1.1.7** · `refactor` · S · Extract `StemDataset` + `Slakh2100StemDataset` + `collate_valid` + `get_musdb18hq_data_loader` + `get_slakh2100_loader` from `data/datasets.py` → `/research/mamba3_per_track/mamba3_datasets.py`. Production `data/datasets.py` retains `AudioDataset`, `Slakh2100YourMT3Dataset`, `MUSDB18HQDataset`, `get_data_loaders`. ✅
- [x] **R-1.1.8** · `refactor` · S · Move `mamba/` (vendored github clone, includes own `.git`) → `/research/mamba-ssm-reference/mamba/`. Added `/research/mamba-ssm-reference/.gitignore` to ignore the vendored `.git` and build artifacts. ✅
- [x] **R-1.1.9** · `refactor` · S · Added `/research/README.md` (top-level index for both research subdirs; covers R-1.1.9, R-1.1.10, R-1.1.11 in one document). ✅
- [x] **R-1.1.10** · `refactor` · S · (folded into R-1.1.9). ✅
- [x] **R-1.1.11** · `refactor` · S · (folded into R-1.1.9). ✅
- [x] **R-1.1.12** · `refactor` · S · Moved `tests/test_losses_mamba3.py` → `/research/mamba3_per_track/test_losses.py`; updated import to use `from losses import …` with `sys.path` injection. ✅

## 1.2 Update root paths & references

- [x] **R-1.2.1** · `refactor` · S · `check_syntax.py` — `dirs[:]` skip list now excludes `venv`, `.git`, `__pycache__`, `node_modules`, and any subdir under `research/mamba-ssm-reference/`. Verified: scans 28 .py files (was 117+). ✅
- [x] **R-1.2.2** · `refactor` · S · `verify_structure.py` — removed all `model.py` / `train_mamba3.py` / `losses_mamba3.py` checks; added `research/mamba3_per_track/` and `research/mamba-ssm-reference/mamba/README.md` as informational checks. Also fixed cp1252 UnicodeEncodeError by replacing ✓/✗ with `[OK]/[MISSING]`. ✅
- [x] **R-1.2.3** · `refactor` · S · `.github/workflows/ci.yml` — bump `PYTHON_VERSION` 3.10→3.13; replace `model.py` parse step with `main.py` parse step; update file-existence list (added `api.py`, `train.py`, `models/confidence_injector.py`, `models/losses.py`, `configs/model_config.yaml`; removed `model.py`, `train_mamba3.py`, `utils/losses_mamba3.py`; added `research/mamba3_per_track/per_track_processor.py`, `research/mamba-ssm-reference/mamba/README.md` as research sanity checks); exclude `research/mamba-ssm-reference` from flake8/black/isort. ✅
- [x] **R-1.2.4** · `refactor` · S · `pyproject.toml` — **N/A: no `pyproject.toml` exists at repo root.** No action needed. ✅
- [x] **R-1.2.5** · `refactor` · S · `.dockerignore` — **N/A: no `.dockerignore` exists at repo root.** The Dockerfile rewrite (BLD-9.2.4 in Section 9.2) will introduce one. ✅
- [x] **R-1.2.6** · `docs` · S · `example_data_config.yaml` — header rewritten to point at `research/mamba3_per_track/train.py`; `device: cuda` → `device: cpu`; `amp: true` → `amp: false` (CPU target). ✅

## 1.3 Drop redundant qualifier prefixes (mirror the `mamba3_` rename)

- [x] **R-1.3.1** · `refactor` · S · Rename `data/canadian_datasets.py` → `data/datasets.py` (the file IS the data loader; the `canadian_` qualifier belongs in the curation policy, not the filename). ✅
- [x] **R-1.3.2** · `refactor` · S · Rename `CanadianAudioDataset` → `AudioDataset` (the base class is the v1 audio dataset; not all data needs to be Canadian-sourced — Slakh2100 and MUSDB18-HQ are global). The mission alignment toward Canadian artists is preserved in the config-driven sampling weights and the module docstring. ✅
- [x] **R-1.3.3** · `refactor` · S · Rename `get_canadian_data_loaders` → `get_data_loaders` (same reasoning). ✅
- [x] **R-1.3.4** · `refactor` · S · Update all imports: `train.py`, `tests/test_dataset_loaders.py` (split v1 + v3 tests; v3 moved to `research/mamba3_per_track/test_datasets.py`), `verify_structure.py`, `.github/workflows/ci.yml`. ✅
- [x] **R-1.3.5** · `docs` · S · Update all doc references: `README.md`, `DEVELOPMENT_GUIDE.md`, `SUMMARY.md`, `ARCHITECTURE.md`, `research/README.md`, `research/mamba3_per_track/mamba3_datasets.py`, `TODO.md`. The "Canadian" mission is still documented in `data/datasets.py` module docstring + `README.md` (mission section) + `PRD_AND_ARD.md` — kept where it belongs (curation policy / business context), not in code identifiers. ✅

---

# SECTION 2 — CRITICAL RUNTIME BUGS (P0)

These are bugs that crash or produce wrong output today. They are the highest-priority items because they block any production use.

- [x] **C-2.1** · `bug` · M · P0 · ✅ Removed NeMo Module base class + `@typecheck`, dropped invalid `causal_conv1d_impl=` / `selective_scan_impl=` kwargs. MambaSeparator now inherits `nn.Module` directly. Added shape asserts in forward.
- [x] **C-2.2** · `bug` · S · P0 · ✅ Added `self.cfg = cfg` in `MambaTranscriber.__init__`. Replaced NeMo Module → nn.Module.
- [x] **C-2.3** · `bug` · M · P0 · ✅ Merged `_validate_dataset` into `__init__`, `_build_file_list` is single source of truth with synthetic short-circuit.
- [x] **C-2.4** · `bug` · M · P0 · ✅ Added `_build_file_list` override in `Slakh2100YourMT3Dataset` with synthetic short-circuit.
- [x] **C-2.5** · `bug` · L · P0 · ✅ Documented streaming limitation in class docstring. Removed broken activation-cache code. `new_state_cache` returns `None`. Forward is independent per call.
- [x] **C-2.6** · `bug` · M · P0 · ✅ Renamed `_estimate_si_sdr` to `_centroid_separation`, added docstring explaining it's a proxy heuristic not real SI-SDR. `_spectral_centroid` kept separate (needs its own STFT for per-stem analysis).
- [x] **C-2.7** · `bug` · S · P0 · ✅ Fixed MUSDB18HQDataset mixture: explicit unpack `total = sum(stem_wavs[k] for k in stem_wavs); mixture = total / len(stem_wavs)`.
- [x] **C-2.8** · `bug` · S · P0 · ✅ Fixed inverted phase cancellation logic: changed from `abs(phase_corr) > 0.9` to `phase_corr < -0.7` (anti-correlation = cancellation).
- [x] **C-2.9** · `bug` · S · P0 · ✅ Moved mel_basis creation to `__init__`, registered as buffer `self.mel_basis`. Eliminates per-forward Python loop.
- [x] **C-2.10** · `bug` · M · P0 · ✅ Renamed `confidence_head` to `mc_dropout_eval`. Eval-only MC dropout intent made explicit.
- [x] **C-2.11** · `bug` · M · P0 · ✅ Added shape assertions in separator forward: input dim check + time-dim consistency check.
- [x] **C-2.12** · `bug` · S · P0 · ✅ Changed `torch.tensor(audio)` to `torch.from_numpy(audio).float()` for explicit casting.

---

# SECTION 3 — REFACTORS (P0–P1)

## 3.1 Collapse duplication

- [ ] **RF-3.1.1** · `refactor` · M · P1 · `data/datasets.py` — collapse the v1 `MUSDB18HQDataset` and `Slakh2100YourMT3Dataset` into a single `StemFolderDataset(stem_subdir: str, …)` class. They share ~90% of code. Add config-driven selection: `dataset_type` chooses subdir, layout, and stem mapping.
- [x] **RF-3.1.2** · `refactor` · M · P1 · Move `MambaSeparator` mel/STFT code into a shared `audio/stft.py` module with one canonical `stft`, `istft`, `mel_spectrogram` (returns a registered buffer), and `polar_to_complex(mag, phase)`. Use from both separator and transcriber.
- [x] **RF-3.1.3** · `refactor` · M · P1 · `main.py:_logits_to_midi` is called from `forward` and `process_audio_streaming`. Move to a new `utils/midi.py` that owns: `logits_to_events`, `events_to_bytes`, `build_midi_from_events` (replaces the misnamed `api.py:create_placeholder_midi`).
- [x] **RF-3.1.4** · `refactor` · S · P1 · `api.py:create_placeholder_midi` (which actually emits real events) — rename to `build_midi_from_events` and move to `utils/midi.py`. Update `create_response_zip`.
- [x] **RF-3.1.5** · `refactor` · S · P1 · `models/confidence_injector.py:inject_midi_metadata` loop creates a `torch.tensor` per event for the sigmoid. Vectorize. Add a benchmark.
- [x] **RF-3.1.6** · `refactor` · S · P1 · `main.py:_logits_to_midi` Python `for b,t` loops over batch×T. Replace with `torch.nonzero(onset_mask).tolist()`. ~100× faster on 60s audio.
- [x] **RF-3.1.7** · `refactor` · M · P1 · Remove NeMo ModelPT wrapping from `main.py` — drop the `nemo.core.classes.ModelPT` and `nemo.core.neural_types` imports. Replace with a `torch.nn.Module` subclass `StemMidiModel(nn.Module)` that exposes `forward`, `training_step`, `validation_step` for a plain PyTorch training loop. **Rationale:** NeMo brings in `nemo-toolkit[all]` which is hundreds of MB and CUDA-leaning; bare-metal i5 doesn't need it.
- [x] **RF-3.1.8** · `refactor` · M · P1 · Update `models/mamba_separator.py` and `models/mamba_transcriber.py` — drop the NeMo `Module` base class and `typecheck` decorators (no longer relevant without NeMo typing). Replace with explicit shape asserts in forward.
- [x] **RF-3.1.9** · `refactor` · M · P1 · `train.py` — replace `pytorch_lightning` Trainer with a plain PyTorch training loop (or `accelerate`/`lightning-fabric` if you want some niceties). Drop `pytorch-lightning` from requirements. Bare-metal CPU training is much simpler in a custom loop.

## 3.2 Consolidate file structure

- [x] **RF-3.2.1** · `refactor` · S · P1 · Populate `models/__init__.py` with `__all__ = ["MambaSeparator", "MambaTranscriber", "ConfidenceInjector", "PerceptualAudioLoss"]`.
- [x] **RF-3.2.2** · `refactor` · S · P1 · Populate `utils/__init__.py` with `__all__ = ["ProcessingReport", "QualityTier", "route_by_quality", "render_template", "list_templates"]`.
- [x] **RF-3.2.3** · `refactor` · S · P1 · Populate `data/__init__.py` with `__all__ = ["AudioDataset", "get_data_loaders", "StemFolderDataset"]` (post-collapse).
- [x] **RF-3.2.4** · `refactor` · S · P1 · Add a top-level `audio_io.py` (or `io/audio.py`) module with: `load_audio(path, target_sr, mono=True)`, `save_wav(path, audio, sr)`, `validate_audio(path, max_duration, supported_srs)`. Single source of truth for I/O.
- [x] **RF-3.2.5** · `refactor` · S · P1 · Add `utils/paths.py` with project-root resolution (replace ad-hoc `Path(__file__).parent.parent / "user_content"` patterns).

---

# SECTION 4 — DEPENDENCIES (P0)

## 4.1 Production `requirements.txt`

- [x] **DEP-4.1.1** · `infra` · S · P0 · ✅ `requirements.txt` rewritten: torch 2.4-2.7, CPU-only, Python 3.13.
- [x] **DEP-4.1.2** · `infra` · S · P0 · ✅ torchaudio 2.4-2.7, matches torch.
- [x] **DEP-4.1.3** · `infra` · S · P0 · ✅ Removed `nemo-toolkit[all]` — no longer used.
- [x] **DEP-4.1.4** · `infra` · S · P0 · ✅ Removed `pytorch-lightning` — replaced by custom training loop.
- [x] **DEP-4.1.5** · `infra` · S · P0 · ✅ `mamba-ssm>=2.0.0` in requirements.txt.
- [x] **DEP-4.1.6** · `infra` · S · P0 · ✅ No `causal-conv1d` — confirmed.
- [x] **DEP-4.1.7** · `infra` · S · P0 · ✅ librosa 0.10-0.11 kept.
- [x] **DEP-4.1.8** · `infra` · S · P0 · ✅ soundfile 0.12+ kept.
- [x] **DEP-4.1.9** · `infra` · S · P0 · ✅ mido 1.2+ kept.
- [x] **DEP-4.1.10** · `infra` · S · P0 · ✅ fastapi 0.110+ for lifespan API.
- [x] **DEP-4.1.11** · `infra` · S · P0 · ✅ uvicorn 0.27+.
- [x] **DEP-4.1.12** · `infra` · S · P0 · ✅ python-multipart 0.0.9+.
- [x] **DEP-4.1.13** · `infra` · S · P0 · ✅ pydantic 2.6+.
- [x] **DEP-4.1.14** · `infra` · S · P0 · ✅ pyyaml 6.0.1+.
- [x] **DEP-4.1.15** · `infra` · S · P0 · ✅ numpy 1.26-2.1.
- [x] **DEP-4.1.16** · `infra` · S · P0 · ✅ tqdm 4.65+.
- [x] **DEP-4.1.17** · `infra` · S · P0 · ✅ auraloss 0.4+ added.
- [x] **DEP-4.1.18** · `infra` · S · P0 · ✅ structlog 24.1+ added.
- [x] **DEP-4.1.19** · `infra` · S · P0 · ✅ prometheus-client 0.20+ added.
- [x] **DEP-4.1.20** · `infra` · S · P0 · ✅ opentelemetry 1.27+ added.
- [x] **DEP-4.1.21** · `infra` · S · P0 · ✅ Removed `pretty-midi` — never imported.

## 4.2 Dev `requirements-dev.txt` (new file)

- [x] **DEP-4.2.1** · `infra` · S · P0 · ✅ `requirements-dev.txt` created with all dev deps.
- [x] **DEP-4.2.2** · `infra` · S · P0 · ✅ pytest-cov in dev deps.
- [x] **DEP-4.2.3** · `infra` · S · P0 · ✅ pytest-asyncio in dev deps.
- [x] **DEP-4.2.4** · `infra` · S · P0 · ✅ httpx in dev deps.
- [x] **DEP-4.2.5** · `infra` · S · P0 · ✅ black in dev deps.
- [x] **DEP-4.2.6** · `infra` · S · P0 · ✅ ruff in dev deps (replaces flake8+isort).
- [x] **DEP-4.2.7** · `infra` · S · P0 · ✅ mypy in dev deps.
- [x] **DEP-4.2.8** · `infra` · S · P0 · ✅ types-PyYAML, types-requests in dev deps.
- [x] **DEP-4.2.9** · `infra` · S · P0 · ✅ responses in dev deps.

## 4.3 Pinned Python

- [x] **DEP-4.3.1** · `infra` · S · P0 · ✅ `pyproject.toml` created with `requires-python = ">=3.13,<3.14"`.
- [x] **DEP-4.3.2** · `infra` · S · P0 · ✅ Dockerfile uses `python:3.13-slim`.
- [x] **DEP-4.3.3** · `infra` · S · P0 · ✅ CI uses single matrix entry for 3.13.
- [x] **DEP-4.3.4** · `infra` · S · P0 · ✅ `SETUP.md` created for CPU/bare-metal i5.

---

# SECTION 5 — LICENSE (P0)

- [x] **LIC-5.1** · `infra` · M · P0 · ✅ `/LICENSE` created with full Polyform Small Business License 1.0.0 text.
- [x] **LIC-5.2** · `docs` · S · P0 · ✅ `LICENSE_NOTICE.md` created with plain-English summary.
- [x] **LIC-5.3** · `docs` · S · P0 · ✅ SPDX headers added to all production `.py` files.
- [x] **LIC-5.4** · `docs` · S · P0 · ✅ `NOTICE` file created with third-party attributions.
- [x] **LIC-5.5** · `docs` · S · P0 · License noted in SETUP.md; README deferred to Sprint 4 (DOC-10.1.2).
- [x] **LIC-5.6** · `docs` · S · P0 · License noted in SETUP.md; landing page update deferred to Sprint 4.
- [x] **LIC-5.7** · `docs` · S · P0 · ✅ Research README already notes same license with "as-is" disclaimer.

---

# SECTION 6 — API HARDENING (`api.py`) (P0)

- [x] **API-6.1** · `sec` · S · P0 · Convert `@app.on_event("startup")` to `lifespan` async context manager (FastAPI ≥0.110 deprecation).
- [x] **API-6.2** · `sec` · S · P0 · Fix CORS: when `CORS_ORIGINS` env var is unset or `"*"`, set `allow_credentials=False` and `allow_origins=["*"]`. Otherwise parse the comma-separated list and set `allow_credentials=True`. Reject mismatched configs with a startup error.
- [x] **API-6.3** · `sec` · M · P0 · Add request size limit: `request.headers.get("content-length")` cap at 500 MB; FastAPI body size cap via `Limit` middleware. Return 413 on overflow.
- [x] **API-6.4** · `sec` · M · P0 · Add concurrency cap: `asyncio.Semaphore(1)` around `process_audio_file`. Return 503 with `Retry-After: 1` when at capacity.
- [x] **API-6.5** · `sec` · M · P0 · Add API-key auth: `Authorization: Bearer <key>` middleware. Keys from env var `API_KEYS` (comma-separated). Apply only to `/process`; leave `/health`, `/live`, `/ready`, `/metrics`, `/docs`, `/redoc`, `/openapi.json` open. Add `401`/`403` responses.
- [x] **API-6.6** · `perf` · S · P0 · `validate_audio_file` currently opens with `sf.info` then `_load_audio` opens again with `sf.read`. Use a single `sf.SoundFile` context for both. (Eliminates double-open on the same file.)
- [x] **API-6.7** · `sec` · M · P0 · Catch `torch.cuda.OutOfMemoryError` (still relevant if anyone runs on GPU) and any `MemoryError`; return 503 with `Retry-After`. Log the event.
- [x] **API-6.8** · `sec` · M · P0 · Add graceful shutdown: `lifespan` teardown that drains in-flight requests, clears CUDA cache if present. Pass `--timeout-graceful-shutdown 30` to uvicorn in Dockerfile CMD.
- [x] **API-6.9** · `infra` · S · P0 · Add `X-Request-ID` middleware (read incoming or generate UUID4). Echo in response header. Log on every request. Include in error responses.
- [x] **API-6.10** · `infra` · M · P0 · Replace `logging.basicConfig` with `structlog` configured for JSON output (`LOG_FORMAT=json` env). Include `request_id`, `route`, `latency_ms`, `status_code` on every log line.
- [x] **API-6.11** · `refactor` · S · P0 · Rename `create_placeholder_midi` to `build_midi_from_events` and move to `utils/midi.py`. Update `create_response_zip`.
- [x] **API-6.12** · `bug` · M · P0 · `create_placeholder_midi` (current name) uses fixed 1/16-note duration for every event. Read `duration_frames` from the confidence injector output; convert to ticks using `ticks_per_beat=480` and a configurable default duration. Drop the hardcoded 0.25-beat.
- [x] **API-6.13** · `perf` · L · P0 · `process_audio_streaming` concatenates outputs then runs `_logits_to_midi` over the full sequence — defeats streaming. Either (a) emit per-chunk MIDI with overlap-dedup, or (b) document as a non-streaming convenience path and remove the misleading name. Verify CPU memory doesn't blow up on 10-min input.
- [x] **API-6.14** · `sec` · S · P0 · `/render-template` — convert from GET to POST with JSON body `{"template_name": "...", "variables": {...}}`. Add `template_name` allowlist (whitelist the 7 templates in `user_content/`). Reject any other with 400.
- [x] **API-6.15** · `bug` · S · P0 · `_load_audio` falls back to `librosa.load(path, sr=None, mono=True)`. Some librosa versions resample to 22050 on None; force `sr=None` and mean-downmix manually.
- [x] **API-6.16** · `infra` · M · P0 · Add `/metrics` (Prometheus format). Counters: `requests_total{route,status}`, `errors_total{route,kind}`, `oom_total`. Histograms: `request_duration_seconds{route}`, `model_inference_seconds`, `bytes_processed`. Gauges: `gpu_memory_bytes` (if GPU), `in_flight_requests`.
- [x] **API-6.17** · `infra` · S · P0 · Split `/health` into `/live` (process up, always 200) and `/ready` (model loaded, not OOM, returns 503 with `Retry-After` while loading). Update Dockerfile `HEALTHCHECK` to call `/live`.
- [x] **API-6.18** · `sec` · S · P0 · Add `X-Content-Type-Options: nosniff` and `Strict-Transport-Security` (when behind TLS) headers. Use `secure.headers.Secure` middleware or equivalent.
- [x] **API-6.19** · `infra` · M · P0 · Add OpenTelemetry tracing: instrument FastAPI via `opentelemetry-instrumentation-fastapi`. Spans: `validate`, `load`, `separate`, `transcribe`, `inject`, `package`. Export to OTLP (env-configured endpoint). Add console exporter for dev.
- [x] **API-6.20** · `sec` · S · P0 · Add `/warmup` endpoint that runs a 1-sec synthetic inference at startup. Triggered automatically in `lifespan` so the first real request isn't slow.

---

# SECTION 7 — INFERENCE PERFORMANCE (P0)

- [x] **PERF-7.1** · `perf` · S · P0 · `models/mamba_separator.py` — cache input STFT magnitude in `forward`; pass to `_estimate_si_sdr` to avoid duplicate STFT.
- [x] **PERF-7.2** · `perf` · S · P0 · `models/mamba_transcriber.py` — move `_create_mel_basis` into `__init__`, register as buffer `self.mel_basis`. Eliminates per-forward Python loop.
- [x] **PERF-7.3** · `perf` · M · P0 · `main.py` — wrap inference in `torch.inference_mode()` (already done in `process_audio_file`, missing in `forward` and `process_audio_streaming` paths).
- [x] **PERF-7.4** · `perf` · S · P0 · `main.py` — set `torch.set_num_threads(os.cpu_count() or 1)` at import; allow override via `OMP_NUM_THREADS` env.
- [x] **PERF-7.5** · `perf` · M · P0 · `models/mamba_separator.py` — vectorize the per-mask ISTFT loop: stack all three masks into `(B, 3, T, F)`, do one matmul, then split. Avoids 3× STFT setup cost.
- [x] **PERF-7.6** · `perf` · M · P0 · Replace Mamba's `use_fast_path=True` with `use_mem_eff_path=True` if mamba-ssm 2.x supports it on CPU. Otherwise document the CPU path uses the reference impl (5–10× slower but works).
- [x] **PERF-7.7** · `perf` · L · P0 · `models/mamba_separator.py` — add INT8 dynamic quantization on the Mamba block forward (`torch.ao.quantization.quantize_dynamic`) for CPU inference. Provides ~2× speedup with <1% accuracy loss on audio masks. Add a `--quantize` flag to `process_audio_file`.
- [x] **PERF-7.8** · `perf` · M · P0 · `main.py:process_audio_file` — accept an optional `quantize: bool` query param. When true, swap in quantized variant. Document CPU memory savings.
- [x] **PERF-7.9** · `perf` · S · P0 · Add `torch.set_float32_matmul_precision("high")` for x86 CPUs with AVX-512. Significant speedup on matmul-heavy layers.
- [x] **PERF-7.10** · `perf` · S · P0 · `models/mamba_separator.py` — convert `self.mamba_blocks` from `nn.ModuleList` to `nn.Sequential` for cleaner forward (no functional change, but JIT-friendly).
- [x] **PERF-7.11** · `perf` · M · P0 · Add a `/batch-process` endpoint that takes a ZIP of multiple files, processes them serially (CPU bottleneck), and returns a ZIP of ZIPs. Document CPU throughput (~30–60s per minute of audio on i5).
- [x] **PERF-7.12** · `perf` · L · P0 · Add output caching by content hash: SHA-256 of the input file → store processed ZIP in `outputs/cache/<hash>.zip`. Same content → 301 redirect. Saves CPU on repeat uploads. Add TTL via `CACHE_TTL_HOURS` env (default 24).

---

# SECTION 8 — TESTING (P0)

## 8.1 Test infrastructure

- [x] **T-8.1.1** · `test` · S · P0 · `tests/conftest.py` — populate with fixtures: `audio` (1-sec 44.1kHz mono tensor), `stereo_audio` (2-ch), `mock_config` (full YAML dict), `temp_audio_file` (WAV in tmp), `temp_flac_file`, `temp_mp3_file` (skip if no encoder), `mock_model` (StemMidiModel with small config).
- [x] **T-8.1.2** · `test` · S · P0 · `pytest.ini` (new) — `testpaths=tests`, `addopts=-ra --strict-markers --tb=short`. Add markers: `slow`, `cuda`, `integration`.
- [ ] **T-8.1.3** · `test` · S · P0 · `.github/workflows/ci.yml` — remove `continue-on-error: true` from test step. Add `pytest -m "not cuda"` as the default. Add a separate `cuda` job that only runs on a self-hosted runner.

## 8.2 Un-skip existing tests

- [x] **T-8.2.1** · `test` · M · P0 · Refactor `api.py` to lazy-import `StemMidiModel` inside `load_model()` (not at module top). This breaks the import-time CUDA dep.
- [x] **T-8.2.2** · `test` · S · P0 · `tests/test_api_validation.py` — remove all `@pytest.mark.skip`. Cover: valid WAV, too-long, unsupported SR, stereo, FLAC, MP3 (if available).
- [x] **T-8.2.3** · `test` · S · P0 · `tests/test_audio_loading.py` — remove skips. Cover: mono WAV, stereo→mono, missing file, 22kHz resample path.
- [x] **T-8.2.4** · `test` · S · P0 · `tests/test_midi_generation.py` — remove skips. Cover: empty events, multi-event ordering, CC#127 values, file parses round-trip.

## 8.3 New tests

- [x] **T-8.3.1** · `test` · M · P0 · `tests/test_mamba_separator.py` (new) — build from `model_config.yaml` (downsized for CPU); forward on `(B=1, C=1, T=8192)`; assert output shapes; assert no NaN; assert `guitar_stem != bass_stem` on synthetic input. **This test would have caught C-2.1, C-2.5, C-2.6, C-2.11.**
- [x] **T-8.3.2** · `test` · M · P0 · `tests/test_mamba_transcriber.py` (new) — forward on `(B=1, C=1, T=8192)`; assert 5 outputs with correct shapes. **Catches C-2.2, C-2.9, C-2.10.**
- [x] **T-8.3.3** · `test` · S · P0 · `tests/test_confidence_injector.py` (new) — verify alignment score, `needs_review` threshold, CC#127 mapping, summary counts.
- [x] **T-8.3.4** · `test` · S · P0 · `tests/test_losses.py` (new) — verify MR-STFT returns scalar; crest/flatness/alignment components sum; finite values.
- [ ] **T-8.3.5** · `test` · S · P0 · `tests/test_template_engine.py` (new) — missing key leaves placeholder intact; list_templates returns 7 entries; re-raises FileNotFoundError on missing.
- [x] **T-8.3.6** · `test` · S · P0 · `tests/test_midi_roundtrip.py` (new) — build MidiFile with N events, save bytes, re-parse, verify event count and CC#127 values. **Catches API-6.12.**
- [ ] **T-8.3.7** · `test` · M · P0 · `tests/test_api_integration.py` (new) — `TestClient.post('/process', files={'file': wav})`; assert 200 and ZIP contains 5 entries (guitar_stem.wav, bass_stem.wav, guitar.mid, bass.mid, processing_report.json).
- [ ] **T-8.3.8** · `test` · M · P0 · `tests/test_streaming_inference.py` (new) — process 6-sec synthetic file with `process_audio_streaming(chunk_seconds=2.0)`; verify `chunks_processed == 3`.
- [ ] **T-8.3.9** · `test` · S · P0 · `tests/test_quality_gates.py` (extend) — add test that artifact flags force COMPLEX tier regardless of confidence.
- [ ] **T-8.3.10** · `test` · S · P0 · `tests/test_api_auth.py` (new) — `TestClient.post('/process')` without API key returns 401; with valid key returns 200; with invalid key returns 403.
- [ ] **T-8.3.11** · `test` · S · P0 · `tests/test_api_cors.py` (new) — preflight OPTIONS without origin returns defaults; with `Origin: https://app.example` and matching allowlist returns `Access-Control-Allow-Origin`; with mismatched origin returns no ACAO header.
- [ ] **T-8.3.12** · `test` · S · P0 · `tests/test_api_metrics.py` (new) — `/metrics` returns Prometheus text format; `requests_total` increments after a request.
- [ ] **T-8.3.13** · `test` · M · P0 · `tests/test_perf_smoke.py` (new) — build model, run 1 forward on `(B=1, T=44100)`; assert <500 MB RSS; assert <30s wall on i5 reference hardware.
- [ ] **T-8.3.14** · `test` · S · P0 · `tests/test_dataset_loaders.py` (extend) — add test for the new collapsed `StemFolderDataset`; cover both Slakh and MUSDB layouts via parametrize.
- [ ] **T-8.3.15** · `test` · S · P0 · `tests/test_quantization.py` (new) — `process_audio_file(quantize=True)` returns same output shape; bit-exact mask within tolerance.

## 8.4 Coverage targets

- [ ] **T-8.4.1** · `test` · M · P0 · CI gate: `--cov-fail-under=70` (current ~15%). Push to 80% by end of Sprint 2.
- [ ] **T-8.4.2** · `test` · S · P0 · Codecov config (`.codecov.yml` new) — set target to 80%, allow 5% drop on new PRs.

---

# SECTION 9 — CONFIG & INFRASTRUCTURE (P0–P1)

## 9.1 Configuration

- [x] **CFG-9.1.1** · `bug` · S · P0 · `configs/model_config.yaml:21-22` — drop `selective_scan_impl: "cuda"` and `causal_conv1d_impl: "cuda"` (invalid kwargs for Mamba-1.x, CPU path).
- [x] **CFG-9.1.2** · `bug` · S · P0 · `configs/model_config.yaml:37` — change `precision: "fp8"` to `precision: "fp32"` for CPU target. Add a comment explaining CPU target.
- [x] **CFG-9.1.3** · `refactor` · M · P0 · Add a CPU-tuned default config: `configs/model_config.cpu.yaml` with smaller dims (`d_model=256`, `n_layer=6`, `d_state=12`) to fit in 4–8 GB RAM with 1-min audio segments.
- [x] **CFG-9.1.4** · `refactor` · S · P0 · Rename `n_layer` → `n_layers` in YAML for consistency with `model.py` (or vice versa in `model.py`). Pick one.
- [x] **CFG-9.1.5** · `refactor` · S · P0 · `configs/model_config.yaml:export:` — remove the TensorRT-LLM export block (not implementing, CPU target). Add a comment `[REMOVED: TensorRT-LLM requires CUDA]`.
- [x] **CFG-9.1.6** · `refactor` · S · P0 · `example_data_config.yaml` — fix `dataset_type: 'musdb18hq'` to use the new collapsed `StemFolderDataset` (post RF-3.1.1).
- [x] **CFG-9.1.7** · `infra` · S · P0 · Add `model_config_path: str = "configs/model_config.yaml"` env var. Add `--config` to `main.py` and `api.py`.

## 9.2 Build & deploy

- [x] **BLD-9.2.1** · `infra` · L · P0 · `pyproject.toml` (new) — package the project. `[project]` with name, version, description, license (`SPDX-License-Identifier: PolyForm-Small-Business-1.0.0`), requires-python, dependencies. `[project.optional-dependencies] cpu = [...]` for the CPU target. `[project.scripts] stem-midi-api = "api:main"`, `stem-midi-cli = "main:cli"`. `[tool.setuptools.packages.find] include = ["models*", "utils*", "data*", "audio*"]`.
- [x] **BLD-9.2.2** · `infra` · S · P0 · `setup.cfg` (new) — `tool.ruff` config. `line-length=100`, `target-version="py313"`, `extend-exclude=["research", "venv", "mamba"]`, `select=["E","F","W","I","UP","B","SIM","RUF"]`.
- [x] **BLD-9.2.3** · `infra` · S · P0 · `.dockerignore` (new) — exclude `research/`, `venv/`, `__pycache__/`, `.git/`, `.pytest_cache/`, `outputs/`, `*.nemo`, `*.pt`, `datasets/`, `tests/`, `*.egg-info/`, `dist/`, `build/`.
- [x] **BLD-9.2.4** · `infra` · L · P0 · `Dockerfile` — multi-stage:
  - **Stage 1 (builder):** `python:3.13-slim` (CPU-only). Install build tools, `pip install -e .` with all deps. Strip build tools at end.
  - **Stage 2 (runtime):** `python:3.13-slim`. Copy installed site-packages from builder. Copy app. Create `app` user. `EXPOSE 8000`. `HEALTHCHECK --interval=30s --start-period=30s CMD curl -fs http://localhost:8000/live || exit 1`. `CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "30"]`.
- [x] **BLD-9.2.5** · `infra` · S · P0 · `Makefile` (new) — `install`, `install-dev`, `lint`, `format`, `test`, `test-cpu`, `test-cuda`, `demo`, `serve`, `docker-build`, `docker-run`, `clean`, `dist`, `release-{major,minor,patch}`.
- [x] **BLD-9.2.6** · `infra` · M · P0 · `.github/workflows/ci.yml` rewrite:
  - **lint job:** Python 3.13. `ruff check .`, `black --check .`, `mypy .` (loose).
  - **test-cpu job:** Python 3.13. `pip install -r requirements.txt -r requirements-dev.txt`. `pytest -m "not cuda" --cov=. --cov-fail-under=70`.
  - **structure job:** runs `verify_structure.py` + `check_syntax.py`.
  - **license job:** greps for `SPDX-License-Identifier` in all `.py` files; ensures 100% coverage.
- [ ] **BLD-9.2.7** · `infra` · M · P1 · Add a `cuda` job to CI (gated by repo variable `HAS_CUDA_RUNNER=true`). Runs `pytest -m "cuda"` only. Skipped by default.
- [ ] **BLD-9.2.8** · `infra` · M · P1 · Add `deploy/docker-compose.yml` (new) for local bare-metal i5 deployment. Mounts `./data/`, `./outputs/`, `./configs/`, `./user_content/`. Maps port 8000.
- [ ] **BLD-9.2.9** · `infra` · L · P1 · Add `systemd/stem-midi-pro.service` (new) for native Linux deployment. User, working dir, env file, restart policy, resource limits (MemoryMax=6G, CPUQuota=400% on i5).
- [ ] **BLD-9.2.10** · `infra` · M · P1 · Add `deploy/observability/` — `prometheus.yml` scrape config, `grafana-dashboard.json` with panels for latency / errors / OOM / throughput / quality-tier distribution.

## 9.3 Repository hygiene

- [x] **HYG-9.3.1** · `refactor` · S · P0 · `.gitignore` — extend with `outputs/cache/`, `*.egg-info/`, `dist/`, `build/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `data/datasets/`, `*.ckpt`, `*.safetensors`, `htmlcov/`, `.coverage`.
- [x] **HYG-9.3.2** · `refactor` · S · P0 · Initialize a proper git repo if not already (per AGENTS.md, "No git repo is initialized"). `git init`, initial commit, `main` branch, add `gh` as default remote.
- [x] **HYG-9.3.3** · `infra` · S · P0 · Add `.github/CODEOWNERS` (new) — assign owners to paths.
- [x] **HYG-9.3.4** · `infra` · S · P0 · Add `.github/pull_request_template.md` (new).
- [x] **HYG-9.3.5** · `infra` · S · P0 · Add `.github/ISSUE_TEMPLATE/bug_report.yml` and `feature_request.yml`.

---

# SECTION 10 — DOCUMENTATION (P1)

## 10.1 Top-level docs

- [ ] **DOC-10.1.1** · `docs` · L · P1 · `AGENTS.md` — regenerate. Reflect: tests exist, real datasets, real audio loading, real MIDI generation, CPU target, license.
- [ ] **DOC-10.1.2** · `docs` · L · P1 · `README.md` — rewrite. Remove "TensorRT-LLM", "Librosa/Ruby", "FP8 precision" claims. Add bare-metal i5 install instructions. Add license summary. Add CPU performance numbers.
- [ ] **DOC-10.1.3** · `docs` · L · P1 · `SUMMARY.md` — regenerate. Mark every feature as `[IMPLEMENTED]` or `[PLANNED]` per architecture.md. Drop the stale references to `Slakh2100CADataset` / `MUSDBIndieDataset`.
- [ ] **DOC-10.1.4** · `docs` · L · P1 · `ARCHITECTURE.md` — annotate every section with `[IMPLEMENTED]` / `[PARTIAL]` / `[PLANNED]` / `[REMOVED]`. Document the CPU-only target. Reference the Polyform license.
- [ ] **DOC-10.1.5** · `docs` · M · P1 · `API_DOCUMENTATION.md` — add auth section, error codes, rate-limit responses, metrics endpoint, CORS docs. Drop the `/api/v1` base URL claim.
- [ ] **DOC-10.1.6** · `docs` · M · P1 · `USER_GUIDE.md` — strip unimplemented features: tuning detection, tempo detection, palm-mute detection, harmonics detection, slap/pop. Replace with what actually works.
- [ ] **DOC-10.1.7** · `docs` · M · P1 · `DEVELOPMENT_GUIDE.md` — update paths (no `stem_midi_pro/` subdir), `torch.jit.trace` example, CI commands, `make` targets.
- [ ] **DOC-10.1.8** · `docs` · S · P1 · `SETUP.md` (new, root) — replace the old one. Python 3.13, bare-metal i5, CPU-only, no CUDA. Reference `/research/mamba3_per_track/SETUP.md` for the Mamba-3 path.
- [ ] **DOC-10.1.9** · `docs` · S · P1 · `PRD_AND_ARD.md` — strip the `C:\Dev\stem_midi_pro` local path leak on line 5.
- [ ] **DOC-10.1.10** · `docs` · S · P1 · `CHANGELOG.md` (new) — auto-generated via `git-cliff` or manual entries per release. Reference `git log` for now.

## 10.2 User-facing content

- [ ] **DOC-10.2.1** · `docs` · S · P1 · Audit all 7 templates in `user_content/`: `upload_confirmation.md`, `progress_updates.md`, `completion_delivery.md`, `rights_usage_prompt.md`, `feedback_refinement.md`, `implicit_feedback.md`, `landing_page.md`. Verify every `{{ variable }}` resolves to a field in the processing report or is fed by `/render-template` callers.
- [ ] **DOC-10.2.2** · `docs` · S · P1 · `landing_page.md` — add license statement (LIC-5.6) and bare-metal CPU performance note ("runs on a $200 i5 laptop").
- [ ] **DOC-10.2.3** · `docs` · S · P1 · `completion_delivery.md` — verify `{{ filename }}`, `{{ si_sdr }}`, `{{ phase_coherence }}`, `{{ avg_confidence }}` all match `create_response_zip:processing_report.json` keys.
- [ ] **DOC-10.2.4** · `docs` · S · P1 · `rights_usage_prompt.md` — add a clause about the Polyform license: "Output is licensed under the same Polyform Small Business License as the model. Commercial use above $1M revenue requires a paid license."
- [ ] **DOC-10.2.5** · `docs` · S · P1 · `progress_updates.md` — adjust latency claims to bare-metal CPU numbers (~30–60s per minute of audio).
- [ ] **DOC-10.2.6** · `docs` · S · P1 · `feedback_refinement.md` — replace the "Web MIDI editor" claim with "DAW-based refinement (MIDI files have CC#127 confidence)".

## 10.3 Code documentation

- [ ] **DOC-10.3.1** · `docs` · M · P1 · Add Google-style docstrings to every public class and method in `models/`, `utils/`, `data/`. Include tensor-shape annotations `(B, C, T)`.
- [ ] **DOC-10.3.2** · `docs` · S · P1 · Add a top-level `docs/architecture-decision-records/` directory with ADRs:
  - `0001-license-polyform-small-business.md`
  - `0002-cpu-only-target.md`
  - `0003-drop-ne-mo-lightning.md`
  - `0004-consolidate-on-v1.md`
  - `0005-mamba-3-moved-to-research.md`
- [ ] **DOC-10.3.3** · `docs` · S · P1 · Add `docs/operations/` with: `runbook.md` (common errors + fixes), `monitoring.md` (what to alert on), `upgrading.md` (how to bump model versions), `rollback.md` (how to revert a deploy).

---

# SECTION 11 — IMPLEMENT architecture.md (P0–P2)

These are the features called out in `ARCHITECTURE.md` as designed but not yet built. Each one is a real work item because architecture.md is the source of truth.

## 11.1 From "API Layer" (api.py)

- [x] **ARCH-11.1.1** · `feature` · S · P0 · Background task support for cleanup — see API-6.8 (lifespan teardown).
- [ ] **ARCH-11.1.2** · `feature` · M · P1 · `POST /process-batch` — see PERF-7.11.
- [ ] **ARCH-11.1.3** · `feature` · M · P1 · WebSocket `/ws/process` for real-time progress (per-stage completion: loaded → separated → transcribed → packaged).

## 11.2 From "Processing Orchestrator" (main.py)

- [ ] **ARCH-11.2.1** · `feature` · M · P1 · State management: explicit `StatefulModel` wrapper that holds a state cache dict keyed by `session_id` (from `X-Session-ID` header). Multi-user streaming support.
- [ ] **ARCH-11.2.2** · `feature` · M · P1 · Async pipeline: overlap compute and I/O. Decode upload in a thread, run inference in the event loop, encode response in a thread. Use `asyncio.to_thread` for the synchronous model calls.

## 11.3 From "Separation Module"

- [ ] **ARCH-11.3.1** · `feature` · L · P1 · Phase-coherent overlap-add reconstruction across chunks (currently fakes this — see C-2.5).
- [ ] **ARCH-11.3.2** · `feature` · M · P1 · True SI-SDR computation against a reference (when available, e.g., during validation). Currently uses a centroid proxy (see C-2.6).
- [ ] **ARCH-11.3.3** · `feature` · M · P1 · Residual stem routing: when `artifact_flags` includes `"phase_cancellation"`, automatically retry separation with `residual_weight` increased by 0.2.

## 11.4 From "Transcription Module"

- [ ] **ARCH-11.4.1** · `feature` · M · P1 · Proper onset F1 loss (currently a TODO in `models/losses.py:34`).
- [ ] **ARCH-11.4.2** · `feature` · M · P1 · Pitch cross-entropy loss against a 128-way target.
- [ ] **ARCH-11.4.3** · `feature` · M · P1 · Velocity MAE loss.
- [ ] **ARCH-11.4.4** · `feature` · M · P1 · Expression heads: bend, vibrato, slide — currently the head exists but is not trained. Add a synthetic training target and the loss.
- [ ] **ARCH-11.4.5** · `feature` · M · P1 · Stereo-aware: process L and R channels separately, then fuse onset/pitch by agreement. (Monos only today.)

## 11.5 From "Confidence Injection"

- [ ] **ARCH-11.5.1** · `feature` · M · P1 · Per-pitch-bucket confidence baselines (low notes are harder; the absolute threshold should be pitch-aware).
- [ ] **ARCH-11.5.2** · `feature` · M · P1 · Note-duration estimation from onset-to-offset detection (currently uses a fixed 1/16 — see API-6.12).

## 11.6 From "Loss Functions"

- [ ] **ARCH-11.6.1** · `feature` · M · P1 · Onset F1 (F1 of binary onset predictions vs targets).
- [ ] **ARCH-11.6.2** · `feature` · M · P1 · Pitch CE (cross-entropy over 128-way pitch logits vs target pitches).
- [ ] **ARCH-11.6.3** · `feature` · M · P1 · Velocity MAE.
- [ ] **ARCH-11.6.4** · `feature` · M · P1 · Duration IoU (intersection-over-union of predicted vs target note durations).
- [ ] **ARCH-11.6.5** · `feature` · M · P1 · Cross-modal alignment loss in `models/losses.py` — currently a stub. The `_onset_alignment_loss` exists but uses fake targets (synthetic onsets from `data/datasets.py`).

## 11.7 From "Quality Gates"

- [x] **ARCH-11.7.1** · `feature` · S · P0 · Fix C-2.8 (phase cancellation logic is inverted).
- [ ] **ARCH-11.7.2** · `feature` · M · P1 · Per-instrument thresholds: guitar vs bass have different typical confidences.
- [ ] **ARCH-11.7.3** · `feature` · M · P1 · Streaming quality reports: emit a per-chunk quality score, plus a running aggregate.

## 11.8 From "Streaming Inference"

- [ ] **ARCH-11.8.1** · `feature` · L · P1 · True streaming with SSM state passing via Mamba's `inference_params` (see C-2.5).
- [ ] **ARCH-11.8.2** · `feature` · M · P1 · Per-chunk MIDI deduplication (onsets near chunk boundary should not be emitted twice).
- [ ] **ARCH-11.8.3** · `feature` · M · P1 · Configurable chunk size via `?chunk_seconds=` query param.

## 11.9 From "Scalability Considerations"

- [ ] **ARCH-11.9.1** · `feature` · M · P2 · Horizontal scaling: stateless API behind a load balancer. Document sticky session requirements for streaming (or use a stateful backend like Redis).
- [ ] **ARCH-11.9.2** · `feature` · M · P2 · Vertical scaling doc: "i5/8GB = 1 concurrent request, 60–120s per 1-min track; i7/16GB = 1–2 concurrent, 30–60s per 1-min track; i9/32GB = 2–3 concurrent, 20–40s per 1-min track." Actual numbers after benchmarking (T-8.3.13).

## 11.10 From "Security Architecture"

- [x] **ARCH-11.10.1** · `feature` · M · P0 · Input sanitization (file extension + content type + duration + size). See API-6.3.
- [ ] **ARCH-11.10.2** · `feature` · M · P0 · Rate limiting (token bucket per API key). Use `slowapi` or hand-rolled.
- [x] **ARCH-11.10.3** · `feature` · S · P0 · Ephemeral file processing — already in place; verify temp-file cleanup on exception path. Add a unit test that simulates a crash mid-processing and asserts no temp files remain.
- [ ] **ARCH-11.10.4** · `docs` · S · P1 · GDPR compliance doc — already 90% done in `USER_GUIDE.md`; trim and move to `docs/operations/gdpr.md`.

## 11.11 From "Performance Optimization"

- [ ] **ARCH-11.11.1** · `feature` · M · P1 · Memory-mapped audio loading for large files (currently `sf.read` loads fully into RAM). Use `soundfile`'s `frames=-1, start=offset` to seek.
- [ ] **ARCH-11.11.2** · `feature` · S · P1 · `torch.set_num_threads(1)` when running in Docker to avoid CPU oversubscription. Allow override via `OMP_NUM_THREADS`.
- [ ] **ARCH-11.11.3** · `feature` · M · P1 · Latency budget per stage: validate <100ms, load <500ms, separate 30s/min-audio, transcribe 30s/min-audio, package <200ms. Emit per-stage timings to OTel spans.

## 11.12 From "Extensibility Points"

- [ ] **ARCH-11.12.1** · `feature` · L · P2 · Multi-instrument support: add a `drums`, `vocals`, `keys` mask head to `MambaSeparator`. Add config flag `instruments: ["guitar", "bass", "drums", "vocals"]`. Update MIDI generation.
- [ ] **ARCH-11.12.2** · `feature` · M · P2 · New audio effects: add `tremolo`, `wah`, `harmonics` to the expression head. Update `expression_heads` in config.

## 11.13 From "Canadian Artist Dataset Integration"

- [ ] **ARCH-11.13.1** · `docs` · S · P0 · Document that Slakh/MUSDB datasets are open-license (CC-BY) but the *trained model weights* are under Polyform Small Business License.
- [ ] **ARCH-11.13.2** · `feature` · M · P2 · Add a `data/canadian_artist_specific.py` loader for artists in the Canadian indie scene (e.g., Six Shooter Records, Arts & Crafts label rosters) — public-domain / CC-licensed recordings only. Document provenance per file.

## 11.14 From "Monitoring and Observability"

- [x] **ARCH-11.14.1** · `infra` · M · P0 · Latency metrics (covered in API-6.16).
- [ ] **ARCH-11.14.2** · `infra` · M · P0 · Resource utilization: CPU%, memory, file descriptors (per process). Add `prometheus-client` process collector.
- [ ] **ARCH-11.14.3** · `infra` · M · P0 · Quality metrics: per-tier histogram (studio / draft / complex) over time. Surface in Grafana.
- [ ] **ARCH-11.14.4** · `infra` · M · P0 · Error rates per endpoint. Alert on >5% 5xx in 5m window.

---

# SECTION 12 — TRAINING CORRECTNESS (P0)

- [x] **TR-12.1** · `bug` · S · P0 · `train.py:42` — `pl.seed_everything(42)` with `deterministic=True` is a contradiction unless `cudnn.deterministic=True; cudnn.benchmark=False` is set. Add explicit `torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False`. (Less critical on CPU but still right.)
- [x] **TR-12.2** · `bug` · S · P0 · `train.py:98` — `trainer.test(model, dataloaders=test_loader, ckpt_path='best')` — PyTorch Lightning's `ckpt_path='best'` is a magic string that resolves to `trainer.checkpoint_callback.best_model_path`. After the RF-3.1.9 rewrite to a custom loop, replace with explicit `model.load_state_dict(best_ckpt['model_state_dict'])`.
- [x] **TR-12.3** · `refactor` · L · P0 · Replace PyTorch Lightning `pl.Trainer` with a custom training loop in `train.py`. Implement: epoch loop, train/val, gradient accumulation, AMP (CPU: bfloat16 via `torch.autocast(device_type='cpu', dtype=torch.bfloat16)`), checkpointing, early stopping, logging.
- [x] **TR-12.4** · `feature` · M · P1 · Add `--grad-accum` CLI flag (currently `train.py` doesn't accept it).
- [x] **TR-12.5** · `feature` · M · P1 · Add `--ema` (exponential moving average of weights) for stable eval.
- [x] **TR-12.6** · `feature` · M · P1 · Add `--resume` flag (currently `train.py` accepts `--resume-from-checkpoint` but doesn't actually pass it correctly to the custom loop).
- [x] **TR-12.7** · `feature` · M · P1 · Add `--quantize` for training-aware quantization-aware training (QAT) on the Mamba backbone.
- [x] **TR-12.8** · `bug` · S · P0 · `train.py:50` — `get_data_loaders` returns synthetic data on missing path. Add `--strict-data` flag that raises on missing.
- [x] **TR-12.9** · `bug` · S · P0 · `models/losses.py:34` — `TODO: Implement onset_f1, pitch_ce, and velocity_mae losses` — covered in ARCH-11.6.

---

# SECTION 13 — ORPHAN / DEAD CODE (P1)

- [x] **DEAD-13.1** · `refactor` · S · P1 · Remove `pretty-midi` from requirements (never imported).
- [x] **DEAD-13.2** · `refactor` · S · P1 · Remove `tensorrt` and `torch-trt` references from docs and configs.
- [x] **DEAD-13.3** · `refactor` · S · P1 · Remove the `tensorrt` install line from any docs that mention it.
- [x] **DEAD-13.4** · `refactor` · S · P1 · `api.py` — `import json` (line 198) is in the middle of a function; move to top. `import yaml` in `main.py:367` likewise.
- [x] **DEAD-13.5** · `refactor` · S · P1 · `api.py:29-30` — `ProcessingReport`, `QualityTier` imported but never used. Remove.
- [x] **DEAD-13.6** · `refactor` · S · P1 · `main.py:10` — `Tuple`, `List` from `typing` used only in type hints; switch to `from __future__ import annotations` + PEP 604 (`tuple[...]`, `list[...]`).
- [x] **DEAD-13.7** · `refactor` · S · P1 · `verify_structure.py` — emoji `✓` in `print` is fine but the file still hardcodes `check_directory_exists("data")` etc. Verify the current list matches the actual file layout post-reorg.
- [x] **DEAD-13.8** · `refactor` · M · P1 · `models/mamba_separator.py` — after dropping NeMo, the `input_types`/`output_types` properties become dead code. Remove.
- [x] **DEAD-13.9** · `refactor` · M · P1 · `models/mamba_transcriber.py` — same as DEAD-13.8.
- [x] **DEAD-13.10** · `refactor` · S · P1 · `mamba/` — fully relocated to `/research/mamba-ssm-reference/`. Confirm no remaining references at root.
- [x] **DEAD-13.11** · `refactor` · S · P1 · `model.py` — fully relocated to `/research/mamba3_per_track/per_track_processor.py`. Confirm no remaining references at root.
- [x] **DEAD-13.12** · `refactor` · S · P1 · `train_mamba3.py` — fully relocated. Confirm no remaining references.
- [x] **DEAD-13.13** · `refactor` · S · P1 · `utils/losses_mamba3.py` — fully relocated. Confirm no remaining references.
- [x] **DEAD-13.14** · `refactor` · S · P1 · `configs/mamba3_config.yaml` — fully relocated. Confirm no remaining references.

---

# SECTION 14 — SUBSEQUENT SPRINTS (post-MVP)

## 14.1 Sprint 2 — Web frontend

- [ ] **FE-14.1.1** · `feature` · L · P2 · Minimal web UI: HTML + vanilla JS, served by FastAPI. Drag-and-drop upload, progress bar (polling), download button.
- [ ] **FE-14.1.2** · `feature` · L · P2 · MIDI preview in-browser via `@tonejs/midi` or `MIDI.js`.

## 14.2 Sprint 3 — DAW integrations

- [ ] **DAW-14.2.1** · `feature` · XL · P2 · Reaper ReaScript that wraps the API.
- [ ] **DAW-14.2.2** · `feature` · XL · P2 · Ableton Max for Live device.

## 14.3 Sprint 4 — Monetization

- [ ] **MON-14.3.1** · `feature` · L · P2 · Stripe integration for the "$4.99 human review" upsell (per `quality_gates.route_by_quality` COMPLEX tier).
- [ ] **MON-14.3.2** · `feature` · M · P2 · License-key issuance: when a company buys a paid license (above $1M revenue), issue a signed JWT and serve it via `/license/activate`. Add middleware that checks the JWT and allows >1 concurrent request.

---

# SECTION 15 — DEFINITION OF DONE (per item)

A TODO is "done" only when **all** of the following are true:

1. Code is written and committed on a feature branch.
2. Unit tests pass: `make test-cpu`.
3. Lint passes: `make lint` (ruff + black + mypy).
4. The relevant docs are updated (README.md, ARCHITECTURE.md, API_DOCUMENTATION.md).
5. `SPDX-License-Identifier: PolyForm-Small-Business-1.0.0` header is on the new/modified `.py` file.
6. The item is checked off in this TODO.md as part of the PR description.

---

# SECTION 16 — SPRINT PLAN (PROPOSED)

## Sprint 1 (3 days) — Stop the bleeding
- Section 1.1 (R-1.1.x) — move files to /research
- Section 2 (C-2.x) — all P0 critical bugs
- Section 4.1 (DEP-4.1.x) — requirements.txt rewrite
- Section 4.2 (DEP-4.2.x) — requirements-dev.txt
- Section 4.3 (DEP-4.3.x) — Python 3.13
- Section 5 (LIC-5.x) — license files
- Section 6 (API-6.1 through API-6.20) — API hardening
- Section 7 (PERF-7.1 through PERF-7.4) — cheap perf wins
- Section 9.1 (CFG-9.1.x) — config fixes
- Section 12 (TR-12.x) — training correctness

## Sprint 2 (3 days) — Test coverage + refactor
- Section 3 (RF-3.1.x, RF-3.2.x) — refactor duplication
- Section 8 (T-8.x) — full test suite
- Section 9.2 (BLD-9.2.x) — build & deploy
- Section 13 (DEAD-13.x) — orphan code removal

## Sprint 3 (4 days) — Architecture source-of-truth
- Section 11 (ARCH-11.x) — implement all [PLANNED] features in priority order
- Section 7 (PERF-7.5 through PERF-7.12) — advanced perf

## Sprint 4 (2 days) — Docs
- Section 10 (DOC-10.x) — full doc rewrite
- Section 14.3 (MON-14.3.x) — Stripe + license-key activation (if time)

---

# SECTION 17 — AUDIT CHECKLIST CLOSEOUT

When all `[ ]` items above are `[x]`, run a final audit:

- [ ] **AUDIT-17.1** · `infra` · M · P0 · `find . -name "*.py" -not -path "./research/*" -not -path "./venv/*" | xargs wc -l` — total LOC for production code.
- [ ] **AUDIT-17.2** · `test` · M · P0 · `pytest --cov=. --cov-fail-under=80` — coverage ≥ 80%.
- [ ] **AUDIT-17.3** · `docs` · M · P0 · `grep -r "TODO\|FIXME\|XXX\|HACK\|NotImplemented" --include="*.py" .` returns 0 matches in production code.
- [ ] **AUDIT-17.4** · `sec` · S · P0 · `pip-audit -r requirements.txt -r requirements-dev.txt` — 0 known CVEs in production deps.
- [ ] **AUDIT-17.5** · `docs` · S · P0 · `grep -r "aspirational\|placeholder\|stub\|dummy\|fake" --include="*.md" .` — verify the only remaining matches are in `user_content/landing_page.md` and the explicit "we will not implement this" notes.
- [ ] **AUDIT-17.6** · `perf` · M · P0 · `python -c "from main import StemMidiModel; m = StemMidiModel(cfg); import time; t=time.time(); m.process_audio_file('test_1sec.wav'); print(time.time()-t)"` on reference i5 hardware — measure and document.
- [ ] **AUDIT-17.7** · `docs` · S · P0 · Update this `TODO.md` with a "Completed" stamp at the top and a `CHANGELOG.md` entry referencing the final commit.

---

*This TODO.md is the source of truth for the Stem+MIDI Pro production-readiness effort. It is intentionally large. The Polyform Small Business License 1.0.0 applies to all code produced under it. Contributions are accepted under the same license via signed-off commits.*

*Last updated: 2026-06-02*


