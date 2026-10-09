# Stem+MIDI Pro — TODO

<!-- STATUS: research -->

> **Status note (2026-10-08):** This historical plan has been partially
> reconciled against the current checkout, but is not a complete or approved
> roadmap. Some completed boxes conflict with checked-in code (for example,
> CPU-only/Python 3.13 decisions and the claimed `ci.yml`), and some proposed
> PolyForm licensing conflicts with the current Apache-2.0 `LICENSE`. Treat
> unresolved items as research until maintainers confirm scope; do not apply
> conflicting license tasks.

> **Historical plan date:** 2026-06-02; status reconciliation: 2026-10-08
> **Current architecture notes:** `ARCHITECTURE.md` describes the checked-in prototype, not a roadmap or completed specification.
> **Target hardware:** unverified; the model configuration and Docker image include CUDA/NVIDIA assumptions.
> **Python:** no supported version is declared or tested in this audit environment.
> **License:** current root `LICENSE` is Apache License 2.0.
> **Repo structure:** root-level modules are the current entrypoints; the duplicate `stem_midi_pro/` tree remains pending a maintainer decision.

---

## Legend

- **Status:** `[ ]` open · `[x]` done · `[~]` in progress · `[!]` blocked or awaiting research/owner decision
- **Effort:** S < 1h · M 1–4h · L 1–3 days · XL 1+ week
- **Priority:** P0 (blocks prod) · P1 (next sprint) · P2 (post-MVP) · P3 (research)
- **Tags:** `bug`, `feature`, `perf`, `sec`, `test`, `docs`, `infra`, `refactor`, `arch`

---

# SECTION 0 — HISTORICAL STRATEGIC DECISIONS

<!-- STATUS: research -->

Only the checked-in Apache-2.0 license is verified; platform, Python, and
canonical-tree decisions remain unconfirmed.

- [~] **D-STRAT-1** · `research` · Root modules act as the current entrypoints and the Mamba-3 experiment lives under `/research/`, but a duplicate legacy tree remains under `stem_midi_pro/`. Treat the intended canonical tree and duplicate's retention as maintainer decisions.
- [!] **D-STRAT-2** · `research` · This historical plan's architecture is not authoritative; current code facts are documented in `ARCHITECTURE.md`. Maintainers should decide whether any proposed product requirements remain in scope.
- [!] **D-STRAT-3** · `research` · **CPU-only target unverified** — the checked-in config and Dockerfile target CUDA/NVIDIA tooling; no CPU-only Mamba installation or inference run is verified. Confirm the platform target before changing runtime dependencies.
- [!] **D-STRAT-4** · `research` · **Python support range undeclared** — there is no root pyproject/CI matrix; Docker uses the base image's system Python. Confirm and test a supported version before pinning.
- [!] **D-STRAT-5** · `research` · **Mamba installation target unverified** — requirements use the PyPI package, but the current model config requests CUDA and no CPU path is tested. Confirm the target before asserting CPU support.
- [x] **D-STRAT-6** · `decision` · **License: Apache License 2.0** — permissive open source with explicit patent grant and contributor terms; owner retains copyright. Place at `/LICENSE`.

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
- [~] **R-1.1.8** · `refactor` · S · A vendored Mamba snapshot exists under `/research/mamba-ssm-reference/mamba/`, but an identical tracked copy remains at `stem_midi_pro/mamba/`. The tracked research snapshot has no embedded `.git`; confirm the duplicate's intended status before removing either tree.
- [x] **R-1.1.9** · `refactor` · S · Added `/research/README.md` (top-level index for both research subdirs; covers R-1.1.9, R-1.1.10, R-1.1.11 in one document). ✅
- [x] **R-1.1.10** · `refactor` · S · (folded into R-1.1.9). ✅
- [x] **R-1.1.11** · `refactor` · S · (folded into R-1.1.9). ✅
- [x] **R-1.1.12** · `refactor` · S · Moved `tests/test_losses_mamba3.py` → `/research/mamba3_per_track/test_losses.py`; updated import to use `from losses import …` with `sys.path` injection. ✅

## 1.2 Update root paths & references

- [x] **R-1.2.1** · `refactor` · `check_syntax.py` skips environment/cache directories and both vendored Mamba snapshots. The exact file count is intentionally omitted because the tree changes.
- [x] **R-1.2.2** · `refactor` · S · `verify_structure.py` — removed all `model.py` / `train_mamba3.py` / `losses_mamba3.py` checks; added `research/mamba3_per_track/` and `research/mamba-ssm-reference/mamba/README.md` as informational checks. Also fixed cp1252 UnicodeEncodeError by replacing ✓/✗ with `[OK]/[MISSING]`. ✅
- [!] **R-1.2.3** · `research` · The current tree has no `.github/workflows/ci.yml`; `.github/workflows/codeql.yml` is the only workflow. The prior completion claim cannot be verified and needs a maintainer decision on whether to add test/lint CI.
- [x] **R-1.2.4** · `refactor` · S · `pyproject.toml` — **N/A: no `pyproject.toml` exists at repo root.** No action needed. ✅
- [x] **R-1.2.5** · `infra` · Added a root `.dockerignore` for credentials, local caches, datasets/checkpoints, tests, research/vendor trees, and the legacy duplicate tree. Docker image build remains unverified.
- [!] **R-1.2.6** · `research` · The root `example_data_config.yaml` is consumed by the root `train.py`, not the Mamba-3 CLI. The old CPU-only/research instructions were replaced, but the platform target remains unverified.

## 1.3 Drop redundant qualifier prefixes (mirror the `mamba3_` rename)

- [x] **R-1.3.1** · `refactor` · S · Rename `data/canadian_datasets.py` → `data/datasets.py` (the file IS the data loader; the `canadian_` qualifier belongs in the curation policy, not the filename). ✅
- [x] **R-1.3.2** · `refactor` · The base class was renamed to `AudioDataset`. The earlier claim about Canadian curation weights/policy is unsupported by the current loaders and has been removed from current docs; see R-1.3.5.
- [x] **R-1.3.3** · `refactor` · S · Rename `get_canadian_data_loaders` → `get_data_loaders` (same reasoning). ✅
- [!] **R-1.3.4** · `research` · Root imports and `verify_structure.py` use the current dataset module, but the prior completion claim references a nonexistent `.github/workflows/ci.yml`. Verify historical test/workflow migration before treating this item as fully complete.
- [!] **R-1.3.5** · `research` · Path references were updated, but the claimed Canadian-artist mission/curation policy is not evidenced by the current Slakh/MUSDB loaders or README. Confirm whether this remains a product requirement before restoring that copy.

---

# SECTION 2 — HISTORICAL RUNTIME / CORRECTNESS LEADS

<!-- STATUS: research -->

This section originated as an earlier audit, not a verified list of current
P0 defects. Items below have been relabeled where claims were corrected or
rechecked; remaining unchecked items still need reproduction before action.

- [~] **C-2.1** · `test` · Root separator/transcriber now use NeMo's exported `NeuralModule` base and zero-argument superclass initialization; the separator stores configured `d_model` so zero-layer test configs do not index an empty `ModuleList`. Dependency-gated construction/forward, optional-target metadata, and mel-feature tests were added; none has run in this environment.
- [~] **C-2.2** · `test` · Assigned the transcriber config, fixed scalar mel-boundary logarithms, and cached the mel basis as a buffer. Added a dependency-gated mel-feature test; it has not run because pytest/PyTorch/NeMo are unavailable here.
- [x] **C-2.3** · `research` · The prior overwrite claim is inaccurate: `_build_file_list()` returns the synthetic list when `self.synthetic` is true. Explicit synthetic selection is now also forced by `dataset_type: synthetic`, including when the root exists.
- [!] **C-2.4** · `research` · A missing root is marked synthetic by the base `_validate_dataset()`, and the loader returns synthetic samples before scanning files. This can silently train on synthetic audio even when `dataset_type` names a real dataset; strict-failure versus fallback behavior needs an owner decision before changing it.
- [ ] **C-2.5** · `bug` · L · P0 · `main.py:process_audio_streaming()` passes `state_cache=` to `MambaSeparator.forward()`, which accepts only `audio`; calling the streaming helper raises `TypeError`. The separator returns a `None` state placeholder. Implementing true SSM streaming or removing the broken helper is an architectural choice; do not silently claim streaming support.
- [x] **C-2.6** · `refactor` · Renamed `_estimate_si_sdr` to `_centroid_separation()` and updated docs/UI copy to identify the heuristic. It still computes extra STFTs; optimizing this is separate from metric correctness.
- [!] **C-2.7** · `research` · The stated 0-D failure is incorrect: summing the 1-D NumPy stem arrays yields a 1-D array. MUSDB mixing still needs dataset-backed validation for missing stems, sample rates, and target semantics.
- [~] **C-2.8** · `test` · The heuristic now flags strongly negative correlation rather than absolute high correlation, with an explicit caveat that it is not a validated phase metric. Added a dependency-gated regression test; not run in this environment.
- [~] **C-2.9** · `test` · The mel basis is now constructed once and registered as a non-persistent buffer; the per-frame Python build was removed from `_mel_spectrogram()`. The dependency-gated test has not run here.
- [!] **C-2.10** · `research` · In evaluation mode the code repeats the confidence head five times while its `Dropout` layer is disabled, so the outputs are identical and are not an MC-dropout ensemble. The code comments/docs now disclose this; changing inference scores requires a deliberate calibration/test decision.
- [!] **C-2.11** · `research` · The specific transpose/aliasing claim is not supported by the current separator path (`input_proj` output remains `(B, T, D)` for Mamba). End-to-end shape validation remains blocked by unavailable model dependencies.
- [!] **C-2.12** · `research` · `_load_audio()` explicitly returns `np.float32`, so `torch.tensor(audio)` currently infers float32; this is not the stated dtype bug. `torch.from_numpy()` could avoid a copy, but is a separate optimization.

---

# SECTION 3 — REFACTORS (P0–P1)

## 3.1 Collapse duplication

- [!] **RF-3.1.1** · `research` · The Slakh/MUSDB adapters have overlapping code, but the historical `~90%` estimate was not measured. Quantify duplication and validate layout/target differences before consolidating behavior in a new `StemFolderDataset`.
- [ ] **RF-3.1.2** · `refactor` · M · P1 · Move `MambaSeparator` mel/STFT code into a shared `audio/stft.py` module with one canonical `stft`, `istft`, `mel_spectrogram` (returns a registered buffer), and `polar_to_complex(mag, phase)`. Use from both separator and transcriber.
- [ ] **RF-3.1.3** · `refactor` · M · P1 · `main.py:_logits_to_midi` is called from `forward` and `process_audio_streaming`. Move to a new `utils/midi.py` that owns: `logits_to_events`, `events_to_bytes`, `build_midi_from_events` (replaces the misnamed `api.py:create_placeholder_midi`).
- [ ] **RF-3.1.4** · `refactor` · S · P1 · `api.py:create_placeholder_midi` (which actually emits real events) — rename to `build_midi_from_events` and move to `utils/midi.py`. Update `create_response_zip`.
- [ ] **RF-3.1.5** · `refactor` · S · P1 · `models/confidence_injector.py:inject_midi_metadata` loop creates a `torch.tensor` per event for the sigmoid. Vectorize. Add a benchmark.
- [!] **RF-3.1.6** · `research` · `main.py:_logits_to_midi()` uses Python batch/frame loops. Vectorization is a possible optimization, but the `~100×` speedup claim is unmeasured; benchmark representative inputs before prioritizing or quoting gains.
- [!] **RF-3.1.7** · `research` · NeMo `ModelPT` is still the root model base and `nemo-toolkit[all]` remains in requirements. Replacing it with bare PyTorch changes the training/checkpoint architecture; do not assume a CPU-only target or remove NeMo without owner approval and dependency-size/compatibility evidence.
- [!] **RF-3.1.8** · `research` · Root separator/transcriber still use NeMo `NeuralModule` and `typecheck`. Dropping NeMo typing is coupled to an architecture/dependency decision, not an isolated refactor.
- [!] **RF-3.1.9** · `research` · Root `train.py` still uses PyTorch Lightning. Replacing the trainer and checkpoint lifecycle is an architectural change; CPU simplicity is not established without a supported target and benchmark.

## 3.2 Consolidate file structure

- [ ] **RF-3.2.1** · `refactor` · S · P1 · Populate `models/__init__.py` with `__all__ = ["MambaSeparator", "MambaTranscriber", "ConfidenceInjector", "PerceptualAudioLoss"]`.
- [ ] **RF-3.2.2** · `refactor` · S · P1 · Populate `utils/__init__.py` with `__all__ = ["ProcessingReport", "QualityTier", "route_by_quality", "render_template", "list_templates"]`.
- [ ] **RF-3.2.3** · `refactor` · S · P1 · Populate `data/__init__.py` with `__all__ = ["AudioDataset", "get_data_loaders", "StemFolderDataset"]` (post-collapse).
- [ ] **RF-3.2.4** · `refactor` · S · P1 · Add a top-level `audio_io.py` (or `io/audio.py`) module with: `load_audio(path, target_sr, mono=True)`, `save_wav(path, audio, sr)`, `validate_audio(path, max_duration, supported_srs)`. Single source of truth for I/O.
- [ ] **RF-3.2.5** · `refactor` · S · P1 · Add `utils/paths.py` with project-root resolution (replace ad-hoc `Path(__file__).parent.parent / "user_content"` patterns).

---

# SECTION 4 — DEPENDENCIES (P0)

## 4.1 Production `requirements.txt`

- [!] **DEP-4.1.1** · `research` · The proposed Torch range and Python 3.13 CPU-wheel rationale are unverified; resolve supported Python/hardware and test the Mamba/Torch combination before pinning.
- [!] **DEP-4.1.2** · `research` · Match `torchaudio` to the resolved `torch` version only after selecting and testing a supported runtime; no lock or install resolution exists.
- [!] **DEP-4.1.3** · `research` · Root code still imports NeMo (`ModelPT`, `NeuralModule`, `typecheck`). Removing `nemo-toolkit[all]` would break the current model stack unless architecture changes are approved and tested.
- [!] **DEP-4.1.4** · `research` · Root `train.py` still imports/uses PyTorch Lightning; removal depends on the unapproved trainer rewrite in RF-3.1.9.
- [!] **DEP-4.1.5** · `research` · `mamba-ssm` is required by both model modules, but a `>=2.0.0` floor and CPU support have not been compatibility-tested.
- [x] **DEP-4.1.6** · `infra` · Root requirements do not include `causal-conv1d`; the config keys are documented as legacy and are not passed into the Mamba constructor.
- [!] **DEP-4.1.7** · `research` · Both manifests specify `librosa>=0.10.0` without an upper bound. Compatibility of the historical `<0.11.0` cap has not been tested; resolve before narrowing.
- [x] **DEP-4.1.8** · `infra` · Both root and legacy manifests already declare `soundfile>=0.12.0`; dependency resolution remains unverified.
- [x] **DEP-4.1.9** · `infra` · Both root and legacy manifests already declare `mido>=1.2.0`; dependency resolution remains unverified.
- [!] **DEP-4.1.10** · `research` · The `fastapi>=0.110.0` floor was proposed for lifespan migration, but lifespan support predates that floor and the current app uses `on_event`. Resolve a compatible dependency set and test lifecycle behavior before raising the minimum.
- [ ] **DEP-4.1.11** · `infra` · S · P0 · `uvicorn[standard]>=0.27.0`.
- [!] **DEP-4.1.12** · `research` · Both manifests were raised to `python-multipart>=0.0.18` for CVE-2024-53981, but newer 2026 advisories report affected versions below 0.0.27, 0.0.30, and 0.0.31. The current floor is not sufficient for a deployment assurance. Review and approve a new floor/lock after compatibility testing; no resolved runtime version was audited.
- [!] **DEP-4.1.13** · `research` · The stated FastAPI/Pydantic v2 requirement has not been verified against a resolved supported dependency set. Confirm the compatibility matrix before raising this floor.
- [!] **DEP-4.1.14** · `research` · Both manifests specify `pyyaml>=6.0`; the proposed `6.0.1` floor was not resolved or compatibility-tested.
- [!] **DEP-4.1.15** · `research` · The NumPy bound was tied to an unconfirmed Python 3.13 target. Resolve and test an installation matrix before adding this range.
- [x] **DEP-4.1.16** · `infra` · The root and legacy manifests already declare `tqdm>=4.65.0`; dependency resolution remains unverified.
- [!] **DEP-4.1.17** · `research` · Root `models/losses.py` implements MR-STFT in PyTorch and does not import `auraloss`; the separate research trainer uses `auraloss` optionally and warns when absent. Do not add it to root requirements unless the runtime ownership/scope changes.
- [ ] **DEP-4.1.18** · `infra` · S · P0 · `structlog>=24.1.0` — structured logging for production.
- [ ] **DEP-4.1.19** · `infra` · S · P0 · `prometheus-client>=0.20.0` — `/metrics` endpoint.
- [ ] **DEP-4.1.20** · `infra` · S · P0 · `opentelemetry-api>=1.27.0`, `opentelemetry-sdk>=1.27.0`, `opentelemetry-instrumentation-fastapi>=0.48b0` — tracing.
- [x] **DEP-4.1.21** · `infra` · Removed the unused `pretty-midi` dependency from both requirement manifests; current first-party code uses `mido`.

## 4.2 Dev `requirements-dev.txt` (new file)

- [ ] **DEP-4.2.1** · `infra` · S · P0 · `pytest>=7.4.0`.
- [ ] **DEP-4.2.2** · `infra` · S · P0 · `pytest-cov>=4.1.0`.
- [ ] **DEP-4.2.3** · `infra` · S · P0 · `pytest-asyncio>=0.23.0`.
- [ ] **DEP-4.2.4** · `infra` · S · P0 · `httpx>=0.27.0` (FastAPI TestClient).
- [ ] **DEP-4.2.5** · `infra` · S · P0 · `black>=23.10.0`.
- [ ] **DEP-4.2.6** · `infra` · S · P0 · `ruff>=0.6.0` (replaces flake8+isort).
- [ ] **DEP-4.2.7** · `infra` · S · P0 · `mypy>=1.7.0`.
- [ ] **DEP-4.2.8** · `infra` · S · P0 · `types-PyYAML`, `types-requests`.
- [ ] **DEP-4.2.9** · `infra` · S · P0 · `responses>=0.24.0` (HTTP mocking).

## 4.3 Pinned Python

- [!] **DEP-4.3.1** · `research` · No Python support range is declared. Choose one only after resolving/installing the model stack and running checks across candidate versions; Python 3.13 is not established.
- [!] **DEP-4.3.2** · `research` · The current Dockerfile uses a CUDA Ubuntu base. Replacing it with `python:3.13-slim` assumes unverified CPU/Mamba support and requires a tested build.
- [!] **DEP-4.3.3** · `research` · A CI workflow and Python matrix need maintainer approval; do not hard-code an unsupported 3.13-only target. `.github/workflows/codeql.yml` is currently the only workflow.
- [!] **DEP-4.3.4** · `research` · A root setup guide promising Python 3.13/CPU/bare-metal i5 support would be misleading until the platform and installation path are validated.

---

# SECTION 5 — LICENSE (P0)

- [x] **LIC-5.1** · `infra` · M · P0 · Add `/LICENSE` with full Apache License 2.0 text (verbatim from `https://www.apache.org/licenses/LICENSE-2.0.txt`).
- [!] **LIC-5.2** · `research` · The proposed small-business terms and example contact conflict with the current Apache-2.0 `LICENSE`. Do not add a commercial restriction notice unless maintainers explicitly approve a license change.
- [!] **LIC-5.3** · `research` · The proposed PolyForm identifier conflicts with the current Apache-2.0 license; do not add it without an owner-approved relicensing decision.
- [!] **LIC-5.4** · `research` · Verify actual shipped third-party notices and licenses before creating `NOTICE`; the historical package list includes dependencies not present in the current manifests.
- [!] **LIC-5.5** · `research` · The proposed commercial license summary conflicts with Apache-2.0; the README now points to the current `LICENSE` without adding those restrictions.
- [!] **LIC-5.6** · `research` · The proposed small-business restriction conflicts with Apache-2.0. Do not add commercial terms to user-facing copy without an owner-approved license change.
- [ ] **LIC-5.7** · `docs` · S · P0 · `/research/` — note in `README.md` that research code is also under the same license, but is "as-is" with no support.

---

# SECTION 6 — HISTORICAL API-HARDENING PROPOSALS

<!-- STATUS: research -->

These proposals have not been approved as an API/security design. Several
change public routes, authentication, resource limits, or lifecycle behavior;
reassess each against current code and obtain maintainer approval before
implementation.

- [ ] **API-6.1** · `sec` · S · P0 · Convert `@app.on_event("startup")` to `lifespan` async context manager (FastAPI ≥0.110 deprecation).
- [!] **API-6.2** · `research` · Current default is a localhost allowlist; explicit `*` disables credentials. Replacing this with wildcard-by-default or startup rejection changes public CORS behavior and requires an owner/security decision.
- [~] **API-6.3** · `sec` · The handler has a 50 MiB per-file cap and a Content-Length check, but FastAPI parses multipart data before calling it; there is no pre-parser total-body limit. Add an approved ASGI/proxy body cap before public deployment; the historical 500 MiB target is not validated.
- [ ] **API-6.4** · `sec` · M · P0 · Add concurrency cap: `asyncio.Semaphore(1)` around `process_audio_file`. Return 503 with `Retry-After: 1` when at capacity.
- [!] **API-6.5** · `research` · Current optional `API_KEY` uses `X-API-Key` on `/process`, `/templates`, and `/render-template`; it is disabled by default. Switching to bearer keys, multiple keys, or route exceptions changes authentication behavior and needs an owner/security decision.
- [ ] **API-6.6** · `perf` · S · P0 · `validate_audio_file` currently opens with `sf.info` then `_load_audio` opens again with `sf.read`. Use a single `sf.SoundFile` context for both. (Eliminates double-open on the same file.)
- [ ] **API-6.7** · `sec` · M · P0 · Catch `torch.cuda.OutOfMemoryError` (still relevant if anyone runs on GPU) and any `MemoryError`; return 503 with `Retry-After`. Log the event.
- [ ] **API-6.8** · `sec` · M · P0 · Add graceful shutdown: `lifespan` teardown that drains in-flight requests, clears CUDA cache if present. Pass `--timeout-graceful-shutdown 30` to uvicorn in Dockerfile CMD.
- [ ] **API-6.9** · `infra` · S · P0 · Add `X-Request-ID` middleware (read incoming or generate UUID4). Echo in response header. Log on every request. Include in error responses.
- [ ] **API-6.10** · `infra` · M · P0 · Replace `logging.basicConfig` with `structlog` configured for JSON output (`LOG_FORMAT=json` env). Include `request_id`, `route`, `latency_ms`, `status_code` on every log line.
- [!] **API-6.11** · `research` · The helper now serializes event data rather than writing an empty track, but its legacy name remains. Moving/renaming a module-level helper may affect callers; decide whether to provide a compatibility wrapper before changing it.
- [!] **API-6.12** · `research` · MIDI events currently carry no duration prediction, so serialization uses a documented fixed 1/16-note default. Do not invent a `duration_frames` contract or change the output format without model/data evidence and maintainer approval.
- [~] **API-6.13** · `research` · `process_audio_streaming()` is now documented in source/docs as experimental and currently broken (`state_cache` is unsupported). It remains callable; decide whether to remove/deprecate it or design true streaming before claiming this is resolved.
- [!] **API-6.14** · `research` · `/render-template` currently accepts the template name as a query parameter and variables as an optional body; path traversal is blocked by a basename-only resolver. Changing route method/shape/status codes is a public API change; decide before implementation.
- [x] **API-6.15** · `research` · `_load_audio()` explicitly uses `sr=None`, preserving the file sample rate, and `mono=True` in the librosa fallback. The historical claim that `None` defaults to 22.05 kHz is not supported; no resampling bug was verified.
- [ ] **API-6.16** · `infra` · M · P0 · Add `/metrics` (Prometheus format). Counters: `requests_total{route,status}`, `errors_total{route,kind}`, `oom_total`. Histograms: `request_duration_seconds{route}`, `model_inference_seconds`, `bytes_processed`. Gauges: `gpu_memory_bytes` (if GPU), `in_flight_requests`.
- [ ] **API-6.17** · `infra` · S · P0 · Split `/health` into `/live` (process up, always 200) and `/ready` (model loaded, not OOM, returns 503 with `Retry-After` while loading). Update Dockerfile `HEALTHCHECK` to call `/live`.
- [ ] **API-6.18** · `sec` · S · P0 · Add `X-Content-Type-Options: nosniff` and `Strict-Transport-Security` (when behind TLS) headers. Use `secure.headers.Secure` middleware or equivalent.
- [ ] **API-6.19** · `infra` · M · P0 · Add OpenTelemetry tracing: instrument FastAPI via `opentelemetry-instrumentation-fastapi`. Spans: `validate`, `load`, `separate`, `transcribe`, `inject`, `package`. Export to OTLP (env-configured endpoint). Add console exporter for dev.
- [ ] **API-6.20** · `sec` · S · P0 · Add `/warmup` endpoint that runs a 1-sec synthetic inference at startup. Triggered automatically in `lifespan` so the first real request isn't slow.

---

# SECTION 7 — HISTORICAL INFERENCE-PERFORMANCE PROPOSALS

<!-- STATUS: research -->

Performance tasks require a validated baseline and target hardware. Do not
implement the historical CPU/H100 assumptions or make latency claims without
measurement.

- [!] **PERF-7.1** · `research` · `_estimate_si_sdr` was renamed `_centroid_separation()` and recomputes STFTs on the separated stems. Caching the input mixture STFT may not eliminate those calls; profile the actual metric path before changing it.
- [~] **PERF-7.2** · `perf` · `_create_mel_basis` now runs in `__init__` and `mel_basis` is a non-persistent buffer used by `_mel_spectrogram()`. Dependency-gated feature test is present but unrun here.
- [!] **PERF-7.3** · `research` · `process_audio_file()` and `process_audio_streaming()` already wrap inference in `torch.inference_mode()`. Wrapping `forward()` itself would disable gradients needed by training; do not apply this historical proposal without a distinct inference interface.
- [ ] **PERF-7.4** · `perf` · S · P0 · `main.py` — set `torch.set_num_threads(os.cpu_count() or 1)` at import; allow override via `OMP_NUM_THREADS` env.
- [!] **PERF-7.5** · `research` · `models/mamba_separator.py` runs one input STFT and three per-mask ISTFT calls; the old note incorrectly said this loop repeats STFT setup. Any batched iSTFT rewrite needs a valid PyTorch API and profiling before adoption.
- [!] **PERF-7.6** · `research` · The current `Mamba` constructor uses `use_fast_path=True`; CPU support and any 5–10× comparison in the historical note are unverified. Check the installed Mamba API and benchmark a supported target before changing the path.
- [!] **PERF-7.7** · `research` · Dynamic INT8 quantization support for the Mamba block and its quality impact are unverified. The historical `~2×`/`<1%` figures have no benchmark evidence; do not add a quantization flag or claim gains without model validation.
- [!] **PERF-7.8** · `research` · A `quantize` option would change inference behavior and API/CLI contracts; defer until PERF-7.7 is validated and maintainers approve the option.
- [!] **PERF-7.9** · `research` · The claimed AVX-512 speedup is unmeasured and the current hardware target is unknown. Benchmark accuracy/latency before changing global matmul settings.
- [ ] **PERF-7.10** · `perf` · S · P0 · `models/mamba_separator.py` — convert `self.mamba_blocks` from `nn.ModuleList` to `nn.Sequential` for cleaner forward (no functional change, but JIT-friendly).
- [!] **PERF-7.11** · `research` · A batch endpoint changes public API/resource limits; no CPU reference target or throughput measurement exists. Do not add the route or publish the historical `30–60s/min` estimate without maintainer approval and load tests.
- [!] **PERF-7.12** · `research` · Content-hash caching changes response semantics and creates persistence/retention, access-control, and deletion requirements. No cache design or repeat-request benchmark is approved; do not promise CPU savings or a 24-hour TTL.

---

# SECTION 8 — TESTING (P0)

## 8.1 Test infrastructure

- [ ] **T-8.1.1** · `test` · S · P0 · `tests/conftest.py` — populate with fixtures: `audio` (1-sec 44.1kHz mono tensor), `stereo_audio` (2-ch), `mock_config` (full YAML dict), `temp_audio_file` (WAV in tmp), `temp_flac_file`, `temp_mp3_file` (skip if no encoder), `mock_model` (StemMidiModel with small config).
- [x] **T-8.1.2** · `test` · `pytest.ini` restricts default collection to `tests/`, uses `-ra --strict-markers --tb=short`, and registers `slow`, `cuda`, and `integration` markers.
- [!] **T-8.1.3** · `research` · No `.github/workflows/ci.yml` or test step exists; `.github/workflows/codeql.yml` is the only workflow. Decide whether to add a test/lint CI workflow rather than attempting to remove a nonexistent `continue-on-error` setting.

## 8.2 Un-skip existing tests

- [~] **T-8.2.1** · `test` · `api.py` now lazy-imports `StemMidiModel` inside `load_model()`, so importing the API no longer imports the full model stack. A missing-checkpoint regression test was added; it remains unrun because pytest and runtime dependencies are unavailable here.
- [~] **T-8.2.2** · `test` · `tests/test_api_validation.py` no longer uses skip markers and covers valid WAV, excessive duration, and unsupported sample rate. Stereo/FLAC/MP3 cases are not covered; tests remain unrun because pytest and runtime dependencies are unavailable here.
- [~] **T-8.2.3** · `test` · `tests/test_audio_loading.py` still skips all three cases while constructing the optional NeMo/Mamba model stack; mono WAV, stereo-to-mono, missing-file, and resampling behavior remain unverified.
- [~] **T-8.2.4** · `test` · `tests/test_midi_generation.py` no longer uses skip markers and covers empty events, note ordering/ticks, exact CC#127 values, and parsing generated MIDI bytes. Tests remain unrun because pytest and `mido` are unavailable here.

## 8.3 New tests

- [~] **T-8.3.1** · `test` · Dependency-gated separator coverage now lives in `tests/test_model_module_setup.py`: a zero-layer forward asserts output shapes and placeholder state. It is unrun here and does not yet assert finiteness, nonidentical stems, or the broken streaming contract.
- [~] **T-8.3.2** · `test` · `tests/test_model_module_setup.py` covers mel-feature shape/finiteness but not the transcriber's complete five-output `forward()` contract. Full-stack test and forward-shape coverage remain pending.
- [ ] **T-8.3.3** · `test` · S · P0 · `tests/test_confidence_injector.py` (new) — verify alignment score, `needs_review` threshold, CC#127 mapping, summary counts.
- [ ] **T-8.3.4** · `test` · S · P0 · `tests/test_losses.py` (new) — verify MR-STFT returns scalar; crest/flatness/alignment components sum; finite values.
- [~] **T-8.3.5** · `test` · `tests/test_template_engine.py` covers missing-key placeholders, seven listed templates, and `FileNotFoundError` for missing/escaping paths. The tests are implemented but unrun here.
- [~] **T-8.3.6** · `test` · Equivalent MIDI-byte round-trip, ordered note, and exact CC#127 assertions are in `tests/test_midi_generation.py` rather than a separate module. Implemented but unrun here. **Catches API-6.12.**
- [!] **T-8.3.7** · `research` · A `/process` 200/ZIP integration test is blocked by the current batch/event contract mismatch in `main.py:forward()`. Fix and validate that contract before making 200 the expected behavior.
- [!] **T-8.3.8** · `research` · The proposed streaming test calls a helper that currently raises `TypeError` because `state_cache` is unsupported. Decide whether to remove/deprecate the helper or implement true streaming before testing `chunks_processed`.
- [!] **T-8.3.9** · `research` · Artifact flags currently do not force COMPLEX tier. Whether to change routing semantics requires maintainer/calibration approval; do not codify the historical expectation without that decision.
- [!] **T-8.3.10** · `research` · Current API returns `401` for both missing and incorrect keys; the proposed `403` expectation is not current behavior. Decide the public auth/status contract before adding assertions.
- [!] **T-8.3.11** · `research` · CORS preflight expectations depend on Starlette behavior and a resolved version. Pin/test the supported dependency set before treating the historical response assertions as acceptance criteria.
- [ ] **T-8.3.12** · `test` · S · P0 · `/metrics` and its counter semantics are not implemented; if approved, test the eventual documented endpoint/metric names.
- [!] **T-8.3.13** · `research` · No reference CPU hardware or benchmark exists. Do not encode `<500 MB` or `<30s` i5 limits until the platform, workload, and measurement method are approved.
- [!] **T-8.3.14** · `research` · `StemFolderDataset` consolidation is not approved or implemented; the proposed parametrized tests depend on the maintainer decision recorded in RF-3.1.1.
- [!] **T-8.3.15** · `research` · Quantization support and `process_audio_file(quantize=True)` are unimplemented and unvalidated; defer tests until maintainers approve and benchmark the change (PERF-7.7/7.8).

## 8.4 Coverage targets

- [ ] **T-8.4.1** · `test` · M · P0 · CI gate: `--cov-fail-under=70` (current ~15%). Push to 80% by end of Sprint 2.
- [ ] **T-8.4.2** · `test` · S · P0 · Codecov config (`.codecov.yml` new) — set target to 80%, allow 5% drop on new PRs.

---

# SECTION 9 — CONFIG & INFRASTRUCTURE (P0–P1)

## 9.1 Configuration

- [!] **CFG-9.1.1** · `research` · The `selective_scan_impl`/`causal_conv1d_impl` keys are annotated as legacy and are not passed into the `Mamba` constructor. Removing them or selecting a CPU path depends on the unresolved runtime target.
- [!] **CFG-9.1.2** · `research` · The config labels `fp8` as legacy; root `train.py` maps it to Lightning precision 16, not FP8. A CPU `fp32` policy is not approved because hardware/runtime support is unverified.
- [!] **CFG-9.1.3** · `research` · A smaller CPU config and 4–8 GB/one-minute capacity target require hardware and model benchmarks; none is available.
- [!] **CFG-9.1.4** · `research` · Root and research models have separate config schemas (`n_layer` versus `n_layers`); choose a supported schema before renaming.
- [!] **CFG-9.1.5** · `research` · The root config labels the TensorRT-LLM export block as legacy/unimplemented. Removing it on a presumed CPU target is not approved; target hardware remains unverified.
- [!] **CFG-9.1.6** · `research` · `example_data_config.yaml` now selects synthetic data explicitly. The real-data loader has not been collapsed into `StemFolderDataset`; changing the sample config to real data would be misleading without a validated dataset path.
- [~] **CFG-9.1.7** · `infra` · `MODEL_CONFIG_PATH` is supported by `api.py` and `main.py` already accepts `--config`; `api.py` does not expose a direct `--config` CLI flag. A second interface is not implemented or required for current documented launch commands.

## 9.2 Build & deploy

- [!] **BLD-9.2.1** · `research` · A package manifest/version source is absent, but the proposed PolyForm license and CPU extras conflict with current Apache licensing and unverified runtime support. Resolve those decisions before packaging.
- [ ] **BLD-9.2.2** · `infra` · S · P0 · `setup.cfg` (new) — `tool.ruff` config. `line-length=100`, `target-version="py313"`, `extend-exclude=["research", "venv", "mamba"]`, `select=["E","F","W","I","UP","B","SIM","RUF"]`.
- [x] **BLD-9.2.3** · `infra` · Added root `.dockerignore` for local secrets/state, datasets/checkpoints, tests, research/vendor trees, and the legacy duplicate. Docker build context/build was not validated because Docker is unavailable.
- [!] **BLD-9.2.4** · `research` · The proposed Python 3.13 CPU-only multi-stage image, health check, and graceful-shutdown command assume an unapproved platform/runtime target. The current Dockerfile uses a CUDA base; reconcile hardware/dependency support and test a build before changing it.
- [ ] **BLD-9.2.5** · `infra` · S · P0 · `Makefile` (new) — `install`, `install-dev`, `lint`, `format`, `test`, `test-cpu`, `test-cuda`, `demo`, `serve`, `docker-build`, `docker-run`, `clean`, `dist`, `release-{major,minor,patch}`.
- [ ] **BLD-9.2.6** · `infra` · M · P0 · `.github/workflows/ci.yml` rewrite:
  - **lint job:** Python 3.13. `ruff check .`, `black --check .`, `mypy .` (loose).
  - **test-cpu job:** Python 3.13. `pip install -r requirements.txt -r requirements-dev.txt`. `pytest -m "not cuda" --cov=. --cov-fail-under=70`.
  - **structure job:** runs `verify_structure.py` + `check_syntax.py`.
  - **license job:** greps for `SPDX-License-Identifier` in all `.py` files; ensures 100% coverage.
- [ ] **BLD-9.2.7** · `infra` · M · P1 · Add a `cuda` job to CI (gated by repo variable `HAS_CUDA_RUNNER=true`). Runs `pytest -m "cuda"` only. Skipped by default.
- [ ] **BLD-9.2.8** · `infra` · M · P1 · Add `deploy/docker-compose.yml` (new) for local bare-metal i5 deployment. Mounts `./data/`, `./outputs/`, `./configs/`, `./user_content/`. Maps port 8000.
- [ ] **BLD-9.2.9** · `infra` · L · P1 · Add `systemd/stem-midi-pro.service` (new) for native Linux deployment. User, working dir, env file, restart policy, resource limits (MemoryMax=6G, CPUQuota=400% on i5).
- [ ] **BLD-9.2.10** · `infra` · M · P1 · Add `deploy/observability/` — `prometheus.yml` scrape config, `grafana-dashboard.json` with panels for latency / errors / OOM / throughput / quality-tier distribution.

## 9.3 Repository hygiene

- [x] **HYG-9.3.1** · `refactor` · S · `.gitignore` now excludes test/lint caches, coverage output, logs, local env files, generated model files, and dataset/output artifacts.
- [x] **HYG-9.3.2** · `research` · A Git repository and `main` branch exist. History review found separate, unrelated histories on `master`, `local-wip`, and current `main`; do not reinitialize or rewrite history. Maintainers should reconcile branch lineage.
- [ ] **HYG-9.3.3** · `infra` · S · P0 · Add `.github/CODEOWNERS` (new) — assign owners to paths.
- [ ] **HYG-9.3.4** · `infra` · S · P0 · Add `.github/pull_request_template.md` (new).
- [ ] **HYG-9.3.5** · `infra` · S · P0 · Add `.github/ISSUE_TEMPLATE/bug_report.yml` and `feature_request.yml`.

---

# SECTION 10 — DOCUMENTATION (P1)

## 10.1 Top-level docs

- [~] **DOC-10.1.1** · `docs` · `AGENTS.md` now describes the current prototype, tests, known data/model gaps, and Apache-2.0 license. CPU-only support and end-to-end real-data/audio/MIDI behavior remain unverified and are not claimed.
- [~] **DOC-10.1.2** · `docs` · `README.md` now describes the prototype, current routes, dependencies, and Apache-2.0 license without the stale TensorRT/FP8/CPU-performance claims. A portable install recipe and measured CPU performance figures remain unavailable.
- [x] **DOC-10.1.3** · `docs` · `SUMMARY.md` was replaced with a current-checkout inventory and explicit implementation gaps; stale dataset-class claims were removed.
- [~] **DOC-10.1.4** · `docs` · `ARCHITECTURE.md` now distinguishes prototype implementation from unverified/gapped behavior and references Apache-2.0. A CPU-only target is not established; some requirements sections remain aspirational.
- [x] **DOC-10.1.5** · `docs` · `API_DOCUMENTATION.md` documents current auth, CORS, constraints, error responses, and routes, and explicitly states that rate limiting/metrics are absent; the stale `/api/v1` claim was removed.
- [x] **DOC-10.1.6** · `docs` · `USER_GUIDE.md` now documents only the local prototype and current API/output limitations; unsupported feature claims were removed.
- [x] **DOC-10.1.7** · `docs` · `DEVELOPMENT_GUIDE.md` now uses root paths and available local checks; removed/nonexistent TorchScript, CI, and Make-target examples were not retained.
- [!] **DOC-10.1.8** · `research` · The requested Python 3.13/CPU-only/i5 instructions are unsupported by checked-in dependencies and runtime evidence. Existing setup notes distinguish the root runtime from the experimental Mamba-3 CLI; confirm a supported platform before publishing stronger instructions.
- [x] **DOC-10.1.9** · `docs` · Removed the local workstation path and labeled the old Part III audit as unverified historical research (its cited commit is unavailable in this repository).
- [!] **DOC-10.1.10** · `research` · Do not infer a release changelog from the GitHub `v0.1.0` release description: it refers to product areas absent from this checkout, while the shallow/grafted local clone has no matching tag. Maintainers should reconcile product/version lineage before adding release notes.
- [!] **DOC-10.1.11** · `research` · `api.py` OpenAPI metadata still uses the description “Professional audio AI service” and version `1.0.0`; changing public schema metadata should follow a maintainer-approved product/version decision.

## 10.2 User-facing content

- [~] **DOC-10.2.1** · `docs` · All seven user-content templates were reviewed and unsupported product claims removed. `/render-template` accepts caller-supplied variables, but there are no in-repo template callers; `filename`/`sample_rate` therefore require an external caller to supply values.
- [!] **DOC-10.2.2** · `research` · `landing_page.md` now identifies the Apache-2.0 software license. The requested CPU claim is unmeasured and intentionally omitted.
- [~] **DOC-10.2.3** · `docs` · Completion copy now distinguishes report fields from quality guarantees; report variable names match the JSON fields, while `filename` is not included in the report. The repository has no in-process caller to populate the template.
- [x] **DOC-10.2.4** · `docs` · `rights_usage_prompt.md` now distinguishes the software's Apache-2.0 license from rights in user audio and avoids claiming the software license grants media rights.
- [!] **DOC-10.2.5** · `research` · Removed the unsupported live-progress claim. The suggested CPU latency range is not measured and must not be published as a performance fact.
- [x] **DOC-10.2.6** · `docs` · Replaced the browser-editor/human-review pitch with a qualified note that generated MIDI can be reviewed in an external DAW; no editor or calibrated CC#127 workflow is promised.
- [!] **DOC-10.2.7** · `research` · `utils/quality_gates.py` still returns draft routing copy advertising an editor, $4.99/24h human review, and a refund. The repository has no corresponding UI/payment/refund service. Its output is part of the Python return contract; obtain an owner decision before changing/removing it, and do not present it as a live offer.

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

# SECTION 11 — HISTORICAL PRODUCT / ARCHITECTURE BACKLOG

<!-- STATUS: research -->

These items originated as a proposed production roadmap. The current root
`ARCHITECTURE.md` is an implementation inventory, not an approved feature
specification. Confirm product scope and review public API, payment, security,
and architecture decisions before implementing any item.

## 11.1 From "API Layer" (api.py)

- [ ] **ARCH-11.1.1** · `feature` · S · P0 · Background task support for cleanup — see API-6.8 (lifespan teardown).
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

- [ ] **ARCH-11.7.1** · `feature` · S · P0 · Fix C-2.8 (phase cancellation logic is inverted).
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

- [ ] **ARCH-11.10.1** · `feature` · M · P0 · Input sanitization (file extension + content type + duration + size). See API-6.3.
- [ ] **ARCH-11.10.2** · `feature` · M · P0 · Rate limiting (token bucket per API key). Use `slowapi` or hand-rolled.
- [ ] **ARCH-11.10.3** · `feature` · S · P0 · Ephemeral file processing — already in place; verify temp-file cleanup on exception path. Add a unit test that simulates a crash mid-processing and asserts no temp files remain.
- [ ] **ARCH-11.10.4** · `docs` · S · P1 · GDPR compliance doc — already 90% done in `USER_GUIDE.md`; trim and move to `docs/operations/gdpr.md`.

## 11.11 From "Performance Optimization"

- [ ] **ARCH-11.11.1** · `feature` · M · P1 · Memory-mapped audio loading for large files (currently `sf.read` loads fully into RAM). Use `soundfile`'s `frames=-1, start=offset` to seek.
- [ ] **ARCH-11.11.2** · `feature` · S · P1 · `torch.set_num_threads(1)` when running in Docker to avoid CPU oversubscription. Allow override via `OMP_NUM_THREADS`.
- [ ] **ARCH-11.11.3** · `feature` · M · P1 · Latency budget per stage: validate <100ms, load <500ms, separate 30s/min-audio, transcribe 30s/min-audio, package <200ms. Emit per-stage timings to OTel spans.

## 11.12 From "Extensibility Points"

- [ ] **ARCH-11.12.1** · `feature` · L · P2 · Multi-instrument support: add a `drums`, `vocals`, `keys` mask head to `MambaSeparator`. Add config flag `instruments: ["guitar", "bass", "drums", "vocals"]`. Update MIDI generation.
- [ ] **ARCH-11.12.2** · `feature` · M · P2 · New audio effects: add `tremolo`, `wah`, `harmonics` to the expression head. Update `expression_heads` in config.

## 11.13 From "Canadian Artist Dataset Integration"

- [ ] **ARCH-11.13.1** · `docs` · S · P0 · Document that Slakh/MUSDB datasets are open-license (CC-BY) but the *trained model weights* are under Apache License 2.0.
- [ ] **ARCH-11.13.2** · `feature` · M · P2 · Add a `data/canadian_artist_specific.py` loader for artists in the Canadian indie scene (e.g., Six Shooter Records, Arts & Crafts label rosters) — public-domain / CC-licensed recordings only. Document provenance per file.

## 11.14 From "Monitoring and Observability"

- [ ] **ARCH-11.14.1** · `infra` · M · P0 · Latency metrics (covered in API-6.16).
- [ ] **ARCH-11.14.2** · `infra` · M · P0 · Resource utilization: CPU%, memory, file descriptors (per process). Add `prometheus-client` process collector.
- [ ] **ARCH-11.14.3** · `infra` · M · P0 · Quality metrics: per-tier histogram (studio / draft / complex) over time. Surface in Grafana.
- [ ] **ARCH-11.14.4** · `infra` · M · P0 · Error rates per endpoint. Alert on >5% 5xx in 5m window.

---

# SECTION 12 — TRAINING CORRECTNESS (P0)

- [!] **TR-12.1** · `research` · The root trainer already sets `deterministic=True` on `pl.Trainer`; the historical assertion that seeding contradicts this was not reproduced. Verify exact Lightning/CUDA semantics before adding global cuDNN settings.
- [!] **TR-12.2** · `research` · The current trainer still uses Lightning and `ckpt_path='best'`; the proposed explicit state-dict load only applies after a custom-loop rewrite. Confirm behavior against the resolved Lightning version before treating this as a defect.
- [!] **TR-12.3** · `research` · Replacing Lightning with a custom loop and adding CPU bfloat16 behavior is an architectural/platform change. The model/data/training path is not validated, and a CPU target is unapproved.
- [ ] **TR-12.4** · `feature` · M · P1 · Add `--grad-accum` CLI flag (currently `train.py` doesn't accept it).
- [ ] **TR-12.5** · `feature` · M · P1 · Add `--ema` (exponential moving average of weights) for stable eval.
- [ ] **TR-12.6** · `feature` · M · P1 · Add `--resume` flag (currently `train.py` accepts `--resume-from-checkpoint` but doesn't actually pass it correctly to the custom loop).
- [ ] **TR-12.7** · `feature` · M · P1 · Add `--quantize` for training-aware quantization-aware training (QAT) on the Mamba backbone.
- [!] **TR-12.8** · `research` · A missing root can silently select synthetic data even for a real `dataset_type`. Whether to fail fast or permit fallback changes training behavior; decide with maintainers before adding a strict-data CLI flag.
- [ ] **TR-12.9** · `bug` · S · P0 · `models/losses.py:34` — `TODO: Implement onset_f1, pitch_ce, and velocity_mae losses` — covered in ARCH-11.6.

---

# SECTION 13 — ORPHAN / DEAD CODE (P1)

- [x] **DEAD-13.1** · `refactor` · Removed unused `pretty-midi` from both root and nested requirement manifests.
- [!] **DEAD-13.2** · `research` · The root config now labels TensorRT export keys as legacy/unimplemented, while TensorRT packages remain only in the duplicate legacy manifest. Removing all mentions from archival docs/configs is a maintainer scope decision, not an active runtime fix.
- [!] **DEAD-13.3** · `research` · The TensorRT installation guidance lives in the duplicate legacy tree and upstream reference. Do not remove historical/vendor references as if they were root runtime dependencies; reconcile the duplicate tree first.
- [~] **DEAD-13.4** · `refactor` · `api.py` now imports `json` at module scope. `main.py` retains local `yaml` imports for configuration loading/CLI startup; moving them globally is optional and could change import behavior.
- [x] **DEAD-13.5** · `refactor` · Removed the unused `ProcessingReport`/`QualityTier` imports from `api.py`.
- [ ] **DEAD-13.6** · `refactor` · S · P1 · `main.py` uses `Tuple`/`List` in annotations; converting to built-in generic syntax is a style-only cleanup, not a runtime defect.
- [x] **DEAD-13.7** · `test` · `verify_structure.py` was run successfully and its expected paths match the current root/research layout; it does not validate runtime behavior.
- [!] **DEAD-13.8** · `research` · Root `MambaSeparator` still subclasses NeMo's `NeuralModule` and uses NeMo type-checking. Removing this requires an approved architecture/dependency decision.
- [!] **DEAD-13.9** · `research` · Root `MambaTranscriber` still subclasses NeMo's `NeuralModule` and uses NeMo type-checking. Removing this requires an approved architecture/dependency decision.
- [!] **DEAD-13.10** · `research` · The root vendor snapshot is under `/research/mamba-ssm-reference/`, but an identical tracked copy remains in `stem_midi_pro/mamba/`; duplicate retention/removal is a maintainer decision.
- [x] **DEAD-13.11** · `refactor` · The former root `model.py` path is absent; its research implementation is `research/mamba3_per_track/per_track_processor.py`. Historical backlog references remain intentionally.
- [x] **DEAD-13.12** · `refactor` · The former `train_mamba3.py` path is absent; the research trainer is `research/mamba3_per_track/train.py`. Historical backlog references remain intentionally.
- [x] **DEAD-13.13** · `refactor` · The former `utils/losses_mamba3.py` path is absent; the research implementation is `research/mamba3_per_track/losses.py`. Historical backlog references remain intentionally.
- [x] **DEAD-13.14** · `refactor` · The former `configs/mamba3_config.yaml` path is absent; the research config is `research/mamba3_per_track/config.yaml`. Historical backlog references remain intentionally.

---

# SECTION 14 — SUBSEQUENT SPRINTS (post-MVP)

## 14.1 Sprint 2 — Web frontend

- [ ] **FE-14.1.1** · `feature` · L · P2 · Minimal web UI: HTML + vanilla JS, served by FastAPI. Drag-and-drop upload, progress bar (polling), download button.
- [ ] **FE-14.1.2** · `feature` · L · P2 · MIDI preview in-browser via `@tonejs/midi` or `MIDI.js`.

## 14.2 Sprint 3 — DAW integrations

- [ ] **DAW-14.2.1** · `feature` · XL · P2 · Reaper ReaScript that wraps the API.
- [ ] **DAW-14.2.2** · `feature` · XL · P2 · Ableton Max for Live device.

## 14.3 Historical monetization proposals

<!-- STATUS: research -->

- [!] **MON-14.3.1** · `research` · Stripe integration for the "$4.99 human review" proposal is not approved; no matching service exists and the current Apache-2.0 license has no commercial restriction.
- [!] **MON-14.3.2** · `research` · License-key/JWT issuance and concurrency entitlements conflict with the current Apache-2.0 license and have no approved product/payment requirements. Do not implement without owner/legal approval.

---

# SECTION 15 — HISTORICAL DEFINITION-OF-DONE PROPOSAL

<!-- STATUS: research -->

The following is not current repository policy: `make test-cpu`, `make lint`,
ruff, mypy, and signed-off commits are not configured or verified in this
checkout. For changes in this audit, use the available checks documented in
`DEVELOPMENT_GUIDE.md`; do not claim tests passed when dependencies are absent.
If SPDX headers are added, they must match the current Apache-2.0 license.

---

# SECTION 16 — HISTORICAL SPRINT PLAN

<!-- STATUS: research -->

These estimates and sprint contents were not approved or re-estimated after
repository history diverged. In particular, the CPU-only/Python 3.13,
production-hardening, and monetization assumptions remain unverified.

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

# SECTION 17 — CURRENT AUDIT CLOSEOUT

<!-- STATUS: in progress -->

This section's original completion criteria were not feasible or valid as
written (the test suite requires unavailable packages, there is no declared
CPU reference target, and some grep criteria included intentional caveats).

- [~] **AUDIT-17.1** · `infra` · Root entrypoints are identified in `README.md` and `AGENTS.md`; source/test counts should be scoped to root first-party code and exclude research, vendor, and the legacy duplicate. The duplicate-tree retention decision remains with maintainers.
- [!] **AUDIT-17.2** · `test` · `pytest` is not installed in the review environment; no test execution or coverage result can be claimed.
- [!] **AUDIT-17.3** · `research` · A zero-marker requirement is not a valid acceptance criterion: intentional TODOs document unimplemented/high-risk work. Inventory and classify them instead of deleting markers blindly.
- [!] **AUDIT-17.4** · `sec` · No dependency lockfile or `requirements-dev.txt` exists and the dependency audit was not run. Resolve an install set and run a current vulnerability scan before release.
- [~] **AUDIT-17.5** · `docs` · Root product copy was revised and duplicate-tree docs/templates now carry research/archival notices. The PRD remains explicitly aspirational and its old Part III is historical; treat any matching terms in context, not as a zero-hit goal.
- [!] **AUDIT-17.6** · `research` · No supported CPU target, trained checkpoint, valid test audio, or working end-to-end inference was available; do not publish a CPU latency figure.
- [!] **AUDIT-17.7** · `research` · Do not stamp this work complete or create a release changelog until maintainers reconcile the GitHub `v0.1.0` release description with this checkout and confirm version lineage; local history is shallow/grafted and contains no matching tag.

---

*This file is an unreconciled historical backlog, not the current source of truth or an approved production-readiness plan. The root `ARCHITECTURE.md` and source describe the current prototype. See the status notes above before acting on any item.*

*Status reconciled for this audit: 2026-10-08*
