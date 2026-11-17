# Stem+MIDI Pro — Verification Matrix

**Audit date:** 2026-11-17
**Branch:** `arena/98583ceb-stem-midi-pro`
**Scope:** functional and performance claims in `README.md`, `SUMMARY.md`, `ARCHITECTURE.md`, `USER_GUIDE.md`, `SETUP.md`, `API_DOCUMENTATION.md`, `DEVELOPMENT_GUIDE.md`, `TODO.md`, `PRD_AND_ARD.md`, `docs/operations/`, `checkpoints/README.md`, user templates, and `docs/architecture-decision-records/*.md`, cross-checked against `main.py`, `api.py`, `models/`, `utils/`, configs, and tests.

## Executive verdict

**The P0 implementation defects listed in this matrix have been remediated at the code-contract level, but the repository remains no-go for a production-quality separation/transcription claim.** Native-Mamba file/API inference requires a validated project checkpoint; the explicit pretrained CPU profile is accepted only when both dependency backends attest to their own weights. Configured backends are constructed or fail explicitly, guitar and bass event streams are independent, MIDI timing/polyphony/expression are serialized correctly, confidence is note-counted, and the report names the separation value as a proxy. The repository still ships no trained project checkpoint and has no reference-i5 benchmark, held-out separation/transcription evaluation, confidence calibration, or DAW import evidence.

The strongest claims that are verified are mechanical:

- the pure-PyTorch fallback imports and runs without `mamba_ssm`;
- the small-model unit tests cover tensor shapes and finite outputs;
- audio validation, API error handling, ZIP packaging, health/auth/CORS/metrics/template endpoints, and basic SMF byte generation work in the tested environment;
- quality-tier classification is deterministic and artifact flags force `complex`.

Those mechanical results do **not** demonstrate usable separation, transcription accuracy, calibrated confidence, expression quality, or the stated reference-hardware performance.

## Remediation status

The following code-level corrections are now covered by focused regression tests in
`tests/test_remediation_contracts.py` plus the existing suite:

- API/CLI native-Mamba file inference rejects missing checkpoints; the explicit pretrained-backend profile is accepted only when both backends attest to their own weights. Random-weight structural forwards are marked with `untrained_weights` and route to manual review.
- `build_separator` and `build_transcriber` are used when `backends` is configured; unknown names and missing optional dependencies fail with actionable errors.
- Guitar and bass are transcribed through separate calls and packaged from separate event dictionaries.
- MIDI converts audio-frame positions using the configured sample rate/hop, schedules absolute ticks, honors durations, preserves chords/overlaps, and maps expression to pitch-wheel/CC1/CC74.
- Confidence is attached to emitted notes, counted at note level, retained canonically in JSON, and serialized as Polyphonic Key Pressure rather than CC#127.
- The separation value is exposed as `separation_proxy` with explicit non-SI-SDR semantics; artifact/untrained flags prevent heuristic studio routing.

These tests prove serialization/orchestration contracts only. They do not prove
that a future checkpoint separates real recordings, transcribes accurately, has
calibrated confidence/expression, or meets a target DAW or hardware benchmark.

## Result definitions

- **Verified** — implemented behavior is supported by source inspection and an executable test or direct probe, within the stated scope.
- **Partial** — a structural/mechanical part works, but an important part of the documented claim is absent, misleading, untested, or only applies to a narrower path.
- **Failed** — the implementation directly contradicts the claim or produces an unusable result for the claimed use.
- **Unverifiable** — the claim is a design/documentation assertion without an executable check, trained/evaluated model, target hardware, or other supporting evidence.

## Test and probe basis

The audit environment was **Python 3.11.2**, not the documented Python 3.13 target. Core dependencies were installed into an ignored `.venv`; `mamba_ssm`, `demucs`, `basic_pitch`, and `torchaudio` were not installed.

| Command/probe | Observed result | What it does and does not establish |
|---|---|---|
| `.venv/bin/python -m pytest -q -m "not cuda"` | **139 passed, 2 skipped** in 8.47 s | Full non-CUDA regression run in Python 3.11.2, including the review-fix and remediation suites. Two checkpoint tests skip because `mamba_ssm` is absent. |
| `.venv/bin/python -m pytest -q -m slow` | **1 passed, 140 deselected** in 1.75 s | Only the slow-marked smoke test; it uses a downsized test config and a 1-second input, not the default model or an i5. |
| Focused remediation regressions | Backend selection, checkpoint refusal/marking, independent guitar/bass events, absolute MIDI timing, polyphony, expression mapping, Poly Pressure confidence, proxy naming, and note-level counts | Verifies serialization/orchestration contracts with fake or random test fixtures. It does not establish learned audio quality. |
| Default config / target-hardware probe | Not rerun as a production benchmark because no checkpoint is shipped and the sandbox is not the reference i5 | Correct decision: random-weight runtime and non-reference timing cannot certify quality or performance. |
| Direct `utils.midi.build_midi_from_events` regression | Focused tests now verify approximately 960 ticks/sec at 120 BPM, duration timing, same-onset chords, overlaps, expression messages, and Polyphonic Key Pressure | Establishes the serializer contract; external DAW import evidence is still required. |

## Critical flow trace

### 1. Audio in → validation/temp file

`POST /process` checks the filename extension and reads the complete upload into a Python `bytes` list before writing a temporary file (`api.py:631-665`). `audio_io.open_and_validate` checks decodability, duration, and sample rate (`audio_io.py:38-77`); the temp file is removed in a `finally` block (`api.py:721-763`). The API ZIP contract and validation/error tests pass.

Caveats:

- “Streams upload” is only chunked reading into memory; it is not streaming-to-disk. A large upload, model tensors, and the in-memory ZIP can coexist.
- API validation uses `soundfile` only. `.mp3` is accepted by extension, but successful MP3 decoding depends on the installed libsndfile build; no independent MP3 integration result is recorded here.
- The model then opens the file again in `main.py:_load_audio` (`main.py:420-432`), so the documented “single `SoundFile` open” optimization is not end-to-end.

### 2. Separation

`StemMidiModel` uses `models.backends.factory` whenever a `backends` section is configured. The native Mamba wrapper, Demucs backend, and fake test backend satisfy the same separator contract; Basic Pitch and the native Mamba wrapper satisfy the transcriber contract. Missing optional packages raise an actionable install/configuration error. No trained checkpoint is shipped, and `api.load_model` refuses to construct a serving model without one.

The native Mamba separator computes independent masks and returns guitar,
bass, residual, and a `[centroid_separation_proxy, phase_coherence]` tensor.
The first value is explicitly a heuristic and is not reference-based SI-SDR.

### 3. Transcription

`main.py` calls the transcriber independently for guitar and bass. Backend-provided events are preserved; frame-logit backends are decoded separately. Event dictionaries carry absolute frame positions, optional duration frames, normalized expression fields, and confidence. The expression-to-MIDI mapping is implemented, but expression quality remains unverified without a trained checkpoint.

### 4. Confidence injection

`ConfidenceInjector` aligns frame confidence and an adaptive frame-level energy envelope to each emitted event. It clamps normalized confidence, attaches note-level JSON metadata, and counts only emitted notes below the threshold. The JSON `cc_127_value` key is a deprecated compatibility alias; the MIDI serializer uses note-specific Polyphonic Key Pressure and records the encoding caveat in the summary.

### 5. MIDI out

`utils/midi.py` schedules all messages on an absolute tick timeline. At the default 120 BPM and 480 PPQ, one second is 960 ticks; frame positions use the configured sample rate and hop length. Duration frames set note-off positions, and priority ordering preserves same-onset chords and overlapping notes. Expression fields map to pitch-wheel, CC1, and CC74. Focused tests cover timing, overlaps, expression, round trips, and the 48 kHz-capable parameter path.

Polyphonic Key Pressure is a standard note-specific MIDI message, but DAW support for using it as a confidence overlay varies. The JSON event data is the canonical interoperable representation; no CC#127 channel-mode message is emitted.

## Verification matrix

### A. Headline, model, and performance claims

| ID | Claim and source | Evidence (file + logic/test) | Result |
|---|---|---|---|
| A-1 | CPU-only, bare-metal Intel i5 target; no GPU/CUDA required (`README.md:5,35-37`; ADR 0002) | `models/_mamba_compat.py:21-103` selects a pure-PyTorch implementation when `mamba_ssm` is absent; model tests run on CPU and no production forward calls CUDA. `requirements.txt` does not require `mamba-ssm`. | **Partial** — the fallback core is CPU-runnable, but the project does not pin a CPU-only PyTorch distribution and the optional pretrained path is not validated here. |
| A-2 | 4–8 GB RAM is sufficient; Linux/macOS/Windows support (`README.md:31-37`; `ARCHITECTURE.md:5`) | No target-hardware memory test, OS matrix, or Python 3.13 run. The only memory test uses a downsized test model and permits 2.5 GB (`tests/test_perf_smoke.py:1-56`). | **Unverifiable** |
| A-3 | Python 3.13 is supported/targeted (`README.md:5,31`; `pyproject.toml:10`; CI) | `pyproject.toml`, Dockerfiles, and CI declare 3.13. Audit execution was Python 3.11.2. | **Partial** — declared and configured, not runtime-validated in this audit. |
| A-4 | Reproducible test result is documented (`README.md`; `SUMMARY.md`) | Final local counts are recorded below after the remediation run; Python 3.11 is available here, while the project target is Python 3.13. Optional `mamba_ssm`/`torchaudio` tests remain environment-dependent. | **Partial** — local regression is reproducible, target-runtime evidence is pending. |
| A-5 | Active development with explicit production boundary (`README.md:7`) | README, API, CLI, and ADR 0007 require a validated project checkpoint for native Mamba file inference and allow only the explicit pretrained backend exception; no trained quality evidence is shipped. | **Verified** as a status statement; production readiness remains **unverified**. |
| A-6 | Mamba-SSM separation and transcription are implemented (`README.md:25-26`; `SUMMARY.md:10-11`; `ARCHITECTURE.md:72-110`) | `MambaSeparator` and `MambaTranscriber` construct and pass shape/finite-output tests (`tests/test_mamba_separator.py`, `tests/test_mamba_transcriber.py`). | **Partial** — the neural graph exists and runs, but the end-to-end product path uses random weights unless an external checkpoint is supplied. |
| A-7 | Pure-PyTorch fallback remains functional when `mamba_ssm` is missing (`README.md:70`; `SUMMARY.md:10`; ADR 0002) | Direct probe reported `MAMBA_BACKEND=pure-torch` with `mamba_ssm` absent; the Mamba separator/transcriber and full small model forward completed. | **Verified** for construction/shape-level CPU execution. It does not establish useful musical output. |
| A-8 | The fallback is “Mamba-SSM”/the official reference implementation (`README.md:5,70`; `_mamba_compat.py` docstring) | The fallback is local code in `_mamba_compat.py:28-103`; it is not the installed `mamba_ssm` package. The package is optional and absent. | **Partial** — it is a Mamba-like pure-PyTorch implementation, not evidence that the official package is a supported CPU runtime. |
| A-9 | CLI/demo process a real file and produce stems/MIDI (`README.md:39-58`; `SUMMARY.md:18`; `USER_GUIDE.md:19-31`) | CLI code calls `StemMidiModel.process_audio_file` (`main.py:459-495`); audio loading and package tests pass. No trained checkpoint is shipped; default construction is random. | **Partial** mechanically; **Failed** as a claim of usable musical output from the documented default invocation. |
| A-10 | CPU-only target and one request at a time (`README.md`; `USER_GUIDE.md`; ADR 0002) | API semaphore and CPU fallback are implemented. No target-i5 benchmark, RAM result, or trained-checkpoint throughput evidence exists; historical seconds-per-minute estimates were removed from current user claims. | **Partial** for the service contract; **Unverifiable** for performance. |
| A-11 | Slakh training produces shippable weights; no trained weights are included (`README.md:97`) | No binary checkpoint is tracked (`checkpoints/README.md` documents only ignored/cache locations). Training code exists, but no quality/evaluation result demonstrates shippable weights. | **Partial** for “no weights included”; **Unverifiable** for “yields weights you can ship.” |
| A-12 | Everything runs locally; no cloud/internet needed after install (`README.md:30`; `USER_GUIDE.md:1,101-102`) | Core Mamba inference has no network call. Optional Demucs/Basic Pitch installation and first-use weight caching are explicitly download-based (`checkpoints/README.md`, `scripts/install_pretrained.py`). | **Partial** — self-hosted inference is local after dependencies/weights exist; the optional advertised pretrained setup is not offline. |

### B. Separation, transcription, expression, confidence, and MIDI

| ID | Claim and source | Evidence (file + logic/test) | Result |
|---|---|---|---|
| B-1 | Mixed audio is split into guitar and bass WAV stems (`README.md:25`; `USER_GUIDE.md:10`; `ARCHITECTURE.md:72-88`) | `MambaSeparator` returns guitar/bass tensors and API writes two WAV files. Tests verify shapes and the ZIP names. Independent sigmoid masks are applied to the same mixture phase (`models/mamba_separator.py:101-116`); no trained checkpoint is shipped. | **Partial** structurally; **Failed** as verified usable separation. |
| B-2 | Residual stem is a production output (`ARCHITECTURE.md:82-87`) | Residual exists only in the internal separator return and streaming local variables. `process_audio_file` and API package return only guitar/bass (`main.py:299-310`; `api.py:597-600`). | **Partial** |
| B-3 | “Phase-coherent” separation and phase-coherence report (`ARCHITECTURE.md:72,118-119`; `USER_GUIDE.md:69-73`) | `phase_score` is a learned head over mean hidden states (`models/mamba_separator.py:118`), not a comparison of output phases. Demucs backend returns constant `0.5` (`models/backends/demucs_separator.py:92-103`). | **Failed** as an actual phase-coherence measurement; the field exists mechanically. |
| B-4 | Onset, pitch, velocity transcription (`README.md:26`; `SUMMARY.md:11`; `ARCHITECTURE.md:89-109`) | Multi-task heads and `_logits_to_midi` exist; tests cover output shapes and bounds. No trained checkpoint or accuracy test exists. Pitch vocab is 128 despite the config comment saying “128 MIDI notes + silence”; there is no separate silence class. | **Partial** |
| B-5 | Bend/vibrato/slide expression is serialized in MIDI (`README.md:28`; `USER_GUIDE.md`) | `_logits_to_midi` preserves normalized expression fields and `utils/midi.py` maps bend→pitch-wheel, vibrato→CC1, slide→CC74. `tests/test_remediation_contracts.py` verifies values. No trained-expression quality evidence exists. | **Verified** as a serialization contract; **Unverifiable** as a learned musical feature. |
| B-6 | Expression is an advisory partial feature (`SUMMARY.md`) | The head and mapping exist, but expression targets/loss/calibration and a trained checkpoint are absent. | **Partial**; no quality claim. |
| B-7 | Both guitar and bass are transcribed to separate MIDI files (`README.md`; `USER_GUIDE.md`) | `_run_transcriber` is called separately for each stem, backend instrument hints are set, and API selects `midi_data["guitar"]` and `midi_data["bass"]`. Fake-backend regression uses distinct notes. | **Verified** structurally; musical independence still depends on weights/backend. |
| B-8 | Per-note confidence is interoperable metadata | `ConfidenceInjector` stores canonical note JSON and `utils/midi.py` emits Polyphonic Key Pressure. The old `cc_127_value` is JSON-only. | **Verified** for the documented representation; DAW overlay support remains variable. |
| B-9 | A DAW can identify/filter confidence per note | Polyphonic Key Pressure carries a note number and value, while JSON is canonical. A DAW that ignores poly pressure must use the JSON/import path. No CC#127 channel-mode message is emitted. | **Partial** — protocol semantics are explicit, but host support is not universal. |
| B-10 | Confidence is meaningful and per-note (`README.md`; `SUMMARY.md`) | Counts and values are attached to emitted events and tested, but MC-dropout values are not calibrated against held-out labels and no trained weights ship. | **Partial** mechanically; **Unverifiable** as calibrated confidence. |
| B-11 | Per-note durations are honored, with a configurable fallback (`SUMMARY.md`; `USER_GUIDE.md`) | `utils/midi.py` honors positive `duration_frames`; Basic Pitch converts note durations; missing durations use the configurable beat fallback. | **Verified** for serialization; native Mamba duration quality is not established. |
| B-12 | MIDI is editable/aligned to audio (`README.md`; `USER_GUIDE.md`) | Absolute ticks use sample rate/hop and 120 BPM/480 PPQ; independent scheduling preserves same-onset chords/overlaps. Focused timing/polyphony tests pass. | **Verified** as a file-timing contract; **Unverifiable** as a DAW import result until external DAW testing. |
| B-13 | MIDI events are valid and sensible (`SUMMARY.md`; tests) | SMF parseability, absolute timing, duration, velocity/range validation, expression, poly pressure, chords, and overlaps are tested. | **Verified** mechanically; musical content depends on the checkpoint. |
| B-14 | A `guitar.mid` and `bass.mid` package is always delivered (`USER_GUIDE.md:22-31`; `SUMMARY.md:24`) | API integration test verifies all five ZIP members using a fake model (`tests/test_api_integration.py:60-96`). | **Verified** for package contract; not for file content quality (see B-7/B-12). |

### C. Quality gates and processing report

| ID | Claim and source | Evidence (file + logic/test) | Result |
|---|---|---|---|
| C-1 | `studio` / `draft` / `complex` tiers use confidence and a separation threshold (`README.md`; `SUMMARY.md`; `ARCHITECTURE.md`) | `ProcessingReport.quality_tier` uses `separation_proxy`, configurable `min_separation_proxy`, confidence thresholds, and artifact/untrained flags. Existing quality tests and remediation tests cover the boundaries. | **Verified** as deterministic routing logic. |
| C-2 | A high-confidence result is certified “studio”/ready as-is | The code calls this a heuristic gate, not SI-SDR or accuracy certification; no trained weights/calibration/held-out evaluation are present. | **Unverifiable** as a quality certification; **Verified** only as conservative routing metadata. |
| C-3 | Artifact flags detect clipping/noise/phase cancellation | `_detect_artifacts` retains heuristic checks; any non-`none` flag forces complex. These checks are not independent noise-floor/phase-quality measurements. | **Partial** |
| C-4 | The processing report exposes honest metrics and status | ZIP JSON includes `separation_proxy`, `metric_semantics`, deprecated alias, phase score, average confidence, encoding, artifact flags, note count, model-weight status, and tier; the API integration contract remains covered. | **Verified** for schema; **Partial** for metric meanings. |
| C-5 | Low-confidence count means low-confidence emitted notes | `ConfidenceInjector` counts only event-aligned note confidences; a focused regression uses 100 frames and two emitted notes. | **Verified** mechanically; calibration remains unverified. |
| C-6 | Draft/complex routing advertises available services | Routing now says `manual_review`/`review_in_daw` and no longer promises a built-in editor, paid human review, refund, or payment flow. | **Verified** as honest routing metadata. |
| C-7 | The report is transparent/an honest routing signal | Names and caveats are explicit, random weights are marked, and proxy/quality semantics are documented. It is still not an accuracy certificate without evaluation data. | **Partial** |

### D. API and operational claims

| ID | Claim and source | Evidence (file + logic/test) | Result |
|---|---|---|---|
| D-1 | FastAPI async service with lifespan startup/shutdown and documented endpoints (`ARCHITECTURE.md:36-53`; `SUMMARY.md:20-35`) | `api.py:319-353` defines lifespan; inference is moved to a thread (`api.py:668-682`). Endpoint tests and source inspection cover the listed routes. | **Verified** for the API shell. |
| D-2 | `POST /process` returns ZIP with stems, MIDI, report (`SUMMARY.md:24`; `USER_GUIDE.md:19-31`) | `api.py:591-623` and `tests/test_api_integration.py` verify the five entries. | **Verified** for packaging; content caveats remain. |
| D-3 | 500 MB cap, 10-minute duration cap, extension/content/sample-rate validation (`SUMMARY.md:24`; `USER_GUIDE.md:49`; `ARCHITECTURE.md:47`) | `api.py:631-665`, `audio_io.py:38-77`, and API validation tests exercise caps/44.1 kHz/FLAC/stereo/invalid content. | **Verified** for tested WAV/FLAC paths. MP3 decoder portability and memory implications are not established. |
| D-4 | Bearer API-key auth only on `/process`, with 401/403 (`SUMMARY.md:25`; `ARCHITECTURE.md:49`) | `require_api_key` uses `secrets.compare_digest` (`api.py:414-423`); auth tests pass. | **Verified** |
| D-5 | CORS allowlist and wildcard credential behavior (`SUMMARY.md:27`; `ARCHITECTURE.md:50`) | Startup parsing and `CORSMiddleware` configuration (`api.py:367-390`); CORS tests pass. | **Verified** |
| D-6 | `/live` always 200; `/ready`/`/health` return readiness and 503 while loading (`SUMMARY.md:28-30`; `ARCHITECTURE.md:38-42`) | `api.py:528-549` implements exactly this; tests pass. | **Verified** for the endpoint contract. The Architecture wording “liveness: model loaded” is not true because `/live` deliberately ignores model state. |
| D-7 | Prometheus `/metrics` exposes request count, duration, and in-flight requests (`SUMMARY.md:31`; `ARCHITECTURE.md:384-403`) | Prometheus and hand-rolled fallback implementations are in `api.py:161-223`; metrics test passes. | **Verified** for the three API metrics. |
| D-8 | `/templates` and `/render-template` expose seven allowlisted user templates (`SUMMARY.md:33`; `ARCHITECTURE.md:45`) | `ALLOWED_TEMPLATES` and basename-safe template engine are used in `api.py:260,771-812`; template tests pass. | **Verified** |
| D-9 | One concurrent request; excess requests fail fast with 503 and `Retry-After` (`SUMMARY.md:34`; API docs; ADR 0002) | `asyncio.Semaphore(1)` is acquired around processing (`api.py:706-736`). A `sem.locked()` pre-check occurs before upload and acquire. Two requests can both pass the pre-check and the second then waits at `acquire`, so fail-fast behavior is race-dependent. | **Partial** — one processing slot is enforced, but the documented fail-fast guarantee is not. |
| D-10 | `/warmup` and automatic 1-second synthetic inference (`SUMMARY.md:35`; `ARCHITECTURE.md:45`) | `api.py:290-316,559-569` implements it and catches warmup failures. No test demonstrates a successful warmup on the full default model. | **Partial** |
| D-11 | Model-loading failures leave `/live` up and `/ready` at 503; OOM returns 503; temp files are cleaned on every path (`ARCHITECTURE.md:47`; API module docs) | Lifespan catches load exceptions (`api.py:326-334`), `_oom_exceptions` and handler map resource failures (`api.py:397-406,745-753`), and process cleanup is in `finally` (`api.py:759-763`). | **Verified** by code; API tests cover key response paths. |
| D-12 | Checkpoint/backend loading makes the service usable (`README.md:7`; `api.py`/checkpoint docs) | `api.load_model` loads a validated project checkpoint when configured, or accepts only the explicit `weights_source: pretrained_dependencies` backend pair; otherwise it catches the failure and keeps `/ready` at 503. No checkpoint is shipped. | **Partial** — loading/readiness machinery exists; usable quality still requires external weights and evaluation. |
| D-13 | No persistent user audio; self-hosted processing stays local (`USER_GUIDE.md:101-102`; `ARCHITECTURE.md:231-240`) | Temp file cleanup and in-memory response are implemented. The API logs the original filename in `processing_started` (`api.py:732`), and legal/privacy compliance cannot be inferred from this code alone. | **Partial** for deletion/local processing; **Unverifiable** for GDPR/compliance claims. |
| D-14 | Security architecture includes input sanitization, no arbitrary code paths, and memory-safe processing (`ARCHITECTURE.md:217-240`) | Extension/content validation, template allowlist, auth, size limits, and temp cleanup exist. No formal security test, rate limiting, or resource isolation is present; architecture itself marks full hardening planned. | **Partial** |
| D-15 | CPU Docker/self-host deployment is implemented; systemd/compose are planned (`SUMMARY.md:51-53`; `ARCHITECTURE.md:243-315`) | `Dockerfile.cpu` exists and binds uvicorn to 0.0.0.0; no systemd unit or compose file exists. The Docker image remains not-ready unless a validated project checkpoint or explicit pretrained-backend config is supplied. | **Partial** |
| D-16 | Horizontal scaling/stateless service, GPU scaling, model parallelism (`ARCHITECTURE.md:197-215`) | Entire section is `[PLANNED]` and describes the superseded GPU design; no load-balancer/deployment/test evidence exists. | **Unverifiable** |

### E. Training, data, and loss claims

| ID | Claim and source | Evidence (file + logic/test) | Result |
|---|---|---|---|
| E-1 | NeMo and PyTorch Lightning were removed; `StemMidiModel` is plain `nn.Module` (`SUMMARY.md:62`; ADR 0003; `ARCHITECTURE.md:62-70`) | `main.py:26` subclasses `nn.Module`; `train.py:1-10` is a custom loop; requirements contain neither package. | **Verified** |
| E-2 | Training loop supports epoch train/val/test, gradient accumulation, clipping, checkpoints, early stopping, resume (`SUMMARY.md:39`; ADR 0003) | `train.py:70-127,181-239` implements these branches and checkpoint files. Forward/training-shape tests pass, but no full default-data training result is recorded in this audit. | **Partial** — code paths exist; end-to-end training and useful convergence are not demonstrated. |
| E-3 | Custom loop includes CPU bfloat16 autocast (`ADR 0003:12`; TODO/TR-12.3) | `train.py` has no `torch.autocast`; TODO explicitly says the custom loop is complete **except** CPU-bfloat16 autocast. | **Failed** |
| E-4 | MR-STFT, spectral flatness, crest factor, onset F1, pitch CE, velocity MAE, duration IoU losses are implemented (`SUMMARY.md:43-45`; `ARCHITECTURE.md:120-129`) | Functions/classes exist and loss tests pass (`models/losses.py`). `StemMidiModel._forward_impl` passes only `pred_onsets`/`target_onsets` to `PerceptualAudioLoss` (`main.py:123-130`); pitch, velocity, and duration terms are never wired into training. | **Partial** — implementations exist, advertised multitask training does not. |
| E-5 | Slakh2100-YourMT3 and MUSDB18-HQ loaders, automatic splits, and synthetic fallback (`SUMMARY.md:42`; `ARCHITECTURE.md:359-376`) | Loader classes and missing-root synthetic fallback exist; tests would exercise them but six are skipped here due to the unnecessary `torchaudio` import. `get_data_loaders` expects pre-existing train/val/test directories rather than creating splits (`data/datasets.py:227-322`). | **Partial** |
| E-6 | Canadian-artist weighting/curation is supported (`README.md:93`; `ARCHITECTURE.md:361-369`) | README says no weighting code exists; `data/datasets.py:1-7,67-72` only logs a generic “Canadian” message. No weighting/config or Canadian dataset is shipped. | **Failed** for implementation; the limitation is honestly stated in README. |
| E-7 | Training on Slakh yields weights that can be shipped (`README.md:97`) | No data, checkpoint, metric, or evaluation report is included; model output is not quality-validated. | **Unverifiable** |
| E-8 | Learning-rate scheduling and model adaptation/domain adaptation are implemented (`ARCHITECTURE.md:371-381`) | `train.py` creates AdamW but no scheduler; no domain-adaptation implementation is present. | **Failed** for scheduling; **Unverifiable** for adaptation claims. |
| E-9 | Loss/report metrics are true SI-SDR and quality metrics (`ARCHITECTURE.md:123-129,388-393`) | Training loss code has no SI-SDR function; separator explicitly calls the inference value a centroid proxy (`models/mamba_separator.py:133-149`). | **Failed** if interpreted as true inference SI-SDR; **Partial** as a report field. |

### F. Streaming claims

| ID | Claim and source | Evidence (file + logic/test) | Result |
|---|---|---|---|
| F-1 | Chunked 2-second inference with configurable overlap (`ARCHITECTURE.md:141-148`; `SUMMARY.md`) | `process_audio_streaming` computes chunk/hop, validates positive duration, and translates local event frames to global positions. The streaming test verifies three no-overlap chunks. | **Verified** for the chunked contract. |
| F-2 | Overlap-add recombination (`ARCHITECTURE.md:146`; `main.py` docstring) | The implementation applies edge crossfade weights, sums overlapping waveform chunks, and normalizes by accumulated weights. | **Verified** mechanically; no external audio-boundary quality result. |
| F-3 | True SSM state passing/state cache (`ARCHITECTURE.md:83,143-146`; old design) | Separator returns `None` for state; streaming restarts state per chunk. | **Failed** as a current feature; **Verified** as an explicit limitation. |
| F-4 | Fixed memory, deterministic latency, persistent state, and boundary handling (`ARCHITECTURE.md:186-195`) | The current method loads the full input and output accumulators; true stateful streaming remains planned. | **Unverifiable** as a future design; **Failed** as current behavior. |
| F-5 | Real-time WebSocket/gRPC streaming (`ARCHITECTURE.md:419-421`) | No WebSocket/gRPC endpoint is present; USER_GUIDE explicitly says real-time streaming is not included. | **Unverifiable** as a planned feature; not a current claim. |

### G. Removed/historical performance and architecture claims

| ID | Claim and source | Evidence | Result |
|---|---|---|---|
| G-1 | TensorRT-LLM, FP8, CUDA graphs, sub-5-ms latency, GPU scale are removed/historical (`SUMMARY.md:58-60`; `ARCHITECTURE.md:316-337`; ADR 0002) | Current summary and architecture explicitly label them removed/not production; production model code uses CPU tensors and no TensorRT export. | **Verified** as a non-current status. |
| G-2 | The old performance section claims O(n) Mamba, hardware selective scan, FP8 2× capacity, constant-memory streaming, and kernel fusion (`ARCHITECTURE.md:316-337`) | The section is `[REMOVED]` and conflicts with the pure-PyTorch sequential scan (`models/_mamba_compat.py:93-99`). | **Verified** only as historical context; **not evidence** for current performance. |
| G-3 | Research Mamba-3 code is separate, as-is, and unsupported (`README.md:85`; `SUMMARY.md:67`; ADR 0004/0005) | Files are under `research/` and the research README/ADR disclaim support. No production import from Mamba-3 research code was found. | **Partial** — repository separation is visible; no automated boundary test or API-stability guarantee exists. |

### H. Future/planned feature claims

| Claim/source | Evidence | Result |
|---|---|---|
| Extensible new instruments, effects, alternate architectures (`ARCHITECTURE.md:339-357`) | Entire section is `[PLANNED]`; current separator has guitar/bass/residual heads only and no tremolo/wah/harmonics mapping. | **Unverifiable** |
| Systemd unit, compose, Grafana dashboards, OpenTelemetry, resource/quality/error dashboards (`SUMMARY.md:52-53`; `ARCHITECTURE.md:384-403`) | No unit/compose/Grafana/OTel implementation. `/metrics` covers only HTTP request metrics. | **Unverifiable** as planned work; **Partial** only for the existing `/metrics` endpoint. |
| Web MIDI editor, batch endpoint, real-time API, multi-instrument support, advanced articulation, AAF/OMF, VST/AU, edge/mobile, federated learning, collaboration, arrangement tools (`ARCHITECTURE.md:414-431`) | No implementation in `api.py` or production model. `USER_GUIDE.md:14,91` explicitly says editor and real-time streaming are not included. | **Unverifiable** as future design; absent today. |
| Tuning, tempo, palm-mute, harmonics, slap/pop detection (`SUMMARY.md:64`; `USER_GUIDE.md:14`) | No code; docs explicitly say not included/removed. | **Verified** as a current limitation. |

## SUMMARY versus ARCHITECTURE reconciliation

| Topic | What `SUMMARY.md` says | What `ARCHITECTURE.md` says | Audited current reality |
|---|---|---|---|
| Separation/transcription | `[PARTIAL]` | `[PARTIAL]` | Factory selection and independent streams are implemented and tested mechanically; no trained weights or quality evidence exists. |
| Expression | `[PARTIAL]`, mapped but untrained | Transcription section is `[PARTIAL]` | Mapping is serialized to standard MIDI messages; learned expression quality remains unverified. |
| Confidence/Poly Pressure | `[IMPLEMENTED]` | `[IMPLEMENTED]` | Note-level JSON and Polyphonic Key Pressure are explicit; `cc_127_value` is compatibility-only and calibration is not established. |
| Quality gates | `[IMPLEMENTED]` | `[IMPLEMENTED]` | Named proxy, untrained/artifact forcing, and manual-DAW routing are implemented; no quality certification is claimed. |
| Streaming | True SSM state passing `[PLANNED]`; current chunks restart state | Streaming section `[PARTIAL]`, true-streaming section `[PLANNED]` | Chunk processing performs waveform overlap-add and global event translation, but is not stateful or fixed-memory. |
| Processing report | `[PARTIAL]` | Data flow `[IMPLEMENTED with caveats]` | JSON names proxy semantics and model status; reference metrics/calibration remain absent. |
| API | `[PARTIAL]` until checkpoint configured | API `[IMPLEMENTED]` | Shell behavior is tested; missing checkpoints keep readiness false rather than serving random weights. Upload remains buffered in memory. |
| Training/data | Loop/loaders/losses `[IMPLEMENTED]` | Canadian integration `[PARTIAL]` | Loop and loss functions exist, but pitch/velocity/duration losses are not wired, data loaders use synthetic fallback, and Canadian weighting/splitting claims are not implemented. |
| Deployment/observability | Docker `[IMPLEMENTED]`, systemd/compose planned, observability `[PARTIAL]` | Deployment `[PLANNED]`, monitoring `[PARTIAL]` | CPU Dockerfile and `/metrics` exist; systemd/compose/Grafana/OTel do not. |
| Production status | SUMMARY is an audit of “works today” | ARCHITECTURE retains historical/planned sections | The implementation boundary is now explicit, but production-quality claims remain blocked by absent trained weights, calibration, target-hardware benchmark, and external DAW/real-audio evidence. |

## ADR verification

| ADR | Functional/performance decision | Evidence and result |
|---|---|---|
| 0001 | PolyForm Small Business licensing and generated-output licensing | ADR 0001 is explicitly superseded by ADR 0006. It is historical, not a current product claim. **Verified as superseded.** |
| 0002 | CPU-only bare-metal i5; no CUDA-only paths; pure-PyTorch Mamba; one request | CPU fallback and no required `mamba_ssm` are verified. Reference hardware timing, 4–8 GB sufficiency, and official-package CPU behavior are not. **Partial / Unverifiable.** |
| 0003 | Remove NeMo/Lightning; plain module and custom loop with gradient accumulation, CPU bf16 autocast, checkpointing, early stopping | Plain module, custom loop, accumulation, checkpointing, and early stopping are present. CPU bf16 autocast is absent. **Partial; bf16 portion Failed.** |
| 0004 | v1 path is the only deployable code path; Mamba-3 moved under research; old dataset names retired | The tree has the research relocation, old names are not used in production imports, and the v1 path now uses the configured backend factory. “Deployable” still requires an explicit trained checkpoint. **Partial.** |
| 0005 | Mamba-3 research is isolated, as-is, with no support/API-stability guarantee | Research files and disclaimers exist; there is no automated import boundary/API guarantee. **Partial.** |
| 0006 | Apache-2.0 for code, trained weights, research; headers/metadata/NOTICE; no paid path | `/LICENSE`, `pyproject.toml`, current README, and most source headers support the current Apache-2.0 metadata. No trained weights are present to verify a weights distribution. ADR 0001 is marked superseded. **Verified** for repository metadata; legal/commercial compliance is outside a code audit. |
| 0007 | Explicit checkpoint and output contracts | ADR 0007 matches the checkpoint gate, factory selection, independent event streams, MIDI expression/confidence mapping, and proxy naming implemented in this remediation. **Verified** mechanically; trained-quality evidence remains external. |

## Remaining evidence gates before claiming usable production output

1. Supply a trained checkpoint through an artifact/release process and run held-out guitar/bass separation and transcription evaluation. Record reference stems, SI-SDR or another reference-based metric, note precision/recall, duration accuracy, and expression accuracy.
2. Calibrate confidence on held-out data and document the calibration set, threshold selection, and expected error rate. The current note-level count is a reporting contract, not calibration evidence.
3. Run a reproducible Python 3.13 benchmark on the stated i5/4–8 GB target with that checkpoint, including peak RSS, stage timings, input duration, and output event counts. Do not restore the historical seconds-per-minute claim without this evidence.
4. Import generated MIDI and stems into representative DAWs (including a DAW that supports Polyphonic Key Pressure) and record timing, overlap, expression, and confidence-lane behavior. JSON remains the fallback for hosts that ignore poly pressure.
5. Install and exercise the optional Demucs + Basic Pitch backend path in a clean environment, including dependency versions and pretrained model acquisition/offline behavior.
6. Implement true stateful streaming and fixed-memory behavior separately if those planned claims are required; the current chunked overlap-add path is not a real-time API.
