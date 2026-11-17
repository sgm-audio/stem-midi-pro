# Stem+MIDI Pro — Verification Matrix

**Audit date:** 2026-11-17  
**Branch:** `arena/98583ceb-stem-midi-pro`  
**Scope:** functional and performance claims in `README.md`, `SUMMARY.md`, `ARCHITECTURE.md`, `USER_GUIDE.md`, and `docs/architecture-decision-records/*.md`, cross-checked against `main.py`, `api.py`, `models/`, `utils/`, configs, and tests.

## Executive verdict

**No-go for a production-ready guitar/bass separation or editable-MIDI claim.** The repository has a runnable CPU-shaped model and a reasonably well-tested API/packaging shell, but the shipped inference path initializes random weights, does not use the advertised pretrained CPU backends, drops expression output, emits the same MIDI for guitar and bass, and writes MIDI timing incorrectly. The quality report also calls a spectral-centroid heuristic `si_sdr` and uses it in the quality gate.

The strongest claims that are verified are mechanical:

- the pure-PyTorch fallback imports and runs without `mamba_ssm`;
- the small-model unit tests cover tensor shapes and finite outputs;
- audio validation, API error handling, ZIP packaging, health/auth/CORS/metrics/template endpoints, and basic SMF byte generation work in the tested environment;
- quality-tier classification is deterministic and artifact flags force `complex`.

Those mechanical results do **not** demonstrate usable separation, transcription accuracy, calibrated confidence, expression, DAW-correct timing, or the stated reference-hardware performance.

## Result definitions

- **Verified** — implemented behavior is supported by source inspection and an executable test or direct probe, within the stated scope.
- **Partial** — a structural/mechanical part works, but an important part of the documented claim is absent, misleading, untested, or only applies to a narrower path.
- **Failed** — the implementation directly contradicts the claim or produces an unusable result for the claimed use.
- **Unverifiable** — the claim is a design/documentation assertion without an executable check, trained/evaluated model, target hardware, or other supporting evidence.

## Test and probe basis

The audit environment was **Python 3.11.2**, not the documented Python 3.13 target. Core dependencies were installed into an ignored `.venv`; `mamba_ssm`, `demucs`, `basic_pitch`, and `torchaudio` were not installed.

| Command/probe | Observed result | What it does and does not establish |
|---|---|---|
| `.venv/bin/python -m pytest -q -m "not cuda"` | **90 passed, 8 skipped, 0 failed** in 4.50 s | Current test result in this environment. Two checkpoint tests skip because `mamba_ssm` is absent; six dataset tests skip because `torchaudio` is absent. It does not reproduce README's `96 passed, 2 skipped` claim and is not a Python 3.13 run. |
| `.venv/bin/python -m pytest -q -m slow` | **1 passed** in 1.61 s | Only `tests/test_perf_smoke.py`; it uses a downsized test config and a 1-second input, not the default model or an i5. |
| Small direct `StemMidiModel` probe | `MAMBA_BACKEND=pure-torch`; forward completed; stems had correct shapes; expression was absent from emitted events | Verifies the fallback and mechanical pipeline. It used a tiny test config and random weights. |
| Default config, 60-second tensor, current sandbox (`OMP_NUM_THREADS=2`) | 16.814 s wall time, RSS about 822 MB → 1,153 MB; **0 MIDI events** | A non-reference measurement only. It is not evidence for an i5 result or audio quality; the default model was randomly initialized. |
| Direct `utils.midi.build_midi_from_events` probe | SMF reparses with Mido, but an onset/duration is scaled by 60 and same-onset notes are serialized one after another | Exposes two DAW-usability defects not covered by the existing round-trip tests. |

## Critical flow trace

### 1. Audio in → validation/temp file

`POST /process` checks the filename extension and reads the complete upload into a Python `bytes` list before writing a temporary file (`api.py:631-665`). `audio_io.open_and_validate` checks decodability, duration, and sample rate (`audio_io.py:38-77`); the temp file is removed in a `finally` block (`api.py:721-763`). The API ZIP contract and validation/error tests pass.

Caveats:

- “Streams upload” is only chunked reading into memory; it is not streaming-to-disk. A large upload, model tensors, and the in-memory ZIP can coexist.
- API validation uses `soundfile` only. `.mp3` is accepted by extension, but successful MP3 decoding depends on the installed libsndfile build; no independent MP3 integration result is recorded here.
- The model then opens the file again in `main.py:_load_audio` (`main.py:420-432`), so the documented “single `SoundFile` open” optimization is not end-to-end.

### 2. Separation

`StemMidiModel` always constructs `MambaSeparator` and `MambaTranscriber` (`main.py:41-48`). `MambaSeparator` computes an STFT, projects magnitudes through Mamba blocks, applies independent sigmoid masks, and performs ISTFT (`models/mamba_separator.py:70-131`). It returns guitar, bass, residual, a `None` state cache, and two metrics.

The `backends` factory exists, including Demucs and Basic Pitch (`models/backends/factory.py:11-52`), but `main.py` never calls `build_separator` or `build_transcriber`. Therefore `configs/model_config.cpu.yaml`'s advertised Demucs/Basic Pitch path is ignored by CLI/API/model construction. The default API config also has no checkpoint and initializes random weights (`api.py:263-287`); no model binary is present in the repository (`checkpoints/README.md` only documents ignored/cache locations).

### 3. Transcription

Only the guitar stem is passed to the transcriber (`main.py:85-88`). `_logits_to_midi` turns onset/pitch/velocity/confidence tensors into events (`main.py:194-218`). The expression tensor is assigned to `_expression` and discarded. No duration head or duration value is produced by this Mamba path.

`main.py:98-99` then aliases the same enhanced event dictionary as both `guitar` and `bass`. The two MIDI files in `api.py:602-604` are consequently built from the same events.

### 4. Confidence injection

`ConfidenceInjector` looks up confidence and “energy” at each onset frame and adds `cc_127_value` (`models/confidence_injector.py:30-63`). It does not attach expression, note duration, or a note identity to a MIDI controller event. `low_confidence_count` counts **all confidence frames below threshold**, not only emitted notes (`models/confidence_injector.py:66-74`). Also, `stem_energy` in `main.py:94` is per-sample channel energy, while its index is a transcription frame index; it is not a frame-window energy envelope.

### 5. MIDI out

`utils/midi.py:53-95` creates a parseable SMF with note-on, note-off, and CC messages. It honors an input `duration_frames` mechanically, but Mamba-generated events do not contain that field, so the default duration is used on the advertised path.

Two material defects make the output not DAW-usable as claimed:

1. `ticks_per_second` is calculated as `ticks_per_beat * (60 / seconds_per_beat)` (`utils/midi.py:64-65`). For 120 BPM and 480 PPQ this is 57,600 ticks/sec; it should be 960 ticks/sec. Onset and duration timing are therefore 60 times too long.
2. Events are emitted as `note_on → note_off → CC` for every event (`utils/midi.py:71-89`). Same-onset notes and overlapping notes are forced to wait for the previous note-off, so chords become serial notes. The existing tests check count/order/parseability, not absolute timing or polyphony.

CC#127 is encoded as a syntactically valid MIDI control-change status, and the tests confirm the bytes survive a Mido round trip. However, it is a channel-mode controller (Poly Mode On), not a general per-note metadata lane. A CC is channel-wide and the implementation places it after the note-off, so a DAW cannot reliably associate it with the preceding note, especially for overlapping notes. The “each note carries confidence that a DAW can use to find uncertain notes” claim is therefore not verified and is operationally false for standard MIDI semantics.

## Verification matrix

### A. Headline, model, and performance claims

| ID | Claim and source | Evidence (file + logic/test) | Result |
|---|---|---|---|
| A-1 | CPU-only, bare-metal Intel i5 target; no GPU/CUDA required (`README.md:5,35-37`; ADR 0002) | `models/_mamba_compat.py:21-103` selects a pure-PyTorch implementation when `mamba_ssm` is absent; model tests run on CPU and no production forward calls CUDA. `requirements.txt` does not require `mamba-ssm`. | **Partial** — the fallback core is CPU-runnable, but the project does not pin a CPU-only PyTorch distribution and the optional pretrained path is not validated here. |
| A-2 | 4–8 GB RAM is sufficient; Linux/macOS/Windows support (`README.md:31-37`; `ARCHITECTURE.md:5`) | No target-hardware memory test, OS matrix, or Python 3.13 run. The only memory test uses a downsized test model and permits 2.5 GB (`tests/test_perf_smoke.py:1-56`). | **Unverifiable** |
| A-3 | Python 3.13 is supported/targeted (`README.md:5,31`; `pyproject.toml:10`; CI) | `pyproject.toml`, Dockerfiles, and CI declare 3.13. Audit execution was Python 3.11.2. | **Partial** — declared and configured, not runtime-validated in this audit. |
| A-4 | README test result: 96 passed, 2 skipped on 2026-11-04; 86 passed on Python 3.12 (`README.md:5`) | Current run: 90 passed, 8 skipped, 0 failed, with two `mamba_ssm` and six `torchaudio` skips. No archived 3.12/3.13 result is in the repository. | **Failed** — the published count is stale or not reproducible from the current tree/environment. |
| A-5 | Active development toward production readiness, with SUMMARY marking what works (`README.md:7`) | The README itself says active development. The current default output has random weights; timing, expression, bass MIDI, and quality metric issues are demonstrated below. | **Verified** as a status statement; **Failed** if read as a production-readiness indication. |
| A-6 | Mamba-SSM separation and transcription are implemented (`README.md:25-26`; `SUMMARY.md:10-11`; `ARCHITECTURE.md:72-110`) | `MambaSeparator` and `MambaTranscriber` construct and pass shape/finite-output tests (`tests/test_mamba_separator.py`, `tests/test_mamba_transcriber.py`). | **Partial** — the neural graph exists and runs, but the end-to-end product path uses random weights unless an external checkpoint is supplied. |
| A-7 | Pure-PyTorch fallback remains functional when `mamba_ssm` is missing (`README.md:70`; `SUMMARY.md:10`; ADR 0002) | Direct probe reported `MAMBA_BACKEND=pure-torch` with `mamba_ssm` absent; the Mamba separator/transcriber and full small model forward completed. | **Verified** for construction/shape-level CPU execution. It does not establish useful musical output. |
| A-8 | The fallback is “Mamba-SSM”/the official reference implementation (`README.md:5,70`; `_mamba_compat.py` docstring) | The fallback is local code in `_mamba_compat.py:28-103`; it is not the installed `mamba_ssm` package. The package is optional and absent. | **Partial** — it is a Mamba-like pure-PyTorch implementation, not evidence that the official package is a supported CPU runtime. |
| A-9 | CLI/demo process a real file and produce stems/MIDI (`README.md:39-58`; `SUMMARY.md:18`; `USER_GUIDE.md:19-31`) | CLI code calls `StemMidiModel.process_audio_file` (`main.py:459-495`); audio loading and package tests pass. No trained checkpoint is shipped; default construction is random. | **Partial** mechanically; **Failed** as a claim of usable musical output from the documented default invocation. |
| A-10 | CPU performance is about 30–60 seconds per minute, with 3-minute and 10-minute projections; one request at a time (`README.md:62-70`; `USER_GUIDE.md:34-45`; ADR 0002) | No i5 benchmark exists. TODO audit item `AUDIT-17.6` is still open. The current sandbox measured 16.814 s for 60 s with the default random model; this is not an i5 and is not a controlled reference result. API has a semaphore, but direct CLI has no concurrency concept. | **Unverifiable** for latency/RAM; **Partial** for API single-job serialization. |
| A-11 | Slakh training produces shippable weights; no trained weights are included (`README.md:97`) | No binary checkpoint is tracked (`checkpoints/README.md` documents only ignored/cache locations). Training code exists, but no quality/evaluation result demonstrates shippable weights. | **Partial** for “no weights included”; **Unverifiable** for “yields weights you can ship.” |
| A-12 | Everything runs locally; no cloud/internet needed after install (`README.md:30`; `USER_GUIDE.md:1,101-102`) | Core Mamba inference has no network call. Optional Demucs/Basic Pitch installation and first-use weight caching are explicitly download-based (`checkpoints/README.md`, `scripts/install_pretrained.py`). | **Partial** — self-hosted inference is local after dependencies/weights exist; the optional advertised pretrained setup is not offline. |

### B. Separation, transcription, expression, confidence, and MIDI

| ID | Claim and source | Evidence (file + logic/test) | Result |
|---|---|---|---|
| B-1 | Mixed audio is split into guitar and bass WAV stems (`README.md:25`; `USER_GUIDE.md:10`; `ARCHITECTURE.md:72-88`) | `MambaSeparator` returns guitar/bass tensors and API writes two WAV files. Tests verify shapes and the ZIP names. Independent sigmoid masks are applied to the same mixture phase (`models/mamba_separator.py:101-116`); no trained checkpoint is shipped. | **Partial** structurally; **Failed** as verified usable separation. |
| B-2 | Residual stem is a production output (`ARCHITECTURE.md:82-87`) | Residual exists only in the internal separator return and streaming local variables. `process_audio_file` and API package return only guitar/bass (`main.py:299-310`; `api.py:597-600`). | **Partial** |
| B-3 | “Phase-coherent” separation and phase-coherence report (`ARCHITECTURE.md:72,118-119`; `USER_GUIDE.md:69-73`) | `phase_score` is a learned head over mean hidden states (`models/mamba_separator.py:118`), not a comparison of output phases. Demucs backend returns constant `0.5` (`models/backends/demucs_separator.py:92-103`). | **Failed** as an actual phase-coherence measurement; the field exists mechanically. |
| B-4 | Onset, pitch, velocity transcription (`README.md:26`; `SUMMARY.md:11`; `ARCHITECTURE.md:89-109`) | Multi-task heads and `_logits_to_midi` exist; tests cover output shapes and bounds. No trained checkpoint or accuracy test exists. Pitch vocab is 128 despite the config comment saying “128 MIDI notes + silence”; there is no separate silence class. | **Partial** |
| B-5 | Bend/vibrato/slide expression is produced in MIDI (`README.md:26`; `USER_GUIDE.md:91`) | `expression_head` produces a tensor (`models/mamba_transcriber.py:57-66,85-90`), but `main.py:86` binds it to `_expression` and never injects it into events. `utils/midi.py` emits no pitch-wheel, aftertouch, or expression CC messages. | **Failed** end to end. The head is present but the documented MIDI feature is neither trained nor serialized. |
| B-6 | Expression is at least an advisory partial feature (`SUMMARY.md:12`; `USER_GUIDE.md:91`) | The head is present, but there is no expression loss/target wiring in the model training call and no event mapping. | **Partial** for “head exists”; **Failed** for “expression data is produced.” |
| B-7 | Both guitar and bass are transcribed to separate MIDI files (`README.md:25-27`; `USER_GUIDE.md:10-12`; `ARCHITECTURE.md:93`) | Only `guitar_stem` is sent to the transcriber (`main.py:85-88`). The result is copied into both `guitar` and `bass` keys (`main.py:98-99`), and both API files use that dictionary (`api.py:602-604`). | **Failed** |
| B-8 | Per-note confidence is injected and exported as CC#127 (`README.md:27`; `SUMMARY.md:13`; `USER_GUIDE.md:12,88`) | `ConfidenceInjector` adds `cc_127_value`; `build_midi_from_events` writes one control-change message per event; MIDI tests pass for value/count round trips. | **Partial** mechanically; **Failed** as standard per-note DAW metadata (see B-9). |
| B-9 | CC#127 lets a DAW identify/filter the confidence of each note (`USER_GUIDE.md:88`; API docs/package description) | Code emits channel-wide controller 127 after note-off (`utils/midi.py:84-89`). CC#127 is a channel-mode controller, not a per-note lane; no note ID or note-relative mapping is encoded. | **Failed** for the claimed meaning, despite valid SMF bytes. |
| B-10 | Confidence is meaningful and per-note (`README.md:26-27`; `SUMMARY.md:13`) | Eval confidence is an MC-dropout mean (`models/mamba_transcriber.py:91-114`), but weights are random and no calibration/accuracy test exists. The summary count is frame-based, not note-based (`models/confidence_injector.py:68-72`); confidence is not included in a training loss. | **Failed** for production confidence semantics; **Partial** for attaching a numeric field to events. |
| B-11 | Per-note durations are honored, with a configurable fallback (`SUMMARY.md:13`; `USER_GUIDE.md:11`) | `utils/midi.py:79-84` honors `duration_frames` and tests cover it. `_logits_to_midi` never creates `duration_frames`, so Mamba outputs always take the default 0.25-beat duration. Basic Pitch can create durations, but that backend is not wired into `main.py`. | **Partial** |
| B-12 | MIDI is editable/DAW-usable and aligned to the audio (`README.md:121`; `USER_GUIDE.md:19,86-91`) | Mido reparses the generated SMF and note counts pass. But `utils/midi.py:64-65` uses 57,600 instead of 960 ticks/sec at 120 BPM, making timing 60× too long. Same-onset events are serialized after the previous note-off. | **Failed** |
| B-13 | MIDI events are valid and sensible (`SUMMARY.md:13`; tests) | SMF header, note-on/off, velocity range, and CC value serialization are tested. There are no absolute-time, overlap, chord, note-range, expression, or DAW import tests. | **Partial** — syntactically valid, musically wrong in important cases. |
| B-14 | A `guitar.mid` and `bass.mid` package is always delivered (`USER_GUIDE.md:22-31`; `SUMMARY.md:24`) | API integration test verifies all five ZIP members using a fake model (`tests/test_api_integration.py:60-96`). | **Verified** for package contract; not for file content quality (see B-7/B-12). |

### C. Quality gates and processing report

| ID | Claim and source | Evidence (file + logic/test) | Result |
|---|---|---|---|
| C-1 | `studio` / `draft` / `complex` tiers use confidence and SI-SDR thresholds (`README.md:28`; `SUMMARY.md:14`; `ARCHITECTURE.md:131-138`; `USER_GUIDE.md:76-83`) | `ProcessingReport.quality_tier` implements configurable thresholds and tests cover studio/draft/complex boundaries (`utils/quality_gates.py:26-48`; `tests/test_quality_gates.py`). Artifact flags force `complex`. | **Verified** as deterministic routing logic. |
| C-2 | A high-confidence, ≥20 dB result is “studio”/ready as-is (`ARCHITECTURE.md:133`; `USER_GUIDE.md:78`) | The code applies that rule, but `si_sdr` is a centroid-separation proxy (`models/mamba_separator.py:120-149`; `utils/quality_gates.py:15-18`) and confidence is uncalibrated/random without a checkpoint. | **Failed** as an evidence-based quality certification; **Verified** only as the rule. |
| C-3 | Artifact flags detect clipping, noise floor, and phase cancellation (`ARCHITECTURE.md:136-138`; `SUMMARY.md:14`) | `_detect_artifacts` checks output peak >.99, mean-square >.1, and anti-correlation <−.9 (`main.py:220-244`). The noise check is not noise-floor estimation, and phase coherence is not measured from phases. | **Partial** |
| C-4 | `si_sdr`, phase coherence, average confidence, artifact flags, low-confidence notes, and quality tier are in `processing_report.json` (`SUMMARY.md:16`; `USER_GUIDE.md:51-73`) | `api.create_response_zip` writes all six fields (`api.py:606-620`); API integration test checks report values. | **Verified** for schema/presence; **Partial** for metric meanings. |
| C-5 | Low-confidence count means the number of low-confidence MIDI notes (`USER_GUIDE.md:64,73`; API docs) | `ConfidenceInjector` counts every confidence frame below threshold, including frames with no emitted note (`models/confidence_injector.py:68-72`). | **Failed** |
| C-6 | Draft routes to an editor; complex offers human review/refund (`utils/quality_gates.py` output referenced by `ARCHITECTURE.md:134-135` and `USER_GUIDE.md:83`) | `route_by_quality` returns UI-shaped dictionaries, but no editor, human-review service, payment, or refund endpoint exists. `USER_GUIDE.md:91` correctly says there is no built-in editor. | **Partial** — routing metadata exists; the offered services do not. |
| C-7 | The report is “transparent”/an honest routing signal (`README.md:28`; `SUMMARY.md:16`) | Field names and artifact list are visible, but proxy SI-SDR, constant/random phase score, frame-counted confidence, and untrained weights make the signal misleading. | **Partial** |

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
| D-12 | Optional checkpoint loading makes the service usable (`README.md:97`; `api.py`/checkpoint docs) | `api.load_model` loads a checkpoint only if `MODEL_CHECKPOINT_PATH` exists; otherwise it constructs a fresh model (`api.py:278-287`). No checkpoint is shipped. | **Partial** — loading machinery exists; default deployment is random-init. |
| D-13 | No persistent user audio; self-hosted processing stays local (`USER_GUIDE.md:101-102`; `ARCHITECTURE.md:231-240`) | Temp file cleanup and in-memory response are implemented. The API logs the original filename in `processing_started` (`api.py:732`), and legal/privacy compliance cannot be inferred from this code alone. | **Partial** for deletion/local processing; **Unverifiable** for GDPR/compliance claims. |
| D-14 | Security architecture includes input sanitization, no arbitrary code paths, and memory-safe processing (`ARCHITECTURE.md:217-240`) | Extension/content validation, template allowlist, auth, size limits, and temp cleanup exist. No formal security test, rate limiting, or resource isolation is present; architecture itself marks full hardening planned. | **Partial** |
| D-15 | CPU Docker/self-host deployment is implemented; systemd/compose are planned (`SUMMARY.md:51-53`; `ARCHITECTURE.md:243-315`) | `Dockerfile.cpu` exists and binds uvicorn to 0.0.0.0; no systemd unit or compose file exists. The Docker image still runs the default random-init model unless a checkpoint/config is supplied. | **Partial** |
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
| E-7 | Training on Slakh yields weights that can be shipped (`README.md:97`) | No data, checkpoint, metric, or evaluation report is included; model output path is not quality-validated. | **Unverifiable** |
| E-8 | Learning-rate scheduling and model adaptation/domain adaptation are implemented (`ARCHITECTURE.md:371-381`) | `train.py` creates AdamW but no scheduler; no domain-adaptation implementation is present. | **Failed** for scheduling; **Unverifiable** for adaptation claims. |
| E-9 | Loss/report metrics are true SI-SDR and quality metrics (`ARCHITECTURE.md:123-129,388-393`) | Training loss code has no SI-SDR function; separator explicitly calls the inference value a centroid proxy (`models/mamba_separator.py:133-149`). | **Failed** if interpreted as true inference SI-SDR; **Partial** as a report field. |

### F. Streaming claims

| ID | Claim and source | Evidence (file + logic/test) | Result |
|---|---|---|---|
| F-1 | Chunked 2-second inference with configurable overlap (`ARCHITECTURE.md:141-148`; `SUMMARY.md:62`) | `process_audio_streaming` computes chunk/hop and the streaming test verifies three no-overlap chunks (`main.py:312-379`; `tests/test_streaming_inference.py`). | **Verified** for basic chunk iteration. |
| F-2 | Overlap-add recombination (`ARCHITECTURE.md:146`; `main.py` docstring) | `main.py:368-381` trims each chunk and concatenates tensors. It does not apply an overlap window or sum/normalize overlapping samples. | **Failed** |
| F-3 | True SSM state passing/state cache (`ARCHITECTURE.md:83,143-146`; old design) | Separator returns `None` for state (`models/mamba_separator.py:125-130`); streaming restarts each chunk. SUMMARY correctly marks it planned. | **Failed** as a current feature; **Verified** as a documented limitation. |
| F-4 | Fixed memory, deterministic latency, persistent state, and boundary handling (`ARCHITECTURE.md:186-195`) | The section is `[PLANNED]`; streaming first loads the full file and accumulates all chunk outputs (`main.py:327-381`). No state/latency test exists. | **Unverifiable** as a future design; **Failed** as current behavior. |
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
| Separation/transcription | `[IMPLEMENTED]` | `[IMPLEMENTED]` | The graphs run and tests pass, but default weights are random and the advertised pretrained backend config is ignored. Treat as structural implementation, not a usable model. |
| Expression | `[PARTIAL]`, head exists but untrained | Transcription section is `[IMPLEMENTED]` with an untrained-expression caveat | SUMMARY is closer, but still overstates delivery: the head is discarded before MIDI serialization. |
| Confidence/CC#127 | `[IMPLEMENTED]` | `[IMPLEMENTED]` | Numeric fields and CC bytes exist; standard MIDI does not provide the claimed per-note semantics, and confidence is not calibrated/note-counted. |
| Quality gates | `[IMPLEMENTED]` | `[IMPLEMENTED]` | Threshold logic is implemented. Its `si_sdr` input is a centroid proxy and its routing offers nonexistent editor/human-review services. |
| Streaming | True SSM state passing `[PLANNED]`; current chunks restart state | Streaming section `[PARTIAL]`, separate true-streaming section `[PLANNED]` | Chunk processing exists; it is not stateful and does not perform overlap-add. Both documents should say “chunk-and-concatenate” until fixed. |
| Processing report | `[IMPLEMENTED]` | Data flow `[IMPLEMENTED with caveats]` | JSON schema is present. Metric and confidence semantics are not reliable enough for a quality gate. |
| API | Endpoints mostly `[IMPLEMENTED]` | API `[IMPLEMENTED]` | Shell behavior is mostly verified. `/process` still invokes random-init Mamba by default; concurrency fail-fast is race-dependent; upload is buffered in memory. |
| Training/data | Loop/loaders/losses `[IMPLEMENTED]` | Canadian integration `[PARTIAL]` | Loop and loss functions exist, but pitch/velocity/duration losses are not wired, data loaders use synthetic fallback, and Canadian weighting/splitting claims are not implemented. |
| Deployment/observability | Docker `[IMPLEMENTED]`, systemd/compose planned, observability `[PARTIAL]` | Deployment `[PLANNED]`, monitoring `[PARTIAL]` | CPU Dockerfile and `/metrics` exist; systemd/compose/Grafana/OTel do not. |
| Production status | SUMMARY is an audit of “works today” | ARCHITECTURE retains historical/planned sections | The README warning is necessary but insufficient: the documented CLI/API path should not be called production-ready until the P0 output defects and checkpoint/backend integration are fixed. |

## ADR verification

| ADR | Functional/performance decision | Evidence and result |
|---|---|---|
| 0001 | PolyForm Small Business licensing and generated-output licensing | ADR 0001 is explicitly superseded by ADR 0006. It is historical, not a current product claim. **Verified as superseded.** |
| 0002 | CPU-only bare-metal i5; no CUDA-only paths; pure-PyTorch Mamba; 30–60 s/min; one request | CPU fallback and no required `mamba_ssm` are verified. Reference hardware timing, 4–8 GB sufficiency, and official-package CPU behavior are not. API has one processing semaphore with a fail-fast race. **Partial / Unverifiable.** |
| 0003 | Remove NeMo/Lightning; plain module and custom loop with gradient accumulation, CPU bf16 autocast, checkpointing, early stopping | Plain module, custom loop, accumulation, checkpointing, and early stopping are present. CPU bf16 autocast is absent. **Partial; bf16 portion Failed.** |
| 0004 | v1 path is the only production path; Mamba-3 moved under research; old dataset names retired | The tree has the research relocation and old names are not used in production imports. However “production path” overstates readiness because main ignores the backend factory and defaults to random Mamba. **Partial.** |
| 0005 | Mamba-3 research is isolated, as-is, with no support/API-stability guarantee | Research files and disclaimers exist; there is no automated import boundary/API guarantee. **Partial.** |
| 0006 | Apache-2.0 for code, trained weights, research; headers/metadata/NOTICE; no paid path | `/LICENSE`, `pyproject.toml`, current README, and most source headers support the current Apache-2.0 metadata. No trained weights are present to verify a weights distribution. ADR 0001 is marked superseded. **Verified** for repository metadata; legal/commercial compliance is outside a code audit. |

## Recommended P0 corrections before claiming usable production output

1. Wire `StemMidiModel` to the selected backend factory, or remove the unused Demucs/Basic Pitch configuration and clearly require a trained Mamba checkpoint. Fail readiness/process rather than silently serving random weights.
2. Fix MIDI time conversion (`ticks_per_second = ticks_per_beat * 1_000_000 / tempo_us_per_beat`, or equivalent), preserve absolute event positions, and schedule note-offs independently so chords/overlaps remain polyphonic.
3. Replace CC#127 confidence tagging with a documented interoperable representation, or explicitly call it nonstandard/channel-wide metadata. Add absolute-time/chord/overlap DAW round-trip tests.
4. Transcribe bass independently, attach durations from a trained duration model or backend, and serialize expression using a specified MIDI mapping. Do not advertise expression until the head is trained and emitted.
5. Rename `si_sdr` to the proxy metric and prevent it from being interpreted as dB SI-SDR; implement reference-based evaluation before using it in a “studio” gate.
6. Define confidence at note level, calibrate it on held-out data, and make `low_confidence_notes` count emitted notes rather than frames.
7. Run and commit a reproducible Python 3.13 benchmark on the stated i5/4–8 GB target, including peak RSS, full default model, checkpoint, separation/transcription stages, and real audio quality metrics.
8. Add tests for the documented failure modes: missing checkpoint, missing optional dependencies, backend selection, 44.1/48 kHz packaging, MP3 decoding, MIDI absolute timing, polyphony, expression, bass/guitar event independence, and full API startup/warmup.
