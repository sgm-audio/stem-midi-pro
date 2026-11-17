# Stem+MIDI Pro — Independent Audit Report

**Repository:** `sgm-audio/stem-midi-pro` @ `6ab1a1e` (branch `arena/d6d3cd74-stem-midi-pro`)
**Audit date:** 2026-11-17
**Target under test:** CPU-only, bare-metal Intel i5, 4–8 GB RAM, Python 3.13
**Auditor stance:** adversarial. Every claim below is backed by code at `file:line` and/or a run I executed in this sandbox.

---

## 0. Verdict

> ## **NO-GO — this is not usable beyond a shape-correct prototype, and it is not currently a
> correct prototype either.**
>
> The Mamba stacks are numerically dead (activations underflow to `~1e-20`, every mask
> saturates to exactly `0.500`), the exported MIDI is time-scaled by **60×** and is monophonic
> by construction, the transcription heads are **never trained** (they receive no gradient), the
> documented 10-minute maximum **cannot be processed on the 4 GB half of the target**, and the
> service will happily return "quality-gated" stems + MIDI for **digital silence** (4 of 8 random
> initialisations produce `NaN` inside the user-facing JSON).
>
> None of this is theoretical: I reproduced each item below on a 2-vCPU / 3.9 GB box with
> `mamba_ssm` absent, exactly as on the target.

**Bottom line by question asked:**

| Question | Answer |
|---|---|
| Is the Mamba path numerically correct on CPU? | **No.** Numerically *dead* in the separator/transcriber (no residual ⇒ 22× shrinkage per block ⇒ 12-layer collapse), and *divergent* on constant/silent input (NaN, 1e31). |
| Does MIDI round-trip? | **No.** 60× timing error, mono-only serialisation, CC#127 written after note-off and is a *Channel Mode* controller, no durations, no expression. |
| Are the performance/memory claims right? | Time: conservative (measured RTF 0.30 ≈ 18 s/min). Memory: **wrong** — `O(duration)` growth of ~23 MB/s; 180 s is OOM-killed on 4 GB; 600 s needs ≈14 GB. |
| Are the failure modes honest? | Mixed. The docs *disclose* the proxy metric and missing weights, but the service has no guard against serving them, and the documented OOM→503 path does not catch the failure that actually occurs. |
| Overclaim vs `SUMMARY.md`? | Yes — several `[IMPLEMENTED]` rows are shape-only, and `[PARTIAL]` understates two of them (expression cannot reach MIDI at all; confidence is not a confidence). |

---

## 1. Method and environment (so you can falsify me)

**What I ran** (scratch scripts kept outside the tree; minimal repros in Appendix A):

| Item | Value |
|---|---|
| CPU / RAM | 2 vCPU, 3.9 GB, **no swap** (i.e. at the *low* end of the claimed target) |
| Python | 3.11.2 (repo targets 3.13 — see caveat §8) |
| torch | 2.5.1, CPU execution, `torch.cuda.is_available() == False` |
| `mamba_ssm` | **absent** → `models/_mamba_compat.MAMBA_BACKEND == "pure-torch"` (the target configuration) |
| Session | ~3.5 h; the only file added to the tree is this report |

Exercised: full forward, `/process` end-to-end via `TestClient`, `main.py` CLI, `demo.py`,
`process_audio_streaming`, MIDI build/parse with `mido`, one real training step, 8 random-seed
sweeps, INT8 quantization, pytest. Everything else is static reading with line references.

---

## 2. Risk map

| Area | Risk | Why |
|---|---|---|
| **Numerical correctness of the Mamba path** | **Critical** | Residual-free block stacking collapses activations; masks saturate to a constant ⇒ the "separator" is a 0.5× gain. Divergent/NaN on constant input. |
| **MIDI fidelity** | **Critical** | 60× timing error, monophonic serialisation of chords, CC#127 misuse, no durations, no expression, no tempo map. |
| **Real performance & memory** | **Critical** | Inference memory `O(T)` at ~23 MB per second of audio; 180 s OOM-kills; 600 s needs ≈14 GB. Training step needs 129 s and 2.7 GB *per sample*. |
| **Honest failure modes** | **High** | No checkpoint ⇒ random weights served silently; OOM surfaces as HTTP 500; NaN can reach the report JSON; silent backend substitution. |
| **Overclaim vs SUMMARY** | **High** | Losses "implemented" but unwired; expression cannot be emitted; confidence is a prediction mean; CPU config's pretrained backends are dead code. |
| **Test-suite strength** | **High** | Every defect above passes the suite. No test asserts a time value, content dependence, or accuracy; CI test job is `continue-on-error`. |

---

## 3. Critical findings

### C-1 · The Mamba stacks are numerically dead — the separator is a 0.5× gain, the transcriber is a constant
**Files:** `models/mamba_separator.py:41-52,96` · `models/mamba_transcriber.py:43-54,82-83` · `models/_mamba_compat.py:60-101`

Bare `Mamba` mixers are stacked with `nn.Sequential` **with no residual connection and no
normalisation between layers**. Upstream `mamba_ssm` always wraps a mixer in a `Block` with
`x = x + mixer(norm(x))`; here the 22×-per-block shrinkage compounds.

Measured (default config, 3 s input, RMS after each block):

```
input rms 1.0e+00 → 4.5e-02 → 1.1e-03 → 2.4e-05 → 5.6e-07 → … → 3.0e-18 → 7.3e-20
cpu config         → 4.3e-02 → 1.1e-03 → 2.5e-05 → 5.8e-07 → … → 3.8e-10
```

Consequences, all measured:

* Every mask head is evaluated at ≈0 input logits ⇒ `sigmoid(0) = 0.5`. Mask min/max over a
  30 s file: **0.491 / 0.509**; mean **0.500084**.
* Therefore `guitar_stem ≈ 0.5 × mixture`; measured output/mixture energy ratio **0.2501**
  (power) — i.e. −6 dB. With the 2-line residual fix the same measurement gives final RMS
  **1.01** (input 1.00).
* Real SI-SDR of the "guitar" stem against a known synthetic guitar source: **−9.82 dB**
  (worse than passing the mix through, which scores **+51.7 dB** on the same estimator).
* The transcriber sees a constant mel frame; its hidden state is ~`1e-13`, onset probability is
  identical on every frame (measured `min = max = mean = 0.5031`), pitch `argmax` is constant
  (0 or 47 depending on init), confidence std `5e-4`.
* Gradients at the mask heads measured `~1e-6` against an active separation loss (standalone
  probe: `mask_heads.guitar.weight` `2e-6`, last block `D` `5e-6`) — the early layers are on a
  vanishing path, so this does not train out of the current initialisation either.

**This is not "untrained" — it is structurally broken.** A trained model with this topology
would still be a near-constant gain, because the collapse happens before the mask heads.

**Remediation (small):** wrap each mixer as `h = h + block(ln(h))` (and add a final `LayerNorm`),
or use `mamba_ssm`'s own `Block`. Add a regression test asserting per-block RMS stays within
`[0.5, 2.0]` of its input.

---

### C-2 · The transcriber diverges on constant (silent) input — NaN reaches the user-facing JSON
**Files:** `models/_mamba_compat.py:85-101` (unbounded `dt`), `main.py:103-108`, `api.py:609-618`

`dt = softplus(dt_proj(dt_raw))` is unbounded and `dt_proj.bias` is zero-initialised (upstream
uses an inverse-softplus initialisation bounded to `[1e-3, 0.1]`). With no residual/norm, a
constant input drives runaway growth for some initialisations.

Seed sweep, 3 s of **digital silence**, `configs/model_config.cpu.yaml`:

| seed | max ∣onset logit∣ | NaN | notes emitted from silence | tier |
|---|---|---|---|---|
| 0 | 0.0055 | no | 259 | complex |
| 1 | 0.0055 | no | 259 | complex |
| 2 | **5.3e+31** | no | 200 | **draft** (avg_conf 0.93) |
| 3 | **NaN** | yes | 2 | complex |
| 4 | **NaN** | yes | 3 | complex |
| 5 | **NaN** | yes | 1 | complex |
| 6 | **2.0e+26** | no | 11 | draft |
| 7 | **NaN** | yes | 0 | complex |

End-to-end via `POST /process` with a 10 s all-zeros WAV: HTTP 200, ZIP containing
`processing_report.json` with `"avg_confidence": NaN` (`json.dumps` emits bare `NaN`, which is
**invalid JSON** per RFC 8259) and a `guitar.mid` with notes `29, 63, 115, 98, …`.

> A silent file produces notes. A musical file produced **zero** notes in the same run.
> Output is anti-correlated with content.

**Remediation:** residual+norm (C-1); `dt` clamp and upstream `dt_proj.bias` init; assert
`torch.isfinite()` on every returned tensor and 422/500 on violation; `json.dumps(..., allow_nan=False)`.

---

### C-3 · Exported MIDI is 60× too slow, and chords are serialised
**File:** `utils/midi.py:64-89` (line 65 is the defect)

```python
# 120 BPM -> 2 beats/sec -> ticks_per_beat * 2 ticks per second
ticks_per_second = ticks_per_beat * (60.0 / (DEFAULT_TEMPO_US / 1_000_000.0))   # 480 * 120 = 57600
```

Correct value is `ticks_per_beat * 1e6 / DEFAULT_TEMPO_US` = **960 ticks/s** at 120 BPM.
Every onset is placed **60× late**.

Measured, end-to-end (`POST /process`, 3 s source):

```
n_notes 259, first onsets (ticks) 0, 668, 1337, 2006 … ; last onset tick 172533 = 179.72 s
MIDI file length 179.8 s  ←→  source 3.0 s
```

Two further defects in the same loop:

1. **Chords are serialised.** Each event writes `note_on → note_off → CC` before the next event,
   and `current_tick` is reset to the onset tick, so same-frame notes are emitted *after* the
   previous note ends. Measured: 3 notes at `onset_frame 0` → absolute ticks `0 / 120 / 240`.
   Polyphony is impossible on the export path.
2. **Timing drift on dense material.** Because the cursor is rewound to `tick` after
   `current_tick += dur_ticks`, notes whose onset is inside the previous note's duration get
   `delta = max(0, negative) = 0` and pile up at the previous note-off.
3. **CC#127 is the wrong controller.** CC 120–127 are *Channel Mode* messages (127 = Poly Mode On);
   DAWs commonly hide or act on them. It is also written **after** `note_off` (i.e. at the note's
   *end*), and a channel controller cannot be per-note without MPE channel allocation. The README's
   "each MIDI note carries its confidence as CC#127 so your DAW can flag uncertain notes" is
   not achievable with this design.

**Remediation:** fix `ticks_per_second`; emit through a sorted (tick, priority) event list with
independent voices so overlapping notes are legal; add `duration_frames` from the model (or a
duration head); write confidence as a text meta event (`Marker`) or an unused CC (e.g. #11/#74)
on per-note channels; add a test that asserts **absolute tick values** for known frames, and a
polyphony test (2 notes, same frame → both `note_on`s at the same tick).

---

### C-4 · The transcription heads are never trained (no gradient at all)
**Files:** `main.py:125-130`, `models/losses.py:192-222`, `data/datasets.py:431-439,528-534`

`_forward_impl` calls the loss with **only** `pred_onsets`/`target_onsets`:

```python
loss, loss_dict = self.loss_fn(
    pred_stems=[guitar_stem, bass_stem], target_stems=[...],
    pred_onsets=onset_logits,
    target_onsets=self._prepare_onset_target(target_onsets, onset_logits))
```

`pred_pitch_logits`, `pred_velocities`, `pred_durations` are never passed, so the pitch CE,
velocity MAE and duration IoU losses are dead. Gradient probe after a real training forward:

```
loss_dict: spectral 1206.727 | flatness 0.019 | crest 0.443
           onset_f1 0.0 | pitch_ce 0.0 | velocity_mae 0.0 | duration_iou 0.0 | alignment 0.004
parameters with grad=None:
  transcriber.pitch_head.*, transcriber.velocity_head.*, transcriber.expression_head.*,
  transcriber.confidence_head.*, separator.mask_heads.residual.*, separator.phase_head.*
```

Worse, the one onset loss that *is* wired is non-differentiable:

```python
pred_bin = (pred_flat[:n] >= threshold).float()   # piecewise constant
```
Calling `onset_f1_loss_fn(...).backward()` alone raises *"element 0 of tensors does not require
grad"*; in the composite it contributes exactly nothing while reporting `onset_f1: 0.0`
("perfect") on an all-negative target.

The only remaining onset supervision is `_onset_alignment_loss`, which compares **two scalars**
(`mean(sigmoid(onset)·energy)` vs `mean(target·energy)`) — a global bias push, weight 0.5 against
a 1206-magnitude spectral term (relative contribution **~3e-6**). It cannot localise onsets.

And the labels are fabricated anyway: both real dataset paths generate *periodic* onsets every
0.5 s at pitch 45 (`data/datasets.py:432-439`, `527-534`), ignoring the MIDI that ships with
Slakh. The real MIDI→target bridge, `utils/midi_targets.py`, is **dead code** — its only importer
is its own unit test.

**Remediation:** pass all heads to the loss; use BCE-with-logits for onsets (or soft-F1);
multi-label pitch (BCE, 128 logits + silence) if polyphony is wanted; wire
`midi_to_targets` into both dataset classes; add a test asserting `pitch_head.weight.grad` is
non-zero and that a 2-note chord yields 2 positive pitch bins.

---

### C-5 · Training on the target hardware is not possible
**Files:** `example_data_config.yaml:9` (`batch_size: 16`), `configs/*.yaml` (`gradient_checkpointing: true`), `train.py:59-104`

| Measurement | Result |
|---|---|
| 1 optimiser step, B=1, 4 s, default config | **128.8 s**, peak RSS **2665 MB** |
| 1 step, B=16, 4 s, `configs/model_config.cpu.yaml` | **killed by the OOM killer** (`exit 137`) |
| 1 step, B=16, 4 s, default config | `RuntimeError: DefaultCPUAllocator: can't allocate memory` |

One step advances 4 s of audio in 129 s (≈32× realtime) with B=1. With 100 synthetic items that
is ~3.6 h/epoch on a *toy* dataset; Slakh/MUSDB is 20 000+ tracks. `gradient_checkpointing: true`
appears in configs and `conftest.py` but **no code reads it** (`grep` finds only config/test hits).

**Remediation:** chunk the scan (see C-6) — that alone removes ~90 % of activation memory; then
implement gradient checkpointing over layers; set a realistic default batch size and document the
achievable epoch time, or train on a rented GPU and ship weights.

---

### C-6 · Inference memory is `O(duration)` and the documented maximum is unreachable
**Files:** `models/_mamba_compat.py:90-91`, `audio_io.py:20` (`MAX_DURATION_SECONDS = 600`)

`dA` and `dBx` are materialised as full `(B, L, d_inner, N)` tensors — no chunking over `d_inner`
or `L` (this footprint is inherited from upstream's `selective_scan_ref`, which is exactly why the
reference scan is not a viable production CPU path):

```python
dA  = torch.exp(dt.unsqueeze(-1) * A)                              # (B, L, d_inner, N)
dBx = dt.unsqueeze(-1) * xc.unsqueeze(-1) * Bmat.unsqueeze(2)      # same shape
```

At 600 s: `L = 51690`, `d_inner = 1536`, `N = 16` → **5.08 GB for one tensor**, ×2 (+ temporaries)
**per layer**. Measured peak RSS, default config, mono, one request:

| Audio | Wall | Peak RSS | Result |
|---|---|---|---|
| 1 s | 0.29 s | 654 MB | ok |
| 10 s | 2.65 s | 875 MB | ok |
| 30 s | 8.46 s | 1353 MB | ok |
| 60 s | 18.0 s | **2037 MB** | ok |
| 120 s | 36.5 s | **3319 MB** | ok — 0.6 GB headroom on a 4 GB box |
| 180 s | — | — | **SIGKILL (OOM)** on 3.9 GB |
| 600 s | — | ≈14 GB (extrapolated from 22.4 MB/s) | impossible on 8 GB |

The README's "A 3-minute song takes roughly 1.5–3 minutes; a 10-minute song (the maximum) takes
5–10 minutes" is **false on the 4 GB half of the target** and false for 10 min on either half.
The `cpu` config is ~4× lighter (60 s → 878 MB) but still `O(T)` and still untrained.

**Remediation:** implement the standard chunked selective scan (carry `h` across chunks; chunk
over `L` in blocks of ~256 frames and over `d_inner`), which bounds memory and simultaneously
gives the real streaming path that C-5/ARCH-11.8.1 wants. Target: <2 GB for 600 s, and a CI
memory test at 600 s.

---

### C-7 · The shipped default is an untrained model, served without any guard
**Files:** `api.py:262-289`, `main.py:435-462`, `README.md:104-107`

`load_model()` builds `StemMidiModel(config)` when `MODEL_CHECKPOINT_PATH` is unset or missing —
no error, no warning in the response, no field in `/model-info`, no capability gate. Because of
C-1 the output is content-independent, yet `/process` returns HTTP 200 with
`quality_tier`, `avg_confidence`, `si_sdr` and a routing decision. In the seed sweep, pure
silence was once routed to **`draft`** (avg_conf 0.93), whose routing payload offers the
"+$4.99, 24 h human review" upsell — i.e. a user can be asked to pay for review of a file
containing no audio.

Also: `main.py` CLI prints a summary and **writes nothing** (`main.py:459-495`), despite README
step 4 ("Process a real audio file from the CLI"). Only the API emits artifacts.

**Remediation:** refuse to serve without a checkpoint (or require
`ALLOW_UNTRAINED=1` and stamp `"model": "untrained"` in report + `X-Model-State` header);
make the CLI write `stems/` + `midi/` + `report.json`; add a content-dependence test.

---

### C-8 · The documented OOM handling does not catch the OOM that happens
**Files:** `api.py:397-406,747` (`_oom_exceptions`), C-6 above

```python
excs: list = [MemoryError]
...
excs.append(torch.cuda.OutOfMemoryError)
```

On CPU, torch raises **`RuntimeError("DefaultCPUAllocator: can't allocate memory …")`** — observed
verbatim in this audit at 180 s inference and at B=16 training. `MemoryError` never occurs.
Result: a memory failure on the flagship 3-minute file returns **HTTP 500
`Processing failed: … DefaultCPUAllocator …`** instead of the documented `503 + Retry-After`,
and leaks the allocator message to the client.

**Remediation:** catch `RuntimeError` whose message matches `allocate.*memory|out of memory` in
addition to `MemoryError`; keep the client message generic.

---

## 4. High findings

### H-1 · The pretrained Demucs / Basic Pitch path is dead code
`models/backends/{factory,demucs_separator,basic_pitch_transcriber}.py` are only imported by
`tests/test_backends.py`. `StemMidiModel.__init__` hardcodes `MambaSeparator` /
`MambaTranscriber` (`main.py:42-43`), so:

* `configs/model_config.cpu.yaml:5-12` — "Default backends: pretrained Demucs + Basic Pitch (no
  mamba_ssm required)" — **is ignored** (verified: `type(model.separator)` is
  `models.mamba_separator.MambaSeparator`, `d_model=256`).
* The backend error messages tell users to `pip install 'stem-midi-pro[pretrained]'`; **that extra
  does not exist** in `pyproject.toml`, and neither `demucs` nor `basic-pitch` is in
  `requirements.txt`.
* `tests/test_backends.py::test_stem_midi_model_with_fake_backends` is named as if it exercised
  the factory; it passes because the *real* tiny Mamba model runs.

This matters because Demucs + Basic Pitch is the only route in this repo to *usable* output
today. Deleting the claim or wiring it are the only honest options.

### H-2 · `si_sdr` in the user-facing report is not SI-SDR — and is not monotone in quality
`models/mamba_separator.py:133-150`: `10*log10(|centroid_guitar − centroid_bass| + 1e-8) + 20`,
clamped `[0, 30]`, where centroids are in FFT **bins**. Measured values 4.75 / 8.14 / 9.51 /
12.51 / 17.1 dB on files whose true SI-SDR was **−9.8 dB**. Because scale is arbitrary
(difference of centroids in bins), two masks that differ by 10 bins hit the clamp at 30 — i.e. a
badly separated pair can score "studio". `utils/quality_gates.py:40-43` then makes
`studio` require `si_sdr ≥ 20`, a threshold with no relation to separation quality.
`SUMMARY.md:12` and `demo.py:78` disclose the proxy, but the report field is still named
`si_sdr` and printed as "SI-SDR: 17.1 dB" in `demo.py` and `scripts/eval_smoke.py`.

**Remediation:** rename the JSON field to `centroid_proxy_db`, label user output accordingly, and
compute true SI-SDR in an offline eval against reference stems (MUSDB test split).

### H-3 · "Confidence" is a prediction mean, and the gate thresholds are unreachable
`models/mamba_transcriber.py:99-112` averages 5 dropout samples of a sigmoid head — that is a
*mean*, not an uncertainty; the ensemble std measured `5e-4`, i.e. degenerate. Then:

* `main.py:105` — `avg_confidence = confidence.mean()` averages over **all frames** (mostly
  silence), not notes; `low_confidence_count` counts **frames**
  (`models/confidence_injector.py:71`) but is surfaced as `low_confidence_notes`.
* `models/confidence_injector.py:44` — `alignment = conf * sigmoid(energy*10)`, where `energy` is
  the stem **mean-square** (order 1e-3 ⇒ `sigmoid ≈ 0.50`). So `alignment ≈ 0.5·conf`, and
  `needs_review = alignment < 0.6` (`:59`) is true for essentially every note. Measured: **259 of
  259** notes flagged; `quantization_suggestion == "grid"` (needs `alignment > 0.85`) is
  unreachable.
* `main.py:94` — `stem_energy` is a **per-sample** array (`(B, T)`), but it is indexed by
  `onset_frame` (a **frame** index, 512 samples wide) at `confidence_injector.py:43`. The energy
  lookup is therefore ~86× early in the track — the "cross-modal alignment" reads the wrong time.

**Remediation:** use MC-dropout *variance* (or a calibration head trained against note-level
correctness), aggregate per note (max/mean over the note's frames), index energy on the frame grid
(`avg_pool1d` by `hop_length`), and normalise energy before the sigmoid.

### H-4 · Expression (bend / vibrato / slide) can never reach the MIDI
`main.py:89` — `onset_logits, pitch_logits, velocity, _expression, confidence = ...`; the tensor is
discarded. There is no pitch-bend emission, no CC mapping, no consumer anywhere in `utils/midi.py`.
`README.md:50` ("MIDI with … expression (bend/vibrato/slide)") is therefore not achievable even
with trained weights; `SUMMARY.md` `[PARTIAL]` ("head exists; not trained") understates it.

### H-5 · Polyphony is structurally impossible
`main.py:204` — `pitches = torch.argmax(pitch_logits, dim=-1)` (one pitch per frame); training
targets are single-label one-hot (and `midi_to_targets` silently drops one note of a chord —
verified: notes 48+64 at the same tick → only 48 survives); the export serialises notes. The
transcriber docstring says "Polyphonic audio-to-MIDI". Any claim of polyphonic transcription is
false at the architecture level.

### H-6 · Streaming path is broken and mislabelled
`main.py:312-405`:
* `final_metrics = torch.tensor([0.0, 0.0])` → `si_sdr` and `phase_coherence` are hardcoded 0 →
  always `complex`.
* `pitch_logits = torch.zeros(...)`, `velocity = torch.zeros(...)` → every emitted note has
  **pitch 0** and velocity 1.
* The overlap region is **discarded** (`trim_len = hop_len`, then `torch.cat`), not overlap-added;
  no window is applied, so chunk boundaries are discontinuous. `chunks_processed` counts
  a double-processed region.
* Same `stem_energy` unit bug as H-3.

It is not exposed by the API (good), but it is documented in `ARCHITECTURE.md §9` and
`test_streaming_inference.py` asserts only `chunks_processed == 3`.

### H-7 · The test suite cannot fail on any of the above
Facts (I ran it: **90 passed, 8 skipped**, Python 3.11 without `torchaudio`; the repo's 96/2 is
consistent once `torchaudio` is installed):

* No test asserts an **absolute MIDI tick** (`test_midi_generation.py::test_multi_event_ordering`
  only checks "non-decreasing") — the 60× bug is invisible.
* No test asserts **content dependence**; no silence test.
* No accuracy test of any kind: `grep si_sdr tests/` yields only fixture constants.
* `test_stem_midi_model_with_fake_backends` passes without touching the backends (H-1).
* The **only** `load_from_checkpoint` tests are `pytest.importorskip("mamba_ssm")`
  (`tests/test_checkpoint_load.py:15,79`) — on the CPU-only target they *always skip*, so the
  code path that would load real weights is never executed on the target. (They don't need
  `mamba_ssm`; the fallback works.) One of them also asserts a `model_state_dict` key that
  `train.py` never writes.
* `tests/test_perf_smoke.py` uses the 32-dim mock config and 1 s of audio; the budget was relaxed
  from 500 MB to 2500 MB with a comment.
* `.github/workflows/ci.yml` runs pytest with `continue-on-error: true` and
  `--cov-fail-under=70` — **the test job cannot fail the build**, and coverage has never been
  measured (AUDIT-17.2 open).

### H-8 · The published performance number has no measurement behind it
`AUDIT-17.6` ("measure and document on reference i5") is **open**, yet "~30–60 s per minute of
audio" appears in `README.md:66`, `API_DOCUMENTATION.md:33,175`, `SETUP.md:68`,
`USER_GUIDE.md:38-41`, `docs/operations/{runbook,monitoring}.md`,
`user_content/{landing_page,progress_updates}.md` and ADR 0002. My measurement (RTF 0.30 ⇒
**18 s/min** for the default config, **4.7 s/min** for the cpu config) happens to be *better* than
claimed, so the number is conservative — but the 10-minute/3-minute *wall-clock* promises in the
same paragraph are wrong because of C-6, and a number published in 9 documents without a
benchmark is a documentation-integrity issue regardless.

---

## 5. Medium findings

| # | Finding | Evidence / fix |
|---|---|---|
| M-1 | **Silent backend substitution.** `models/_mamba_compat.py:22-28` swaps in a bespoke implementation with no log line, no metric, no `/model-info` field. ADR 0002 says "mamba-ssm runs via its pure-PyTorch reference implementation on CPU". mamba-ssm does ship `selective_scan_ref` (`research/mamba-ssm-reference/mamba/mamba_ssm/ops/selective_scan_interface.py:115`), but this code does **not** use it: `models/_mamba_compat.py` is a bespoke re-implementation that omits upstream's residual/LayerNorm `Block` wrapper (`modules/block.py:12-54`) and its `dt_proj.bias` initialisation, so the CPU path has never been validated against upstream. If `mamba_ssm` is installed but its extension fails at runtime with `OSError`/`RuntimeError` (not `ImportError`), the fallback does not engage. | Log the backend at startup, expose it in `/model-info`, catch `Exception` on import, and add a numeric-parity test vs a reference scan. |
| M-2 | **Doc/code contradictions.** `data/datasets.py:4-6` claims Canadian-artist weighting (README says not implemented); `configs/model_config.yaml:1,6` still says "NeMo ModelPT"; `configs/model_config.yaml:9` says "target <5ms via CUDA graph"; `CHANGELOG.md` says a "50 MiB cap" (code: 500 MB); `example_data_config.yaml:1-3` is headed "for the Mamba-3 research path" though it is `train.py`'s default; `SUMMARY.md` marks the four transcription losses `[IMPLEMENTED]` while `PRD_AND_ARD.md` P7 admits only alignment is wired. | Doc sweep gated on a test that greps for stale strings. |
| M-3 | **Silent target coercion.** `main.py:137-163` flattens, truncates and zero-pads `target_onsets` to the model grid; the datasets hardcode `hop_length = 512` (`data/datasets.py:146,432,527`) irrespective of config. Supervised alignment silently degrades whenever hop/n_fft differ (the mock config uses hop 128). | Fail loudly on frame-count mismatch; derive hop from config in the datasets. |
| M-4 | **Upload path triples the audio in RAM.** `api.py:631-660` buffers the whole body (`chunks` + `b"".join`), writes a temp file, then `main._load_audio` re-decodes with `sf.read` (float64) → float64→float32 copy. Duration is validated only *after* the full upload. Combined with C-6 this is what tips 3-minute files over. | Stream to disk with a chunked hash, decode via the already-open `SoundFile` handle (`audio_io.load_native_mono`) instead of re-reading, and cap duration from the header before buffering. |
| M-5 | **Audio I/O quality.** `SUPPORTED_BIT_DEPTHS` is defined but never enforced (`audio_io.py:17`); `_load_audio` catches *every* exception and falls back to librosa (masks truncated/corrupt files); resampling is `F.interpolate(mode="linear")` (`main.py:287,332`), an aliasing interpolator, on 48 kHz input. | Enforce bit depth, narrow the except, use `soxr`/`librosa.resample`. |
| M-6 | **INT8 quantization claim unsubstantiated.** Measured on 10 s: fp32 1.92 s vs int8 1.83 s (~5 %, inside noise) because the Python scan dominates; the clone also adds a full fp32 copy (~250 MB) in RAM. `tests/test_quantization.py` only checks the ZIP shape via a fake model. | Either drop the feature or move the scan to a vectorised/compiled path first; assert a speedup or delete the claim. |
| M-7 | **Invalid JSON possible.** `json.dumps` in `api.py:609-618` defaults to `allow_nan=True` → `NaN` tokens (observed). | `allow_nan=False`, and validate report fields before packaging. |
| M-8 | **Thread oversubscription.** `main.py:17` sets `torch.set_num_threads(os.cpu_count())` at import, ignoring cgroups/affinity; in a container or co-hosted service this thrashes. | Read `OMP_NUM_THREADS` / `sched_getaffinity`, default to `min(4, cores)`. |

---

## 6. Low / informational

* `utils/audio.py` is dead code duplicating `audio_io.py`; `utils/midi_targets.py` is dead (C-4).
* `assert`-based shape invariants (`models/mamba_separator.py:71-93`) vanish under `python -O`.
* `models/backends/base.device_from_cfg` will select CUDA if present, contradicting "CUDA path removed".
* `research/mamba-ssm-reference/` (3.6 MB, 97 files) is vendored without a pinned commit hash in
  the ADR; `.git` is excluded — acceptable, but pin the SHA in `research/README.md`.
* `LICENSE`/SPDX sweep is incomplete in `scripts/` and some `models/backends/` files (noted in
  CHANGELOG); the CI license job would flag it if it weren't scoped to exclude `research/`.

---

## 7. Claim-by-claim verification (against `SUMMARY.md` / `README.md`)

| Claim | Verdict | Basis |
|---|---|---|
| "CPU-only; CUDA path removed (ADR 0002)" | **Verified** | No CUDA maths; only `torch.cuda.empty_cache()`/OOM type probing remain. |
| "`mamba-ssm` runs its pure-PyTorch reference implementation on CPU" | **False** | Own re-implementation (`models/_mamba_compat.py`), no residual, no `dt_bias`, collapses (C-1). |
| "Stem separation [IMPLEMENTED]" | **False in effect** | Stems = 0.5 × mix; true SI-SDR −9.8 dB (C-1). |
| "MIDI transcription (onset/pitch/velocity) [IMPLEMENTED]" | **False in effect** | Heads untrained (C-4); logits constant (C-1/C-2); export 60× off (C-3). |
| "Per-note confidence → CC#127 [IMPLEMENTED]" | **False in effect** | Not per-note; wrong CC class; written at note-off; ~all notes flagged (H-3, C-3). |
| "Expression (bend, vibrato, slide) [PARTIAL]" | **Overstated** | Tensor discarded before MIDI (H-4). |
| "Quality gates studio/draft/complex [IMPLEMENTED]" | **Paper-only** | Inputs are a fake dB and a fake confidence; `studio` unreachable; `draft` reachable on silence (C-7). |
| "Processing report (`si_sdr`, `phase_coherence`, …) [IMPLEMENTED]" | **Misleading names** | `si_sdr` = centroid proxy; `phase_coherence` = sigmoid of a mean hidden state (H-2). |
| "Losses … [IMPLEMENTED]" | **Misleading** | Functions exist; only MR-STFT/flatness/crest + a scalar alignment term are wired (C-4). |
| "Backends include mamba, demucs, basic_pitch, fake" | **Exists, unwired** | H-1. |
| "Data loaders [IMPLEMENTED]" | **Partly false** | Loaders read audio; labels are fabricated periodic onsets; Slakh `stems/guitar.wav` may not exist → silent zero targets. |
| "`~30–60 s per minute of audio`" | **Unsubstantiated but conservative** | Measured 18 s/min (default), 4.7 s/min (cpu config); AUDIT-17.6 open (H-8). |
| "3-min song 1.5–3 min; 10-min 5–10 min" | **False** | OOM at 180 s on 4 GB; 600 s ≈14 GB (C-6). |
| "pytest 96 passed, 2 skipped" | **Reproducible** | 90 passed / 8 skipped here (no `torchaudio`); but CI cannot fail and the 2 skips hide the checkpoint path (H-7). |
| "No trained weights; demo uses synthetic audio" | **Verified and honest** | But nothing stops the API serving the untrained model (C-7). |

---

## 8. Measurement appendix

Environment caveat: the repo targets Python 3.13; I ran 3.11.2 (no cp313 interpreter in this
sandbox). Nothing I found is version-sensitive except the exact test count; all numerical findings
are plain torch math and version-independent. The box (2 vCPU / 3.9 GB, no swap) sits at the
*low* end of the claimed target, so memory numbers are a lower bound on real-world failure.

```
# inference, configs/model_config.yaml, 62.2 M params, 2 threads
1s  → 0.29 s,  654 MB       30s → 8.46 s, 1353 MB      120s → 36.5 s, 3319 MB
10s → 2.65 s,  875 MB       60s → 18.0 s, 2037 MB      180s → SIGKILL (OOM)
# inference, configs/model_config.cpu.yaml, 5.5 M params
60s → 4.69 s,  878 MB
# training (one step incl. backward+AdamW)
B=1,  4 s, default      → 128.8 s, 2665 MB
B=16, 4 s, cpu config   → SIGKILL (OOM killer, exit 137)
# MIDI
3 s input → guitar.mid length 179.8 s, 259 notes, last onset at 179.72 s
# separation quality on a known synthetic 2-source mix
true SI-SDR(guitar) = −9.82 dB | SI-SDR(guitar vs mix) = +51.72 dB | reported proxy = 4.75 dB
# masks / activations
mask min/max 0.491/0.509, mean 0.500084 | stem/mix power 0.2501
block RMS: 1.0 → 4.5e-02 → 1.1e-03 → … → 7.3e-20   (with residual: 1.01)
```

---

## 9. Remediation plan (ordered by leverage)

**Tier 0 — make the model a model (days, not weeks)**
1. Add residual + pre-norm around every mixer; clamp `dt`; initialise `dt_proj.bias` as upstream.
   (≈15 lines; converts C-1/C-2 from "garbage" to "trainable".)
2. Chunk the selective scan over `L` with state carry (bounds memory, enables real streaming;
   fixes C-6 and unblocks C-5).
3. Add regression tests: per-block RMS within `[0.5, 2.0]`; `isfinite` on all outputs for
   silence, DC, full-scale noise; output must differ for two different inputs.

**Tier 1 — make the artifacts honest (days)**
4. Fix `ticks_per_second`; rebuild the SMF writer as a sorted event list with independent voices;
   emit durations; replace CC#127 with a text meta event or an unused CC; tests for absolute
   ticks, chord simultaneity, and note duration.
5. Wire the heads into the loss (BCE onsets, multi-label pitch, velocity MAE, duration IoU), wire
   `midi_to_targets` into the datasets, delete the fabricated periodic labels.
6. Refuse to serve an uninitialised model; stamp model state in `/model-info` and the report; make
   the CLI write artifacts.

**Tier 2 — truth in the API surface (days)**
7. Rename `si_sdr` → `centroid_proxy_db` everywhere; compute real SI-SDR in `scripts/eval_smoke.py`
   against MUSDB test references; publish the numbers.
8. OOM → 503 for CPU `RuntimeError`; stream uploads; single decode; `allow_nan=False`;
   per-note confidence aggregation on the frame grid.
9. Wire or delete the Demucs / Basic Pitch backends and the `model_config.cpu.yaml` claim; if
   wired, add the missing `pretrained` extra.

**Tier 3 — process (week)**
10. Remove `continue-on-error` from CI; enforce coverage; un-skip/repair the checkpoint tests; add
    a 600 s memory test and an end-to-end accuracy smoke with a rendered ground truth.
11. Benchmark on real i5 hardware (AUDIT-17.6) and rewrite the performance paragraph in all nine
    documents from measured data, including the memory ceiling per RAM size.

---

## 10. Go / No-Go

**NO-GO** for anything beyond a prototype: not for a pilot, not for a paid tier, not for
distribution. Reasons, in order of severity:

1. The ML core is numerically dead (C-1) and unstable (C-2) — it is not a separation or
   transcription model, it is a 0.5× gain plus a constant-logit emitter.
2. The MIDI it produces is not usable in a DAW (C-3): wrong time base, no polyphony, no
   durations, meaningless CC.
3. It cannot be trained to fix (2) on the target hardware (C-4, C-5), and cannot be run for the
   advertised file lengths on the advertised RAM (C-6).
4. Its failure modes are not honest in the ways that matter commercially (C-7, C-8): silent
   garbage, NaN in the report, 500 instead of 503, and a paid-upsell path reachable on silence.

**Re-review criteria (all must hold) for a "prototype / internal demo" GO:**

- [ ] Residual/norm fix merged, with the per-block RMS and finiteness tests, plus a monotonic
      content-dependence test (silence → 0 notes, two different inputs → different output).
- [ ] MIDI absolute-timing, polyphony and duration tests pass; 60 s file → events within ±1 frame
      of the analysed onset times.
- [ ] 600 s audio processed under 2 GB with the default config on a 4 GB box, verified in CI.
- [ ] A checkpoint exists whose pitch/onset heads receive non-zero gradients during training, and
      a held-out SI-SDR / onset-F1 number is published (target: not zero).
- [ ] `/process` returns 503 (not 500) on allocator failure; report JSON is RFC-valid; untrained
      mode is refused or clearly stamped.
- [ ] `README`/`SUMMARY`/`ARCHITECTURE` performance and feature tables regenerated from these
      measurements.

Until then: treat every performance number, quality tier and confidence value in this repository
as fabricated.

---

## Appendix A — minimal repros

**A.1 · Activation collapse / 0.5-gain separator** (uses only the public API)

```python
import torch, yaml
from models.mamba_separator import MambaSeparator
cfg = yaml.safe_load(open("configs/model_config.yaml"))
sep = MambaSeparator(cfg).eval()
x = torch.randn(1, 1, 3 * 44100) * 0.1
with torch.inference_mode():
    spec = torch.stft(x.squeeze(1), 2048, 512, window=sep.window, return_complex=True)
    h = sep.input_proj(torch.abs(spec).transpose(1, 2))
    print("input rms", h.pow(2).mean().sqrt().item())          # 1.0
    for i, blk in enumerate(sep.mamba_blocks):
        h = blk(h)
        print(i, h.pow(2).mean().sqrt().item())                 # 4.5e-2 … 7.3e-20
    g, b, r, _, _ = sep(x)
    print("stem/mix power", ((g**2).mean() / (x**2).mean()).item())   # 0.25  → guitar = 0.5·mix
# fix: h = h + blk(h)   (add pre-norm for good measure)
```

**A.2 · MIDI is 60× slow**

```python
from utils.midi import build_midi_from_events
import mido, io
ev = [{"note": 60, "onset_frame": 43, "velocity": 90, "confidence": 0.8}]   # 43 frames ≈ 0.5 s
mid = mido.MidiFile(file=io.BytesIO(build_midi_from_events({"midi_events": ev}, "guitar")))
t = 0
for m in mid.tracks[0]:
    t += m.time
print("absolute tick", t)        # 28755 + 120 ; should be ≈ 480 (0.5 s @ 120 BPM, 480 tpq)
```

**A.3 · Silence produces notes / NaN**

```python
import torch, yaml
from main import StemMidiModel
m = StemMidiModel(yaml.safe_load(open("configs/model_config.cpu.yaml"))).eval()
out = m(torch.zeros(1, 1, 10 * 44100))
print(out["processing_report"].avg_confidence,        # 0.49 / NaN depending on init
      len(out["midi_metadata"]["midi_events"]))       # 0 … 259 notes from pure silence
```
