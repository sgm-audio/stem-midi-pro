# Audit: what is still broken on `main` after PRs #11 and #14

**Date:** 2026-11-17
**Question:** PR #17 conflicts with `main` (14 files). Two fix passes for the same
reliability review have already merged (#11 "Resolve all static-review findings",
#14 "Resolve remaining documented findings: H-6, C-2.6"). **What is genuinely still
missing, and is a second fix pass justified?**

**Answer:** yes. `main` at `ee8dd62` still carried **5 findings** from
`docs/reviews/2026-11-17-reliability-review.md`, including two that are worse than
"unfixed": one is a hard crash, one is an OOM kill. Everything below was measured on
`ee8dd62`, not inferred from reading the diff.

Method: `git worktree add /tmp/mainwt2 origin/main`, copy this branch's regression
suite in, run it, then reproduce each failure by hand. Nothing in `main`'s tree was
modified.

---

## Still broken on `main` (5 findings)

### 1. §1/§2 — 600 s is still OOM-killed — **P0**

`main` chunked the Mamba selective scan (`_mamba_compat.py`, `scan_chunk`) but left
the inverse STFT whole-file (`mamba_separator.py:157`, plain `torch.istft`).

```
$ python /tmp/mem600.py /tmp/mainwt2 main
Killed                     # exit 137
[  968.67] Out of memory: Killed process 2195 (python) total-vm:7081688kB, anon-rss:3838572kB
```

Same workload on this branch:

```
ours 600s OK  peak 2386 MB  144.0s  finite True
```

`torch.istft` allocates 4-5x its output size internally (measured: 1216 MB of scratch
for a 264 MB output) and `requires_grad=False` does not avoid it. Chunking the scan
alone does not bound the peak, because the STFT and the istft dominate from ~4 min up.

### 2. §3 — loud input overflows the SSM to NaN; silent input **raises** — **P0**

`mamba_transcriber._mel_spectrogram` has a floor but **no ceiling**:

```python
log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-8))   # main
```

Unbounded above, so a loud or DC-heavy stem reaches log-mel 19.36 and the recurrence
overflows fp32. Reproduction:

```
$ python /tmp/prove3.py /tmp/mainwt2
RuntimeError: guitar: model output 'onset_logits' contains NaN/Inf;
refusing to emit corrupt stems/MIDI.
```

Note this is *worse* than the original finding. The 2026-11-17 review documented
silent input producing NaN frames that were then clamped into a file. `main` added
`_ensure_finite`, so a **completely silent 5 s WAV now raises RuntimeError instead of
producing output at all** — an all-silence file is a legitimate input (a rest, a
count-in, a muted track).

This branch clamps both sides and normalises per block, so silence yields a finite,
all-zero stem and centroid 0.0. Covered by `test_log_mel_is_bounded`,
`test_silence_forward_is_finite`.

### 3. §4 — the gate turns NaN into `draft` — **P0**

`main` renamed the metric (C-2.6) but never added a non-finite check:

```
$ python /tmp/prove_main.py /tmp/mainwt2
=== §4: NaN metrics ===
  NaN si_sdr -> tier=draft  action=editor_launch
```

`NaN >= x` is always `False`, so the STUDIO branch is skipped and the DRAFT branch is
taken on confidence alone. A blown-up render is presented as "Draft Quality — refine in
our editor", not as a failure. `main` also has no `as_dict()`, so the NaN then reaches
`json.dumps` and produces invalid RFC 8259 JSON (§12).

### 4. §9 — STUDIO is still gated on the saturation-prone proxy — **P1**

`main` fixed the *naming* half of C-2.6 but kept the gate:

```python
if (self.avg_confidence >= self.studio_confidence_threshold
        and self.centroid_separation_db >= self.min_centroid_separation_db):   # main
```

Measured on `main`:

```
one bin (20.00 dB) -> tier=studio        # 1 FFT bin of separation clears the bar
50 low-conf notes  -> tier=studio        # unreliable notes do not matter
```

Renaming `si_sdr` to `centroid_separation_db` makes the report honest; it does not make
the number discriminating. One FFT bin is trivially reached by noise, so STUDIO remains
"one bin plus 0.85 confidence" away from anything.

This branch drops the proxy from the tier decision and requires
`low_confidence_notes == 0`; `min_si_sdr` is ignored with a `DeprecationWarning`.

### 5. §4b — silent input ships fabricated MIDI — **P0**

`main` has no silence predicate anywhere (`_stem_is_silent`: 0 occurrences). Suppressing
this was not possible to measure directly on `main` because the silence path raises
first (§3 above), but the underlying defect is present: nothing suppresses onset events
when a stem is silent, so as soon as §3 is fixed the fabricated-note bug reappears. On
this branch the pre-fix behaviour was 8-128 note-ons; it is now 0 for both stems.

---

## Genuinely fixed on `main` — do **not** re-apply

Verified by reading the code, not by trusting the commit message:

| Finding | Where | Note |
|---|---|---|
| §6 MIDI cursor 60x | `utils/midi.py:117` | Different formula (`ticks_per_beat * 1e6 / DEFAULT_TEMPO_US`), same correct result. My `test_midi_onset_ticks_match_frame_math` still fails on `main` only because it imports the old event shape. |
| §7 overlapping same-pitch notes | `utils/midi.py` | Resolves overlaps, validates 0-127, raises on invalid. |
| §13 heads get no gradient | `train.py` | Loss now receives the pitch/velocity heads. |
| §15 `midi_bass is midi_guitar` | `main.py`, `train.py` | Bass transcribed from the bass stem. |
| §17 onset threshold | `models/mamba_transcriber.py` | Threshold fixed. |
| §18/§19/§20 datasets + targets | `data/datasets.py` (+255), `utils/midi_targets.py` (+90) | Reworked separately; broader than my version. |
| H-1, H-3..H-9, M-1..M-7 | `tests/test_review_fixes.py` (30 tests) | A different, complementary finding set. |

My suite reports ~29 failures against `main`, but a large share are **API-shape
mismatches**, not live bugs: I use `ProcessingReport(si_sdr=...)` where `main` now takes
`centroid_separation_db=`, and I assert `model._backends_used` which does not exist
there. Those must be adapted, not "fixed".

---

## Consequence

**Option 2 is the right call, and the residual PR is small.** After rebasing onto
`ee8dd62`, the genuinely-missing work is confined to five areas:

1. `models/mamba_separator.py` — windowed istft + windowed per-stem masking (memory).
2. `models/mamba_transcriber.py` — bound the log-mel on **both** sides.
3. `utils/quality_gates.py` — non-finite check, drop the proxy from the tier decision,
   `low_confidence_notes == 0` for STUDIO, JSON-safe `as_dict()`.
4. `main.py` — one silence predicate shared by the artifact detector and the
   transcriber.
5. Tests + docs for the above, adapted to `main`'s renamed fields.

Everything else in PR #17 either duplicates `main` or is superseded by it and should be
dropped rather than merged.

---

## Resolution (this PR)

Rebased onto `ee8dd62` and split into per-finding commits. Everything that
`main` already fixed was dropped rather than merged, so the diff contains only
the five residual fixes plus their tests.

| Commit | Finding | Verified |
|---|---|---|
| `4a65375` | §1/§2 memory | 600 s separator: **Killed (137)** → 2401 MB / 149.2 s, finite |
| `b2e5bb7` | §3 log-mel ceiling | zeros / DC ±1e6 / loud sine → finite, capped 6.908; silent WAV no longer raises |
| `2fe6778` | §4, §9, §12, §4b | NaN → `complex` not `draft`; 1-bin proxy no longer gates; silence → `['silent_output']`, 0 notes both stems |
| `82bcc55` | follow-ups | 4-arg `_selective_scan` preserved; `torch.utils.checkpoint` imported (pre-existing failure) |
| `d242c33` | tests | 16 tests, **14 fail on pristine `main`** |

**End-to-end after the rebase**

| Workload | `main` (`ee8dd62`) | This branch |
|---|---|---|
| 600 s separator | Killed (137), OOM ~3.8 GB | **2401 MB / 149.2 s** |
| 600 s full pipeline | (unreachable) | **2430 MB / 236.2 s, exit 0** |
| Silent 5 s WAV | `RuntimeError: onset_logits contains NaN/Inf` | report returned, `['silent_output']`, 0 notes |
| Silent `.mid` payload | 8–128 fabricated note-ons | **0 notes** |
| NaN metric tier | `draft` / `editor_launch` | `complex` / `fallback_options` |
| One FFT bin (20.00 dB) | `studio` | proxy not consulted |
| Suite | 4 failed, 116 passed | **146 passed, 2 skipped, 0 failed** |

The four pre-existing `main` failures were environmental (`librosa` absent) plus
the unimported `torch.utils.checkpoint`; the latter is fixed here, the former
passes once `librosa` is installed.

**Not re-applied** (already correct on `main`): §6 MIDI timing, §7 same-pitch
overlaps, §13 head gradients, §15 bass transcription, §17 onset threshold,
§18/§19/§20 datasets and targets, and the whole H/M finding set in
`tests/test_review_fixes.py`.
