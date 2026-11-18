# Closing the last seven findings — and what "closed" is allowed to mean

**Date:** 2026-11-18
**Branch:** `arena/f0de80e9-stem-midi-pro` (from `c2f960b`)
**Scope:** `TRANS-1`, `CONF-1`, `GATE-4`, `SEP-1`, `SEP-2`, `CPU-3`, `EXPR-1` — the
seven findings the previous remediation left open.
**Result:** `493 passed, 4 skipped, 0 failed, 0 xfailed`; `ruff` and `black` clean.

The previous document closed 36 findings and left seven open, each with a reason.
The reasons were reasonable *at the time* and four of them turned out to be wrong.
That is the interesting part of this record, so each finding below states what I
believed, what I measured, and what changed.

---

## Summary

| ID | Was believed blocked on | What measurement showed | Outcome |
|---|---|---|---|
| **TRANS-1** | "needs a no-note logit + retraining" | The energy needed is already computed on the right frame grid; only the logits were passed | **Fixed**, per-frame energy gate |
| **CONF-1** | "needs trained weights; the numbers describe an untrained head" | The head had **no loss term anywhere in the repo** — retraining was not the blocker, *having no way to train it* was | **Fixed**, supervision + corroboration |
| **GATE-4** | "needs a reference-free metric and a corpus" | The metric was the problem, and its failure is monotone and provable; a corpus is needed to *tune* a threshold, not to replace a broken measure | **Fixed**, mask crosstalk |
| **SEP-1** | "architectural, changes what the residual stem is" | Already fixed upstream by softmax masks; the entry was stale | **Verified fixed**, pinned |
| **SEP-2** | "the hidden state decays 1 → 4e-10" | The measurement chained the **bare mixers**, bypassing the residual the model actually uses | **Verified fixed**, measurement corrected |
| **CPU-3** | "changes model output, no output-preserving property" | Threading state changes the pipeline by 1.5e-04 and improves it by **1.000×** | **Closed as measured**, documented |
| **EXPR-1** | "product decision; head untrained" | Shipped opt-in, labelled, documented, and the docs already matched the code | **Closed**, contract pinned |

Four of the seven were not blocked at all. Two were already fixed and the
*measurement* was wrong. One is genuinely not worth doing, and is now closed
with numbers rather than an opinion.

---

## TRANS-1 — no note without energy in the frame

**What it was.** The onset head has no "no-note" class. Peak-picking asked only
"is the activation above threshold?", so an above-threshold frame on a stem with
no energy became a note. Silence rendered 0 notes only because the untrained head
happened to sit below 0.5 — a property of the initialisation, not a decision.

**What I had assumed.** That fixing it required a new logit, hence retraining.

**What was actually true.** `_match_frame_grid` already computes per-frame stem
energy **on the same frame grid the onset candidates index** — it was being
passed to the confidence injector and nowhere else. The gate needs no new target
and no new weight.

**The fix.** `_logits_to_midi(..., frame_energy=)` refuses any onset frame whose
energy is below `transcriber.energy_floor_db` (default −60 dB) **relative to that
stem's own peak frame**. Relative, so it is level-independent.

| Case | Notes before | Notes after |
|---|---|---|
| saturated onsets, zero-energy stem | 200 | **0** |
| real take with a 10-frame silent gap | 200 | **190** (exactly the 10 silent) |
| uniformly quiet take (1e-6 of the loud one) | 200 | **200** (level-independent) |
| tail 70 dB below peak | 200 | **150** |

Guards: `tests/test_pipeline_integrity.py` ×5, including level-independence and
the configurable floor. Confirmed to bite: disabling the gate fails 3 of them.

---

## CONF-1 — the number every tier depends on

Two independent defects. Only the first is what the entry described.

### (a) The head had no loss term

`models/losses.py` trained onset, pitch, velocity, duration and alignment.
`train.py` optimises exactly what that loss returns. **Confidence appeared in
neither.** The head kept its random initialisation for the entire life of the
model, and it is a three-layer MLP behind dropout, so it produced an almost
constant output. Measured on 2 s inputs with the shipped architecture:

| content | mean confidence |
|---|---|
| clean music | 0.95543 |
| hard-clipped square | 0.95609 |
| white noise | 0.95566 |
| digital silence | 0.95571 |

Spread across content classes: **0.00079**. STUDIO's bar: **0.85**. So every
render — including digital silence — cleared its own quality threshold.

This also corrects the previous write-up: it said CONF-1 "needs trained weights,
the 0.516-clipping vs 0.300-clean number is a statement about an *untrained*
head". True as far as it went, and beside the point — the head could not be
trained at all.

**The fix.** A `confidence` term in `PerceptualAudioLoss`, targeting per-frame
onset **correctness** (did the model's onset prediction agree with the target?).
That target is observable from labels the dataset already provides, so no new
annotation is needed. Two implementation details that each cost me a bug:

* With an `(B, T, 1)` onset target the shape did not match the `(B, T)`
  confidence tensor and the term **silently stayed zero** — a loss term that
  quietly does nothing, which is the same class of defect as the one being
  fixed. It now raises.
* A plain weighted sum let the term shrink to ~7% of the gradient against a
  spectral-dominated total (0.693 vs 4.82). It is now scaled to hold a fixed
  `confidence_share` (0.1) of the objective, so its influence does not depend on
  how large the perceptual terms happen to be on a batch. Scaling it inside
  `torch.no_grad()` detached it, so only the gain is computed there.

Verified trainable: a constant 0.5 vector converges to per-frame correctness
(mean error 0.11) and stops being constant. Confirmed to bite: removing the term
fails both guards.

### (b) Confidence alone was self-certification

Even a trained head can be confidently wrong, and the tier decision rested on
nothing else. STUDIO now additionally requires **corroboration**: separation
better than the GATE-4 crosstalk ceiling. Confidence remains **required**, so
this can only ever remove a badge, never manufacture one.

Four-tier behaviour, measured:

| confidence | crosstalk | tier |
|---|---|---|
| 0.96 | 0.90 (worse than a flat mask) | complex |
| 0.96 | 0.30 | studio |
| 0.50 | 0.10 | complex |
| 0.75 | 0.10 | draft |
| 0.96 | 0.10, ceiling `null` | draft |

The last row is the interesting one: disabling the separation gate removes the
*evidence*, not the *bar*, so a confident render falls to draft rather than
self-certifying.

**Backward compatibility.** The corroboration rule initially broke nine
existing tests, all of which build reports with the legacy centroid proxy
(`centroid_separation_db=25.0`). Those are not wrong — `route_by_quality` is a
public helper and a legacy value is not evidence either way. The dataclass now
records whether `separation_proxy` was supplied as crosstalk or inherited, and
only a real crosstalk reading can corroborate. An older caller keeps working
instead of being silently downgraded for a field it never set.

Guards: `tests/test_confidence_honesty.py` ×9.

---

## GATE-4 — a separation metric that rewarded distortion

**The failure, provable.** The old metric was
`clamp(10·log10(|centroid_guitar − centroid_bass|) + 20, 0, 30)` — a gap between
two spectral centroids in FFT-bin units, mapped through an arbitrary affine
transform and clamped. One 220 Hz tone, clipped progressively harder:

| clip peak | 0.30 | 0.20 | 0.12 | 0.08 | 0.05 |
|---|---|---|---|---|---|
| **old proxy** | 15.64 | 23.97 | 26.89 | 27.80 | 28.73 |

The worse the audio, the better the score, and a hard-clipped square rated the
maximum **30.00**. Clipping creates harmonics, which drag the two centroids
apart, which the metric read as "separation".

**What I had assumed.** That replacing it needed "a reference-free metric **and a
corpus** to set a threshold against". The corpus is needed to *calibrate a
threshold*. It is not needed to establish that the existing measure is invalid,
nor to build one whose endpoints are arithmetic.

**The replacement.** **Mask crosstalk**: the share of each stem's energy that
sits in time-frequency bins the stem does not dominate. Reference-free (works on
a real upload, unlike SI-SDR), bounded [0, 1], lower is better.

Its scale is arithmetic, not fitted:

| case | crosstalk |
|---|---|
| oracle band split at 400 Hz | **0.0000** |
| untrained shipped init | 0.4677 |
| flat one-third masks | **0.6667** (= 2/3, exactly) |

And it does not reward distortion — flat at 0.47–0.51 across the clipping sweep
above. It is now an actual gate (`max_separation_crosstalk: 0.60`, a ceiling,
`null` disables) instead of metadata that nothing read.

**What remains genuinely open, stated plainly.** The *default* (0.60) is set
from the analytic no-separation value and the untrained model, not from a
corpus of separated material. It reliably refuses a separator worse than a flat
mask; it is not a calibrated perceptual quality bar. Setting one requires
reference material and is a separate exercise. That is a narrower claim than
"reference-free metric and a corpus are needed to fix the metric".

Guards: `tests/test_separation_metric.py` ×7. Confirmed to bite: restoring the
old metric fails the inversion guard; removing the gate fails the tier guard.

---

## SEP-1 — the residual stem was decorative

Softmax across stems makes the three masks sum to 1 per bin, so the stems
partition the mixture by construction:

| masks | `‖g+b+r − mix‖ / ‖mix‖` |
|---|---|
| softmax (default) | **1.15e-07** |
| independent sigmoids (old default) | 5.27e-01 |

Already fixed upstream; the entry was stale. The `independent` escape hatch
still measures 0.527 and is now pinned as the contrast, so a future default flip
fails loudly. Guard: `tests/test_numerics.py::test_masks_partition_the_mixture`.

---

## SEP-2 — the collapse was real, in the measurement

The entry quoted a per-block ladder of `1 → 4.9e-2 → … → 4.0e-10` with the
comment "no per-block norm / residual add here". That comment was accurate about
the *script* and wrong about the *model*: it chained `sep.mamba_blocks[i]`
directly, while `forward` does `hidden = hidden + block(norm(hidden))`.

Through the real path:

```
after layer  0 : h=1.001e+00   block_out=4.507e-02
after layer  6 : h=1.007e+00   block_out=4.444e-02
after layer 11 : h=1.013e+00   block_out=4.211e-02
after final_norm: 1.000e+00
```

Each block contributes ~0.044 and the residual holds the state at order one. The
transcriber behaves the same (1.79 → 1.80 across 8 layers). Bypassing the
residual still collapses to 8.7e-20 — which is exactly what the old script
measured.

`scripts/verify_numerics.py` and `tests/test_numerics.py` now replay the real
stack, and `test_bare_mixers_are_the_only_collapsing_stack` pins the contrast in
both directions so nobody can quietly change which one is measured again.

---

## CPU-3 — measured, and the answer is "not worth it"

The mechanism already exists and is exact where it applies. Measured at three
levels (`scripts/verify_cpu3_state.py` reproduces every number):

| level | state threaded vs one long call |
|---|---|
| **block** — two calls vs one | **0.0e+00** |
| **separator** — chunk vs whole | 1.6e-02 |
| **separator** — same, state dropped | 1.6e-02 |
| **separator** — carried vs dropped difference | 1.0e-03 |
| **pipeline** — chunk vs whole, threaded | 4.9e-03 |
| **pipeline** — chunk vs whole, dropped | 4.9e-03 |
| **pipeline** — carried vs dropped difference | 1.5e-04 |
| **pipeline** — accuracy improvement | **1.000×** |

The reason is that each chunk goes through its own `torch.stft`, and the
transform's edge padding is much larger than the state's contribution: at the
separator level the two outputs differ by 1.0e-03 of the reference, 17× smaller
than the edge effect swamping them. In the real chunked path the H-4 crossfade
already absorbs the boundary discontinuity, so threading state in the default
path would change every render by a measured-nonzero amount for a measured-zero
benefit.

It is therefore **not** threaded by default, and a caller with genuinely
contiguous non-overlapping pieces may thread it for the block-exact result.

I also removed a false claim in the module docstring: it asserted a 4e-9
equality and cited `tests/test_streaming_state.py`, which **does not exist in
this repository**. Guards: `tests/test_streaming_inference.py`.

---

## EXPR-1 — the contract was already honest

Checked the three parts rather than asserting them:

| claim | verified |
|---|---|
| default render writes no expression | no `pitchwheel`, no CC lane |
| `include_expression=True` writes the documented mapping | pitch-wheel + CC1 (vibrato) + CC74 (slide) |
| ships off | all three configs `midi.include_expression: false` |

The `cpu` and `mamba` configs did not set the key at all and inherited the code
default — correct behaviour, invisible in the config files. They now state it
explicitly. Guards: `tests/test_expression_export.py` ×4.

---

## How to check all of this

```bash
python -m pytest -q                        # 493 passed, 4 skipped, 0 failed
python scripts/verify_numerics.py          # SEP-1 / SEP-2 / CONF-1 numbers
python scripts/verify_cpu3_state.py        # CPU-3 verdict, all three levels
ruff check . && black --check .
```

Each guard was mutation-tested: reverting the fix and confirming the guard fails.

| revert | guards that fail |
|---|---|
| disable the TRANS-1 energy gate | 3 |
| remove the confidence loss term | 2 |
| restore the old centroid metric | 1 (the inversion guard) |
| remove the crosstalk tier gate | 1 |
| default to `independent` masks | 1 |

## Honest limits

* **No trained checkpoint ships with this repo**, so nothing here demonstrates
  that the confidence head, once trained, is *well calibrated* — only that it is
  now trainable and that the tier can no longer be earned on the head's word
  alone. Calibration is measurable only against trained weights.
* **`max_separation_crosstalk: 0.60`** is anchored to arithmetic (the 2/3
  no-separation value) and to the untrained model's 0.47. It refuses
  non-separations; it is not a perceptual quality threshold. That needs a corpus.
* **CPU-3's** decision rests on the shipped 50%-overlap crossfaded path. A
  different chunking strategy (no overlap, or much longer chunks) would want its
  own measurement.
