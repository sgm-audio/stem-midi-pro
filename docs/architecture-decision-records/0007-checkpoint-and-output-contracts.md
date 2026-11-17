# ADR 0007: Explicit Checkpoint and Output Contracts

**Status:** Accepted
**Date:** 2026-11-17

## Context

The original inference path could construct the Mamba model with random
parameters when no checkpoint was configured. The backend factory existed but
was not used by the orchestrator, guitar and bass MIDI were aliased, expression
heads were discarded, and the MIDI serializer used channel-wide CC#127 for
per-note confidence. The report also exposed a spectral-centroid heuristic as
`si_sdr`.

Those behaviors are unsafe for an application that produces audio and MIDI
artifacts: a syntactically valid ZIP can still contain unusable or misleading
output.

## Decision

1. `StemMidiModel` continues to support random-weight `forward` calls for
   training and structural tests, but file/stream inference requires a loaded
   checkpoint unless an explicit test/development opt-in is present.
2. API startup requires `MODEL_CHECKPOINT_PATH` (or the config's
   `restore_from_path`) for native Mamba/project weights and remains not-ready
   when that checkpoint is missing or invalid. The explicit CPU profile may
   instead select the Demucs + Basic Pitch pair with
   `weights_source: pretrained_dependencies`; both constructors must attest to
   their own pretrained weights. The CLI applies the same rule.
3. `StemMidiModel` constructs the configured separator and transcriber through
   `models.backends.factory`. Unknown backends and missing optional dependencies
   fail with actionable errors rather than silently selecting another backend.
4. Guitar and bass are transcribed independently. Each output keeps absolute
   audio-frame positions, duration frames, overlapping notes, and same-onset
   chords. Expression fields map to pitch-wheel, CC1, and CC74 messages.
5. Note confidence is canonical in JSON and is encoded in SMF as standard MIDI
   Polyphonic Key Pressure. The old `cc_127_value` name remains only as a JSON
   compatibility alias; CC#127 is not emitted.
6. `separation_proxy` is the public metric name. The deprecated `si_sdr` alias
   is marked non-SI-SDR, and the heuristic gate is not an accuracy certificate.

## Consequences

- A deployment without trained weights fails readiness instead of serving
  random output.
- Optional Demucs/Basic Pitch deployments must install and configure their
  dependencies explicitly.
- DAWs that ignore Polyphonic Key Pressure should consume the JSON metadata or
  use a project-specific import step; MIDI controller semantics are not
  assumed.
- The repository still cannot claim trained separation quality, calibrated
  confidence, useful expression, or reference-i5 performance without external
  evaluation artifacts. Those remain evidence requirements, not documentation
  defaults.
