# Stem+MIDI Pro User Guide

## Introduction

Stem+MIDI Pro separates a mixed recording into **guitar and bass stems** and transcribes them into **editable MIDI drafts**, with per-note confidence scores. Native-Mamba file/API inference requires a validated project checkpoint; the explicit CPU profile may instead use its separately installed Demucs + Basic Pitch pretrained weights. No trained project weights are included in this repository. CPU execution is supported, but the stated Intel i5 performance target has not been independently benchmarked here.

## What It Actually Does Today

1. **Stem separation** — produces independent `guitar_stem.wav` and `bass_stem.wav` files (plus a residual internally) using the configured backend.
2. **MIDI transcription** — produces independent `guitar.mid` and `bass.mid` files with absolute audio-frame timing, pitches, velocities, duration-aware note-offs, same-onset chords, and overlapping notes.
3. **Expression mapping** — expression fields, when supplied by a backend, map to pitch-wheel (bend), CC1 (vibrato), and CC74 (slide). The mapping is implemented but **off by default** (`include_expression=True` turns it on), because the bundled head is untrained; exported events are labelled with that provenance. Expression quality still depends on a trained checkpoint.
4. **Confidence metadata** — JSON note fields are canonical. MIDI also carries Polyphonic Key Pressure (0–127), a standard note-specific message whose interpretation varies by DAW. CC#127 is not emitted; `cc_127_value` is only a deprecated JSON alias.
5. **Quality report** — `processing_report.json` exposes a named centroid-separation proxy (not SI-SDR), phase coherence, note-level confidence counts, artifact flags, model-weight status, and a heuristic routing tier.

**Not included (yet):** tempo detection, tuning detection, palm-mute / harmonics / slap-pop detection, a web-based MIDI editor, or real-time streaming. Refinement happens in your own DAW. Earlier versions of this guide described those features aspirationally; this version describes the software as it is.

## Getting Your Results

Processing is done via the API or CLI (see [SETUP.md](SETUP.md), [API_DOCUMENTATION.md](API_DOCUMENTATION.md)):

```bash
# CLI — trained checkpoint required
python main.py --audio mysong.wav \
  --config configs/model_config.yaml \
  --checkpoint /path/to/trained_checkpoint.pt

# or via the API — set this before starting uvicorn
export MODEL_CHECKPOINT_PATH=/path/to/trained_checkpoint.pt
curl -X POST http://localhost:8000/process -F "file=@mysong.wav" -o output.zip
```

You get a package containing:

- `guitar_stem.wav`, `bass_stem.wav` — separated stems
- `guitar.mid`, `bass.mid` — MIDI transcriptions
- `processing_report.json` — metrics and quality tier

## How Long It Takes (CPU)

On the reference bare-metal i5:

No target-hardware timing result is certified in this repository. The current test environment is Python 3.11 rather than the Python 3.13 target, and a meaningful benchmark requires validated model weights, real audio, and the reference i5. One API file is processed at a time by design.

## Input Requirements

- **Formats:** WAV, FLAC, MP3
- **Sample rate:** 44.1 kHz or 48 kHz
- **Duration:** up to 10 minutes, up to 500 MB
- **Channels:** mono or stereo (stereo is downmixed to mono)

For best results: clean recordings with clearly audible guitar and bass, minimal noise, and healthy headroom (peaks around −3 dB, no clipping).

## Understanding the Results

### The Processing Report

```json
{
  "separation_proxy": 21.3,
  "si_sdr": 21.3,
  "metric_semantics": "mask_crosstalk_not_si_sdr",
  "phase_coherence": 0.94,
  "avg_confidence": 0.80,
  "confidence_encoding": "polyphonic_key_pressure",
  "artifact_flags": ["none"],
  "low_confidence_notes": 1,
  "model_weights_loaded": true,
  "quality_tier": "draft"
}
```

- **separation_proxy** — a centroid-separation heuristic used for conservative routing; it is not SI-SDR and has no certified dB interpretation. `si_sdr` is a deprecated compatibility alias only.
- **metric_semantics** — explicit warning for consumers that the proxy is not reference-based SI-SDR.
- **phase_coherence** — model-reported phase score; closer to 1.0 is not independent audio-quality certification.
- **avg_confidence** — average confidence over emitted guitar and bass notes (0–1).
- **confidence_encoding** — MIDI Polyphonic Key Pressure; JSON remains canonical when a DAW ignores poly pressure.
- **artifact_flags** — detected problems such as clipping or `untrained_weights`.
- **low_confidence_notes** — count of emitted MIDI notes below the configured confidence threshold, not all low-confidence frames.
- **model_weights_loaded** — whether a validated checkpoint was loaded.
- **quality_tier** — heuristic routing decision; `studio` does not certify audio quality.

### Quality Tiers

**🟢 Studio gate** — confidence ≥ 85%, no low-confidence notes, and no artifact or invalid flags. The reported centroid proxy is observability only; this is a heuristic route, not SI-SDR certification or a promise that the unedited output is correct. A report from untrained weights cannot reach this tier.

**🟡 Draft** — confidence ≥ 70% without blocking artifacts, but below the Studio conditions. Review and refine notes in your DAW.

**🔴 Complex** — invalid metrics, confidence < 70%, artifacts, or untrained weights. Raw output may be delivered for inspection, but treat it as a sketch.

## Working with the MIDI in Your DAW

1. Import the WAV stems and the MIDI files. Note timing is absolute and aligned to the audio-frame grid; no tempo detection is performed.
2. Assign guitar/bass virtual instruments.
3. Use the JSON event metadata as the canonical confidence overlay. MIDI files also contain Polyphonic Key Pressure values: 96–127 high, 64–95 medium, 0–63 low. Some DAWs expose poly pressure as per-note aftertouch; others ignore it. Do not look for CC#127 — it is not emitted.
4. Edit pitch/timing/velocity of low-confidence notes by ear against the stems. Same-onset and overlapping notes are represented as independent MIDI note events.
5. If you enabled expression export, pitch-wheel is bend, CC1 is vibrato, and CC74 is the slide/brightness proxy. Mapping does not imply that the checkpoint learned useful expression.

There is no built-in MIDI editor — your DAW is the editor. MIDI import support for poly pressure, pitch bend, and CC lanes varies, so verify the rendered instrument in the target DAW.

## Tips for Best Results

- **Trim silence** at the start and end before processing.
- **Clean sources win.** Noise, heavy reverb, or dense arrangements lower confidence.
- **Distorted/very fast material** often lands in Draft or Complex — that's normal, not a failure.
- **One instrument focus:** the separator is trained for guitar + bass; other instruments may bleed into the stems.

## Privacy & Data Handling

When self-hosted (the supported deployment), **nothing leaves your machine**. Audio is written to a temporary file during processing and deleted afterwards; nothing is stored or uploaded. If you use a hosted instance run by someone else, their privacy policy applies — ask them.

## Rights

Only process audio you own or have permission to use — your own recordings, licensed material, or public-domain/CC-licensed works. Output stems and MIDI inherit the rights situation of the input; commercial redistribution of isolated stems from someone else's recording requires the rights holder's permission. The software itself is Apache-2.0 (see [LICENSE](LICENSE)).

## Troubleshooting

| Symptom | Likely cause | What to try |
|---|---|---|
| Low confidence everywhere | Noisy/lo-fi source, dense mix | Use a cleaner recording; check the artifact flags |
| Stems sound thin or phasey | Source stereo phase issues | Check the original mix; phase_coherence < 0.5 is a warning sign |
| MIDI misses notes | Very fast passages, heavy distortion | Edit in DAW; consider splitting the song into sections |
| Extra/ghost notes | Fret noise, string buzz, background sounds | Review low-confidence JSON/poly-pressure notes first |
| `503 Model not loaded` | Server still starting or required weights are missing | Set `MODEL_CHECKPOINT_PATH` for native Mamba, or use the explicit pretrained CPU profile, wait for `/ready`, then retry |

## Getting Help

- [API_DOCUMENTATION.md](API_DOCUMENTATION.md) — endpoint reference
- [SETUP.md](SETUP.md) — installation
- [TODO.md](TODO.md) — known issues and planned work (check here first if something seems broken)

---
*Stem+MIDI Pro: stem separation + editable MIDI drafts. Not magic—just math that respects your craft.*
