# Stem+MIDI Pro User Guide

## Introduction

Stem+MIDI Pro separates a mixed recording into **guitar and bass stems** and transcribes them into **editable MIDI files**, with per-note confidence scores so you can see exactly what the AI is sure about. It runs locally on ordinary hardware — a bare-metal Intel i5 with 4–8 GB RAM is enough. No internet connection is needed once the software is installed.

## What It Actually Does Today

1. **Stem separation** — produces `guitar_stem.wav` and `bass_stem.wav` from a mixed recording.
2. **MIDI transcription** — produces `guitar.mid` and `bass.mid` with note onsets, pitches, and velocities. Note durations are currently a fixed 16th-note length (improvement tracked in the project TODO).
3. **Confidence metadata** — every MIDI note carries a CC#127 value (0–127) representing the model's confidence, so your DAW can highlight uncertain notes.
4. **Quality report** — `processing_report.json` with separation metrics (SI-SDR estimate, phase coherence), average confidence, artifact flags, and an overall quality tier.

**Not included (yet):** tempo detection, tuning detection, palm-mute / harmonics / slap-pop detection, a web-based MIDI editor, or real-time streaming. Refinement happens in your own DAW. Earlier versions of this guide described those features aspirationally; this version describes the software as it is.

## Getting Your Results

Processing is done via the API or CLI (see [SETUP.md](SETUP.md), [API_DOCUMENTATION.md](API_DOCUMENTATION.md)):

```bash
# CLI
python main.py --audio mysong.wav --config configs/model_config.yaml

# or via the API
curl -X POST http://localhost:8000/process -F "file=@mysong.wav" -o output.zip
```

You get a package containing:

- `guitar_stem.wav`, `bass_stem.wav` — separated stems
- `guitar.mid`, `bass.mid` — MIDI transcriptions
- `processing_report.json` — metrics and quality tier

## How Long It Takes (CPU)

On the reference bare-metal i5:

- **~30–60 seconds of processing per 1 minute of audio**
- 1-minute song: ~30–60 s
- 3-minute song: ~1.5–3 minutes
- 10-minute song (maximum): ~5–10 minutes

Longer files are fine — they just take proportionally longer. One file processes at a time.

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
  "si_sdr": 21.3,
  "phase_coherence": 0.94,
  "avg_confidence": 0.89,
  "artifact_flags": [],
  "low_confidence_notes": 12,
  "quality_tier": "studio"
}
```

- **si_sdr** — separation-quality estimate in dB; higher is better, ≥ 20 dB is the "studio" bar. (Note: this is currently an internal estimate — see the project TODO, item C-2.6.)
- **phase_coherence** — how phase-aligned the two stems are; closer to 1.0 is better.
- **avg_confidence** — average transcription confidence (0–1).
- **artifact_flags** — detected problems such as clipping.
- **low_confidence_notes** — count of MIDI notes below the confidence threshold.
- **quality_tier** — overall routing decision.

### Quality Tiers

**🟢 Studio** — confidence ≥ 85% and SI-SDR ≥ 20 dB. Ready to use as-is.

**🟡 Draft** — confidence 70–85%. Good starting point; expect to clean up some notes in your DAW.

**🔴 Complex** — confidence < 70% or artifacts detected. The raw output is still delivered, but treat it as a sketch: dense mixes, heavy distortion, or very fast playing often land here.

## Working with the MIDI in Your DAW

1. Import the WAV stems and the MIDI files at the same project tempo (MIDI note timing is absolute — aligned to the audio, not to a detected tempo grid).
2. Assign guitar/bass virtual instruments.
3. **Use CC#127 to find uncertain notes.** Each note's confidence is encoded as a CC#127 controller value: 96–127 high confidence, 64–95 medium, 0–63 low. Most DAWs (Reaper, Logic, Cubase, Ableton) let you view or filter by controller data — use that to jump straight to the notes that need attention.
4. Edit pitch/timing/velocity of the low-confidence notes by ear against the stems.

There is no built-in MIDI editor — your DAW is the editor. MIDI expression data (bend, vibrato, slide) is produced by the model's expression head but is currently untrained, so treat it as advisory only.

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
| Extra/ghost notes | Fret noise, string buzz, background sounds | Delete flagged low-CC#127 notes first |
| `503 Model not loaded` | Server still starting | Wait for startup to finish, retry |

## Getting Help

- [API_DOCUMENTATION.md](API_DOCUMENTATION.md) — endpoint reference
- [SETUP.md](SETUP.md) — installation
- [TODO.md](TODO.md) — known issues and planned work (check here first if something seems broken)

---
*Stem+MIDI Pro: stem separation + editable MIDI drafts. Not magic—just math that respects your craft.*
