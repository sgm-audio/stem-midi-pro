# Stem+MIDI Pro

Guitar/bass stem separation + editable MIDI drafts.
Not magic—just math that respects your craft.

## ✓ What You Get
- Guitar/bass stems (WAV) from a configured backend and a validated checkpoint
- DAW-oriented MIDI with independent notes, absolute timing, durations, overlaps, expression mapping, and note-level confidence JSON
- MIDI Polyphonic Key Pressure confidence when supported by the DAW; CC#127 is not emitted
- Runs locally on CPU; target-hardware performance is not certified until a trained-checkpoint benchmark is available

## For Producers Who Need Drafts, Not Promises

### How It Works
1. **Upload** your audio file (WAV/FLAC/MP3, <10 min)
2. **AI Processing** separates stems and transcribes each stem to MIDI with confidence scoring
3. **Review & Edit** low-confidence notes in your own DAW; use the JSON event metadata or Polyphonic Key Pressure where supported
4. **Export** a package with stems, MIDI, and an explicitly caveated processing report

### Quality You Can Verify
- See the emitted note metadata and which notes need review
- Polyphonic Key Pressure is note-specific but DAW support varies; JSON is canonical
- Every job ships with `processing_report.json`: centroid separation proxy (not SI-SDR), phase score, confidence, model status, and quality tier
- Your audio is processed in a temp file deleted immediately after the request

### Perfect For
- Creating backing tracks from existing recordings
- Generating MIDI arrangements for remixing or arrangement
- Transcribing guitar/bass parts for notation or learning
- Quick demos and idea generation

### License
Stem+MIDI Pro is free and open source under the **Apache License 2.0** — free for individuals and businesses, including commercial use. See [LICENSE](../LICENSE).

[Get Started Free]
