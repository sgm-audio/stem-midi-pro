# Stem+MIDI Pro

## Experimental audio-to-stems and MIDI prototype

Stem+MIDI Pro is a research prototype for separating guitar and bass audio and
producing MIDI drafts. It is not a hosted product or a validated transcription
service. This repository contains no trained checkpoint or browser interface;
without a compatible checkpoint, the model initializes with random weights.

## Current limitations

- End-to-end model inference and output quality have not been validated.
- The pipeline transcribes guitar only; the bass MIDI file reuses the guitar
event stream.
- MIDI uses a fixed 120 BPM tempo and default note durations because note
durations are not predicted.
- Reported confidence and separation metrics are heuristics, not calibrated
quality measurements.
- Upload processing writes a temporary file and attempts cleanup. This is not
secure erasure or a retention guarantee.

The API accepts `.wav`, `.flac`, and `.mp3` filename suffixes, subject to the
validation limits documented in the repository's API notes. This copy is
informational development material, not an offer of service or performance
promise.

The software is licensed under Apache-2.0; see the repository `LICENSE`.
That software license does not grant rights to any audio you upload or create.
