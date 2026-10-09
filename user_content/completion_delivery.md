# Prototype output summary

**Input file:** {{filename}}

The API attempts to package guitar and bass WAV stems, `guitar.mid`, `bass.mid`,
and `processing_report.json`. Processing and output quality have not been
validated end to end.

## Report values

- `si_sdr`: {{si_sdr}} — currently a spectral-centroid heuristic, not SI-SDR.
- `phase_coherence`: {{phase_coherence}} — currently a model score, not a
  measured coherence metric.
- `avg_confidence`: {{avg_confidence}} — an uncalibrated confidence-head
  average.
- `low_confidence_notes`: {{low_confidence_notes}} — currently frame-based,
  not a reliable note count.

## MIDI limitations

The current pipeline transcribes the guitar stem only and reuses that event
stream for `bass.mid`. MIDI uses a fixed 120 BPM tempo and default 1/16-note
durations; note durations are not predicted. Review any generated files before
using them in a project.

This repository does not include a hosted MIDI editor, DAW project template,
or human-review service. This Markdown file is draft copy; it does not mean a
package was successfully generated.
