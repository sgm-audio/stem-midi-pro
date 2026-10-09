# Prototype processing status

This prototype does not report live processing stages or provide a live MIDI
preview. The following values can be rendered after processing only when the
caller supplies them:

- File: {{filename}}
- Confidence-head average (uncalibrated): {{avg_confidence}}
- `low_confidence_notes` report field (currently frame-based, not a reliable
  count of notes): {{low_confidence_notes}}

These values are model heuristics, not validated quality measurements.
