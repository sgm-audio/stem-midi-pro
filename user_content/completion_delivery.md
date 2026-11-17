✨ Ready: {{filename}} processed successfully
<!-- Variables: filename is supplied by the caller (the upload's original name).
     separation_proxy, phase_coherence, avg_confidence match
     processing_report.json keys (see api.py create_response_zip). The old
     si_sdr field is only a deprecated compatibility alias and is not SI-SDR. -->

📦 Your package includes:
• guitar_stem.wav · bass_stem.wav (centroid separation proxy: {{separation_proxy}}, Phase Coherence: {{phase_coherence}})
• guitar.mid · bass.mid (Avg. Confidence: {{avg_confidence}})
• processing_report.json (full metrics + DAW import notes)

🎛️ DAW Tips:
• Import stems and MIDI together — note timing is aligned to the audio
• Low-confidence notes are in the JSON event metadata and may be encoded as Polyphonic Key Pressure; DAW support varies
• Note durations follow the model's `duration_frames` output (configurable default when absent)

[Download All] [Import into Your DAW] [Share Feedback]
