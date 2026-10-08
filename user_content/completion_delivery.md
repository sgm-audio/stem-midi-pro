✨ Ready: {{filename}} processed successfully
<!-- Variables: filename is supplied by the caller (the upload's original name).
     si_sdr, phase_coherence, avg_confidence match processing_report.json keys
     exactly (see api.py create_response_zip). -->

📦 Your package includes:
• guitar_stem.wav · bass_stem.wav (SI-SDR: {{si_sdr}}dB, Phase Coherence: {{phase_coherence}})
• guitar.mid · bass.mid (Avg. Confidence: {{avg_confidence}})
• processing_report.json (full metrics + DAW import notes)

🎛️ DAW Tips:
• Import stems and MIDI together — note timing is aligned to the audio
• Low-confidence MIDI notes are tagged with CC#127 values <64 — check those first
• Note durations are fixed 16th notes in this version

[Download All] [Import into Your DAW] [Share Feedback]
