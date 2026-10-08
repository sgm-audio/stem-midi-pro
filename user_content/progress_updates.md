🔄 Processing {{filename}} ...
<!-- Template variables supplied by the caller at render time:
     filename, avg_confidence, low_confidence_notes (from processing_report.json keys).
     Per-stage live progress (current_stage, total_stages) is NOT available yet —
     the pipeline reports only at completion. -->
This runs on CPU: expect roughly 30–60 seconds of processing per minute of audio
(a 3-minute song typically takes 1.5–3 minutes).

When done you'll get:
• Confidence: {{avg_confidence}}
• Low-confidence notes: {{low_confidence_notes}}

[Estimating time remaining…] [Processing continues in background if you close this]
