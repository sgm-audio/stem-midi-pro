🔄 Processing {{filename}} ...
<!-- Template variables supplied by the caller at render time:
     filename, avg_confidence, low_confidence_notes (from processing_report.json keys).
     Per-stage live progress (current_stage, total_stages) is NOT available yet —
     the pipeline reports only at completion. -->
This runs on CPU. No reference-hardware timing guarantee is published until a
trained-checkpoint benchmark is recorded; processing time depends on backend,
checkpoint, input length, and hardware.

When done you'll get:
• Confidence: {{avg_confidence}}
• Low-confidence notes: {{low_confidence_notes}}

[Estimating time remaining…] [Processing runs synchronously — you get the package when it's done]
