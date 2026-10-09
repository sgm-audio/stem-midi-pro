# Prototype upload confirmation

**File:** {{filename}}

**Sample rate:** {{sample_rate}} Hz

The local prototype may attempt to separate guitar and bass stems and create a
guitar MIDI draft. The model path and output quality have not been validated
end to end. Without a compatible checkpoint, the model uses random weights.

Uploads are written to a temporary file during processing. The service attempts
to remove that file afterward, but this is not secure erasure or a retention
guarantee. Do not upload sensitive recordings to an untrusted deployment.

This is development copy, not a live upload interface or processing-status
promise.
