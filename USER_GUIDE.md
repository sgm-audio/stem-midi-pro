# Stem+MIDI Pro User Guide

## Introduction

Stem+MIDI Pro is a professional audio AI service that separates audio recordings into individual instrument stems (guitar and bass) and transcribes them into editable MIDI files. This guide will help you get the best results from the service, whether you're a musician, producer, audio engineer, or hobbyist.

## What Stem+MIDI Pro Does

### Core Features
1. **Stem Separation**: Splits mixed audio into separate guitar and bass tracks
2. **MIDI Transcription**: Converts audio to MIDI with note, timing, velocity, and expression data
3. **Quality Assessment**: Provides confidence scores and quality ratings for transparency
4. **DAW Integration**: Outputs are ready to use in Reaper, Logic Pro, Ableton Live, Pro Tools, etc.
5. **Canadian Artist Focus**: Optimized for and trained on Canadian audio content

## Getting Started

### System Requirements
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Internet connection
- Audio file to process (see supported formats below)
- No software installation required for the web interface

### Supported Audio Formats
- **File Types**: WAV, FLAC, MP3
- **Sample Rates**: 44.1 kHz or 48 kHz (other rates will be resampled)
- **Bit Depths**: 16-bit or 24-bit recommended (32-bit float accepted)
- **Duration**: Maximum 10 minutes per file
- **Channels**: Mono or stereo (stereo will be converted to mono for processing)

### Recommended Audio Quality
For best results:
- Use high-quality recordings with minimal background noise
- Ensure guitar and bass are clearly audible in the mix
- Avoid excessive compression or limiting that reduces dynamic range
- Peak levels should be below -1dB to prevent clipping
- Recordings with clear transients (pick attacks, finger plucks) work best

## Using the Web Interface

### Step 1: Upload Your Audio
1. Visit the Stem+MIDI Pro web interface
2. Review and accept the rights and usage confirmation
3. Click the upload area or drag & drop your audio file
4. Wait for validation (format, duration, sample rate checks)

### Step 2: Review Processing Information
After upload, you'll see:
- File name and duration
- Detected sample rate
- Estimated processing time
- Genre detection (if applicable)
- Processing options based on file characteristics

### Step 3: Processing Stages
The service processes your audio in several stages:
1. **Upload Validation**: Checks file properties
2. **Pre-processing**: Resampling and normalization if needed
3. **Stem Separation**: AI separates guitar and bass using Mamba-SSM
4. **MIDI Transcription**: AI transcribes the guitar stem to MIDI
5. **Quality Assessment**: Confidence scoring and artifact detection
6. **Routing Decision**: Determines output quality tier
7. **Output Generation**: Creates stems, MIDI files, and report

### Step 4: Review Results
Upon completion, you'll see:
- Quality badge (Studio/Draft/Complex)
- Processing report with metrics
- Routing recommendation based on quality
- Download options

## Understanding Your Results

### Audio Stems
You'll receive two WAV files:
- **guitar_stem.wav**: Isolated guitar track
- **bass_stem.wav**: Isolated bass track

**Stem Quality Indicators**:
- **SI-SDR**: Signal-to-Distortion Ratio (higher is better, >20dB is excellent)
- **Phase Coherence**: Measures timing alignment between stems (closer to 1.0 is better)
- **Artifact Flags**: Any detected issues like clipping or noise

### MIDI Files
You'll receive two MIDI files:
- **guitar.mid**: Transcribed guitar performance
- **bass.mid**: Transcribed bass performance

**MIDI Quality Indicators**:
- **Onset F1**: Accuracy of note start times (higher is better)
- **Average Confidence**: Overall confidence in transcription (0-100%)
- **Low Confidence Notes**: Number of notes below confidence threshold
- **Expression Data**: Bends, vibrato, slides detected and encoded

### Processing Report
The `processing_report.json` file contains:
```json
{
  "si_sdr": 21.3,
  "phase_coherence": 0.94,
  "avg_confidence": 0.89,
  "artifact_flags": ["none"],
  "low_confidence_notes": 12,
  "quality_tier": "studio"
}
```

## Quality Tiers Explained

### 🟢 Studio Quality
- **Criteria**: Confidence ≥85% AND SI-SDR ≥20dB
- **What it means**: Professional-grade separation and transcription
- **Output**: Ready for immediate use in productions
- **User Action**: Direct download recommended

### 🟡 Draft Quality
- **Criteria**: 70% ≤ Confidence < 85%
- **What it means**: Good starting point needing minor refinement
- **Output**: Editable MIDI with confidence visualization
- **User Action**: Use the built-in MIDI editor to refine low-confidence notes

### 🔴 Complex Material
- **Criteria**: Confidence <70% OR significant artifacts detected
- **What it means**: Challenging material that may need special handling
- **Output**: Options for raw download, human review, or refund
- **User Action**: Choose your preferred path forward

## Working with MIDI Files

### Importing into Your DAW
1. Import the WAV stems at your project's original tempo
2. Import the MIDI files onto MIDI tracks
3. Assign appropriate virtual instruments (guitar/bass VSTs)
4. Enable the tempo sync feature if your DAW supports it
5. Align the MIDI notes with the audio stems for verification

### Understanding MIDI Metadata
Stem+MIDI Pro enriches MIDI files with useful metadata:

**Confidence Visualization (CC#127)**:
- Each MIDI note has a Continuous Controller value (CC#127) representing confidence
- Values 0-63: Low confidence (<50%)
- Values 64-95: Medium confidence (50-75%)
- Values 96-127: High confidence (75-100%)
- Most DAWs can display or filter by CC values

**Expression Data**:
- **Pitch Bend**: Encoded as standard MIDI pitch bend messages
- **Vibrato**: Encoded as modulation wheel (CC#1) or pitch bend variations
- **Slides/Portamento**: Encoded as portamento time (CC#5) or glide parameters
- **Palm Mutes**: Detected and can be mapped to articulation switches

### Editing Low-Confidence Notes
If you receive a Draft Quality result:
1. Open the MIDI file in your DAW or the built-in editor
2. Look for notes highlighted in amber/yellow (low confidence)
3. Listen to each note against the original audio
4. Adjust pitch, timing, or velocity as needed
5. Consider using quantization options:
   - **"Grid"**: Snap to musical grid (good for high-confidence sections)
   - **"Human Feel"**: Preserve natural timing variations (recommended for low-confidence areas)
6. Save your edited MIDI and re-export if needed

## Tips for Best Results

### Before Uploading
- **Trim silence**: Remove long sections of silence at start/end
- **Normalize volume**: Aim for peak around -3dB to leave headroom
- **Check phase**: Ensure stereo tracks aren't severely out of phase
- **Consider BPM**: While the service detects tempo, knowing your approximate BPM helps

### For Specific Genres
- **Clean Guitar (Acoustic, Jazz)**: Usually processes very well, expect Studio quality
- **Overdriven Guitar (Rock, Metal)**: May need more processing, expect Draft quality
- **Bass (Fingerstyle)**: Usually good separation
- **Bass (Pick/Slap)**: May show more articulation complexity
- **Mixed Ensembles**: Service is optimized for guitar/bass; other instruments may appear in stems

### Troubleshooting Common Issues
- **Low Confidence Throughout**: Check audio quality, consider re-recording or using a cleaner source
- **Stems Sound Weak**: May indicate phase issues in original stereo recording
- **MIDI Misses Notes**: Often occurs with very fast playing or heavy distortion
- **False Notes Detected**: Usually from string noise, fret buzz, or background sounds
- **Timing Seems Off**: Verify your DAW's tempo matches the detected tempo in the report

## Advanced Features

### Expression Detection
Stem+MIDI Pro detects and encodes:
- **Bends**: Pitch changes via finger pressure or whammy bar
- **Vibrato**: Cyclical pitch variation
- **Slides**: Fretted hand slides between notes
- **Palm Mutes**: Right-hand muting technique
- **Harmonics**: Natural and artificial harmonics
- **Slap/Pop**: Bass-specific techniques

### Tuning Detection
The service automatically detects:
- Standard tuning (EADGBE)
- Drop D (DADGBE)
- Half-step down (EbAbDbGbBbEb)
- Custom tunings (reported in processing report)

### Tempo Mapping
- Automatic tempo detection and mapping
- MIDI files are tempo-synced to original audio
- Tempo changes within the song are preserved
- Report includes detected average tempo and confidence

## Frequently Asked Questions

### Q: How long does processing take?
A: Processing time depends on file length and server load:
- 1-minute song: ~15-30 seconds
- 3-minute song: ~45-90 seconds
- 10-minute song: ~3-5 minutes
Times shown before processing are estimates based on current server load.

### Q: Is my audio stored or used for anything else?
A: No. Your audio is processed in-memory only and automatically deleted after 24 hours. We do not store, analyze, or use your audio for any purpose beyond providing the service you requested. See our privacy policy for details.

### Q: Can I process copyrighted music?
A: You should only upload audio you have rights to process. The service is intended for:
- Your own original recordings
- Music you have licensed or have permission to process
- Public domain or Creative Commons licensed works
- Fair use for educational, criticism, or parody purposes (check local laws)

### Q: Why do I see low-confidence notes?
A: Low-confidence notes occur when:
- Audio is noisy or low fidelity
- Notes are played very fast or with complex articulation
- There's significant bleed from other instruments
- The performance has unusual techniques or effects
- The mix has heavy processing (reverb, distortion, etc.)

### Q: Can I get better results by processing stems separately?
A: The service is optimized for mixed audio. Processing already-separated stems may actually reduce quality because:
- The separation model expects mixed input
- Already-processed audio may have artifacts
- You lose the benefit of joint optimization
For best results, upload the original mixed audio.

### Q: What if I'm not happy with the results?
A: Depending on the quality tier:
- **Studio**: Contact support if you believe there's an error
- **Draft**: Use the editor to refine results
- **Complex**: Choose human review (+$4.99, 24h turnaround) or request a credit refund

## Best Practices for Different Use Cases

### For Creating Backing Tracks
1. Upload your full band recording
2. Download guitar and bass stems
3. Mute or reduce the original guitar/bass in your DAW
4. Use the stems as-is or with light processing
5. Align any additional instruments to the stems

### For Transcription and Learning
1. Upload the song you want to learn
2. Download the MIDI files
3. Import into notation software or DAW
4. Use the confidence scores to focus practice on uncertain areas
5. Slow down or loop sections for detailed study

### For Remixing and Production
1. Upload stems from your multitrack recordings
2. Process to get clean guitar/bass isolation
3. Use the separated stems in your mix
4. Use the MIDI to trigger different virtual instruments
5. Extract grooves or melodies for new compositions

### For Audio Restoration
1. Upload older recordings with bleed issues
2. Process to isolate problematic instruments
3. Use the cleaner stems in restoration workflow
4. Combine with spectral editing tools for best results

## Privacy and Data Handling

### What We Collect
- Processing metrics (SI-SDR, confidence, processing time)
- Anonymous usage statistics (feature usage, error rates)
- No personal data unless you voluntarily provide it in feedback

### What We Don't Collect
- Audio content (processed in-memory only)
- File names or metadata beyond processing needs
- Personal identifiers
- Usage tied to individual users

### Data Retention
- Audio files: Deleted after 24 hours
- Processing logs: Aggregated and anonymized, kept for 30 days
- Error reports: Kept for 90 days for debugging
- Account information: If you create an account, retained per our privacy policy

### Your Rights
- You can request deletion of any data we retain
- You can opt out of anonymous analytics
- You can export your data if you have an account
- We comply with GDPR, CCPA, and other applicable regulations

## Getting Help and Support

### Documentation
- This user guide
- API documentation (for developers)
- Release notes and changelog
- FAQ section on the website

### Community
- User forums for tips and troubleshooting
- Feature request board
- Bug reporting system
- Example projects and use cases

### Technical Support
- Email support for paid users
- Priority support for enterprise customers
- Community support for free tier users
- Status page for service availability

### Feedback
We actively seek user feedback to improve the service:
- In-app feedback forms
- Optional MIDI upload for training (anonymized)
- Feature voting system
- Regular user surveys

## Conclusion

Stem+MIDI Pro aims to provide professional-quality audio separation and transcription while being transparent about AI limitations and giving you control over the results. By understanding the quality metrics and using the provided tools, you can get excellent results for music production, learning, restoration, and creative projects.

Remember that AI is a tool to augment your creativity, not replace it. Use the confidence scores and editing features to guide your workflow, and trust your ears as the final judge of quality.

Happy music-making! 🎵

---
*User Guide Version: 1.0*
*Last Updated: 2026-05-31*
*Stem+MIDI Pro: Studio-grade stem separation + editable MIDI drafts. Not magic—just math that respects your craft.*