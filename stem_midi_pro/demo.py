#!/usr/bin/env python3
"""
Demo script showing how to use the Stem+MIDI Pro model
"""

import torch
import numpy as np
from main import StemMidiModel
import yaml

def create_dummy_audio(duration_sec=10, sample_rate=44100):
    """Create a dummy audio signal for testing"""
    t = np.linspace(0, duration_sec, int(duration_sec * sample_rate))
    # Create a simple guitar-like signal (two sine waves + noise)
    audio = (
        0.3 * np.sin(2 * np.pi * 110 * t) +  # A2 note
        0.2 * np.sin(2 * np.pi * 220 * t) +  # A3 note
        0.1 * np.random.randn(len(t))        # Noise
    ).astype(np.float32)
    return audio

def main():
    print("Stem+MIDI Pro Demo")
    print("=" * 50)
    
    # Load configuration
    with open("configs/model_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config: {config['name']}")
    
    # Initialize model
    print("\nInitializing model...")
    model = StemMidiModel(config)
    print("Model initialized successfully!")
    
    # Create dummy audio input
    print("\nCreating test audio signal...")
    dummy_audio = create_dummy_audio(duration_sec=5)  # 5 seconds
    audio_tensor = torch.from_numpy(dummy_audio).unsqueeze(0).unsqueeze(0)  # Add batch/channels
    print(f"Audio shape: {audio_tensor.shape}")
    
    # Run inference
    print("\nRunning inference...")
    with torch.inference_mode():
        outputs = model.forward(audio_tensor)
    
    # Display results
    print("\nResults:")
    print("-" * 30)
    
    # Stem shapes
    print(f"Guitar stem shape: {outputs['guitar_stem'].shape}")
    print(f"Bass stem shape: {outputs['bass_stem'].shape}")
    
    # Processing report
    report = outputs['processing_report']
    print(f"\nProcessing Report:")
    print(f"  SI-SDR: {report.si_sdr:.1f} dB")
    print(f"  Phase Coherence: {report.phase_coherence:.3f}")
    print(f"  Avg Confidence: {report.avg_confidence:.0%}")
    print(f"  Artifact Flags: {report.artifact_flags}")
    print(f"  Low Confidence Notes: {report.low_confidence_notes}")
    
    # Quality tier
    print(f"\nQuality Tier: {report.quality_tier.value.upper()}")
    
    # Routing decision
    routing = outputs['routing_decision']
    print(f"\nRouting Decision:")
    print(f"  Action: {routing['action']}")
    print(f"  Badge: {routing['badge']}")
    print(f"  Message: {routing['message']}")
    
    # MIDI metadata sample
    midi_meta = outputs['midi_metadata']
    print(f"\nMIDI Metadata:")
    print(f"  Total Events: {len(midi_meta['midi_events'])}")
    print(f"  Avg Confidence: {midi_meta['summary']['avg_confidence']:.0%}")
    print(f"  Low Confidence Count: {midi_meta['summary']['low_confidence_count']}")
    print(f"  Alignment Warnings: {midi_meta['summary']['alignment_warnings']}")
    
    if midi_meta['midi_events']:
        print(f"\nSample MIDI Event:")
        event = midi_meta['midi_events'][0]
        print(f"  Note: {event['note']} (MIDI)")
        print(f"  Onset Frame: {event['onset_frame']}")
        print(f"  Velocity: {event['velocity']}")
        print(f"  Confidence: {event['confidence']:.0%}")
        print(f"  Needs Review: {event['needs_review']}")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()