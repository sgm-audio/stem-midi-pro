#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""
Demo script showing how to use the Stem+MIDI Pro model
"""

import logging
import torch
import numpy as np
from main import StemMidiModel
import yaml

logger = logging.getLogger(__name__)

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
    
    # Prefer fake backends for a fast smoke test without demucs/basic-pitch/mamba.
    # Use configs/model_config.cpu.yaml for real pretrained backends.
    config = {
        "name": "demo_fake",
        "backends": {"separator": "fake", "transcriber": "fake", "device": "cpu"},
        "audio": {
            "sample_rate": 44100,
            "n_fft": 2048,
            "hop_length": 512,
            "n_mels": 80,
            "chunk_duration_sec": 2.0,
            "overlap_ratio": 0.5,
        },
        "separator": {"d_model": 64, "n_layer": 1, "d_state": 8, "d_conv": 4, "expand": 2},
        "transcriber": {
            "d_model": 64,
            "n_layer": 1,
            "d_state": 8,
            "onset_head_dim": 32,
            "pitch_vocab_size": 128,
            "velocity_bins": 128,
            "expression_heads": ["bend", "vibrato", "slide"],
            "onset_threshold": 0.5,
        },
        "training": {"precision": "fp32", "gradient_checkpointing": False},
        "loss": {
            "mr_stft_weight": 1.0,
            "spectral_flatness_weight": 0.1,
            "crest_factor_weight": 0.05,
            "onset_f1_weight": 1.0,
            "pitch_ce_weight": 0.8,
            "velocity_mae_weight": 0.3,
            "cross_modal_alignment_weight": 0.5,
        },
        "quality_gates": {
            "studio_confidence_threshold": 0.85,
            "draft_confidence_threshold": 0.70,
            "min_confidence": 0.6,
            "min_si_sdr": 20.0,
        },
    }
    print(f"Loaded config: {config['name']} (fake backends — no heavy deps)")

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
    # Strip non-ASCII (emoji badges break Windows cp1252 consoles)
    badge = routing["badge"].encode("ascii", "ignore").decode("ascii").strip()
    print(f"  Badge: {badge}")
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
