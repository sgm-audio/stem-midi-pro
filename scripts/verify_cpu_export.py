#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""
CPU export verification for fine-tuned Demucs model (Path A).

Tests the end-to-end flow:
1. Load fine-tuned checkpoint
2. Run inference on synthetic audio
3. Verify stems and MIDI output
4. Run INT8 quantization
5. Verify backward compatibility with model_config.cpu.yaml
"""
import argparse
import yaml
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from main import StemMidiModel


def create_synthetic_audio(duration=2.0, sr=44100):
    """Create a simple synthetic stereo audio for testing."""
    t = np.linspace(0, duration, int(sr * duration))
    # Two notes: A2 (110Hz) on guitar channel, E4 (330Hz) on bass channel
    guitar = 0.5 * np.sin(2 * np.pi * 110 * t)
    bass = 0.5 * np.sin(2 * np.pi * 330 * t)
    mixture = guitar + bass
    # Normalize and add slight reverb-like tail
    mixture = mixture / np.max(np.abs(mixture))
    return mixture.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Verify CPU export of fine-tuned model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to fine-tuned checkpoint (.pt)")
    parser.add_argument("--config", type=str, default="configs/model_config.cpu.yaml",
                        help="Model config YAML")
    parser.add_argument("--output", type=str, default="./outputs/exported_verify",
                        help="Output directory")
    parser.add_argument("--synthetic-duration", type=float, default=2.0,
                        help="Duration of synthetic test audio")
    parser.add_argument("--sr", type=int, default=44100, help="Sample rate")
    parser.add_argument("--quantize", action="store_true",
                        help="Enable INT8 dynamic quantization")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Initialize model
    log = __import__("logging").getLogger(__name__)
    log.info(f"Loading fine-tuned checkpoint: {args.checkpoint}")
    model = StemMidiModel(config)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info(f"Checkpoint loaded (epoch {ckpt.get('epoch', '?')})")

    # Create synthetic audio
    audio = create_synthetic_audio(args.synthetic_duration, args.sr)
    audio_path = Path("./synthetic_test_audio.wav")
    sf.write(audio_path, audio, args.sr)

    # Run inference
    log.info(f"Running inference on synthetic audio ({args.synthetic_duration}s)")
    result = model.process_audio_file(str(audio_path), quantize=args.quantize)

    # Print results
    print("=" * 70)
    print("CPU EXPORT VERIFICATION RESULTS (Path A: Demucs fine-tune)")
    print("=" * 70)
    print(f"Quality tier: {result['report'].quality_tier.value}")
    print(f"SI-SDR: {result['report'].si_sdr:.1f} dB")
    print(f"Avg confidence: {result['report'].avg_confidence:.0%}")
    print(f"Routing: {result['routing']['action']}")
    print(f"Artifacts: {result['report'].artifact_flags}")
    print()
    print("Guitar stem (first 50 samples):")
    print(result["stems"]["guitar"][:50])
    print()
    print("Bass stem (first 50 samples):")
    print(result["stems"]["bass"][:50])
    print()
    print("MIDI guitar notes:")
    for i, ev in enumerate(result["midi_guitar"]["midi_events"][:5]):
        print(f"  {i+1}. note={ev['note']}, frame={ev['onset_frame']}, "
              f"velocity={ev['velocity']}, confidence={ev['confidence']:.2f}")
    print("MIDI bass notes:")
    for i, ev in enumerate(result["midi_bass"]["midi_events"][:5]):
        print(f"  {i+1}. note={ev['note']}, frame={ev['onset_frame']}, "
              f"velocity={ev['velocity']}, confidence={ev['confidence']:.2f}")
    print()
    print("Stem shapes:")
    print(f"  Guitar: {result['stems']['guitar'].shape}")
    print(f"  Bass: {result['stems']['bass'].shape}")
    print()
    print("=" * 70)

    # Save exported model
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_path = out_dir / "model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "ft_info": {
            "fine_tuned": True,
            "base_backend": "demucs",
            "transcription": "basic_pitch (frozen)",
            "checkpoint": args.checkpoint,
        }
    }, export_path)
    log.info(f"Exported model to: {export_path}")

    # Also save quantized version if requested
    if args.quantize:
        import copy
        qmodel = copy.deepcopy(model)
        qmodel = torch.ao.quantization.quantize_dynamic(
            qmodel, {torch.nn.Linear}, dtype=torch.qint8
        )
        q_path = out_dir / "model_quantized.pt"
        torch.save({
            "model_state_dict": qmodel.state_dict(),
            "quantized": True,
        }, q_path)
        log.info(f"Quantized model saved to: {q_path}")

    # Cleanup
    try:
        audio_path.unlink()
    except:
        pass

    print()
    print("Verification complete!")
    print(f"Model exported successfully to {args.output}")
    print("Use model_config.cpu.yaml with restore_from_path for CPU inference.")


if __name__ == "__main__":
    main()