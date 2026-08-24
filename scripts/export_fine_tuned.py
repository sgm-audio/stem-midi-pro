#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""
Export fine-tuned Demucs model for CPU inference.

Usage:
    python scripts/export_fine_tuned.py \
        --checkpoint ./outputs/checkpoints/best.pt \
        --config configs/model_config.cpu.yaml \
        --output ./outputs/exported \
        --audio ./data/test_audio.wav
"""
import argparse
import yaml
import torch
import numpy as np
import soundfile as sf
from pathlib import Path

from main import StemMidiModel


def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned Demucs model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to fine-tuned checkpoint (.pt)")
    parser.add_argument("--config", type=str, default="configs/model_config.cpu.yaml",
                        help="Model config YAML")
    parser.add_argument("--output", type=str, default="./outputs/exported",
                        help="Output directory")
    parser.add_argument("--audio", type=str, default=None,
                        help="Input audio path (optional, for test run)")
    parser.add_argument("--quantize", action="store_true",
                        help="Enable INT8 dynamic quantization for CPU")
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
    log.info(f"Checkpoint loaded (epoch {ckpt.get('epoch', '?')}, "
             f"best_val_loss={ckpt.get('best_val_loss', '?'):.4f}")

    # Test inference if audio provided
    if args.audio and Path(args.audio).exists():
        log.info(f"Running inference on: {args.audio}")
        audio, sr = sf.read(args.audio)
        if audio.ndim > 1 and audio.shape[1] > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        result = model.process_audio_file(args.audio, quantize=args.quantize)

        print("=" * 60)
        print("FINE-TUNED MODEL INFERENCE RESULTS")
        print("=" * 60)
        print(f"Quality tier: {result['report'].quality_tier.value}")
        print(f"SI-SDR: {result['report'].si_sdr:.1f} dB")
        print(f"Avg confidence: {result['report'].avg_confidence:.0%}")
        print(f"Routing: {result['routing']['action']}")
        print(f"Artifacts: {result['report'].artifact_flags}")
        print()
        print("Guitar stem (first 100 samples):")
        print(result["stems"]["guitar"][:100])
        print()
        print("Bass stem (first 100 samples):")
        print(result["stems"]["bass"][:100])
        print()
        print("MIDI guitar notes:")
        for ev in result["midi_guitar"]["midi_events"][:5]:
            print(f"  note={ev['note']}, frame={ev['onset_frame']}, "
                  f"velocity={ev['velocity']}, confidence={ev['confidence']:.2f}")
        print("MIDI bass notes:")
        for ev in result["midi_bass"]["midi_events"][:5]:
            print(f"  note={ev['note']}, frame={ev['onset_frame']}, "
                  f"velocity={ev['velocity']}, confidence={ev['confidence']:.2f}")
        print("=" * 60)

    # Save exported model
    export_dir = Path(args.output)
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / "model.pt"
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
    log.info(f"Exported fine-tuned model to: {export_path}")

    # Also save a CPU-only version with quantized Linear if requested
    if args.quantize:
        import copy
        qmodel = copy.deepcopy(model)
        qmodel = torch.ao.quantization.quantize_dynamic(
            qmodel, {torch.nn.Linear}, dtype=torch.qint8
        )
        q_path = export_dir / "model_quantized.pt"
        torch.save({
            "model_state_dict": qmodel.state_dict(),
            "quantized": True,
        }, q_path)
        log.info(f"Quantized model saved to: {q_path}")

    print()
    print("Export complete!")
    print(f"Use model_config.cpu.yaml with restore_from_path pointing to {export_path}")
    print("for CPU inference with the fine-tuned Demucs separator.")


if __name__ == "__main__":
    main()