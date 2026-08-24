#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""
Demucs fine-tuning script for Stem+MIDI Pro (Path A).
Trains the Demucs separator on MUSDB18HQ with guitar/bass emphasis.
Transcription head stays Basic Pitch (frozen/pretrained).
"""
import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from data.datasets import AudioDataset, get_data_loaders
from main import StemMidiModel


def main():
    parser = argparse.ArgumentParser(description="Demucs fine-tune (Path A)")
    parser.add_argument("--musdb-path", type=str, required=True,
                        help="Path to MUSDB18HQ dataset root")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (CPU target: 4-8 GB)")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Output directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load configs
    model_config = yaml.safe_load(open("configs/model_config.cpu.yaml"))
    data_config = yaml.safe_load(open("example_data_config.yaml"))

    # Override data config for MUSDB path
    data_config["dataset_type"] = "musdb18hq"
    data_config["root_dir"] = args.musdb_path
    data_config["batch_size"] = args.batch_size
    data_config["train"]["n_steps"] = args.epochs * 1000  # approximate
    data_config["train"]["lr"] = args.lr
    data_config["train"]["amp"] = False  # CPU-only

    # Seed
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Dataset / loaders
    log = __import__("logging").getLogger(__name__)
    log.info("Loading MUSDB18HQ dataset...")
    train_loader, val_loader, test_loader = get_data_loaders(data_config)
    log.info(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}")

    # Model — init from pretrained Demucs, freeze transcription
    log.info("Initializing model from scratch (Demucs init)...")
    model = StemMidiModel(model_config)

    # Freeze transcriber (Basic Pitch — keep pretrained weights)
    if hasattr(model, "transcriber") and hasattr(model.transcriber, "parameters"):
        for p in model.transcriber.parameters():
            p.requires_grad = False
        log.info("Basic Pitch transcriber frozen (Phase 1 weights).")

    # Only separator parameters will be trained
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    log.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # Optimizer
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        log.info(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # Training loop
    os.makedirs(f"{args.output_dir}/checkpoints", exist_ok=True)
    early_stop_patience = 10
    epochs_without_improvement = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            audio = batch["audio"]
            targets = {
                "target_guitar": batch["target_guitar"],
                "target_bass": batch["target_bass"],
                "target_onsets": batch["target_onsets"],
                "target_pitch": batch["target_pitch"],
                "target_velocity": batch.get("target_velocity"),
            }

            # CPU AMP: bfloat16
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                outputs = model(audio, **targets)
                loss = outputs["loss"] / args.grad_accum

            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.grad_accum
            pbar.set_postfix({"loss": loss.item() * args.grad_accum})

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.inference_mode():
            for batch in val_loader:
                audio = batch["audio"]
                targets = {
                    "target_guitar": batch["target_guitar"],
                    "target_bass": batch["target_bass"],
                    "target_onsets": batch["target_onsets"],
                    "target_pitch": batch["target_pitch"],
                    "target_velocity": batch.get("target_velocity"),
                }
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    outputs = model(audio, **targets)
                val_loss += outputs["loss"].item()

        avg_val_loss = val_loss / len(val_loader)
        log.info(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            ckpt_path = f"{args.output_dir}/checkpoints/best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }, ckpt_path)
            log.info(f"Saved best checkpoint (val_loss={best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                log.info(f"Early stopping at epoch {epoch}")
                break

    # Final test
    log.info("Running final test...")
    model.eval()
    test_loss = 0
    with torch.inference_mode():
        for batch in test_loader:
            audio = batch["audio"]
            targets = {
                "target_guitar": batch["target_guitar"],
                "target_bass": batch["target_bass"],
                "target_onsets": batch["target_onsets"],
                "target_pitch": batch["target_pitch"],
                "target_velocity": batch.get("target_velocity"),
            }
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                outputs = model(audio, **targets)
            test_loss += outputs["loss"].item()

    avg_test_loss = test_loss / len(test_loader)
    log.info(f"Test loss: {avg_test_loss:.4f}")
    log.info(f"Training complete! Checkpoints in {args.output_dir}/checkpoints")

    # Save final model info
    with open(f"{args.output_dir}/FINAL.md", "w") as f:
        f.write(f"# Demucs Fine-Tune (Path A) - Complete\n")
        f.write(f"- Epochs: {args.epochs}\n")
        f.write(f"- Best val loss: {best_val_loss:.4f}\n")
        f.write(f"- Test loss: {avg_test_loss:.4f}\n")
        f.write(f"- Trainable params: {sum(p.numel() for p in trainable_params):,}\n")
        f.write(f"- Separator fine-tuned; Basic Pitch transcription frozen\n")
        f.write(f"- Output: ./outputs/checkpoints/best.pt\n")


if __name__ == "__main__":
    main()