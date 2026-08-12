#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""
Training script for Stem+MIDI Pro.
CPU-only, bare-metal i5 target. No PyTorch Lightning.
"""

import argparse
import logging
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from main import StemMidiModel
from data.datasets import get_data_loaders

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Stem+MIDI Pro model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.mamba.yaml",
        help="Model config (use mamba backends for training)",
    )
    parser.add_argument("--data-config", type=str, default="example_data_config.yaml")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--strict-data", action="store_true",
                        help="Raise on missing dataset instead of falling back to synthetic")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--ema", action="store_true",
                        help="Use exponential moving average of weights")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--quantize", action="store_true",
                        help="Enable quantization-aware training (QAT) on Mamba backbone")
    args = parser.parse_args()

    # Load configs
    with open(args.config, "r") as f:
        model_config = yaml.safe_load(f)
    with open(args.data_config, "r") as f:
        data_config = yaml.safe_load(f)

    # TR-12.1: deterministic settings
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # CPU thread settings
    torch.set_num_threads(os.cpu_count() or 1)

    # Initialize model
    log.info("Initializing Stem+MIDI Pro model...")
    model = StemMidiModel(model_config)

    if args.quantize:
        log.warning(
            "QAT (--quantize) is not implemented; ignoring flag. "
            "Use process_audio_file(quantize=True) for INT8 dynamic inference quant."
        )

    # Setup data loaders
    log.info("Loading datasets...")
    train_loader, val_loader, test_loader = get_data_loaders(data_config)
    log.info(f"Train batches: {len(train_loader)}")
    log.info(f"Val batches: {len(val_loader)}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # EMA
    ema_model = None
    if args.ema:
        ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=lambda x, y: 0.999 * x + 0.001 * y)

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

    for epoch in range(start_epoch, args.max_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            audio = batch["audio"]
            targets = {
                "target_guitar": batch["target_guitar"],
                "target_bass": batch["target_bass"],
                "target_onsets": batch["target_onsets"],
                "target_pitch": batch["target_pitch"],
            }

            # CPU AMP: bfloat16
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                outputs = model(audio, **targets)
                loss = outputs["loss"] / args.grad_accum

            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

    # Test
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
            }
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                outputs = model(audio, **targets)
            test_loss += outputs["loss"].item()

    avg_test_loss = test_loss / len(test_loader)
    log.info(f"Test loss: {avg_test_loss:.4f}")
    log.info(f"Training complete! Checkpoints in {args.output_dir}/checkpoints")


if __name__ == "__main__":
    main()