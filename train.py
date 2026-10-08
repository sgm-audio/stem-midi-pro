#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Training script for Stem+MIDI Pro.

Plain PyTorch training loop (TR-12.3) — PyTorch Lightning was removed
(DEP-4.1.4); NeMo was removed earlier (RF-3.1.7). CPU-only target.

Features: epoch train/val loop, gradient accumulation (TR-12.4), grad
clipping, best/last checkpointing, early stopping, --strict-data, --resume.
"""

import argparse
import logging
import os
import time

import torch
import yaml

from data.datasets import get_data_loaders
from main import StemMidiModel

log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stem+MIDI Pro model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="example_data_config.yaml",
        help="Path to data configuration file",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./outputs", help="Directory for logs and checkpoints"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=100, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--gpus", type=int, default=0, help="Deprecated: CPU-only target, this flag is ignored"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=1, help="Gradient accumulation steps (TR-12.4)"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Alias for --resume-from-checkpoint"
    )
    parser.add_argument(
        "--strict-data",
        action="store_true",
        help="Raise an error if the dataset path is missing instead of "
        "falling back to synthetic data",
    )
    return parser.parse_args()


def run_epoch(model, loader, optimizer=None, grad_accum=1, log_every=50, tag="train"):
    """One pass over a loader. Training mode when optimizer is given."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    n_batches = 0
    t0 = time.time()

    ctx = torch.enable_grad() if training else torch.inference_mode()
    with ctx:
        for batch_idx, batch in enumerate(loader):
            outputs = model(
                batch["audio"],
                target_guitar=batch["target_guitar"],
                target_bass=batch["target_bass"],
                target_onsets=batch["target_onsets"],
                target_pitch=batch["target_pitch"],
            )
            loss = outputs["loss"] / grad_accum

            if training:
                loss.backward()
                if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * grad_accum
            n_batches += 1

            if training and log_every and (batch_idx + 1) % log_every == 0:
                log.info(
                    "%s batch %d/%d loss=%.4f",
                    tag,
                    batch_idx + 1,
                    len(loader),
                    total_loss / n_batches,
                )

    mean_loss = total_loss / max(n_batches, 1)
    log.info(
        "%s done: mean_loss=%.4f batches=%d wall=%.1fs", tag, mean_loss, n_batches, time.time() - t0
    )
    return mean_loss


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "epoch": epoch,
            "val_loss": val_loss,
        },
        path,
    )


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.gpus:
        log.warning("--gpus is deprecated: Stem+MIDI Pro is CPU-only (D-STRAT-3); ignoring")

    # Load configurations
    with open(args.config) as f:
        model_config = yaml.safe_load(f)
    with open(args.data_config) as f:
        data_config = yaml.safe_load(f)

    # TR-12.1: cudnn flags BEFORE seeding so deterministic behavior is coherent.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)

    # Initialize model
    log.info("Initializing Stem+MIDI Pro model...")
    model = StemMidiModel(model_config)

    # Setup data loaders
    log.info("Loading datasets...")
    if args.strict_data:
        data_path = (
            data_config.get("dataset", {}).get("path")
            or data_config.get("root_dir")
            or data_config.get("path")
        )
        if not data_path or not os.path.exists(data_path):
            raise FileNotFoundError(
                f"--strict-data: dataset path missing or not found: {data_path!r}"
            )
    train_loader, val_loader, test_loader = get_data_loaders(data_config)
    log.info("Train batches: %d", len(train_loader))
    log.info("Val batches: %d", len(val_loader))
    log.info("Test batches: %d", len(test_loader))

    train_cfg = data_config.get("train", {})
    lr = float(train_cfg.get("lr", 1e-3))
    log_every = int(train_cfg.get("log_every", 50))
    patience = int(train_cfg.get("early_stop_patience", 10))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    best_path = os.path.join(ckpt_dir, "best.pt")
    last_path = os.path.join(ckpt_dir, "last.pt")

    # Resume (TR-12.6): restore model + optimizer + epoch/val_loss from last.ckpt-style file
    start_epoch = 0
    best_val = float("inf")
    resume_ckpt = args.resume_from_checkpoint or args.resume
    if resume_ckpt:
        if not os.path.exists(resume_ckpt):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        if ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val = float(ckpt.get("val_loss", best_val))
        log.info("Resumed from %s (epoch=%d, val_loss=%.4f)", resume_ckpt, start_epoch, best_val)

    # Training loop (TR-12.3)
    log.info(
        "Starting training (max_epochs=%d, grad_accum=%d)...", args.max_epochs, args.grad_accum
    )
    epochs_without_improvement = 0
    for epoch in range(start_epoch, args.max_epochs):
        log.info("Epoch %d/%d", epoch + 1, args.max_epochs)
        run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            grad_accum=args.grad_accum,
            log_every=log_every,
            tag="train",
        )
        val_loss = run_epoch(model, val_loader, tag="val")

        save_checkpoint(model, optimizer, epoch, val_loss, last_path)
        if val_loss < best_val:
            best_val = val_loss
            # TR-12.2: explicit best-checkpoint tracking (no PL magic strings)
            save_checkpoint(model, optimizer, epoch, val_loss, best_path)
            epochs_without_improvement = 0
            log.info("New best val_loss=%.4f -> %s", best_val, best_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                log.info("Early stopping after %d epochs without improvement", patience)
                break

    # Final test with the best checkpoint explicitly loaded (TR-12.2)
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        log.info(
            "Loaded best checkpoint for test (epoch=%d, val_loss=%.4f)",
            ckpt.get("epoch", -1),
            ckpt.get("val_loss", float("nan")),
        )
    log.info("Running final test...")
    test_loss = run_epoch(model, test_loader, tag="test")

    log.info("Training complete! Checkpoints in %s", ckpt_dir)
    log.info("Best val_loss=%.4f  test_loss=%.4f", best_val, test_loss)


if __name__ == "__main__":
    main()
