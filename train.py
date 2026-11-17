#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Training script for Stem+MIDI Pro.

Plain PyTorch training loop (TR-12.3) — PyTorch Lightning was removed
(DEP-4.1.4); NeMo was removed earlier (RF-3.1.7). CPU-only target.

Features: epoch train/val loop, gradient accumulation (TR-12.4), grad
clipping, best/last checkpointing, early stopping, --strict-data, --resume.
"""

import argparse
import contextlib
import logging
import os
import time

import numpy as np
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
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Explicitly permit training on SYNTHETIC fallback data. Without "
        "this flag, train.py refuses to run when any split is synthetic, so a "
        "sine-wave checkpoint can never be mistaken for a real one (C-5).",
    )
    return parser.parse_args()


def resolve_training_options(model_config: dict, data_config: dict) -> dict:
    """Resolve the training knobs the loop honors (L: config was ignored).

    Reads ``model_config['training']`` (precision, gradient_checkpointing,
    grad_clip, device, val_every, n_steps) with ``data_config['train']`` as a
    legacy fallback for grad_clip/lr-style keys. Unsupported values fail
    loudly instead of silently training differently from the config.
    """
    tr = model_config.get("training", {}) or {}
    dtr = data_config.get("train", {}) or {}

    precision = str(tr.get("precision", "fp32")).lower()
    if precision in ("fp32", "float32", "fp", "none"):
        autocast_dtype = None
    elif precision in ("bf16", "bfloat16"):
        # CPU target: bfloat16 autocast (TR-12.3); fp16 is not supported on CPU.
        autocast_dtype = torch.bfloat16
    else:
        raise ValueError(
            f"training.precision={precision!r} not supported on the CPU target "
            "(use 'fp32' or 'bf16')"
        )

    device_name = str(tr.get("device", "cpu")).lower()
    if device_name.startswith("cuda"):
        log.warning(
            "training.device=%s ignored: Stem+MIDI Pro is CPU-only (D-STRAT-3); using cpu",
            device_name,
        )
        device_name = "cpu"

    grad_clip = float(tr.get("grad_clip", dtr.get("grad_clip", 1.0)))
    val_every = int(tr.get("val_every", dtr.get("val_every", 1)))
    if val_every < 1:
        raise ValueError(f"training.val_every={val_every} must be >= 1")
    n_steps_raw = tr.get("n_steps", dtr.get("n_steps"))
    n_steps = int(n_steps_raw) if n_steps_raw else None
    if n_steps is not None and n_steps < 1:
        raise ValueError(f"training.n_steps={n_steps} must be >= 1")

    return {
        "autocast_dtype": autocast_dtype,
        "precision": precision,
        "device": torch.device(device_name),
        "grad_clip": grad_clip,
        "val_every": val_every,
        "n_steps": n_steps,
    }


def run_epoch(model, loader, optimizer=None, grad_accum=1, log_every=50, tag="train", opts=None):
    """One pass over a loader. Training mode when optimizer is given.

    ``opts`` is the dict returned by :func:`resolve_training_options`; when
    omitted, CPU/fp32/grad_clip=1.0 defaults are used.
    """
    training = optimizer is not None
    model.train() if training else model.eval()
    opts = opts or {}

    device = opts.get("device") or torch.device("cpu")
    autocast_dtype = opts.get("autocast_dtype")
    grad_clip = float(opts.get("grad_clip", 1.0))

    total_loss = 0.0
    n_batches = 0
    n_optim_steps = 0
    t0 = time.time()

    ctx = torch.enable_grad() if training else torch.inference_mode()
    with ctx:
        for batch_idx, batch in enumerate(loader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            if autocast_dtype is not None:
                fwd_ctx = torch.autocast("cpu", dtype=autocast_dtype)
            else:
                fwd_ctx = contextlib.nullcontext()
            with fwd_ctx:
                outputs = model(
                    batch["audio"],
                    target_guitar=batch["target_guitar"],
                    target_bass=batch["target_bass"],
                    target_onsets=batch["target_onsets"],
                    target_pitch=batch["target_pitch"],
                    target_velocity=batch.get("target_velocity"),
                )
            loss = outputs["loss"] / grad_accum

            if training:
                loss.backward()
                if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(loader):
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    n_optim_steps += 1

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
    return mean_loss, n_optim_steps


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

    # TR-12.1: deterministic behavior — seed ALL RNG sources (numpy drives
    # dataset augmentation, random drives crops; torch seeds alone are not
    # enough and cudnn flags are no-ops on the CPU-only target).
    torch.manual_seed(42)
    import random

    random.seed(42)
    np.random.seed(42)

    # Initialize model
    log.info("Initializing Stem+MIDI Pro model...")
    model = StemMidiModel(model_config)
    # Setup data loaders. strict-data: raise on missing split dirs; default:
    # refuse synthetic fallback unless --allow-synthetic (C-5).
    log.info("Loading datasets...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_config,
        strict=args.strict_data,
        allow_synthetic=args.allow_synthetic,
    )
    log.info("Train batches: %d", len(train_loader))
    log.info("Val batches: %d", len(val_loader))
    log.info("Test batches: %d", len(test_loader))
    for name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        ds = loader.dataset
        if getattr(ds, "synthetic", False) or getattr(ds, "items_missing_labels", 0):
            log.warning(
                "Dataset[%s]: synthetic=%s tracks_with_fabricated_labels=%d — "
                "checkpoints from this run are NOT trained on real music.",
                name,
                getattr(ds, "synthetic", "?"),
                getattr(ds, "items_missing_labels", 0),
            )

    train_cfg = data_config.get("train", {})
    lr = float(train_cfg.get("lr", 1e-3))
    log_every = int(train_cfg.get("log_every", 50))
    patience = int(train_cfg.get("early_stop_patience", 10))

    # Honor the config's training knobs (precision/device/grad_clip/val_every/
    # n_steps) instead of hardcoding CPU/fp32/1.0 (L finding).
    opts = resolve_training_options(model_config, data_config)
    model.to(opts["device"])
    log.info(
        "Training options: device=%s precision=%s grad_clip=%s val_every=%d n_steps=%s",
        opts["device"],
        opts["precision"],
        opts["grad_clip"],
        opts["val_every"],
        opts["n_steps"],
    )

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
    optimizer_steps = 0
    last_val = float("nan")
    for epoch in range(start_epoch, args.max_epochs):
        log.info("Epoch %d/%d", epoch + 1, args.max_epochs)
        _, steps = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            grad_accum=args.grad_accum,
            log_every=log_every,
            tag="train",
            opts=opts,
        )
        optimizer_steps += steps

        is_val_epoch = (epoch - start_epoch) % opts["val_every"] == opts["val_every"] - 1
        if is_val_epoch:
            last_val, _ = run_epoch(model, val_loader, tag="val", opts=opts)
            val_loss = last_val
        else:
            val_loss = last_val

        save_checkpoint(model, optimizer, epoch, val_loss, last_path)
        if is_val_epoch:
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

        if opts["n_steps"] is not None and optimizer_steps >= opts["n_steps"]:
            log.info(
                "Stopping: reached n_steps=%d optimizer steps (epoch %d)",
                opts["n_steps"],
                epoch + 1,
            )
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
    test_loss, _ = run_epoch(model, test_loader, tag="test", opts=opts)

    log.info("Training complete! Checkpoints in %s", ckpt_dir)
    log.info("Best val_loss=%.4f  test_loss=%.4f", best_val, test_loss)


if __name__ == "__main__":
    main()
