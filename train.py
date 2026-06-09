#!/usr/bin/env python3
"""
Training script for Stem+MIDI Pro.
"""

import argparse
import logging
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from main import StemMidiModel
from data.datasets import get_data_loaders

log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train Stem+MIDI Pro model')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                        help='Path to model configuration file')
    parser.add_argument('--data-config', type=str, default='example_data_config.yaml',
                        help='Path to data configuration file')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Directory for logs and checkpoints')
    parser.add_argument('--max-epochs', type=int, default=100,
                        help='Maximum number of training epochs')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use (0 for CPU)')
    parser.add_argument('--resume-from-checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    args = parser.parse_args()
    
    # Load configurations
    with open(args.config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    with open(args.data_config, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Set random seeds for reproducibility
    pl.seed_everything(42)
    
    # Initialize model
    log.info("Initializing Stem+MIDI Pro model...")
    model = StemMidiModel(model_config)
    
    # Setup data loaders
    log.info("Loading datasets...")
    train_loader, val_loader, test_loader = get_data_loaders(data_config)
    log.info(f"Train batches: {len(train_loader)}")
    log.info(f"Val batches: {len(val_loader)}")
    log.info(f"Test batches: {len(test_loader)}")
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.output_dir}/checkpoints",
        filename='stem-midi-pro-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Setup logger
    tb_logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name='stem-midi-pro',
        version=None
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=tb_logger,
        log_every_n_steps=10,
        val_check_interval=0.5,  # Validate twice per epoch
        precision=16 if model_config.get('training', {}).get('precision') == 'fp8' else 32,
        accelerator='gpu' if args.gpus > 0 and torch.cuda.is_available() else 'cpu',
        devices=args.gpus if args.gpus > 0 else 1,
        deterministic=True
    )
    
    # Start training
    log.info("Starting training...")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from_checkpoint
    )
    
    # Test after training
    log.info("Running final test...")
    trainer.test(model, dataloaders=test_loader, ckpt_path='best')
    
    log.info(f"Training complete! Checkpoints saved to {args.output_dir}/checkpoints")
    log.info(f"Logs saved to {args.output_dir}/stem-midi-pro/*")

if __name__ == '__main__':
    main()