"""Tests for Mamba-3 dataset loaders (research only)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_stem_dataset_synthetic_fallback():
    """StemDataset should return synthetic fallback when root doesn't exist."""
    from mamba3_datasets import StemDataset

    cfg = type('MockCfg', (), {'sample_rate': 44100, 'hop_length': 512})()
    dataset = StemDataset(
        root="/nonexistent/path", cfg=cfg,
        clip_seconds=1.0, augment=False,
    )

    item = dataset[0]
    assert 'target' in item
    assert 'other_stems' in item
    assert 'target_crepe' in item
    assert 'other_crepe' in item
    assert 'stem_name' in item
    assert 'valid' in item
    assert item['valid'] == False


def test_stem_dataset_output_shapes():
    """StemDataset should produce consistent tensor shapes."""
    from mamba3_datasets import StemDataset

    cfg = type('MockCfg', (), {'sample_rate': 44100, 'hop_length': 512})()
    dataset = StemDataset(
        root="/nonexistent/path", cfg=cfg,
        clip_seconds=2.0, augment=False,
    )

    item = dataset[0]
    expected_samples = 88200  # 2s @ 44100
    T_clip = 1 + expected_samples // 512
    n_stems = len(dataset.stems) - 1

    assert item['target'].shape[0] == expected_samples
    assert item['other_stems'].shape[0] == n_stems
    assert item['other_stems'].shape[1] == expected_samples
    assert item['target_crepe'].shape == (T_clip, 360)
    assert item['other_crepe'].shape == (n_stems, T_clip, 360)


def test_slakh2100_stem_dataset():
    """Slakh2100StemDataset should fall back to synthetic when no data."""
    from mamba3_datasets import Slakh2100StemDataset

    cfg = type('MockCfg', (), {'sample_rate': 44100, 'hop_length': 512})()
    dataset = Slakh2100StemDataset(
        root="/nonexistent/path", cfg=cfg,
        clip_seconds=2.0, augment=False,
    )

    item = dataset[0]
    assert 'target' in item
    assert 'stem_name' in item


def test_collate_valid():
    """collate_valid should filter invalid items and stack valid ones."""
    from mamba3_datasets import collate_valid

    valid_item = {
        'target': torch.zeros(88200),
        'other_stems': torch.zeros(4, 88200),
        'target_crepe': torch.zeros(173, 360),
        'other_crepe': torch.zeros(4, 173, 360),
        'target_pitched': 0.5,
        'other_pitched': torch.zeros(4),
        'stem_name': 'bass',
        'valid': True,
    }
    invalid_item = {**valid_item, 'valid': False}

    batch = [valid_item, invalid_item, valid_item]
    result = collate_valid(batch)

    assert result is not None
    assert result['target'].shape[0] == 2


def test_collate_valid_all_invalid():
    """collate_valid should return None when all items are invalid."""
    from mamba3_datasets import collate_valid

    invalid = {
        'target': torch.zeros(88200), 'other_stems': torch.zeros(4, 88200),
        'target_crepe': torch.zeros(173, 360), 'other_crepe': torch.zeros(4, 173, 360),
        'target_pitched': 0.0, 'other_pitched': torch.zeros(4),
        'stem_name': 'none', 'valid': False,
    }

    result = collate_valid([invalid, invalid])
    assert result is None
