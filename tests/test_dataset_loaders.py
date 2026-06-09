"""Tests for v1 dataset loaders (data/datasets.py)."""
import pytest
import torch


def test_synthetic_dataset():
    """AudioDataset should produce synthetic data when root doesn't exist."""
    pytest.importorskip("torch")
    import torch
    from data.datasets import AudioDataset

    dataset = AudioDataset(
        root_dir="/nonexistent/path",
        sample_rate=44100,
        segment_length=1.0,
        augment=False,
    )

    item = dataset[0]
    assert 'audio' in item
    assert 'target_guitar' in item
    assert 'target_bass' in item
    assert 'target_onsets' in item
    assert 'target_pitch' in item
    assert isinstance(item['audio'], torch.Tensor)


def test_dataset_getitem_shape():
    """Dataset items should have matching shapes."""
    pytest.importorskip("torch")
    import torch
    from data.datasets import AudioDataset

    dataset = AudioDataset(
        root_dir="/nonexistent/path",
        sample_rate=44100,
        segment_length=1.0,
        augment=False,
    )

    item = dataset[0]
    expected_samples = 44100
    assert item['audio'].shape[-1] == expected_samples
    assert item['target_guitar'].shape[-1] == expected_samples
    assert item['target_bass'].shape[-1] == expected_samples


def test_data_loader_config_keys():
    """get_data_loaders should return correct structure for synthetic type."""
    pytest.importorskip("torch")
    from data.datasets import get_data_loaders

    config = {
        'dataset_type': 'synthetic',
        'root_dir': '/nonexistent/path',
        'batch_size': 2,
        'num_workers': 0,
        'sample_rate': 44100,
        'segment_length': 1.0,
    }

    train_loader, val_loader, test_loader = get_data_loaders(config)

    for loader in [train_loader, val_loader, test_loader]:
        batch = next(iter(loader))
        assert 'audio' in batch
        assert 'target_guitar' in batch
        assert 'target_bass' in batch
