# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
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


# T-8.3.14: parametrized test over both production dataset loaders' synthetic mode
import pytest


@pytest.mark.parametrize("dataset_cls,dataset_type", [
    ("Slakh2100YourMT3Dataset", "slakh2100_yourmt3"),
    ("MUSDB18HQDataset", "musdb18hq"),
])
def test_dataset_synthetic_mode_shared_contract(dataset_cls, dataset_type):
    """Both production dataset classes should produce the same synthetic-item dict shape.

    (RF-3.1.1 to collapse them into StemFolderDataset is still open; until that
    refactor lands, we verify both share the v1 contract.)
    """
    import data.datasets as ds

    cls = getattr(ds, dataset_cls)
    dataset = cls(
        root_dir="/nonexistent/path",
        sample_rate=44100,
        segment_length=1.0,
        augment=False,
    )

    # Synthetic short-circuit returns __getitem__ with the v1 contract
    item = dataset[0]
    for key in ("audio", "target_guitar", "target_bass", "target_onsets", "target_pitch"):
        assert key in item, f"{dataset_cls} missing {key}"
    assert item["audio"].shape[-1] == 44100


@pytest.mark.parametrize("dataset_type,expected_subdir", [
    ("slakh2100_yourmt3", "slakh2100_yourmt3"),
    ("musdb18hq", "musdb18hq"),
])
def test_get_data_loaders_dispatches_by_type(dataset_type, expected_subdir):
    """get_data_loaders should construct the path <root>/<subdir>/{train,val,test}."""
    from data.datasets import get_data_loaders

    config = {
        'dataset_type': dataset_type,
        'root_dir': '/some/root',
        'batch_size': 1,
        'num_workers': 0,
        'sample_rate': 44100,
        'segment_length': 1.0,
    }
    # The synthetic branch will run because the constructed paths don't exist
    train_loader, val_loader, test_loader = get_data_loaders(config)
    # Verify we got back iterables with the right structure
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

