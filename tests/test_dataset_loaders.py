"""Tests for v1 dataset loaders (data/datasets.py)."""

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("librosa")
pytest.importorskip("soundfile")


def test_synthetic_dataset():
    """AudioDataset should produce synthetic data when root doesn't exist."""
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


def test_default_instrument_lists_are_not_shared():
    """Mutating one dataset's default instruments should not affect another."""
    from data.datasets import AudioDataset

    first = AudioDataset(
        root_dir="/nonexistent/path",
        sample_rate=44100,
        segment_length=1.0,
        augment=False,
    )
    second = AudioDataset(
        root_dir="/nonexistent/path",
        sample_rate=44100,
        segment_length=1.0,
        augment=False,
    )

    first.instruments.append("drums")

    assert second.instruments == ["guitar", "bass"]


def test_data_loader_config_keys():
    """get_data_loaders should return correct structure for synthetic type."""
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


def test_unknown_dataset_type_fails_instead_of_using_synthetic():
    """A misspelled dataset type must not silently train on synthetic audio."""
    from data.datasets import get_data_loaders

    with pytest.raises(ValueError, match="Unsupported dataset_type"):
        get_data_loaders({"dataset_type": "synthetik", "root_dir": "/unused"})


def test_synthetic_dataset_type_works_with_existing_root(tmp_path):
    """Explicit synthetic selection should not scan an existing directory."""
    from data.datasets import get_data_loaders

    (tmp_path / "not-a-track.txt").write_text("fixture", encoding="utf-8")
    config = {
        'dataset_type': 'synthetic',
        'root_dir': str(tmp_path),
        'batch_size': 1,
        'num_workers': 0,
        'sample_rate': 44100,
        'segment_length': 1.0,
    }

    loaders = get_data_loaders(config)

    assert all(loader.dataset.synthetic for loader in loaders)
