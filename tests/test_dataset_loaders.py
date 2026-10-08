# SPDX-License-Identifier: Apache-2.0
"""Tests for v1 dataset loaders (data/datasets.py) — T-8.3.14.

Covers the AudioDataset synthetic fallback and both Slakh2100-YourMT3 and
MUSDB18-HQ layouts via parametrize. All tests run in synthetic mode (missing
root_dir) so no real dataset is needed. Heavy imports are lazy so collection
works without torch installed.
"""

import pytest


def _import_datasets():
    pytest.importorskip("torch")
    pytest.importorskip("soundfile")
    pytest.importorskip("librosa")
    pytest.importorskip("torchaudio")
    try:
        from data import datasets
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"Cannot import data.datasets: {exc}")
    return datasets


def test_synthetic_dataset():
    """AudioDataset should produce synthetic data when root doesn't exist."""
    torch = pytest.importorskip("torch")
    datasets = _import_datasets()

    dataset = datasets.AudioDataset(
        root_dir="/nonexistent/path",
        sample_rate=44100,
        segment_length=1.0,
        augment=False,
    )

    item = dataset[0]
    assert "audio" in item
    assert "target_guitar" in item
    assert "target_bass" in item
    assert "target_onsets" in item
    assert "target_pitch" in item
    assert isinstance(item["audio"], torch.Tensor)
    assert dataset.synthetic is True


def test_dataset_getitem_shape():
    """Dataset items should have matching shapes."""
    pytest.importorskip("torch")
    datasets = _import_datasets()

    dataset = datasets.AudioDataset(
        root_dir="/nonexistent/path",
        sample_rate=44100,
        segment_length=1.0,
        augment=False,
    )

    item = dataset[0]
    expected_samples = 44100
    assert item["audio"].shape[-1] == expected_samples
    assert item["target_guitar"].shape[-1] == expected_samples
    assert item["target_bass"].shape[-1] == expected_samples


def test_data_loader_config_keys():
    """get_data_loaders should return correct structure for synthetic type."""
    datasets = _import_datasets()

    config = {
        "dataset_type": "synthetic",
        "root_dir": "/nonexistent/path",
        "batch_size": 2,
        "num_workers": 0,
        "sample_rate": 44100,
        "segment_length": 1.0,
    }

    train_loader, val_loader, test_loader = datasets.get_data_loaders(config)

    for loader in [train_loader, val_loader, test_loader]:
        batch = next(iter(loader))
        assert "audio" in batch
        assert "target_guitar" in batch
        assert "target_bass" in batch


# ---------------------------------------------------------------------------
# T-8.3.14: both dataset layouts via parametrize (synthetic mode)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls_name", ["Slakh2100YourMT3Dataset", "MUSDB18HQDataset"])
def test_layout_falls_back_to_synthetic(cls_name):
    """Both Slakh and MUSDB loaders fall back to synthetic data when the
    root directory is missing — without raising.

    # Slakh path expects fix per TODO C-2.3 / C-2.4 (file_list overwrite and
    # non-synthetic listdir on a missing root).
    """
    datasets = _import_datasets()
    cls = getattr(datasets, cls_name, None)
    if cls is None:
        pytest.skip(f"{cls_name} not yet in data.datasets (RF-3.1 pending)")
    dataset = cls(
        root_dir="/nonexistent/path/for/tests",
        sample_rate=44100,
        segment_length=1.0,
        augment=False,
    )
    assert dataset.synthetic is True
    item = dataset[0]
    assert "audio" in item


def test_synthetic_item_keys_and_lengths():
    """Synthetic items contain the full contract for the training loop."""
    datasets = _import_datasets()

    dataset = datasets.AudioDataset(
        root_dir="/nonexistent/path",
        sample_rate=44100,
        segment_length=0.5,
        augment=False,
    )
    item = dataset[1]
    assert len(dataset) == len(dataset.file_list)
    assert item["audio"].shape[-1] == int(44100 * 0.5)
