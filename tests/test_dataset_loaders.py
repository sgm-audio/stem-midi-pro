# SPDX-License-Identifier: Apache-2.0
"""Tests for v1 dataset loaders (data/datasets.py) — T-8.3.14.

Covers the AudioDataset synthetic fallback and both Slakh2100-YourMT3 and
MUSDB18-HQ layouts via parametrize. All tests run in synthetic mode (missing
root_dir) so no real dataset is needed. Heavy imports are lazy so collection
works without torch installed.
"""

import pytest

from tests.honesty import require


def _import_datasets():
    # T-8.1.3 honesty fix (converges with main's removal of the torchaudio gate):
    # this used to gate on torchaudio, which data/datasets.py does not import (it
    # uses librosa + soundfile only). The gate made the whole file skip on every
    # spec-compliant CPU install — a clean `pip install -r requirements.txt` box has
    # no torchaudio, so 6 tests silently did not run. require() skips with a
    # registered reason on convenience runs and ERRORS under --strict-deps, which is
    # what CI uses.
    require("torch", "dataset tensors")
    require("soundfile", "dataset audio reads")
    require("librosa", "dataset resampling")
    try:
        from data import datasets
    except (ImportError, RuntimeError) as exc:
        pytest.fail(f"data.datasets does not import ({exc.__class__.__name__}: {exc})")
    return datasets


def test_synthetic_dataset():
    """AudioDataset should produce synthetic data when root doesn't exist."""
    torch = require("torch", "dataset tensor assertions")
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
    require("torch", "dataset shape assertions")
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
