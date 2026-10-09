"""Tests for API model loading configuration failures."""

import pytest

from api import load_model


def test_missing_explicit_checkpoint_fails_before_model_import(monkeypatch):
    """A configured but missing checkpoint must not silently use random weights."""
    monkeypatch.setenv("MODEL_CONFIG_PATH", "configs/model_config.yaml")
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", "/definitely/missing/model.nemo")

    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        load_model()
