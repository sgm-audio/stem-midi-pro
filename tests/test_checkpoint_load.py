# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Tests for checkpoint loading (main.load_from_checkpoint + api lifespan wiring)."""
from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import yaml


def test_load_from_checkpoint_accepts_model_state_dict_key(tmp_path):
    """train.py-style checkpoints use model_state_dict key."""
    pytest.importorskip("mamba_ssm")
    from main import StemMidiModel, load_from_checkpoint

    cfg = {
        "name": "ckpt_test",
        "audio": {
            "sample_rate": 44100,
            "n_fft": 2048,
            "hop_length": 512,
            "n_mels": 80,
            "chunk_duration_sec": 2.0,
            "overlap_ratio": 0.5,
        },
        "separator": {
            "d_model": 64,
            "n_layer": 1,
            "d_state": 8,
            "d_conv": 4,
            "expand": 2,
        },
        "transcriber": {
            "d_model": 64,
            "n_layer": 1,
            "d_state": 8,
            "onset_head_dim": 32,
            "pitch_vocab_size": 128,
            "velocity_bins": 128,
            "expression_heads": ["bend"],
            "onset_threshold": 0.5,
        },
        "training": {"precision": "fp32", "gradient_checkpointing": False},
        "loss": {
            "mr_stft_weight": 1.0,
            "spectral_flatness_weight": 0.1,
            "crest_factor_weight": 0.05,
            "onset_f1_weight": 1.0,
            "pitch_ce_weight": 0.8,
            "velocity_mae_weight": 0.3,
            "cross_modal_alignment_weight": 0.5,
        },
        "quality_gates": {
            "studio_confidence_threshold": 0.85,
            "draft_confidence_threshold": 0.70,
            "min_confidence": 0.6,
            "min_si_sdr": 20.0,
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")

    model = StemMidiModel(cfg)
    ckpt_path = tmp_path / "best.pt"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 1}, ckpt_path)

    loaded = load_from_checkpoint(str(ckpt_path), str(cfg_path))
    assert isinstance(loaded, StemMidiModel)
    # Spot-check one parameter matches
    for a, b in zip(model.parameters(), loaded.parameters()):
        assert torch.allclose(a, b)
        break


def test_load_from_checkpoint_accepts_bare_state_dict(tmp_path):
    """Bare state_dict (no wrapper key) must also load."""
    pytest.importorskip("mamba_ssm")
    from main import StemMidiModel, load_from_checkpoint

    cfg = {
        "name": "ckpt_test",
        "audio": {
            "sample_rate": 44100,
            "n_fft": 2048,
            "hop_length": 512,
            "n_mels": 80,
            "chunk_duration_sec": 2.0,
            "overlap_ratio": 0.5,
        },
        "separator": {
            "d_model": 64,
            "n_layer": 1,
            "d_state": 8,
            "d_conv": 4,
            "expand": 2,
        },
        "transcriber": {
            "d_model": 64,
            "n_layer": 1,
            "d_state": 8,
            "onset_head_dim": 32,
            "pitch_vocab_size": 128,
            "velocity_bins": 128,
            "expression_heads": ["bend"],
            "onset_threshold": 0.5,
        },
        "training": {"precision": "fp32", "gradient_checkpointing": False},
        "loss": {
            "mr_stft_weight": 1.0,
            "spectral_flatness_weight": 0.1,
            "crest_factor_weight": 0.05,
            "onset_f1_weight": 1.0,
            "pitch_ce_weight": 0.8,
            "velocity_mae_weight": 0.3,
            "cross_modal_alignment_weight": 0.5,
        },
        "quality_gates": {
            "studio_confidence_threshold": 0.85,
            "draft_confidence_threshold": 0.70,
            "min_confidence": 0.6,
            "min_si_sdr": 20.0,
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")
    model = StemMidiModel(cfg)
    ckpt_path = tmp_path / "bare.pt"
    torch.save(model.state_dict(), ckpt_path)

    loaded = load_from_checkpoint(str(ckpt_path), str(cfg_path))
    assert isinstance(loaded, StemMidiModel)


def test_lifespan_calls_module_load_from_checkpoint(tmp_path, monkeypatch):
    """api lifespan must call main.load_from_checkpoint(path, config_path), not instance method."""
    import api as api_mod

    cfg = {
        "name": "lifespan_test",
        "audio": {"sample_rate": 44100, "n_fft": 2048, "hop_length": 512, "n_mels": 80},
        "separator": {"d_model": 64, "n_layer": 1, "d_state": 8, "d_conv": 4, "expand": 2},
        "transcriber": {
            "d_model": 64,
            "n_layer": 1,
            "d_state": 8,
            "onset_head_dim": 32,
            "pitch_vocab_size": 128,
            "velocity_bins": 128,
            "expression_heads": ["bend"],
            "onset_threshold": 0.5,
        },
        "training": {"precision": "fp32"},
        "loss": {
            "mr_stft_weight": 1.0,
            "spectral_flatness_weight": 0.1,
            "crest_factor_weight": 0.05,
            "onset_f1_weight": 1.0,
            "pitch_ce_weight": 0.8,
            "velocity_mae_weight": 0.3,
            "cross_modal_alignment_weight": 0.5,
        },
        "quality_gates": {
            "studio_confidence_threshold": 0.85,
            "draft_confidence_threshold": 0.70,
            "min_confidence": 0.6,
            "min_si_sdr": 20.0,
        },
    }
    cfg_path = tmp_path / "model_config.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")
    ckpt_path = tmp_path / "weights.pt"
    ckpt_path.write_bytes(b"not-a-real-checkpoint")

    mock_model = MagicMock()
    mock_model.eval = MagicMock(return_value=None)
    mock_model.forward = MagicMock(return_value={})

    load_calls: list[tuple] = []

    def fake_import():
        api_mod.StemMidiModel = MagicMock(return_value=mock_model)

        def _load(path, config_path):
            load_calls.append((path, config_path))
            return mock_model

        api_mod.load_from_checkpoint = _load

    monkeypatch.setenv("MODEL_CONFIG_PATH", str(cfg_path))
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", str(ckpt_path))
    monkeypatch.setattr(api_mod, "_import_model", fake_import)
    monkeypatch.setattr(api_mod, "StemMidiModel", None)
    monkeypatch.setattr(api_mod, "load_from_checkpoint", None)

    # Run lifespan startup only
    async def _run():
        async with api_mod.lifespan(api_mod.app):
            pass

    import asyncio

    asyncio.run(_run())

    assert len(load_calls) == 1
    assert load_calls[0][0] == str(ckpt_path)
    assert load_calls[0][1] == str(cfg_path)
    mock_model.eval.assert_called()
