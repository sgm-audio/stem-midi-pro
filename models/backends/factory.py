# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Factory for separator / transcriber backends."""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

_SEPARATORS = {
    "mamba": "models.backends.mamba_backend:MambaSeparatorBackend",
    "demucs": "models.backends.demucs_separator:DemucsSeparatorBackend",
    "fake": "models.backends.fake_backend:FakeSeparatorBackend",
}

_TRANSCRIBERS = {
    "mamba": "models.backends.mamba_backend:MambaTranscriberBackend",
    "basic_pitch": "models.backends.basic_pitch_transcriber:BasicPitchTranscriberBackend",
    "fake": "models.backends.fake_backend:FakeTranscriberBackend",
}


def _load_class(path: str):
    module_path, cls_name = path.split(":")
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def _backend_names(cfg: dict) -> tuple[str, str]:
    backends = cfg.get("backends") or {}
    # Default: mamba (legacy). Production configs should set demucs + basic_pitch.
    sep = str(backends.get("separator", "mamba")).lower()
    tr = str(backends.get("transcriber", "mamba")).lower()
    return sep, tr


def build_separator(cfg: dict) -> Any:
    name, _ = _backend_names(cfg)
    if name not in _SEPARATORS:
        raise ValueError(
            f"Unknown separator backend {name!r}. Choose from: {sorted(_SEPARATORS)}"
        )
    log.info("Building separator backend: %s", name)
    return _load_class(_SEPARATORS[name])(cfg)


def build_transcriber(cfg: dict) -> Any:
    _, name = _backend_names(cfg)
    if name not in _TRANSCRIBERS:
        raise ValueError(
            f"Unknown transcriber backend {name!r}. Choose from: {sorted(_TRANSCRIBERS)}"
        )
    log.info("Building transcriber backend: %s", name)
    return _load_class(_TRANSCRIBERS[name])(cfg)
