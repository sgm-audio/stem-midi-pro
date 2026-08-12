# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Model components for Stem+MIDI Pro.

Submodules imported lazily via __getattr__ so that test collection and
non-model pathways (e.g. utils, data) do not require the mamba_ssm
CUDA package at import time. Direct `from models.mamba_separator import ...`
still works eagerly when needed.
"""

__all__ = ["MambaSeparator", "MambaTranscriber", "ConfidenceInjector", "PerceptualAudioLoss"]

_LAZY = {
    "MambaSeparator": "models.mamba_separator",
    "MambaTranscriber": "models.mamba_transcriber",
    "ConfidenceInjector": "models.confidence_injector",
    "PerceptualAudioLoss": "models.losses",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name])
        cls = getattr(module, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module 'models' has no attribute {name!r}")
