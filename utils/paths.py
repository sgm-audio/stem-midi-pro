# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Path utilities for project-root resolution."""
from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT: Path | None = None


def get_project_root() -> Path:
    """Return the project root directory (parent of the file calling this)."""
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        import __main__
        if hasattr(__main__, "__file__"):
            _PROJECT_ROOT = Path(__main__.__file__).resolve().parent
        else:
            _PROJECT_ROOT = Path.cwd()
    return _PROJECT_ROOT


def config_path(filename: str = "model_config.yaml") -> Path:
    """Return path to config directory."""
    return get_project_root() / "configs" / filename


def user_content_path(filename: str = "") -> Path:
    """Return path to user_content directory or a specific template."""
    p = get_project_root() / "user_content"
    return p / filename if filename else p


def outputs_path(subpath: str = "") -> Path:
    """Return path to outputs directory."""
    p = get_project_root() / "outputs"
    if subpath:
        p = p / subpath
    p.mkdir(parents=True, exist_ok=True)
    return p
