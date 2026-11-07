# SPDX-License-Identifier: Apache-2.0
"""Path utilities for project-root resolution."""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT: Path | None = None


def get_project_root() -> Path:
    """Return the project root directory (parent of the ``utils/`` package)."""
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        # __file__ is this module (repo_root/utils/paths.py); go up two levels
        # to reach the repo root. Using __main__.__file__ is wrong under
        # uvicorn/gunicorn where __main__ is the server entry point.
        _PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
