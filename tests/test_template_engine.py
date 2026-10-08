# SPDX-License-Identifier: Apache-2.0
"""Template engine tests (T-8.3.5)."""

import re
from pathlib import Path

import pytest

from utils.template_engine import list_templates, render_template


def test_missing_key_leaves_placeholder():
    """Unprovided variables must remain as {{ var }} in the rendered output."""
    for name in list_templates():
        src = (Path("user_content") / name).read_text(encoding="utf-8")
        rendered = render_template(name, {})
        src_placeholders = set(re.findall(r"\{\{\s*\w+\s*\}\}", src))
        rendered_placeholders = set(re.findall(r"\{\{\s*\w+\s*\}\}", rendered))
        # Rendering with an empty variables dict is a no-op on placeholders.
        assert rendered_placeholders == src_placeholders, f"{name}"
        if not src_placeholders:
            assert rendered == src


def test_known_key_is_replaced():
    """Provided variable must substitute its placeholder."""
    for name in list_templates():
        src = (Path("user_content") / name).read_text(encoding="utf-8")
        keys = set(re.findall(r"\{\{\s*(\w+)\s*\}\}", src))
        if not keys:
            continue
        variables = {k: f"__{k}__" for k in keys}
        rendered = render_template(name, variables)
        assert "{{" not in rendered, f"{name}: leftover placeholder"
        assert f"__{next(iter(keys))}__" in rendered


def test_list_templates_returns_seven():
    """user_content/ ships exactly 7 templates."""
    from pathlib import Path

    templates = list_templates()
    assert isinstance(templates, list)
    assert len(templates) == 7
    assert templates == sorted(templates)
    for name in templates:
        assert name.endswith(".md")
        assert name == Path(name).name  # name only, no path parts


def test_missing_template_raises():
    """A non-existent template must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        render_template("does_not_exist_12345.md", {})


def test_path_traversal_rejected():
    """Path traversal must raise FileNotFoundError (security)."""
    with pytest.raises(FileNotFoundError):
        render_template("../TODO.md", {})
    with pytest.raises(FileNotFoundError):
        render_template("..\\TODO.md", {})
