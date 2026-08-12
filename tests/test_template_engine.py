# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Tests for utils.template_engine."""
from __future__ import annotations

import pytest

from utils.template_engine import list_templates, render_template


def test_list_templates_returns_seven():
    """list_templates should return the 7 user-facing content templates."""
    templates = list_templates()
    expected = {
        "upload_confirmation.md",
        "progress_updates.md",
        "completion_delivery.md",
        "rights_usage_prompt.md",
        "feedback_refinement.md",
        "implicit_feedback.md",
        "landing_page.md",
    }
    assert set(templates) == expected
    assert len(templates) == 7


def test_missing_key_leaves_placeholder_intact():
    """When a variable is missing, the {{ placeholder }} should remain in the output."""
    rendered = render_template("completion_delivery.md", {"filename": "test.wav"})
    assert "{{" in rendered, "Unresolved placeholder should remain untouched"
    # filename substitution should still happen
    assert "test.wav" in rendered


def test_render_substitutes_known_keys():
    """Known keys should be replaced in the rendered template."""
    rendered = render_template(
        "completion_delivery.md",
        {"filename": "my_song.wav", "avg_confidence": "0.92"},
    )
    assert "my_song.wav" in rendered
    assert "0.92" in rendered


def test_render_unknown_template_raises_file_not_found():
    """Unknown template name should re-raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        render_template("does_not_exist.md", {})


def test_render_with_none_variables_uses_empty():
    """Passing variables=None should not crash; placeholders remain."""
    rendered = render_template("landing_page.md", None)
    assert isinstance(rendered, str)
    assert len(rendered) > 0


def test_render_all_templates_have_valid_syntax():
    """Every template in the dir should render without raising."""
    for template_name in list_templates():
        rendered = render_template(template_name, {})
        assert isinstance(rendered, str)
        assert len(rendered) > 0
