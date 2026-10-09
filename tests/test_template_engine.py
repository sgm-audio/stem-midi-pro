"""Tests for safe user-content template rendering."""

import pytest

from utils.template_engine import list_templates, render_template


def test_render_template_replaces_supplied_values():
    """Supplied variables should replace matching placeholders."""
    rendered = render_template(
        "completion_delivery.md",
        {"filename": "demo.wav", "avg_confidence": "0.8"},
    )

    assert "demo.wav" in rendered
    assert "0.8" in rendered


def test_render_template_leaves_missing_values_visible():
    """Unprovided template values should remain visible for the caller."""
    rendered = render_template("completion_delivery.md", {})

    assert "{{filename}}" in rendered


@pytest.mark.parametrize(
    "template_name",
    ["../README.md", "/etc/passwd", "nested/template.md"],
)
def test_render_template_rejects_paths_outside_template_directory(template_name):
    """Template paths must be basenames within user_content/."""
    with pytest.raises(FileNotFoundError):
        render_template(template_name)


def test_render_template_missing_file_raises_file_not_found():
    """An unknown template should preserve FileNotFoundError for the caller."""
    with pytest.raises(FileNotFoundError):
        render_template("does-not-exist.md")


def test_user_content_templates_do_not_repeat_unverified_product_claims():
    """Draft product copy must not promise unsupported service capabilities."""
    templates = list_templates()
    assert len(templates) == 7

    content = "\n".join(render_template(name) for name in templates).casefold()
    unsupported_claims = (
        "studio-grade",
        "processed on nvidia h100s",
        "processed in-memory only",
        "deleted after 24h",
        "web-based editor",
        "human-reviewed refinement available",
        "reaper/logic template",
        "[view live midi preview]",
        "[request human review]",
    )
    for claim in unsupported_claims:
        assert claim not in content
