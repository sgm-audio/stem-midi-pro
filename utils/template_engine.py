# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""
Simple template engine for user-facing content.
Replaces {{ variable }} placeholders with provided values.
No external dependencies (no jinja2 needed).
"""
from __future__ import annotations

import re
from typing import Dict, Optional
from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "user_content"

def render_template(template_name: str, variables: Optional[Dict[str, str]] = None) -> str:
    """
    Render a template file from user_content/ with the given variables.

    Args:
        template_name: Name of the template file (e.g., 'completion_delivery.md')
        variables: Dict of variable names to values (values are converted to str)

    Returns:
        Rendered string with {{ placeholders }} replaced

    Raises:
        FileNotFoundError: If template doesn't exist
    """
    template_path = TEMPLATE_DIR / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    content = template_path.read_text(encoding='utf-8')
    variables = variables or {}

    def replacer(match):
        key = match.group(1).strip()
        return str(variables.get(key, match.group(0)))

    return re.sub(r'\{\{\s*(\w+)\s*\}\}', replacer, content)

def list_templates() -> list[str]:
    """List all available template files."""
    return sorted([p.name for p in TEMPLATE_DIR.glob("*.md")])

