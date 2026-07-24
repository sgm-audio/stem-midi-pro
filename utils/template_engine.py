"""
Simple template engine for user-facing content.
Replaces {{ variable }} placeholders with provided values.
No external dependencies (no jinja2 needed).
"""
import re
from typing import Dict, Optional
from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "user_content"


def _safe_template_path(template_name: str) -> Path:
    """Resolve template_name under TEMPLATE_DIR; reject path traversal."""
    if not template_name or not isinstance(template_name, str):
        raise FileNotFoundError("Template not found")
    # Basename-only: reject ../, absolute paths, nested separators
    if template_name != Path(template_name).name:
        raise FileNotFoundError(f"Template not found: {template_name}")
    if ".." in template_name or "/" in template_name or "\\" in template_name:
        raise FileNotFoundError(f"Template not found: {template_name}")

    root = TEMPLATE_DIR.resolve()
    template_path = (root / template_name).resolve()
    try:
        template_path.relative_to(root)
    except ValueError as exc:
        raise FileNotFoundError(f"Template not found: {template_name}") from exc
    if not template_path.is_file():
        raise FileNotFoundError(f"Template not found: {template_path}")
    return template_path


def render_template(template_name: str, variables: Optional[Dict[str, str]] = None) -> str:
    """
    Render a template file from user_content/ with the given variables.

    Args:
        template_name: Name of the template file (e.g., 'completion_delivery.md')
        variables: Dict of variable names to values (values are converted to str)

    Returns:
        Rendered string with {{ placeholders }} replaced

    Raises:
        FileNotFoundError: If template doesn't exist or path escapes TEMPLATE_DIR
    """
    template_path = _safe_template_path(template_name)

    content = template_path.read_text(encoding='utf-8')
    variables = variables or {}

    def replacer(match):
        key = match.group(1).strip()
        return str(variables.get(key, match.group(0)))

    return re.sub(r'\{\{\s*(\w+)\s*\}\}', replacer, content)

def list_templates() -> list[str]:
    """List all available template files."""
    return sorted([p.name for p in TEMPLATE_DIR.glob("*.md")])
