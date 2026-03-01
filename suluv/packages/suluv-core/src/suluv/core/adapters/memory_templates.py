"""InMemoryTemplateEngine — simple string-template document generation."""

from __future__ import annotations

import string
from typing import Any

from suluv.core.ports.template_engine import (
    TemplateEnginePort,
    DocumentTemplate,
    GeneratedDocument,
)


class InMemoryTemplateEngine(TemplateEnginePort):
    """In-memory template engine using Python string.Template."""

    async def render(
        self, template: DocumentTemplate, variables: dict[str, Any]
    ) -> GeneratedDocument:
        content_str = template.template_content
        if not content_str and template.template_path:
            # In production, read from file; in-memory just uses content
            content_str = f"[template from {template.template_path}]"

        try:
            rendered = string.Template(content_str).safe_substitute(variables)
        except Exception:
            rendered = content_str

        content_type = {
            "text": "text/plain",
            "html": "text/html",
            "pdf": "application/pdf",
        }.get(template.output_format, "text/plain")

        return GeneratedDocument(
            name=template.name,
            content=rendered.encode("utf-8"),
            content_type=content_type,
        )

    async def validate_template(self, template: DocumentTemplate) -> list[str]:
        errors = []
        if not template.name:
            errors.append("Template name is required")
        if not template.template_content and not template.template_path:
            errors.append("Template must have content or a path")
        return errors
