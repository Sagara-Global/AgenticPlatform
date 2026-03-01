"""TemplateEnginePort — document generation from templates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentTemplate:
    """A template for generating documents."""

    name: str
    template_content: str = ""
    template_path: str | None = None
    output_format: str = "text"  # "text", "html", "pdf"
    merge_fields: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedDocument:
    """A document produced by template rendering."""

    name: str
    content: bytes
    content_type: str = "text/plain"
    metadata: dict[str, Any] = field(default_factory=dict)


class TemplateEnginePort(ABC):
    """Port for rendering templates into documents."""

    @abstractmethod
    async def render(
        self, template: DocumentTemplate, variables: dict[str, Any]
    ) -> GeneratedDocument:
        """Render a template with variables to produce a document."""
        ...

    @abstractmethod
    async def validate_template(self, template: DocumentTemplate) -> list[str]:
        """Validate a template. Returns list of errors (empty = valid)."""
        ...
