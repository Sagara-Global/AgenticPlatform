"""@suluv_tool decorator — turns a function into a SuluvTool."""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


@dataclass
class ToolSchema:
    """JSON Schema-like description of a tool for LLM function-calling."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class SuluvTool:
    """A tool that an agent can invoke.

    Wraps a sync or async function with schema metadata, input
    validation, and timeout support.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> None:
        self.fn = fn
        self.name = name or fn.__name__
        self.description = description or (fn.__doc__ or "").strip()
        self.timeout = timeout
        self._is_async = asyncio.iscoroutinefunction(fn)

        # Auto-generate parameters from type hints
        self.parameters = parameters or self._infer_parameters(fn)
        self.schema = ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    async def execute(self, **kwargs: Any) -> Any:
        """Run the tool function with optional timeout."""
        if self._is_async:
            coro = self.fn(**kwargs)
        else:
            loop = asyncio.get_event_loop()
            coro = loop.run_in_executor(None, lambda: self.fn(**kwargs))

        if self.timeout:
            return await asyncio.wait_for(coro, timeout=self.timeout)
        return await coro

    @staticmethod
    def _infer_parameters(fn: Callable[..., Any]) -> dict[str, Any]:
        """Infer JSON Schema parameters from function signature."""
        sig = inspect.signature(fn)
        hints = fn.__annotations__ if hasattr(fn, "__annotations__") else {}
        properties: dict[str, Any] = {}
        required: list[str] = []

        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        for name, param in sig.parameters.items():
            if name in ("self", "cls"):
                continue
            hint = hints.get(name, Any)
            prop: dict[str, Any] = {
                "type": type_map.get(hint, "string")
            }
            if param.default is inspect.Parameter.empty:
                required.append(name)
            else:
                prop["default"] = param.default
            properties[name] = prop

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.schema.to_dict()

    def __repr__(self) -> str:
        return f"SuluvTool(name={self.name!r})"


def suluv_tool(
    name: str | None = None,
    description: str | None = None,
    timeout: float | None = None,
    parameters: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], SuluvTool]:
    """Decorator to turn a function into a SuluvTool.

    Usage::

        @suluv_tool(name="search", description="Search the web")
        async def search(query: str) -> str:
            ...
    """

    def decorator(fn: Callable[..., Any]) -> SuluvTool:
        return SuluvTool(
            fn=fn,
            name=name,
            description=description,
            timeout=timeout,
            parameters=parameters,
        )

    return decorator
