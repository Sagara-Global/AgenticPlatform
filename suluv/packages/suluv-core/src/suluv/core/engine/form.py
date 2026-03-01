"""FormNode — captures structured user input via a form schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from suluv.core.types import NodeType
from suluv.core.engine.node import GraphNode, NodeInput, NodeOutput


@dataclass
class FormField:
    """Single field in a form schema."""

    name: str
    field_type: str = "text"  # text | number | select | checkbox | date | file
    label: str = ""
    required: bool = False
    options: list[str] = field(default_factory=list)  # for select
    default: Any = None
    validation: str | None = None  # regex or expression


@dataclass
class FormSchema:
    """Defines the structure of a form."""

    title: str
    fields: list[FormField]
    submit_label: str = "Submit"
    description: str = ""


class FormNode(GraphNode):
    """Present a form to a user and collect validated input.

    Emits a human task with form metadata.  The task queue consumer
    renders the form and submits the response.  Validates required
    fields before returning output.
    """

    def __init__(
        self,
        schema: FormSchema,
        assignee: str | None = None,
        node_id: str | None = None,
        name: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            node_id=node_id, node_type=NodeType.FORM, name=name, **kwargs
        )
        self._schema = schema
        self._assignee = assignee

    @property
    def schema(self) -> FormSchema:
        return self._schema

    async def execute(self, input: NodeInput) -> NodeOutput:
        # If data is already provided (form already filled), validate and return
        if isinstance(input.data, dict) and input.data.get("_form_response"):
            response = input.data["_form_response"]
            errors = self._validate(response)
            if errors:
                return NodeOutput(data={"errors": errors}, success=False)
            return NodeOutput(data=response, success=True)

        task_queue = (input.context or {}).get("task_queue")
        if task_queue is None:
            return NodeOutput(
                data=None,
                success=False,
                error="No task_queue in context for form submission",
            )

        # Emit a form task
        from suluv.core.ports.human_task_queue import HumanTask, TaskStatus

        task = HumanTask(
            task_id="",  # adapter fills it
            title=self._schema.title,
            description=self._schema.description,
            assignee=self._assignee,
            payload={
                "type": "form",
                "schema": {
                    "title": self._schema.title,
                    "fields": [
                        {
                            "name": f.name,
                            "field_type": f.field_type,
                            "label": f.label or f.name,
                            "required": f.required,
                            "options": f.options,
                            "default": f.default,
                            "validation": f.validation,
                        }
                        for f in self._schema.fields
                    ],
                    "submit_label": self._schema.submit_label,
                },
            },
            status=TaskStatus.PENDING,
        )
        await task_queue.emit(task)

        return NodeOutput(
            data=None,
            success=True,
            metadata={"waiting": True, "form": self._schema.title},
        )

    def _validate(self, response: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        for f in self._schema.fields:
            val = response.get(f.name)
            if f.required and (val is None or val == ""):
                errors.append(f"Field '{f.name}' is required")
        return errors
