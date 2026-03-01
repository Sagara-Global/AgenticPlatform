"""Process engine — full Business Process Management system."""

from suluv.core.process.definition import ProcessDefinition
from suluv.core.process.version import ProcessVersion, ProcessVersionRegistry
from suluv.core.process.variables import ProcessVariables, VariableScope
from suluv.core.process.stage import ProcessStage
from suluv.core.process.step import ProcessStep
from suluv.core.process.instance import ProcessInstanceManager

__all__ = [
    "ProcessDefinition",
    "ProcessVersion",
    "ProcessVersionRegistry",
    "ProcessVariables",
    "VariableScope",
    "ProcessStage",
    "ProcessStep",
    "ProcessInstanceManager",
]
