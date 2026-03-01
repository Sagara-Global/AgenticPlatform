"""
Suluv Core — Agentic Business Process Framework.

3 levels of complexity:
  Level 1: SuluvAgent       — standalone agent + LLM + tools
  Level 2: GraphRuntime     — multi-agent orchestration with nodes + edges
  Level 3: ProcessDefinition — business workflows that compile to graphs
"""

from suluv.core.types import (
    SuluvID,
    NodeID,
    EdgeID,
    ExecutionID,
    SessionID,
    OrgID,
    UserID,
    ProcessID,
    InstanceID,
    NodeState,
    ErrorPolicy,
)

__all__ = [
    "SuluvID",
    "NodeID",
    "EdgeID",
    "ExecutionID",
    "SessionID",
    "OrgID",
    "UserID",
    "ProcessID",
    "InstanceID",
    "NodeState",
    "ErrorPolicy",
]
