"""Calculator tool — safe math expression evaluator."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from suluv.core.tools.decorator import suluv_tool

# Safe operators whitelist
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Safe math functions
_SAFE_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "int": int,
    "float": float,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
    "ceil": math.ceil,
    "floor": math.floor,
    "pow": math.pow,
}


def _safe_eval(node: ast.AST) -> Any:
    """Recursively evaluate an AST node with only safe operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    elif isinstance(node, ast.BinOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op(_safe_eval(node.operand))
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCS:
            func = _SAFE_FUNCS[node.func.id]
            args = [_safe_eval(a) for a in node.args]
            if callable(func):
                return func(*args)
            return func  # constants like pi, e
        raise ValueError(f"Unsupported function: {ast.dump(node.func)}")
    elif isinstance(node, ast.Name):
        if node.id in _SAFE_FUNCS:
            val = _SAFE_FUNCS[node.id]
            if not callable(val):
                return val  # pi, e
        raise ValueError(f"Unknown name: {node.id}")
    else:
        raise ValueError(f"Unsupported expression: {type(node).__name__}")


@suluv_tool(
    name="calculator",
    description=(
        "Evaluate a mathematical expression safely. Supports: +, -, *, /, //, %, ** "
        "and functions: sqrt, log, sin, cos, tan, abs, round, min, max, ceil, floor. "
        "Constants: pi, e. Example: 'sqrt(144) + 2 ** 3'"
    ),
)
def calculator(expression: str) -> str:
    """Evaluate a math expression safely without exec/eval."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        return str(result)
    except Exception as e:
        return f"Error: {e}"
