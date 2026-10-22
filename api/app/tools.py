"""Tool registry. Add a tool with @tool(name, description, json_schema)."""
import ast
import json
import operator
from collections.abc import Callable
from datetime import UTC, datetime

TOOLS: dict[str, tuple[Callable[..., object], dict]] = {}


def tool(name: str, description: str, parameters: dict):
    def deco(fn):
        TOOLS[name] = (fn, {
            "type": "function",
            "function": {"name": name, "description": description, "parameters": parameters},
        })
        return fn
    return deco


def specs() -> list[dict]:
    return [spec for _, spec in TOOLS.values()]


class ToolCallError(ValueError):
    """The model's tool call is malformed. The message tells the model how to fix it."""


def validate_call(name: str, raw_args: str | None) -> dict:
    """Check a tool call against the registry and its JSON schema; return the parsed args."""
    if name not in TOOLS:
        raise ToolCallError(f"unknown tool '{name}'. Available tools: {', '.join(TOOLS)}")
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError as e:
        raise ToolCallError(f"arguments are not valid JSON ({e}). Send a JSON object.") from e
    if not isinstance(args, dict):
        raise ToolCallError("arguments must be a JSON object")
    schema = TOOLS[name][1]["function"]["parameters"]
    missing = [k for k in schema.get("required", []) if k not in args]
    unknown = [k for k in args if k not in schema.get("properties", {})]
    if missing or unknown:
        raise ToolCallError(f"bad arguments for '{name}': missing {missing}, unknown {unknown}. "
                            f"Expected schema: {json.dumps(schema)}")
    return args


def run_tool(name: str, raw_args: str | None) -> tuple[str, bool]:
    """Run a tool; returns (JSON result, ok). Errors go back to the model as data so it can repair."""
    try:
        args = validate_call(name, raw_args)
        return json.dumps(TOOLS[name][0](**args), ensure_ascii=False), True
    except Exception as e:  # noqa: BLE001
        msg = str(e) if isinstance(e, ToolCallError) else f"{type(e).__name__}: {e}"
        return json.dumps({"error": msg}, ensure_ascii=False), False


# ---------------------------------------------------------------- built-ins

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


@tool("calculate", "Evaluate an arithmetic expression, e.g. '1250000 * 0.18'.", {
    "type": "object",
    "properties": {"expression": {"type": "string"}},
    "required": ["expression"],
})
def calculate(expression: str) -> dict:
    return {"result": _eval(ast.parse(expression, mode="eval").body)}


@tool("current_time", "Current date and time in UTC (ISO 8601).", {
    "type": "object", "properties": {},
})
def current_time() -> dict:
    return {"utc": datetime.now(UTC).isoformat(timespec="seconds")}
