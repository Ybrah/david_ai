"""Harness state machine around the small model.

    call model ─► tool calls? ─► validate + run each ─► any failed? ─► repair ─► call model
        │
        └─► reply ─► validate ─► ok ─► return
                         └─► failed ─► inject the problem as feedback ─► repair ─► call model

Every failure goes back to the model as data so it can fix itself. Each failure uses one repair; past
MAX_REPAIRS, or after MAX_TOOL_ROUNDS model calls, the run ends with a typed HarnessError.
No frontier escalation yet (VISION §4, Phase 0).
"""
import logging

from pydantic import BaseModel, Field

from . import config, tools
from .llm import extra_body, get_client

log = logging.getLogger("david.harness")


class ToolCall(BaseModel):
    name: str
    args: str
    result: str
    ok: bool


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0  # reported by OpenRouter; 0 for self-hosted


class Run(BaseModel):
    reply: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    rounds: int = 0  # model calls
    repairs: int = 0


class HarnessError(Exception):
    def __init__(self, code: str, message: str, run: Run):
        super().__init__(message)
        self.code, self.message, self.run = code, message, run


def check_reply(content: str | None) -> str | None:
    """Validate a final answer. Returns the problem to send back to the model, or None if valid."""
    text = (content or "").strip()
    if not text:
        return "Your reply was empty. Answer the user's last message."
    if "<tool_call>" in text:
        return ("You wrote a tool call as plain text. Call the tool through function calling, "
                "or answer the user directly.")
    return None


async def run(messages: list[dict], temperature: float, max_tokens: int) -> Run:
    """Run the harness. `messages` (system prompt first) is extended in place with the transcript."""
    out = Run()
    client = get_client()

    for _ in range(config.MAX_TOOL_ROUNDS):
        out.rounds += 1
        try:
            resp = await client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                tools=tools.specs(),
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body(),
            )
        except Exception as e:
            log.exception("model API error")
            raise HarnessError("upstream_error", f"model API error: {type(e).__name__}", out) from e
        _add_usage(out.usage, resp.usage)
        msg = resp.choices[0].message

        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })
            errors = []
            for tc in msg.tool_calls:
                result, ok = tools.run_tool(tc.function.name, tc.function.arguments)
                out.tool_calls.append(ToolCall(name=tc.function.name, args=tc.function.arguments or "",
                                               result=result, ok=ok))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                if not ok:
                    errors.append(f"{tc.function.name}: {result}")
            if errors:
                _use_repair(out, "tool_call_failed", "; ".join(errors))
            continue

        problem = check_reply(msg.content)
        if problem is None:
            out.reply = msg.content.strip()
            return out
        _use_repair(out, "invalid_reply", problem)
        messages.append({"role": "assistant", "content": msg.content or ""})
        messages.append({"role": "user", "content": f"[David harness] {problem}"})

    raise HarnessError("max_rounds", f"no valid answer after {config.MAX_TOOL_ROUNDS} model calls", out)


def _use_repair(out: Run, code: str, problem: str) -> None:
    out.repairs += 1
    if out.repairs > config.MAX_REPAIRS:
        raise HarnessError(code, problem, out)
    log.info("repair %d/%d (%s): %s", out.repairs, config.MAX_REPAIRS, code, problem)


def _add_usage(usage: Usage, raw) -> None:
    if raw is None:
        return
    usage.prompt_tokens += raw.prompt_tokens or 0
    usage.completion_tokens += raw.completion_tokens or 0
    usage.cost_usd += getattr(raw, "cost", None) or 0.0
