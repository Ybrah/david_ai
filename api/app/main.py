"""David API: gateway (auth, rate limit, audit) + harness over a small model.

Run locally:  uvicorn app.main:app --reload
"""
import logging
import time
import uuid
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config, gateway, harness
from .harness import ToolCall, Usage
from .llm import get_client

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="David API", version="0.2.0")


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    temperature: float = Field(0.3, ge=0, le=2)
    max_tokens: int = Field(1024, ge=1, le=4096)


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[ToolCall]
    usage: Usage
    repairs: int
    latency_ms: int


class CompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class CompletionRequest(BaseModel):
    """OpenAI-compatible subset. `model` is accepted but the gateway picks the model (LLM_MODEL)."""
    model: str | None = None
    messages: list[CompletionMessage] = Field(min_length=1)
    temperature: float = Field(0.3, ge=0, le=2)
    max_tokens: int = Field(1024, ge=1, le=4096)
    stream: bool = False


async def _run(kid: str, endpoint: str, system: str, messages: list[dict],
               temperature: float, max_tokens: int) -> tuple[harness.Run, int]:
    t0 = time.perf_counter()
    try:
        run = await harness.run([{"role": "system", "content": system}, *messages], temperature, max_tokens)
    except harness.HarnessError as e:
        latency = int((time.perf_counter() - t0) * 1000)
        gateway.audit(kid, endpoint, messages, e.run, e.code, latency)
        raise HTTPException(502, {"error": e.code, "message": e.message,
                                  "usage": e.run.usage.model_dump()}) from e
    latency = int((time.perf_counter() - t0) * 1000)
    gateway.audit(kid, endpoint, messages, run, "ok", latency)
    return run, latency


@app.get("/health")
async def health():
    return {"ok": True, "model": config.LLM_MODEL}


@app.post("/warmup", dependencies=[Depends(gateway.authenticate)])
async def warmup():
    """Check the model API is reachable (and wake the GPU container when self-hosting on Modal)."""
    t0 = time.perf_counter()
    await get_client().models.list()
    return {"ok": True, "latency_ms": int((time.perf_counter() - t0) * 1000)}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, kid: str = Depends(gateway.authenticate)):
    run, latency = await _run(kid, "/chat", config.SYSTEM_PROMPT, [m.model_dump() for m in req.messages],
                              req.temperature, req.max_tokens)
    return ChatResponse(reply=run.reply, tool_calls=run.tool_calls, usage=run.usage,
                        repairs=run.repairs, latency_ms=latency)


@app.post("/v1/chat/completions")
async def chat_completions(req: CompletionRequest, kid: str = Depends(gateway.authenticate)):
    """Drop-in for OpenAI clients. Tools run server-side; the trace is in the extra `david` field."""
    if req.stream:
        raise HTTPException(400, "stream=true is not supported yet")
    system = "\n\n".join([config.SYSTEM_PROMPT] + [m.content for m in req.messages if m.role == "system"])
    convo = [m.model_dump() for m in req.messages if m.role != "system"]
    if not convo:
        raise HTTPException(422, "messages needs at least one user message")

    run, latency = await _run(kid, "/v1/chat/completions", system, convo, req.temperature, req.max_tokens)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": config.LLM_MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": run.reply},
                     "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": run.usage.prompt_tokens,
            "completion_tokens": run.usage.completion_tokens,
            "total_tokens": run.usage.prompt_tokens + run.usage.completion_tokens,
            "cost": run.usage.cost_usd,
        },
        "david": {"tool_calls": [t.model_dump() for t in run.tool_calls], "repairs": run.repairs,
                  "rounds": run.rounds, "latency_ms": latency},
    }
