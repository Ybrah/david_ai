"""David gateway: API-key auth, per-key rate limit, audit log. Runs in-process, no extra service."""
import hashlib
import json
import logging
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException

from . import config
from .harness import Run

audit_log = logging.getLogger("david.audit")
_recent: dict[str, deque[float]] = defaultdict(deque)  # key id -> request times in the last minute


def _sha(text: str, n: int) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:n]


def authenticate(authorization: str | None = Header(default=None)) -> str:
    """FastAPI dependency: check the API key, apply the rate limit, return a key id for the audit log."""
    if not config.API_KEYS:
        kid = "dev"  # dev mode: no keys configured
    else:
        token = (authorization or "").removeprefix("Bearer ").strip()
        if token not in config.API_KEYS:
            raise HTTPException(401, "invalid API key")
        kid = _sha(token, 12)  # never log the key itself
    _rate_limit(kid)
    return kid


def _rate_limit(kid: str) -> None:
    if config.RATE_LIMIT_RPM <= 0:
        return
    now = time.monotonic()
    window = _recent[kid]
    while window and now - window[0] >= 60:
        window.popleft()
    if len(window) >= config.RATE_LIMIT_RPM:
        retry = int(60 - (now - window[0])) + 1
        raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(retry)})
    window.append(now)


def audit(kid: str, endpoint: str, messages: list[dict], run: Run, status: str, latency_ms: int) -> None:
    """One JSON line per request: metadata and a payload hash, never the content."""
    audit_log.info(json.dumps({
        "key": kid,
        "endpoint": endpoint,
        "status": status,
        "model": config.LLM_MODEL,
        "payload_sha256": _sha(json.dumps(messages, sort_keys=True, ensure_ascii=False), 16),
        "rounds": run.rounds,
        "repairs": run.repairs,
        "tools": [t.name for t in run.tool_calls],
        "prompt_tokens": run.usage.prompt_tokens,
        "completion_tokens": run.usage.completion_tokens,
        "cost_usd": run.usage.cost_usd,
        "latency_ms": latency_ms,
    }))
