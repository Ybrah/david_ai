# David

## North star
Cheap, sovereign LLM inference for African businesses. Narrow, repeatable tasks run on small
open-weight models (4–30B), wrapped in strong harnesses (schemas, validators, tools). The metric
that matters is cost per task. The full vision is in [`docs/VISION.md`](docs/VISION.md).

The moat is the harness, not the metal (VISION §1). In Phase 0 we call commodity inference APIs
(OpenRouter / NVIDIA NIM) and own no GPUs. Self-hosted SGLang comes later.

## We start very simple
**Current focus: ship the first harness end to end on a commodity API.**
- The model is `qwen/qwen3-30b-a3b-instruct-2507` on OpenRouter (the default; swap with `LLM_MODEL`). Set `LLM_API_KEY`.
- The David gateway is in-process in FastAPI (`api/app/gateway.py`): keys, per-key rate limit, audit log,
  cost per request. No LiteLLM: with OpenRouter as the only upstream it adds nothing yet.
- The harness (`api/app/harness.py`): tool loop, then validators, then one repair pass, then a typed error.
  There's no escalation.
- Endpoints: `/chat` and an OpenAI-compatible `/v1/chat/completions` (non-streaming).
- Hosted on Railway (EU West) with real `DAVID_API_KEYS`.
- Done when the README's TVA `curl` gets a correct reply from the deployed URL.
- `inference/serve.py` (Modal SGLang) stays dormant. It is the Phase 1 self-hosting path, so don't delete it.

**Not yet. Don't build ahead.** Self-host only after volume (~20M tokens/month) or an anchor
client's in-region requirement justifies it (VISION §4).
- Next, only when the user asks: the Python sandbox, web search, frontier escalation (removed for
  now), and streaming (VISION §3, §6).
- Later: LiteLLM (once there are several upstreams), `/v1/tasks/*` with schema validators, the d1 router,
  failure mining and fine-tunes, self-hosted GPUs, and the on-prem T1 tier.

Add them one at a time.

**Keep these even now:**
- Host our own services in the EU (Railway EU West).
- No PII or NDA client documents through Phase 0 APIs (VISION §5).
- Report tokens, cost and latency on every request (`usage`, `latency_ms`, the `david.audit` log line).
- Keep the model swappable: everything goes through the OpenAI-compatible client in
  `api/app/llm.py`, and the model name comes from config. This is what makes the pivot a key swap.
- Keep diffs small and add no new infrastructure without a reason.

## Repo map
| Path | What |
|---|---|
| `api/app/main.py` | `/health`, `/warmup`, `/chat`, `/v1/chat/completions` (thin) |
| `api/app/gateway.py` | API-key auth, per-key rate limit, `david.audit` JSON log |
| `api/app/harness.py` | State machine: tool loop, validators, repair, typed `HarnessError` |
| `api/app/tools.py` | `@tool` registry (`calculate`, `current_time`) + tool-call validation |
| `api/app/config.py` | Env-driven settings, system prompt |
| `api/app/llm.py` | OpenAI-compatible client for any endpoint (OpenRouter now); resolves the Modal URL if `LLM_BASE_URL` is set empty |
| `api/tests/` | pytest with a scripted fake model (no network needed) |
| `inference/serve.py` | Modal SGLang server (L40S, scale-to-zero); Phase 1 self-host path, not used in Phase 0 |
| `.github/workflows/` | `ci.yml` (ruff + pytest) |

## Commands
```bash
cd api && pytest -q && ruff check .                        # same checks as CI
LLM_API_KEY=sk-or-... uvicorn app.main:app --reload       # Phase 0: OpenRouter (default)
# Phase 1 (self-hosting), not used now:
modal deploy inference/serve.py                            # GPU server
modal run inference/serve.py                               # smoke test (needs MODAL_PROXY_TOKEN)
```
