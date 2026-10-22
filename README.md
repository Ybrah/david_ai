# David

Cheap, sovereign LLM inference — small models, strong harnesses.

Where this is going: [`docs/VISION.md`](docs/VISION.md). Where we are: Phase 0, a harness around a
small model on OpenRouter. No GPUs of our own yet.

```
client ──► api/  (FastAPI on Railway, EU West)
              gateway   auth · rate limit · audit log · cost per request
              harness   tool loop · validators · 1 repair pass · typed errors
              │  OpenAI-compatible
              ▼
           OpenRouter   qwen/qwen3-30b-a3b-instruct-2507 (swap with LLM_MODEL)
```

| Folder | What | Deploys to |
|---|---|---|
| `api/` | Gateway + harness: keys, tool loop, validators, repair | Railway, root dir `/api` |
| `inference/` | Self-hosted GPU server (SGLang). Phase 1, not used now | Modal, `modal deploy inference/serve.py` |

## 1. Model API (OpenRouter)

Create a key at openrouter.ai → Keys, and put a credit limit on it. Any model that supports tool
calling works (`https://openrouter.ai/api/v1/models?supported_parameters=tools`).

No PII or NDA client documents through Phase 0: OpenRouter's providers are mostly outside the EU
(VISION §5).

## 2. API (Railway)

1. New project → Deploy from GitHub repo → this repo.
2. Service **Settings → Root Directory**: `/api`. Region: **EU West**.
3. Variables (see `.env.example`):
   - `DAVID_API_KEYS`: keys for your clients
   - `LLM_API_KEY`: your OpenRouter key
   - optional: `LLM_MODEL`, `RATE_LIMIT_RPM`, `MAX_REPAIRS`
4. Railway uses `api/railway.toml` (start command + `/health` check).

## 3. Call it

```bash
curl -s $API/chat -H "Authorization: Bearer change-me" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"TVA à 18% sur 1 250 000 FCFA ?"}]}'
```

Returns `reply`, `tool_calls` (each with `ok`), `usage` (tokens + `cost_usd`), `repairs`, `latency_ms`.

OpenAI clients can use the drop-in endpoint instead (non-streaming; tools run server-side):

```python
from openai import OpenAI

client = OpenAI(base_url=f"{API}/v1", api_key="change-me")
r = client.chat.completions.create(model="david", messages=[{"role": "user", "content": "TVA 18% sur 1 250 000 ?"}])
print(r.choices[0].message.content, r.model_extra["david"])  # david = tool trace, repairs, latency
```

## How the harness works

`api/app/harness.py`: call the model → run any tool calls through validators (known tool, JSON args
matching the schema) → validate the final reply (non-empty, no tool call written as text). A failure
goes back to the model as feedback and uses a repair. Past `MAX_REPAIRS` (default 1), or after
`MAX_TOOL_ROUNDS` model calls, the API returns `502` with a typed `error`: `tool_call_failed`,
`invalid_reply`, `max_rounds` or `upstream_error`. No frontier escalation yet.

`api/app/gateway.py` writes one `david.audit` JSON line per request: key hash, status, tokens,
cost, tools used and a payload hash. It never logs content.

## Local dev

```bash
cd api && pip install -r requirements-dev.txt
pytest -q                                   # mocked model, no network needed
LLM_API_KEY=sk-or-... uvicorn app.main:app --reload
```

Offline? `llama-server -hf Qwen/Qwen3-8B-GGUF:Q4_K_M --jinja --port 30000`, then run with
`LLM_BASE_URL=http://localhost:30000/v1 LLM_MODEL=qwen3`.

## Add a tool

`api/app/tools.py`:

```python
@tool("lookup_account", "Find a SYSCOHADA account by number.", {
    "type": "object", "properties": {"number": {"type": "string"}}, "required": ["number"],
})
def lookup_account(number: str) -> dict:
    return {"number": number, "label": PLAN.get(number, "unknown")}
```

The harness validates calls against this schema, so list every argument in `properties`.

## CI

- `ci.yml`: ruff + pytest on `api/`.

## Phase 1: self-hosting on Modal (not used now)

```bash
pip install modal openai && modal setup
modal deploy inference/serve.py
export MODAL_PROXY_TOKEN="wk-xxx.ws-yyy"   # Modal dashboard → Proxy Auth Tokens
modal run inference/serve.py                     # smoke test
```

First boot downloads weights (~6 GB) to a Volume; later cold starts take ~1–2 min. Idle cost with
`min_containers=0`: €0.

Then point the API at it: `LLM_BASE_URL=<modal url>/v1` and `MODAL_PROXY_TOKEN`, plus
`LLM_MODEL=Qwen/Qwen3-8B-AWQ`. Or set `LLM_BASE_URL=` (empty) with `MODAL_TOKEN_ID` +
`MODAL_TOKEN_SECRET` so the Modal SDK resolves the URL. `POST /warmup` wakes the GPU container
before real traffic.
