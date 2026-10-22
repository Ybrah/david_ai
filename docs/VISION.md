# Project David: Harness-First Strategy

**Cheap, sovereign LLM inference for Africa — small models, strong harnesses.**

*Draft v0.3 — 2026-10-21*

---

## 1. The Refined Thesis: Decoupling the Moat from the Metal

The original thesis holds: **for narrow business tasks and developer endpoints, a well-harnessed 4B–30B model delivers frontier-quality outcomes at 10× to 100× lower cost.** The intelligence lives in the system around the weights—schemas, validators, execution sandboxes, and repair loops.

However, the infrastructure required to host these models (CUDA drivers, VRAM sizing, GPU orchestration, idling costs) is a distraction from building that intelligence.

**The Pivot:** We are decoupling the harness IP from the infrastructure burden. Project David will launch by routing inference to commodity API providers (OpenRouter / NVIDIA NIM) for ultra-cheap small models. This allows us to focus 100% of our engineering cycles on our true moat—the execution harness—with $0 fixed infrastructure costs. Only once the task schemas stabilize and volume justifies it will we swap the API keys for self-hosted SGLang clusters to lock in unit economics and data sovereignty.

## 2. What David Is

David is an **API Infrastructure Provider (Inference + Harness as a Service)** for African developers, fintechs, and enterprises.

We expose standard drop-in endpoints (`/v1/chat/completions` and `/v1/tasks/{task_name}`). When developers hit our API, they aren't just getting text back. They are getting:

1. **Batteries-Included Execution:** Server-side tool execution (Python sandboxes, web search) without writing glue code.
2. **Deterministic Reliability:** Outputs that have passed through hard validators (schema checks, code exit codes).
3. **Automated Error Repair:** If a model hallucinates a tool call, David catches the exception and forces the model to repair it before the developer ever sees the response.
4. **Transparent Escalation:** Automatic routing to heavy frontier models if the small model fails validation twice.

## 3. The Architecture (Phase 0: The Harness Engine)

In Phase 0, David acts as a highly intelligent middleware layer. We do not host the LLM weights; we host the execution and validation logic.

```
┌──────────────────────────────────────────────────────────────┐
│  Developer Client (App / Bot / ERP)                          │
│  POST /v1/chat/completions (with "harness_tools" enabled)    │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│  David Gateway (LiteLLM Core)                                │
│  auth · rate limits · cost accounting · audit log            │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│  Harness Middleware (The Moat)                               │
│  - System Context & Guardrail Injection                      │
│  - Tool Interceptor (pauses stream on tool request)          │
│  - Wasm/gVisor Code Sandbox (executes Python securely)       │
│  - Context Extractor (compresses Web Search results)         │
│  - Deterministic Validators (exit_code == 0, schema match)   │
│  - Repair Loop (re-prompts model with traceback on fail)     │
└──────┬──────────────────────────────────────────────┬────────┘
       │ standard path (>90%)                         │ escalation
       ▼ (Pure API Calls)                             ▼
┌──────────────────────────────────────┐     ┌─────────────────────────────────┐
│  Commodity Inference APIs            │     │  Frontier Escalation APIs       │
│  (OpenRouter / NVIDIA NIM)           │     │  (OpenRouter / Direct)          │
│  - Qwen 2.5 Coder 14B (~$0.15/M)     │     │  - Claude 3.5 Sonnet            │
│  - Mistral Small / Llama 3.1 8B      │     │  - DeepSeek V3                  │
└──────────────────────────────────────┘     └─────────────────────────────────┘
```

**Why this architecture wins Day 1:**

* **Instant Model Swaps:** If a better 14B model drops tomorrow, we change one string in our config. No downloading 20GB files or re-quantizing.
* **Focus on the Hard Problem:** Building a secure, sub-second Python execution sandbox is hard. Routing tokens is a solved problem. We focus on the sandbox.

## 4. The Two-Step Migration Roadmap

The strategy moves from proving the intelligence (Phase 0) to owning the unit economics and data sovereignty (Phase 1+).

### Phase 0: The Developer API (Weeks 1–6)

* **Goal:** Ship the API gateway and the core tools with zero idle compute costs.
* **Hosting:** Deploy the Gateway and Harness Middleware on fast, scalable CPU hosts (Fly.io, Render, or Modal).
* **Inference:** OpenRouter API (targeting `qwen-2.5-coder-14b`).
* **Deliverables:**
  * Fully compliant `/v1/chat/completions` endpoint.
  * Server-side Python code sandbox (max 3s timeout, no network).
  * Web search context compressor.
  * The State Machine: Automatic 1-pass repair on tool failure, followed by Frontier escalation.
* **Economics:** Variable cost only. Sub-cents per API call.

### Phase 1: Structured Tasks & Self-Hosting (Months 2–4)

* **Goal:** Lock in enterprise margins and begin sovereign data hosting.
* **Trigger:** When token volume exceeds ~20M tokens/month, OR when anchor clients (`invoice_to_entry` for Koori) require payloads to stay in-region.
* **Hosting (T2 Tier):** Spin up bare-metal EU GPUs (e.g., Hetzner RTX 4090s at €200/mo or OVH/Scaleway).
* **Inference:** Swap OpenRouter API keys for internal SGLang IP addresses. The Harness Middleware does not change.
* **Deliverables:**
  * Launch `/v1/tasks/` endpoints with strict JSON schema validators.
  * Deploy the `d1-3B` CPU router to natively pre-sort task difficulty.
  * Begin offline failure mining to fine-tune our own models.
* **Economics:** Fixed compute floor (€200-500/mo) enabling infinite margin scaling on local traffic.

### Phase 2: The Sovereign Edge (Months 4–7)

* **Goal:** Absolute zero-cloud environments for African banks, ministries, and logistics hubs.
* **Hosting (T1 Tier):** Pre-configured local hardware appliances shipped to client cabinets.
* **Inference:** `llama.cpp` or SGLang running entirely offline.
* **Deliverables:**
  * Turnkey Docker images with automated failovers (UPS/LTE).
  * Local-language task expansions (ASR for Wolof, Bambara).

## 5. Honest Tradeoffs of the Pivot

* **Data Sovereignty Delayed:** In Phase 0, data still flows through OpenRouter/US providers. *Mitigation:* We use this phase strictly for non-sensitive developer traffic and internal task building. No PII or strict NDA enterprise documents run until Phase 1 self-hosting is live.
* **API Latency Limits:** We incur the network latency of hitting OpenRouter APIs rather than querying localhost VRAM. *Mitigation:* We are building asynchronous business tasks and tool loops where a 1,500ms TTFT is perfectly acceptable.

## 6. Next Actions

1. Spin up the LiteLLM gateway and route to OpenRouter (`qwen-2.5-coder`).
2. Build the Wasm/Firecracker micro-sandbox for Python execution.
3. Wire the interceptor: Catch the tool call, run the sandbox, evaluate the exit code, and trigger the repair loop.
