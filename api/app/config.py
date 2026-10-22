import os

# Upstream model API (Phase 0: OpenRouter). Set LLM_BASE_URL to a self-hosted SGLang URL to swap,
# or set it empty to let the Modal SDK resolve the deployed server URL (needs MODAL_TOKEN_ID/SECRET).
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("MODAL_PROXY_TOKEN", "none")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3-30b-a3b-instruct-2507")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "300"))  # covers Modal cold starts
IS_OPENROUTER = "openrouter.ai" in LLM_BASE_URL

MODAL_APP = os.getenv("DAVID_MODAL_APP", "david-sglang")
MODAL_SERVER = os.getenv("DAVID_MODAL_SERVER", "SGLang")

API_KEYS = {k.strip() for k in os.getenv("DAVID_API_KEYS", "").split(",") if k.strip()}
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "60"))  # per API key; 0 = off
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "5"))
MAX_REPAIRS = int(os.getenv("MAX_REPAIRS", "1"))  # failed validations the model may fix per request
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "false").lower() == "true"

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT") or """You are a concise assistant for West African business users.
Answer in the user's language (French or English).
Use tools for any arithmetic or date/time question; never compute by hand.
If you don't know, say so."""
