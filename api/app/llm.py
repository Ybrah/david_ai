"""OpenAI-compatible client to the model API (OpenRouter, or SGLang via Modal URL resolution)."""
import logging
from functools import lru_cache

from openai import AsyncOpenAI

from . import config

log = logging.getLogger(__name__)


def resolve_base_url() -> str:
    if config.LLM_BASE_URL:
        return config.LLM_BASE_URL.rstrip("/")
    import modal  # only needed when resolving via the SDK

    url = modal.Server.from_name(config.MODAL_APP, config.MODAL_SERVER).get_url()
    log.info("resolved model server via Modal SDK: %s", url)
    return f"{url.rstrip('/')}/v1"


def extra_body() -> dict:
    """Provider-specific request fields."""
    if config.IS_OPENROUTER:
        return {"usage": {"include": True}}  # adds usage.cost (USD) to every response
    return {"chat_template_kwargs": {"enable_thinking": config.ENABLE_THINKING}}


@lru_cache(maxsize=1)
def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=resolve_base_url(),
        api_key=config.LLM_API_KEY,
        timeout=config.LLM_TIMEOUT,
    )
