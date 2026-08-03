import logging
from functools import lru_cache

from app.core.config import settings

logger = logging.getLogger("vie.prompts")


@lru_cache(maxsize=1)
def _client():
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


def get_prompt(name: str, fallback: str, *, version: int | None = None) -> str:
    """Fetches a managed prompt from Langfuse, falling back to an inline default.

    Every video-LLM prompt (two-pass grounding, claim verification, ...) is
    named and versioned in Langfuse so prompt changes don't require a
    redeploy. Without Langfuse credentials configured (e.g. local dev), this
    transparently returns `fallback` so the pipeline still runs.
    """
    client = _client()
    if client is None:
        return fallback
    try:
        prompt = client.get_prompt(name, version=version)
        return prompt.prompt
    except Exception:
        logger.exception("failed to fetch prompt %r from Langfuse, using fallback", name)
        return fallback
