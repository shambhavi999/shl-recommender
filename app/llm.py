"""
app/llm.py

Thin wrapper around the Groq client.

Design:
  – Two public functions: `chat()` for free-text replies,
    `chat_json()` for structured JSON output (slot extraction).
  – Tenacity retry with exponential backoff on transient errors.
  – Hard 28-second timeout wall (evaluator cap is 30 s; we leave 2 s buffer).
  – Falls back gracefully: if JSON parsing fails, returns a safe default dict.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings

try:
    from groq import Groq, APIError, APITimeoutError, RateLimitError
    _HAS_GROQ = True
except ImportError:
    _HAS_GROQ = False

_client = None
_TIMEOUT = 28  # seconds


def _get_client():
    global _client
    if _client is None:
        if not _HAS_GROQ:
            raise RuntimeError("groq package not installed. Run: pip install groq")
        settings = get_settings()
        if not settings.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to .env or export it in your shell."
            )
        _client = Groq(api_key=settings.GROQ_API_KEY, timeout=_TIMEOUT)
    return _client


@retry(
    retry=retry_if_exception_type((Exception,)),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
def chat(messages: list[dict], max_tokens: int | None = None) -> str:
    """
    Call the LLM and return the assistant's text reply.
    Raises RuntimeError on failure after retries.
    """
    settings = get_settings()
    client = _get_client()

    try:
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=messages,
            max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        # Log and re-raise; tenacity will retry
        print(f"[llm] Error calling LLM: {type(exc).__name__}: {exc}", flush=True)
        raise


def chat_json(messages: list[dict], fallback: dict) -> dict:
    """
    Call the LLM expecting a JSON response.
    Returns parsed dict, or ``fallback`` if parsing fails.

    We ask for a JSON reply via the system prompt (we don't use Groq's
    json_object mode here to preserve compatibility with any OpenAI-compatible
    backend the user may swap in).
    """
    try:
        raw = chat(messages, max_tokens=400)
        # Strip markdown code fences if the model wraps the JSON
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        # Find the outermost { … } in case there's preamble text
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError, Exception) as exc:
        print(f"[llm] JSON parse failed: {exc}; using fallback.", flush=True)
        return fallback
