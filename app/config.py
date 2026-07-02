"""
app/config.py

Central configuration.  All tuneable constants live here;
nothing else imports os.environ directly.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent


@lru_cache(maxsize=1)
def get_settings():
    return _Settings()


class _Settings:
    # ── LLM ────────────────────────────────────────────────────────────
    # Primary: Groq (fast, free tier).  Swap GROQ_API_KEY for your own.
    # Fallback: any OpenAI-compatible endpoint by overriding OPENAI_API_KEY
    # and LLM_BASE_URL.
    GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
    LLM_MODEL: str = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
    # Max tokens for the agent reply (keeps us under 30 s timeout)
    LLM_MAX_TOKENS: int = int(os.environ.get("LLM_MAX_TOKENS", "700"))
    LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.2"))

    # ── Retrieval ───────────────────────────────────────────────────────
    CATALOG_PATH: str = os.environ.get(
        "CATALOG_PATH", str(BASE_DIR / "data" / "catalog.json")
    )
    EMBED_MODEL: str = os.environ.get(
        "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    # Number of candidates from each retriever before RRF fusion
    RETRIEVAL_TOP_K: int = int(os.environ.get("RETRIEVAL_TOP_K", "20"))
    # Final shortlist size cap (1–10 per spec)
    MAX_RECOMMENDATIONS: int = 10
    # RRF constant (k=60 is the standard default)
    RRF_K: int = 60
    # BM25 / dense blend weight (higher = more BM25 exact-match influence)
    BM25_WEIGHT: float = float(os.environ.get("BM25_WEIGHT", "0.5"))

    # ── Agent policy ────────────────────────────────────────────────────
    # How many required slots must be filled before we recommend
    MIN_SLOTS_TO_RECOMMEND: int = 1  # role/description at minimum
    # The evaluator caps conversations at 8 turns total; we stop before that
    MAX_TURNS: int = 8
