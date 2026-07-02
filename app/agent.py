"""
app/agent.py

The agent pipeline for one turn of the conversation.

Every POST /chat call flows through ``run_agent(messages)`` which:

  1. GUARD      – Reject prompt-injection attempts before calling the LLM.
  2. EXTRACT    – Call LLM#1 (small/structured) to pull slots from conversation.
  3. POLICY     – Decide intent: clarify / recommend / refine / compare / refuse.
  4. RETRIEVE   – Run hybrid search with the query built from slots.
  5. REPLY      – Call LLM#2 (main) to generate the reply, grounded in retrieved docs.
  6. PARSE      – Extract structured recommendations from the LLM reply.
  7. VALIDATE   – Ensure every recommended URL exists in the catalog.

This two-LLM architecture means:
  – Slot extraction is fast and deterministic (small output, structured JSON).
  – The main reply call never has to guess slots; they're injected into the prompt.
  – Both calls are independently testable (see tests/).
"""
from __future__ import annotations

import re
from typing import Optional

from app.catalog import Assessment, get_retriever
from app.config import get_settings
from app.llm import chat, chat_json
from app.prompts import (
    build_agent_messages,
    build_compare_query,
    slot_extraction_messages,
)

# ── Guardrail patterns (checked BEFORE any LLM call) ────────────────────────
_INJECTION_PATTERNS = [
    r"ignore\s+(previous|above|all|prior)\s+instructions",
    r"(forget|disregard)\s+(your|the|all)\s+(instructions|rules|prompt|system)",
    r"you\s+are\s+now\s+(a\s+)?(different|new|unrestricted)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"do\s+anything\s+now",
    r"jailbreak",
    r"DAN\s*mode",
    r"override\s+your",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def _is_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text))


# ── Recommendation extraction ────────────────────────────────────────────────

def _extract_recommendations(
    reply: str,
    retrieved: list[Assessment],
    catalog_urls: set[str],
) -> list[dict]:
    """
    Pull structured recommendations from the LLM reply text.

    Strategy:
      1. Find every catalog URL mentioned in the reply.
      2. Cross-reference against the retrieved list to get name + test_type.
      3. Fall back to matching by assessment name if URL isn't in reply.
    """
    found: list[dict] = []
    seen_urls: set[str] = set()

    # Index retrieved items by URL and normalised name
    url_to_item = {a["url"]: a for a in retrieved}
    name_to_item = {a["name"].lower(): a for a in retrieved}

    # Match URLs mentioned in reply
    url_pattern = re.compile(r"https?://www\.shl\.com/[^\s\)\]\>\"']+")
    for m in url_pattern.finditer(reply):
        url = m.group(0).rstrip(".,;")
        # Normalise: strip trailing slash variations
        url_norm = url.rstrip("/") + "/"
        # Must be a real catalog URL
        candidate = url_to_item.get(url) or url_to_item.get(url_norm)
        if candidate and url_norm not in seen_urls:
            seen_urls.add(url_norm)
            found.append(_make_rec(candidate))

    # If no URLs extracted, fall back to name matching
    if not found:
        for a in retrieved:
            name_lower = a["name"].lower()
            # Check if the assessment name appears in the reply (case-insensitive)
            if re.search(re.escape(name_lower), reply, re.IGNORECASE):
                url_norm = a["url"].rstrip("/") + "/"
                if url_norm not in seen_urls:
                    seen_urls.add(url_norm)
                    found.append(_make_rec(a))

    # Safety: if we still have nothing but the LLM clearly recommended,
    # surface the top retrieved items (capped at MAX_RECOMMENDATIONS).
    settings = get_settings()
    if not found and retrieved and _looks_like_recommendation(reply):
        for a in retrieved[: settings.MAX_RECOMMENDATIONS]:
            found.append(_make_rec(a))

    # Validate: drop any URL not in the live catalog set (anti-hallucination)
    valid_found = []
    for rec in found:
        url = rec["url"].rstrip("/") + "/"
        if url in {u.rstrip("/") + "/" for u in catalog_urls}:
            valid_found.append(rec)
        else:
            print(f"[agent] Dropping hallucinated URL: {rec['url']}", flush=True)

    return valid_found[: settings.MAX_RECOMMENDATIONS]


def _make_rec(a: Assessment) -> dict:
    test_type = a.get("test_type", [])
    # API spec: test_type is a single letter string; join if multiple
    type_str = " ".join(test_type) if test_type else "K"
    return {"name": a["name"], "url": a["url"], "test_type": type_str}


def _looks_like_recommendation(reply: str) -> bool:
    indicators = [
        "recommend", "assessment", "suggest", "shortlist",
        "here are", "following", "consider", "test for"
    ]
    return any(ind in reply.lower() for ind in indicators)


# ── Turn counter ─────────────────────────────────────────────────────────────

def _count_turns(messages: list[dict]) -> int:
    """
    Count user + assistant turns already in the history.
    The evaluator caps at 8 total turns (user + assistant combined).
    """
    return len(messages)


# ── Slot defaults ────────────────────────────────────────────────────────────

_DEFAULT_SLOTS = {
    "role": None,
    "skills": [],
    "job_levels": [],
    "test_type_filter": [],
    "duration_max_mins": None,
    "language": None,
    "additional_context": None,
    "intent": "clarify",
}


# ── Retrieval query builder ───────────────────────────────────────────────────

def _build_query(slots: dict, messages: list[dict]) -> str:
    """
    Construct a retrieval query from slots.
    We favour explicit slots over raw message text for precision,
    then append the last user message as a fallback signal.
    """
    parts = []
    if slots.get("role"):
        parts.append(slots["role"])
    if slots.get("skills"):
        parts.extend(slots["skills"])
    if slots.get("additional_context"):
        parts.append(slots["additional_context"])
    if slots.get("job_levels"):
        parts.extend(slots["job_levels"])
    # Always include the last user utterance to catch terms not yet in slots
    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    if last_user:
        parts.append(last_user)
    return " ".join(parts) if parts else ""


# ── Main entry point ─────────────────────────────────────────────────────────

def run_agent(messages: list[dict]) -> dict:
    """
    Process one conversation turn and return the API response dict.

    Returns:
        {
          "reply": str,
          "recommendations": list[dict],   # [] when clarifying / refusing
          "end_of_conversation": bool
        }
    """
    settings = get_settings()
    retriever = get_retriever()

    # ── Turn cap guard ───────────────────────────────────────────────────────
    turn_count = _count_turns(messages)
    if turn_count >= settings.MAX_TURNS:
        return {
            "reply": (
                "We've reached the maximum conversation length. "
                "Please start a new conversation to continue."
            ),
            "recommendations": [],
            "end_of_conversation": True,
        }

    # ── Prompt injection guard ───────────────────────────────────────────────
    last_user_msg = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    if _is_injection(last_user_msg):
        return {
            "reply": (
                "I'm only able to help with SHL assessment selection. "
                "If you have a genuine assessment question, I'm happy to help!"
            ),
            "recommendations": [],
            "end_of_conversation": False,
        }

    # ── Step 1: Slot extraction ──────────────────────────────────────────────
    slot_messages = slot_extraction_messages(messages)
    slots = chat_json(slot_messages, fallback=dict(_DEFAULT_SLOTS))
    # Ensure all keys exist (LLM may omit optional ones)
    for k, v in _DEFAULT_SLOTS.items():
        slots.setdefault(k, v)

    intent: str = slots.get("intent", "clarify")

    # ── Step 2: Retrieve relevant assessments ────────────────────────────────
    retrieved: list[Assessment] = []

    if intent in ("recommend", "refine", "compare"):
        if intent == "compare":
            # For compare, we extract the specific assessment names the user mentioned
            compare_names = _extract_named_assessments(messages)
            query = build_compare_query(compare_names) if compare_names else _build_query(slots, messages)
            # Look up each named assessment directly first
            for name in compare_names:
                a = retriever.get_by_name(name)
                if a and a not in retrieved:
                    retrieved.append(a)
        else:
            query = _build_query(slots, messages)

        if query:
            filter_types = slots.get("test_type_filter") or None
            extra = retriever.search(
                query,
                top_k=settings.RETRIEVAL_TOP_K,
                filter_test_types=filter_types,
            )
            # Merge: direct lookups first, then search results (dedup by url)
            existing_urls = {a["url"] for a in retrieved}
            for a in extra:
                if a["url"] not in existing_urls:
                    retrieved.append(a)
                    existing_urls.add(a["url"])

        # Apply duration filter if set (post-retrieval, since duration isn't always indexed)
        max_dur = slots.get("duration_max_mins")
        if max_dur:
            retrieved = [
                a for a in retrieved
                if not a.get("duration_minutes") or a["duration_minutes"] <= int(max_dur)
            ]

        # Cap to 2× MAX to give the LLM choice without overwhelming the context
        retrieved = retrieved[: settings.MAX_RECOMMENDATIONS * 2]

    # ── Step 3: Build and call main agent LLM ────────────────────────────────
    agent_messages = build_agent_messages(messages, slots, retrieved, intent)
    try:
        reply = chat(agent_messages)
    except Exception as exc:
        return {
            "reply": "I'm having trouble connecting right now. Please try again in a moment.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    # ── Step 4: Extract structured recommendations from reply ─────────────────
    catalog_urls = retriever.get_catalog_urls()
    recommendations: list[dict] = []

    if intent in ("recommend", "refine"):
        recommendations = _extract_recommendations(reply, retrieved, catalog_urls)

    # ── Step 5: Determine end_of_conversation ────────────────────────────────
    # We mark conversation complete when the agent provided a recommendation
    # AND the intent was a final recommendation (not a clarification or refusal).
    end_of_conversation = (
        intent in ("recommend", "refine")
        and len(recommendations) > 0
        and _agent_considers_done(reply)
    )

    return {
        "reply": reply,
        "recommendations": recommendations,
        "end_of_conversation": end_of_conversation,
    }


def _extract_named_assessments(messages: list[dict]) -> list[str]:
    """
    Find specific SHL assessment names mentioned by the user (for compare mode).
    Heuristic: look for quoted terms or known patterns like "OPQ", "GSA", "MQ", "Verify".
    """
    text = " ".join(m["content"] for m in messages if m["role"] == "user")
    found = []
    # Quoted names
    found += re.findall(r'"([^"]+)"', text)
    found += re.findall(r"'([^']+)'", text)
    # Known SHL product acronyms (uppercase, 2-6 chars)
    found += re.findall(r"\b(OPQ\w*|MQ\w*|GSA\w*|Verify\w*|SJT\w*|CCSQ\w*)\b", text, re.IGNORECASE)
    return list(dict.fromkeys(found))  # preserve order, deduplicate


def _agent_considers_done(reply: str) -> bool:
    """
    Detect whether the LLM reply signals the conversation is wrapping up.
    We look for phrases that indicate the agent has delivered the shortlist.
    """
    done_signals = [
        r"let me know if you.{0,30}(adjust|refine|change|add|remove)",
        r"hope (this|these) help",
        r"feel free to reach out",
        r"good luck",
        r"any (other|further) question",
    ]
    return any(re.search(sig, reply, re.IGNORECASE) for sig in done_signals)
