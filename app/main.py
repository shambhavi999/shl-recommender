"""
app/main.py

FastAPI service — two endpoints:
  GET  /health  →  {"status": "ok"}
  POST /chat    →  AgentResponse

The API is fully stateless. Every /chat call receives the complete
conversation history in the request body. No session state is stored.

Schema is non-negotiable per the assignment spec:
  Request:  {"messages": [{"role": str, "content": str}, ...]}
  Response: {"reply": str, "recommendations": [...], "end_of_conversation": bool}
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from app.agent import run_agent
from app.catalog import get_retriever


# ── Lifespan: pre-warm the catalog and retrieval indices at startup ───────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Pre-loading catalog and building retrieval indices…", flush=True)
    get_retriever()  # triggers the singleton build; expensive on first call
    print("[startup] Ready.", flush=True)
    yield


app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "Conversational agent that helps hiring managers find the right "
        "SHL Individual Test Assessments for their roles."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins (required for the SHL evaluator to reach the endpoint)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message content must not be empty")
        return v


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def at_least_one_message(cls, v):
        if not v:
            raise ValueError("messages must contain at least one message")
        if v[-1].role != "user":
            raise ValueError("The last message must have role='user'")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Readiness check — returns 200 as soon as the service is up."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main conversational endpoint.

    Accepts the full conversation history and returns the agent's
    next reply plus (when ready) a structured assessment shortlist.
    """
    messages = [m.model_dump() for m in request.messages]
    try:
        result = run_agent(messages)
    except RuntimeError as exc:
        # Surface configuration errors (missing API key, missing catalog) as 503
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        print(f"[chat] Unhandled error: {exc}", flush=True)
        raise HTTPException(status_code=500, detail="Internal server error")

    return ChatResponse(
        reply=result["reply"],
        recommendations=[
            Recommendation(**r) for r in result.get("recommendations", [])
        ],
        end_of_conversation=result.get("end_of_conversation", False),
    )


# ── Global error handler ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"[error] {type(exc).__name__}: {exc}", flush=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
