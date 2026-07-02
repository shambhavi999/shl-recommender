# SHL Assessment Recommender

A conversational agent that helps hiring managers find the right SHL Individual Test Assessments through dialogue.

Built for the **SHL Labs AI Intern Take-Home Assignment**.

---

## Architecture

```
POST /chat
    │
    ├─ 1. GUARD        Reject prompt-injection attempts before calling LLM
    ├─ 2. EXTRACT      LLM call #1 (structured/small): extract slots from conversation
    ├─ 3. POLICY       Decide intent: clarify / recommend / refine / compare / refuse
    ├─ 4. RETRIEVE     Hybrid BM25 + dense (FAISS) search with RRF fusion
    ├─ 5. REPLY        LLM call #2: generate grounded reply from retrieved catalog docs
    ├─ 6. PARSE        Extract structured recommendations from reply text
    └─ 7. VALIDATE     Drop any URL not in the real scraped catalog (anti-hallucination)
```

**Why two LLM calls?**
- Call #1 is small and deterministic (JSON output, ~50 tokens). It reliably extracts slots.
- Call #2 gets pre-computed slots and pre-retrieved context injected into the prompt. It only has to write a good reply — not re-derive everything from scratch.
- Both calls are independently testable.

**Why hybrid retrieval?**
- BM25 excels at exact-match ("Java 8", "OPQ32r").
- Dense (all-MiniLM-L6-v2 + FAISS) finds semantic matches ("culture fit" → OPQ32r).
- Reciprocal Rank Fusion (RRF) merges both lists without score normalisation.

---

## Setup

### 1. Clone & install

```bash
git clone <your-repo-url>
cd shl-recommender
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY
# Get a free key at https://console.groq.com
```

### 3. Scrape the SHL catalog

```bash
python scripts/scrape_catalog.py --out data/catalog.json
# Takes ~5 minutes (polite crawl of ~380 pages)
# For quick testing: --limit 40 --skip-details
```

### 4. Run the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Test the health endpoint

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### 6. Test the chat endpoint

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I am hiring a mid-level Java developer who works with stakeholders"},
      {"role": "assistant", "content": "Sure. What is the seniority level?"},
      {"role": "user", "content": "Mid-level, around 4 years experience"}
    ]
  }'
```

---

## Evaluation

### Run unit tests

```bash
pytest tests/ -v
```

### Run the local eval harness

```bash
# Health check first
curl http://localhost:8000/health

# Run behaviour probes + trace replay
python eval/run_eval.py --endpoint http://localhost:8000 --traces eval/traces/ -v

# Behaviour probes only (fast)
python eval/run_eval.py --endpoint http://localhost:8000 --probes-only
```

Download the 10 public traces from the assignment link and place them in `eval/traces/` before running full trace evaluation.

---

## Deployment

### Render (recommended — free tier supports cold starts ≤ 2 min)

1. Push to GitHub.
2. New Web Service → connect repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
5. Add env vars: `GROQ_API_KEY`, `LLM_MODEL`, `CATALOG_PATH`.
6. Upload `data/catalog.json` as a persistent disk mount or bake it into the Docker image.

### Docker

```bash
# Build catalog first
python scripts/scrape_catalog.py --out data/catalog.json

# Build image
docker build -t shl-recommender .

# Run
docker run -p 8000:8000 \
  -e GROQ_API_KEY=your_key \
  shl-recommender
```

---

## API Reference

### `GET /health`

```json
{"status": "ok"}
```

### `POST /chat`

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I am hiring a Java developer"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "Mid-level, 4 years"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are 5 assessments for a mid-level Java developer...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

`recommendations` is empty when the agent is clarifying or refusing.
`end_of_conversation` is `true` when the agent has delivered its final shortlist.

---

## Project Structure

```
shl-recommender/
├── app/
│   ├── main.py        FastAPI app, endpoints, schema models
│   ├── agent.py       Core agent pipeline (guard → extract → retrieve → reply → validate)
│   ├── catalog.py     Catalog loader + HybridRetriever (BM25 + FAISS + RRF)
│   ├── llm.py         Groq client wrapper with retry logic
│   ├── prompts.py     All prompt templates (slot extraction + agent system prompt)
│   └── config.py      Centralised settings
├── scripts/
│   └── scrape_catalog.py  Full catalog scraper (listing + detail pages)
├── data/
│   └── catalog_seed.json  Seed data (subset; run scraper for full catalog)
├── eval/
│   ├── run_eval.py    Local evaluation harness (Recall@10 + behaviour probes)
│   └── traces/        Conversation traces for evaluation
├── tests/
│   └── test_agent.py  pytest unit tests
├── Dockerfile
├── requirements.txt
└── .env.example
```
