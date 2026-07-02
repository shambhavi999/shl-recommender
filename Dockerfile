FROM python:3.11-slim

# Install build tools (needed for faiss-cpu / numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so cold starts are fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy source
COPY . .

# Ensure catalog exists (CI will have run the scraper before building the image)
RUN test -f data/catalog.json || (echo "ERROR: data/catalog.json missing. Run: python scripts/scrape_catalog.py --out data/catalog.json" && exit 1)

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
