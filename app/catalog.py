"""
app/catalog.py

Loads the SHL catalog and builds the hybrid retrieval engine.

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │                   HybridRetriever                   │
  │                                                     │
  │  query ──► BM25 (keyword)  ──► ranked list         │
  │        └─► Dense (FAISS)   ──► ranked list         │
  │                        └──► RRF fusion ──► top-K   │
  └─────────────────────────────────────────────────────┘

Why hybrid?
  – BM25 excels at exact-term queries ("Java 8", "OPQ32r").
  – Dense retrieval finds semantically similar assessments even
    when the user phrase doesn't match the catalog name verbatim
    ("check if someone is detail-oriented" → personality tests).
  – Reciprocal Rank Fusion (RRF) combines both lists without
    needing any score normalisation.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import get_settings

# ── lazy imports so the module loads even without optional deps ──────────────
try:
    from rank_bm25 import BM25Okapi  # type: ignore
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

try:
    import faiss  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_DENSE = True
except ImportError:
    _HAS_DENSE = False

# ── type aliases ────────────────────────────────────────────────────────────
Assessment = dict  # {"id", "name", "url", "test_type": list[str], ...}
TEST_TYPE_FULL = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def _tokenise(text: str) -> list[str]:
    """Lower-case, split on non-alphanumeric, return tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_document_text(a: Assessment) -> str:
    """Concatenate all fields into a single searchable string."""
    parts = [a.get("name", "")]
    if a.get("description"):
        parts.append(a["description"])
    for letter in a.get("test_type", []):
        parts.append(TEST_TYPE_FULL.get(letter, letter))
    parts.extend(a.get("job_levels", []))
    return " ".join(filter(None, parts))


class HybridRetriever:
    """
    Singleton retriever — call ``get_retriever()`` instead of instantiating
    directly to share state across requests.
    """

    def __init__(self, catalog: list[Assessment], settings) -> None:
        self._catalog = catalog
        self._s = settings
        self._docs = [_build_document_text(a) for a in catalog]

        # ── BM25 ────────────────────────────────────────────────────────
        # Guard against empty corpus (causes ZeroDivisionError in rank_bm25)
        self._bm25: Optional[object] = None
        self._bm25_idx_map: list[int] = []

        if _HAS_BM25 and self._docs:
            tokenised = [_tokenise(d) for d in self._docs]
            # Only keep docs that produce at least one token
            non_empty_pairs = [(i, t) for i, t in enumerate(tokenised) if t]
            if non_empty_pairs:
                self._bm25_idx_map = [i for i, _ in non_empty_pairs]
                self._bm25 = BM25Okapi([t for _, t in non_empty_pairs])
            else:
                print("[catalog] WARNING: all documents tokenised to empty — BM25 disabled.", flush=True)

        # ── Dense / FAISS ───────────────────────────────────────────────
        self._encoder: Optional[object] = None
        self._index: Optional[object] = None
        if _HAS_DENSE and self._docs:
            try:
                self._encoder = SentenceTransformer(settings.EMBED_MODEL)
                embeddings = self._encoder.encode(
                    self._docs, normalize_embeddings=True, show_progress_bar=False
                )
                dim = embeddings.shape[1]
                self._index = faiss.IndexFlatIP(dim)  # inner-product on L2-normalised = cosine
                self._index.add(embeddings.astype("float32"))
            except Exception as exc:
                print(f"[catalog] Dense index failed to build: {exc}; falling back to BM25-only.", flush=True)
                self._encoder = None
                self._index = None

    # ── public API ───────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filter_test_types: list[str] | None = None,
    ) -> list[Assessment]:
        """
        Hybrid search: BM25 + dense cosine, fused with RRF.
        Returns up to ``top_k`` assessments.

        Parameters
        ----------
        query: str
            Natural-language query derived from the conversation.
        top_k: int | None
            Number of results to return (defaults to settings.RETRIEVAL_TOP_K).
        filter_test_types: list[str] | None
            If set, only return assessments whose test_type overlaps this list,
            e.g. ["P"] to restrict to Personality tests.
        """
        top_k = top_k or self._s.RETRIEVAL_TOP_K
        query = query.strip()
        if not query:
            return []

        n = len(self._catalog)
        if n == 0:
            return []

        candidates = min(top_k * 3, n)  # retrieve more, then filter & fuse

        bm25_ranks: dict[int, int] = {}
        dense_ranks: dict[int, int] = {}

        # ── BM25 ────────────────────────────────────────────────────────
        if self._bm25 and self._bm25_idx_map:
            tokens = _tokenise(query)
            if tokens:  # don't score empty token list
                scores = self._bm25.get_scores(tokens)
                ranked = np.argsort(scores)[::-1][:candidates]
                for rank, bm25_pos in enumerate(ranked):
                    bm25_pos = int(bm25_pos)
                    if bm25_pos < len(self._bm25_idx_map):
                        catalog_idx = self._bm25_idx_map[bm25_pos]
                        bm25_ranks[catalog_idx] = rank + 1

        # ── Dense ───────────────────────────────────────────────────────
        if self._encoder and self._index:
            q_emb = self._encoder.encode([query], normalize_embeddings=True)
            _, indices = self._index.search(q_emb.astype("float32"), candidates)
            for rank, idx in enumerate(indices[0]):
                if idx >= 0:
                    dense_ranks[int(idx)] = rank + 1

        # ── RRF fusion ──────────────────────────────────────────────────
        all_ids = set(bm25_ranks) | set(dense_ranks)

        # If neither retriever returned anything, fall back to first N catalog items
        if not all_ids:
            return self._catalog[:top_k]

        k = self._s.RRF_K
        w_bm25 = self._s.BM25_WEIGHT
        w_dense = 1.0 - w_bm25

        def rrf_score(idx: int) -> float:
            s = 0.0
            if idx in bm25_ranks:
                s += w_bm25 / (k + bm25_ranks[idx])
            if idx in dense_ranks:
                s += w_dense / (k + dense_ranks[idx])
            return s

        ranked_all = sorted(all_ids, key=rrf_score, reverse=True)

        # ── Optional test-type filter ────────────────────────────────────
        results = []
        for idx in ranked_all:
            if idx >= len(self._catalog):
                continue
            a = self._catalog[idx]
            if filter_test_types:
                if not any(t in a.get("test_type", []) for t in filter_test_types):
                    continue
            results.append(a)
            if len(results) >= top_k:
                break

        return results

    def get_by_name(self, name: str) -> Optional[Assessment]:
        """Exact + normalised-name lookup (used for compare mode)."""
        name_lower = name.lower().strip()
        for a in self._catalog:
            if a["name"].lower() == name_lower:
                return a
        # Fuzzy: any catalog name that contains the query as a substring
        for a in self._catalog:
            if name_lower in a["name"].lower():
                return a
        return None

    @property
    def catalog(self) -> list[Assessment]:
        return self._catalog

    def get_catalog_urls(self) -> set[str]:
        return {a["url"] for a in self._catalog}


# ── module-level singleton ───────────────────────────────────────────────────
_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = _build_retriever()
    return _retriever


def _build_retriever() -> HybridRetriever:
    settings = get_settings()

    # Resolve path — works on both Windows and Linux
    raw_path = settings.CATALOG_PATH
    path = Path(raw_path)
    if not path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        path = project_root / raw_path

    catalog: list[Assessment] = []

    if path.exists():
        with open(path, encoding="utf-8") as f:
            catalog = json.load(f)
        print(f"[catalog] Loaded catalog from {path}", flush=True)
    else:
        # Fallback: load the seed file shipped with the repo
        project_root = Path(__file__).resolve().parent.parent
        seed = project_root / "data" / "catalog_seed.json"
        if seed.exists():
            with open(seed, encoding="utf-8") as f:
                catalog = json.load(f)
            print(
                f"[catalog] WARNING: {path} not found; using seed ({len(catalog)} items). "
                "Run: python scripts/scrape_catalog.py --out data/catalog.json",
                flush=True,
            )
        else:
            raise RuntimeError(
                f"No catalog found at {path} and no seed at {seed}.\n"
                "Run: python scripts/scrape_catalog.py --out data/catalog.json"
            )

    if not catalog:
        raise RuntimeError(
            "Catalog loaded but is empty. "
            "Check data/catalog_seed.json or re-run the scraper."
        )

    # Ensure every entry has required fields with safe defaults
    for i, a in enumerate(catalog, start=1):
        a.setdefault("id", f"shl-{i:04d}")
        a.setdefault("test_type", [])
        a.setdefault("description", "")
        a.setdefault("job_levels", [])
        a.setdefault("languages", [])

    print(f"[catalog] Loaded {len(catalog)} assessments; building indices…", flush=True)
    retriever = HybridRetriever(catalog, settings)
    print("[catalog] Indices ready.", flush=True)
    return retriever