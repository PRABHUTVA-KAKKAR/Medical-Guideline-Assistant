import json
import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .models import Chunk, RetrievalResult

load_dotenv()

ABBR = {
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "tb": "tuberculosis",
    "mi": "myocardial infarction",
    "copd": "chronic obstructive pulmonary disease",
    "ari": "acute respiratory infection",
    "uti": "urinary tract infection",
    "bp": "blood pressure",
}

_INDEX = None


@dataclass
class LoadedIndex:
    vectorizer: object
    matrix: object
    bm25: object
    chunks: List[Chunk]
    manifest: dict
    idf: Dict[str, float] = field(default_factory=dict)
    max_idf: float = 1.0


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def normalize_query(query: str) -> str:
    text = (query or "").lower()
    for abbr, full in ABBR.items():
        text = re.sub(rf"\b{re.escape(abbr)}\b", full, text)
    return re.sub(r"\s+", " ", text).strip()


def load_index(index_dir: str | Path = "index") -> LoadedIndex:
    global _INDEX
    d = Path(index_dir)
    chunks: List[Chunk] = []
    with open(d / "chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(Chunk.model_validate_json(line))
    with open(d / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    payload = joblib.load(d / "tfidf.joblib")
    vectorizer = payload["vectorizer"]
    matrix = payload["matrix"]
    with open(d / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)["bm25"]
    idf_map: Dict[str, float] = {}
    max_idf = 1.0
    try:
        vocab = vectorizer.vocabulary_
        idfs = vectorizer.idf_
        idf_map = {t: float(idfs[i]) for t, i in vocab.items()}
        max_idf = float(idfs.max()) if len(idfs) else 1.0
    except Exception:
        pass
    _INDEX = LoadedIndex(
        vectorizer=vectorizer, matrix=matrix, bm25=bm25,
        chunks=chunks, manifest=manifest, idf=idf_map, max_idf=max_idf,
    )
    return _INDEX


def _get_index(index=None) -> LoadedIndex:
    if index is not None:
        return index
    global _INDEX
    if _INDEX is None:
        _INDEX = load_index("index")
    return _INDEX


def _z_norm(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    std = float(a.std()) if a.size else 0.0
    if std < 1e-9:
        return np.zeros_like(a)
    return (a - float(a.mean())) / std


def hybrid_search(query: str, k: int = 8, index=None) -> List[RetrievalResult]:
    idx = _get_index(index)
    norm = normalize_query(query)
    q_vec = idx.vectorizer.transform([norm])
    dense = np.asarray((idx.matrix * q_vec.T).toarray()).ravel()
    scores = np.asarray(idx.bm25.get_scores(tokenize(norm)), dtype=float)
    n = min(len(idx.chunks), len(dense), len(scores))
    dense = dense[:n]
    scores = scores[:n]
    fused = 0.6 * _z_norm(dense) + 0.4 * _z_norm(scores)
    order = np.argsort(-fused)[: max(0, k)]
    out: List[RetrievalResult] = []
    for j in order:
        j = int(j)
        out.append(
            RetrievalResult(
                chunk=idx.chunks[j],
                dense_score=float(dense[j]),
                bm25_score=float(scores[j]),
                fused_score=float(fused[j]),
            )
        )
    return out


def _idf_weight(token: str, idf: Dict[str, float], max_idf: float) -> float:
    if token in ENGLISH_STOP_WORDS or len(token) < 2:
        return 0.0
    return idf.get(token, max_idf)


def _coverage(query_tokens: List[str], chunk_tokens: List[str], idf, max_idf) -> float:
    if not query_tokens:
        return 0.0
    cset = set(chunk_tokens)
    num = 0.0
    den = 0.0
    for t in set(query_tokens):
        w = _idf_weight(t, idf, max_idf)
        den += w
        if t in cset:
            num += w
    if den <= 0.0:
        return 0.0
    return num / den


def rerank_and_confidence(results, query="") -> Tuple[List[RetrievalResult], float]:
    if isinstance(results, str) and isinstance(query, list):
        results, query = query, results
    res = list(results or [])
    if not res:
        return [], 0.0
    q = normalize_query(query if isinstance(query, str) else "")
    qtok = tokenize(q)
    idx = _INDEX
    idf = idx.idf if idx is not None else {}
    max_idf = idx.max_idf if idx is not None else 1.0
    scored = []
    for r in res:
        lex = _coverage(qtok, tokenize(r.chunk.text), idf, max_idf)
        scored.append((r, lex))
    fused_vals = np.array([r.fused_score for r, _ in scored], dtype=float)
    lo, hi = float(fused_vals.min()), float(fused_vals.max())
    span = hi - lo
    ranked = []
    for r, lex in scored:
        fn = 0.5 if span < 1e-9 else (r.fused_score - lo) / span
        combined = 0.2 * fn + 0.8 * lex
        ranked.append((combined, lex, r))
    ranked.sort(key=lambda t: t[0], reverse=True)
    ordered = [r for _, _, r in ranked]
    # Square coverage so isolated single-term matches score low. Large
    # corpora match generic query terms by chance, so partial overlap alone
    # must not read as grounded.
    confidence = float(max(0.0, min(1.0, ranked[0][1] ** 2)))
    return ordered, confidence


_QDRANT = None


def _qdrant():
    global _QDRANT
    if _QDRANT is None:
        from qdrant_client import QdrantClient

        _QDRANT = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    return _QDRANT


def dense_search(query: str, k: int = 8, collection: str | None = None) -> List[RetrievalResult]:
    """Dense retrieval (diagram's retrieval pipeline): query embedding ->
    Qdrant cosine Top-K -> same RetrievalResult shape as hybrid_search, so
    generate/guardrails/API need no changes. Confidence gating stays lexical
    (conservative); Production RAG can learn a cosine-aware rerank later."""
    from .embed import embed_query

    coll = collection or os.getenv("QDRANT_COLLECTION", "mohfw_guidelines")
    vec = embed_query(normalize_query(query))
    client = _qdrant()
    try:
        hits = client.query_points(coll, query=vec, limit=k).points
        scored = [(h.payload, float(h.score)) for h in hits]
    except (AttributeError, TypeError):
        hits = client.search(coll, query_vector=vec, limit=k)
        scored = [(h.payload, float(h.score)) for h in hits]
    out: List[RetrievalResult] = []
    for payload, score in scored:
        payload = payload or {}
        src = payload.get("source", "")
        out.append(
            RetrievalResult(
                chunk=Chunk(
                    chunk_id=payload.get("chunk_id", ""),
                    doc_id=payload.get("doc_id", ""),
                    title=payload.get("title", ""),
                    source_filename="" if str(src).startswith("http") else str(src),
                    source_url=str(src) if str(src).startswith("http") else "",
                    section=payload.get("section", ""),
                    text=payload.get("text", ""),
                ),
                dense_score=score,
                bm25_score=0.0,
                fused_score=score,
            )
        )
    return out


def search(query: str, k: int = 8, index=None) -> List[RetrievalResult]:
    """Dispatcher: dense Qdrant cosine when RETRIEVAL_BACKEND=dense,
    else legacy keyword hybrid. Dense failures fall back to keyword so the
    API never breaks when Qdrant is down or un-ingested."""
    if os.getenv("RETRIEVAL_BACKEND", "dense") == "dense":
        try:
            return dense_search(query, k=k)
        except Exception:
            pass
    return hybrid_search(query, k=k, index=index)
