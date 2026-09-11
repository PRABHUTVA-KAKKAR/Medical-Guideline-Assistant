from pathlib import Path
from typing import Any, Dict, List, Optional
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .generate import compose
from .guardrails import classify_input
from .models import GroundedAnswer
from .retrieve import dense_search, hybrid_search, load_index

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "index"

app = FastAPI(title="RAG Medical Guideline Assistant")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    k: Optional[int] = 4


_IDX = None


def _get_index():
    global _IDX
    if _IDX is None:
        _IDX = load_index(str(INDEX_DIR))
    return _IDX


def _index_manifest() -> Dict[str, Any]:
    try:
        return _get_index().manifest
    except Exception:
        return {"version": "0", "chunk_count": 0, "doc_count": 0}


@app.get("/api/health")
def health() -> Dict[str, Any]:
    m = _index_manifest()
    return {
        "status": "ok",
        "index_version": str(m.get("version", "0")),
        "chunk_count": int(m.get("chunk_count", 0)),
        "doc_count": int(m.get("doc_count", 0)),
    }


@app.get("/api/docs-list")
def docs_list() -> List[Dict[str, Any]]:
    m = _index_manifest()
    docs = m.get("docs", [])
    return docs if isinstance(docs, list) else []


@app.post("/api/query", response_model=GroundedAnswer)
def query(req: QueryRequest) -> GroundedAnswer:
    verdict = classify_input(req.query)
    k = req.k if req.k and 1 <= req.k <= 20 else 4
    if verdict != "PROCEED":
        return compose(req.query, [], verdict)
    backend = "keyword"
    try:
        if os.getenv("RETRIEVAL_BACKEND", "dense") == "dense":
            results = dense_search(req.query, k=k)
            backend = "dense"
        else:
            results = hybrid_search(req.query, k=k, index=_get_index())
    except Exception:
        results = hybrid_search(req.query, k=k, index=_get_index())
    return compose(req.query, results, "PROCEED", backend=backend)


_frontend = ROOT / "frontend"
if _frontend.is_dir():
    try:
        app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="frontend")
    except Exception:
        pass
