from typing import List, Literal
from pydantic import BaseModel

ChunkId = str
DocId = str

Verdict = Literal[
    "PROCEED",
    "ANSWER",
    "REFUSE_OUT_OF_SCOPE",
    "REFUSE_PERSONALIZED",
    "REFUSE_EMERGENCY",
    "REFUSE_LOW_CONFIDENCE",
]

LICENSE_DEFAULT = "MOHFW STG, government guideline"


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    source_filename: str
    source_url: str = ""
    version: str = ""
    license: str = LICENSE_DEFAULT
    section: str = ""
    text: str
    char_start: int = 0


class DocManifestEntry(BaseModel):
    doc_id: str
    title: str
    filename: str
    source_url: str = ""
    version: str = ""
    license: str = LICENSE_DEFAULT
    pages: int = 0
    num_chunks: int = 0


class IndexManifest(BaseModel):
    version: str
    created_at: str
    embedding: str = "tfidf-charword-hybrid"
    chunk_count: int = 0
    doc_count: int = 0
    files: List[str] = []


class RetrievalResult(BaseModel):
    chunk: Chunk
    dense_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    source: str = ""
    section: str = ""


class GroundedAnswer(BaseModel):
    answer: str
    citations: List[Citation] = []
    confidence: float = 0.0
    verdict: Verdict = "ANSWER"
    disclaimer: str = ""
