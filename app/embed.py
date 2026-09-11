"""Single home for OpenAI embeddings (Normal RAG, per architecture diagram).

Rule from the diagram: chunk embeddings and the query embedding MUST use the
same model, otherwise cosine scores are meaningless. Both paths below read
OPENAI_EMBED_MODEL, so they cannot drift apart.
"""

import os
import time
from typing import List

from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
EMBED_BATCH = 100
EMBED_RETRIES = 4


def _client():
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Paste it into .env first.")
    return OpenAI()


def embed_texts(texts: List[str], batch: int = EMBED_BATCH) -> List[List[float]]:
    """Embed many texts in batches with retry. Returns one vector per input."""
    client = _client()
    out: List[List[float]] = []
    for start in range(0, len(texts), batch):
        piece = texts[start : start + batch]
        for attempt in range(EMBED_RETRIES):
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=piece)
                out.extend([d.embedding for d in resp.data])
                break
            except Exception:
                if attempt == EMBED_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)
        print(f"embedded {min(start + batch, len(texts))}/{len(texts)}", flush=True)
    return out


def embed_query(query: str) -> List[float]:
    """Embed a single user query with the same model (the diagram's TARGET)."""
    return embed_texts([query], batch=1)[0]
