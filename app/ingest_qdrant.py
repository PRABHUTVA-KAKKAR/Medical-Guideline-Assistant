"""Ingestion pipeline v2 (Normal RAG): chunks -> embeddings -> Qdrant.

Reuses index/chunks.jsonl byte-for-byte (stable chunk ids + metadata), embeds
with OPENAI_EMBED_MODEL, upserts to Qdrant with cosine distance. Idempotent:
reruns skip everything when the collection already holds all chunks.

Usage:  python -m app.ingest_qdrant [--limit N]
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _cfg():
    return (
        os.getenv("QDRANT_URL", "http://localhost:6333"),
        os.getenv("QDRANT_COLLECTION", "mohfw_guidelines"),
    )


def load_chunks(limit=None):
    from .models import Chunk

    chunks = []
    with open(ROOT / "index" / "chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(Chunk.model_validate_json(line))
    return chunks if limit is None else chunks[:limit]


def main() -> None:
    from .embed import embed_texts

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    url, collection = _cfg()
    chunks = load_chunks(args.limit)
    print(f"chunks={len(chunks)} qdrant={url} collection={collection}")

    client = QdrantClient(url=url)
    names = [c.name for c in client.get_collections().collections]
    if collection in names and client.count(collection).count == len(chunks):
        print("collection already complete, nothing to do")
        return

    vectors = embed_texts([c.text for c in chunks])
    dim = len(vectors[0])
    if collection in names:
        client.delete_collection(collection)
    client.create_collection(
        collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )
    step = 256
    for start in range(0, len(chunks), step):
        points = []
        for i in range(start, min(start + step, len(chunks))):
            c = chunks[i]
            points.append(
                PointStruct(
                    id=i,
                    vector=vectors[i],
                    payload={
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "title": c.title,
                        "section": c.section,
                        "source": c.source_url or c.source_filename,
                        "text": c.text,
                    },
                )
            )
        client.upsert(collection, points)
        print(f"upserted {min(start + step, len(chunks))}/{len(chunks)}", flush=True)
    print(f"done count={client.count(collection).count}")


if __name__ == "__main__":
    main()
