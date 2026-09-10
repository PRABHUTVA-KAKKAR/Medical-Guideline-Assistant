import argparse
import csv
import json
import pickle
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from .models import LICENSE_DEFAULT, Chunk, DocManifestEntry

CHUNK_SIZE = 1000
OVERLAP = 150
MIN_PAGE_CHARS = 50
EMBEDDING_NAME = "tfidf-charword-hybrid"
INDEX_VERSION = "1"


def _trailing_key(name: str) -> str:
    m = re.search(r"(\d+)(?:\.pdf)?$", name, re.IGNORECASE)
    return m.group(1) if m else name.lower()


def load_url_map(pdf_dir: Path) -> Dict[str, Tuple[str, str]]:
    candidates = [pdf_dir / "stg_urls.csv", Path("Data") / "stg_urls.csv"]
    csv_path = next((p for p in candidates if p.exists()), None)
    mapping: Dict[str, Tuple[str, str]] = {}
    if csv_path is None:
        return mapping
    with open(csv_path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = (row.get("title") or "").strip()
            url = (row.get("url") or "").strip()
            if not url:
                continue
            key = _trailing_key(url.split("/")[-1])
            mapping[key] = (title, url)
    return mapping


def clean_text(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw or "")
    # PDFs often emit private-use glyphs for bullets and symbols. Drop them
    # plus control chars so chunks hold plain readable text.
    text = "".join(ch for ch in text if unicodedata.category(ch) not in ("Co", "Cc", "Cf", "Cs"))
    return re.sub(r"\s+", " ", text).strip()


def extract_section(raw_page_text: str) -> str:
    for line in (raw_page_text or "").splitlines():
        s = line.strip()
        if len(s) >= 3:
            return s[:120]
    return ""


def chunk_page_text(text: str) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    n = len(text)
    start = 0
    while start < n:
        if start > 0:
            # Never start mid-word. Chunk starts otherwise slice terms in
            # half and answers show fragments.
            gap = text.find(" ", start, start + 40)
            if gap != -1:
                start = gap + 1
        if n - start <= CHUNK_SIZE:
            piece = text[start:].strip()
            if len(piece) >= MIN_PAGE_CHARS:
                out.append((piece, start))
            break
        window = text[start : start + CHUNK_SIZE]
        cut = window.rfind(" ")
        if cut > 700:
            end = start + cut
        else:
            end = start + CHUNK_SIZE
        piece = text[start:end].strip()
        if len(piece) >= MIN_PAGE_CHARS:
            out.append((piece, start))
        nxt = end - OVERLAP
        start = nxt + 1 if nxt <= start else nxt
        while start < n and text[start] == " ":
            start += 1
    return out


def ingest_pdf(path: Path, url_map: Dict[str, Tuple[str, str]]) -> Tuple[DocManifestEntry, List[Chunk]]:
    doc_id = path.stem
    key = _trailing_key(path.name)
    title, source_url = url_map.get(key, ("", ""))
    if not title:
        title = re.sub(r"[-_]+", " ", doc_id).strip()
    reader = PdfReader(str(path))
    pages = len(reader.pages)
    chunks: List[Chunk] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        section = extract_section(raw)
        cleaned = clean_text(raw)
        if len(cleaned) < MIN_PAGE_CHARS:
            continue
        for idx, (piece, cstart) in enumerate(chunk_page_text(cleaned)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#p{i}-c{idx}",
                    doc_id=doc_id,
                    title=title,
                    source_filename=path.name,
                    source_url=source_url,
                    version="",
                    license=LICENSE_DEFAULT,
                    section=section,
                    text=piece,
                    char_start=cstart,
                )
            )
    entry = DocManifestEntry(
        doc_id=doc_id,
        title=title,
        filename=path.name,
        source_url=source_url,
        version="",
        license=LICENSE_DEFAULT,
        pages=pages,
        num_chunks=len(chunks),
    )
    return entry, chunks


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def run(pdf_dir: Path, out_dir: Path, limit: int | None = None) -> None:
    pdf_dir = Path(pdf_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    url_map = load_url_map(pdf_dir)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if limit is not None:
        pdfs = pdfs[:limit]
    all_chunks: List[Chunk] = []
    docs: List[DocManifestEntry] = []
    for pdf in pdfs:
        entry, chunks = ingest_pdf(pdf, url_map)
        docs.append(entry)
        all_chunks.extend(chunks)
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(c.model_dump_json() + "\n")
    manifest = {
        "version": INDEX_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding": EMBEDDING_NAME,
        "chunk_count": len(all_chunks),
        "doc_count": len(docs),
        "files": ["chunks.jsonl", "manifest.json", "tfidf.joblib", "bm25.pkl"],
        "docs": [d.model_dump() for d in docs],
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    corpus = [c.text for c in all_chunks]
    if corpus:
        vec = TfidfVectorizer(
            ngram_range=(1, 2), max_features=50000, sublinear_tf=True, lowercase=True, stop_words="english"
        )
        matrix = vec.fit_transform(corpus)
        joblib.dump({"vectorizer": vec, "matrix": matrix}, out_dir / "tfidf.joblib")
        tokenized = [tokenize(t) for t in corpus]
        bm25 = BM25Okapi(tokenized)
        with open(out_dir / "bm25.pkl", "wb") as f:
            pickle.dump({"tokens": tokenized, "bm25": bm25}, f)
    else:
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)
        joblib.dump({"vectorizer": vec, "matrix": vec.fit_transform([]) if False else None}, out_dir / "tfidf.joblib")
        with open(out_dir / "bm25.pkl", "wb") as f:
            pickle.dump({"tokens": [], "bm25": BM25Okapi([[]])}, f)
    print(f"docs={len(docs)} chunks={len(all_chunks)} out={out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default="Data")
    ap.add_argument("--out", default="index")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(Path(args.pdf_dir), Path(args.out), args.limit)


if __name__ == "__main__":
    main()
