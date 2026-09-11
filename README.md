# RAG Medical Guideline Assistant

Grounded question answering over 61 MOHFW Standard Treatment Guideline PDFs. Retrieval feeds cited chunks to an OpenAI chat model; every factual sentence carries a `[chunk_id]` citation. Branch `main` holds the original keyword-only snapshot; `feat/normal-rag` adds dense retrieval (OpenAI embeddings + Qdrant cosine search).

## How it works

Ingestion (offline): PDFs → chunking (`app/ingest.py`, 1000 chars / 150 overlap, stable `#p-c` ids) → one embedding per chunk (`app/embed.py`, `text-embedding-3-small`) → Qdrant collection with metadata payload (`chunk_id`, `doc_id`, `title`, `section`, `source`, `text`).

Retrieval (online): query → query embedding with the same model → Qdrant cosine Top-K (`dense_search` in `app/retrieve.py`) → LLM writes the answer from those chunks only (`app/generate.py`), refusals via `app/guardrails.py`. File map: `app/api.py` (FastAPI + UI hosting), `app/evaluate.py` (12-query benchmark), `frontend/` (demo page).

## Install

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env   # then paste OPENAI_API_KEY into .env
```

## Build the index

```powershell
python -m app.ingest --limit 3
python -m app.ingest
```

The first command is a smoke run. The second indexes all of `Data/*.pdf`. Output goes to `index/`. Reruns overwrite the same files with stable chunk ids.

## Dense retrieval (Normal RAG, `feat/normal-rag`)

```powershell
docker run -p 6333:6333 -v ${PWD}/qdrant_data:/qdrant/storage qdrant/qdrant
python -m app.ingest_qdrant --limit 3   # smoke: embed 3 chunks, upsert, verify count
python -m app.ingest_qdrant             # full: embed 9218 chunks with text-embedding-3-small, upsert to Qdrant
```

Env vars (see `.env.example`):

| Var | Purpose |
|---|---|
| `OPENAI_API_KEY` | Secret, never committed (`.env` is gitignored) |
| `OPENAI_MODEL` | Chat model for answers (`gpt-4o-mini`) |
| `OPENAI_EMBED_MODEL` | Single embedding model for chunks AND queries (`text-embedding-3-small`) |
| `QDRANT_URL` / `QDRANT_COLLECTION` | Vector DB location (`http://localhost:6333`, `mohfw_guidelines`) |
| `RETRIEVAL_BACKEND` | `dense` (Qdrant cosine, falls back to keyword if Qdrant is down) or `keyword` (local TF-IDF+BM25) |

Reruns are idempotent and skip when the collection already holds all chunks. Qdrant dashboard: `http://localhost:6333/dashboard`.

## Run the API

```powershell
uvicorn app.api:app --port 8000
```

## Query example

```powershell
curl -Method POST http://localhost:8000/api/query -ContentType "application/json" -Body '{"query": "What is the diagnosis and treatment of malaria per guidelines?", "k": 8}'
```

Response shape:

```json
{
  "answer": "The diagnosis of malaria involves using either Rapid Diagnostic Tests (RDT) or microscopy ... [51-ii-diagnosis-and-treatment-of-malaria-892#p2-c2]",
  "citations": [{"chunk_id": "51-ii-diagnosis-and-treatment-of-malaria-892#p2-c2", "doc_id": "51-ii-diagnosis-and-treatment-of-malaria-892", "title": "(ii) Diagnosis and treatment of Malaria", "source": "https://clinicalestablishments.mohfw.gov.in/.../892.pdf", "section": "..."}],
  "confidence": 0.789,
  "verdict": "ANSWER",
  "disclaimer": "Information from MOHFW Standard Treatment Guidelines for education only. Consult a qualified clinician for personal advice."
}
```

Other endpoints: `GET /api/health`, `GET /api/docs-list`. A minimal demo page is served at `/` from `frontend/`.

## Troubleshooting

- `RETRIEVAL_BACKEND=dense` but Qdrant is down: the API auto-falls-back to keyword search, so queries still work.
- Fresh machine: run `app.ingest` first (builds `index/`), then `app.ingest_qdrant` (fills Qdrant). Without Qdrant data, dense mode falls back until you ingest.
- Low confidence (~0.5) on a correct answer: usually meta-words in the query ("what does the document say") that never appear in chunks. Confidence blends lexical overlap with cosine; see Thresholds below.

## Evaluate

```powershell
python -m app.evaluate
```

Runs 12 benchmark queries and writes `eval/report.json`. Current scores: recall 1.0, refusal accuracy 1.0, faithfulness 1.0, safety compliance 1.0.

## Tests

```powershell
pytest -q
```

## Thresholds

`REFUSAL_THRESHOLD = 0.30` in `app/generate.py`. Keyword confidence is squared IDF weighted query term coverage of the top reranked chunk. Dense confidence blends that with rescaled Qdrant cosine (`_dense_confidence` in `app/retrieve.py`, anchors 0.40/0.70 fitted on a small probe set). Scores below the threshold return `REFUSE_LOW_CONFIDENCE` with the text `I don't have enough information in the guidelines to answer this.`

## Scope and safety boundaries

Input verdicts: `REFUSE_EMERGENCY` beats `REFUSE_PERSONALIZED` beats `REFUSE_OUT_OF_SCOPE`, else `PROCEED`. The table lives in `app/guardrails.py` as `INTENT_REGISTRY`. Output is checked against `SAFETY_PATTERNS`.

The assistant refuses emergencies, personal diagnosis and dosage requests, and topics outside the corpus. Refusals carry empty citations. All answers carry the disclaimer. This tool is for education only. It is not a substitute for a clinician.

## Corpus manifest

Source PDFs live in `Data/`. Titles and source URLs come from `Data/stg_urls.csv`. The built index holds 61 docs and 9218 chunks. Files: `index/chunks.jsonl`, `index/manifest.json`, `index/tfidf.joblib`, `index/bm25.pkl` (keyword fallback). Dense vectors live in Qdrant (`mohfw_guidelines` collection, cosine distance, `text-embedding-3-small`).

## Open decisions

Docs 22 to 41 are titled `a) Full Document` in the source CSV, so citations for those show the generic title plus the exact doc id. Doc 24 is the hypertension full document. Abbreviation expansion covers HTN, DM, TB, MI, COPD, ARI, UTI, and BP only.
