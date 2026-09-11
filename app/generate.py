import os
import re
from typing import List

from dotenv import load_dotenv
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .guardrails import DISCLAIMER, REFUSAL_TEXT, check_output
from .models import Citation, GroundedAnswer, RetrievalResult, Verdict
from .retrieve import normalize_query, rerank_and_confidence, tokenize

load_dotenv()

REFUSAL_THRESHOLD = 0.12
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = (
    "You answer ONLY from the provided MOHFW Standard Treatment Guideline chunks. "
    "This is for general education, never personal medical advice, diagnosis, or dosage. "
    "Rules: every factual sentence must end with a citation marker like [chunk-id] "
    "copied exactly from the chunk list. Use no other brackets. If the chunks lack "
    "the answer, reply exactly: I don't have enough information in the guidelines to answer this."
)


def _use_llm() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _build_citations(top: List[RetrievalResult]) -> List[Citation]:
    out: List[Citation] = []
    for r in top:
        src = r.chunk.source_url or r.chunk.source_filename
        out.append(
            Citation(
                chunk_id=r.chunk.chunk_id,
                doc_id=r.chunk.doc_id,
                title=r.chunk.title,
                source=src,
                section=r.chunk.section,
            )
        )
    return out


def _context_block(top: List[RetrievalResult]) -> str:
    parts = []
    for r in top:
        parts.append(f"[{r.chunk.chunk_id}] ({r.chunk.title}) {r.chunk.text[:1200]}")
    return "\n\n".join(parts)


def _llm_answer(query: str, top: List[RetrievalResult], confidence: float) -> GroundedAnswer:
    from openai import OpenAI

    citations = _build_citations(top)
    valid_ids = {c.chunk_id for c in citations}
    client = OpenAI()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {query}\n\nGuideline chunks:\n{_context_block(top)}",
            },
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    markers = re.findall(r"\[([^\]]+)\]", text)
    if not text or not any(m.strip() in valid_ids for m in markers):
        raise ValueError("LLM output missing valid citation markers")
    if not check_output(text):
        raise ValueError("LLM output failed safety check")
    return GroundedAnswer(
        answer=text,
        citations=citations,
        confidence=float(confidence),
        verdict="ANSWER",
        disclaimer=DISCLAIMER,
    )


# --- LEGACY extractive fallback (no API key / LLM failure) ---
# To be REMOVED in Production RAG once embeddings + LLM generation are mandatory.


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\[\(])", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def _query_terms(query: str) -> List[str]:
    return [
        t for t in set(tokenize(normalize_query(query)))
        if t not in ENGLISH_STOP_WORDS and len(t) > 2
    ]


def _pick_sentence(text: str, terms: List[str]) -> str:
    sents = _sentences(text)
    long_enough = [s for s in sents if len(s) >= 40]
    pool = long_enough or sents
    if terms and pool:
        topical = [s for s in pool if any(t in s.lower() for t in terms)]
        ranked = sorted(
            topical,
            key=lambda s: (sum(1 for t in terms if t in s.lower()), "http" not in s.lower(), -sents.index(s)),
            reverse=True,
        )
        if ranked:
            return ranked[0][:350]
    for s in pool:
        return s[:350]
    return (text or "").strip()[:350]


def _cite_sentence(sentence: str, chunk_id: str) -> str:
    s = sentence.strip()
    if not s:
        return ""
    if not re.search(r"[.!?]$", s):
        s += "."
    return f"{s} [{chunk_id}]"


def _extractive_answer(query: str, top: List[RetrievalResult], confidence: float) -> GroundedAnswer:
    citations = _build_citations(top)
    sentences: List[str] = []
    terms = _query_terms(query)
    for r in top:
        cited = _cite_sentence(_pick_sentence(r.chunk.text, terms), r.chunk.chunk_id)
        if cited:
            sentences.append(cited)
    sentences = sentences[:4]
    if confidence < 0.35 and top:
        sentences.append(
            f"Coverage of this point in the cited chunks is partial [{top[0].chunk.chunk_id}]"
        )
    return GroundedAnswer(
        answer=" ".join(sentences),
        citations=citations,
        confidence=float(confidence),
        verdict="ANSWER",
        disclaimer=DISCLAIMER,
    )


def _refusal(key: str, confidence: float = 0.0) -> GroundedAnswer:
    key = key if key in REFUSAL_TEXT else "REFUSE_LOW_CONFIDENCE"
    return GroundedAnswer(
        answer=REFUSAL_TEXT[key],
        citations=[],
        confidence=float(confidence),
        verdict=key,  # type: ignore[typeddict-item]
        disclaimer=DISCLAIMER,
    )


def compose(query: str, results: List[RetrievalResult], verdict: Verdict = "PROCEED") -> GroundedAnswer:
    if verdict != "PROCEED":
        return _refusal(verdict)
    if not results:
        return _refusal("REFUSE_LOW_CONFIDENCE")
    ranked, confidence = rerank_and_confidence(results, query)
    if confidence < REFUSAL_THRESHOLD:
        return _refusal("REFUSE_LOW_CONFIDENCE", float(confidence))
    top = ranked[:3]
    if _use_llm():
        try:
            return _llm_answer(query, top, float(confidence))
        except Exception:
            pass
    return _extractive_answer(query, top, float(confidence))
