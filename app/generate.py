import re
from typing import List
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from .guardrails import DISCLAIMER, REFUSAL_TEXT
from .models import Citation, GroundedAnswer, RetrievalResult, Verdict
from .retrieve import normalize_query, rerank_and_confidence, tokenize

REFUSAL_THRESHOLD = 0.12


def _sentences(text: str) -> List[str]:
    # No split after abbreviations (Dte., e.g., i.e.) and only before a
    # capital, digit, quote, or citation marker. Stops markers detaching
    # from their sentence and stops fragments like "Dte." or "i.e.".
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
        # Most query terms wins, earliest breaks ties. Link lists carry no
        # substance, so sentences with URLs only serve as fallback.
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


def compose(query: str, results: List[RetrievalResult], verdict: Verdict = "PROCEED") -> GroundedAnswer:
    if verdict != "PROCEED":
        key = verdict if verdict in REFUSAL_TEXT else "REFUSE_LOW_CONFIDENCE"
        return GroundedAnswer(
            answer=REFUSAL_TEXT[key],
            citations=[],
            confidence=0.0,
            verdict=key,  # type: ignore[typeddict-item]
            disclaimer=DISCLAIMER,
        )
    if not results:
        return GroundedAnswer(
            answer=REFUSAL_TEXT["REFUSE_LOW_CONFIDENCE"],
            citations=[],
            confidence=0.0,
            verdict="REFUSE_LOW_CONFIDENCE",
            disclaimer=DISCLAIMER,
        )
    ranked, confidence = rerank_and_confidence(results, query)
    if confidence < REFUSAL_THRESHOLD:
        return GroundedAnswer(
            answer=REFUSAL_TEXT["REFUSE_LOW_CONFIDENCE"],
            citations=[],
            confidence=float(confidence),
            verdict="REFUSE_LOW_CONFIDENCE",
            disclaimer=DISCLAIMER,
        )
    top = ranked[:3]
    sentences: List[str] = []
    citations: List[Citation] = []
    terms = _query_terms(query)
    for r in top:
        pick = _pick_sentence(r.chunk.text, terms)
        cited = _cite_sentence(pick, r.chunk.chunk_id)
        if cited:
            sentences.append(cited)
        src = r.chunk.source_url or r.chunk.source_filename
        citations.append(
            Citation(
                chunk_id=r.chunk.chunk_id,
                doc_id=r.chunk.doc_id,
                title=r.chunk.title,
                source=src,
                section=r.chunk.section,
            )
        )
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
