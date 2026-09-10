import re

from fastapi.testclient import TestClient

from app import api as api_module
from app.generate import compose
from app.guardrails import classify_input
from app.ingest import chunk_page_text
from app.retrieve import hybrid_search, load_index

FIXED_TEXT = ("malaria diagnosis treatment guidelines " * 60).strip()


def test_chunk_ids_stable():
    first = chunk_page_text(FIXED_TEXT)
    second = chunk_page_text(FIXED_TEXT)
    assert first == second
    assert len(first) >= 2
    assert first[0][1] == 0
    for piece, start in first:
        assert len(piece) >= 50
        assert start >= 0


def test_hybrid_search_returns_malaria_doc():
    idx = load_index("index")
    results = hybrid_search("malaria diagnosis treatment", k=5, index=idx)
    assert len(results) > 0
    assert any("malaria" in r.chunk.doc_id.lower() for r in results)
    assert results[0].fused_score >= results[-1].fused_score


def test_guardrail_refuses_personalized_dose_request():
    verdict = classify_input("I have chest pain, what dose should I take?")
    assert verdict in ("REFUSE_PERSONALIZED", "REFUSE_EMERGENCY")
    ans = compose("I have chest pain, what dose should I take?", [], verdict)
    assert ans.verdict == verdict
    assert ans.citations == []


def test_cricket_query_gets_low_confidence_refusal():
    idx = load_index("index")
    results = hybrid_search("cricket world cup rules", k=5, index=idx)
    ans = compose("cricket world cup rules", results, "PROCEED")
    assert ans.verdict == "REFUSE_LOW_CONFIDENCE"
    assert "I don't have enough information in the guidelines to answer this." in ans.answer
    assert classify_input("cricket world cup rules") == "REFUSE_OUT_OF_SCOPE"


def test_compose_citations_exist_in_input_chunks():
    idx = load_index("index")
    results = hybrid_search("malaria diagnosis treatment", k=5, index=idx)
    ans = compose("malaria diagnosis treatment", results, "PROCEED")
    assert ans.verdict == "ANSWER"
    assert len(ans.citations) > 0
    valid_ids = {r.chunk.chunk_id for r in results}
    for c in ans.citations:
        assert c.chunk_id in valid_ids
    assert len(ans.answer.split()) > 10
    markers = re.findall(r"\[([^\]]+)\]", ans.answer)
    assert len(markers) >= 2
    for m in markers:
        assert m in valid_ids


def test_api_health_reports_index():
    client = TestClient(api_module.app)
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["chunk_count"] > 0
    assert body["doc_count"] > 0


def test_api_query_answers_malaria():
    client = TestClient(api_module.app)
    r = client.post("/api/query", json={"query": "malaria diagnosis treatment", "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "ANSWER"
    assert len(body["citations"]) > 0
