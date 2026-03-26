"""Tests for the FastAPI layer (Phase 8)."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import QueryRequest, QueryResponse, HealthResponse


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """Test client with Neo4j mocked (not connected)."""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture
def mock_graph_result():
    return {
        "user_query": "How do transformers work?",
        "final_response": "Transformers use self-attention to process sequences in parallel. " * 5,
        "inferred_facts": [
            'influencia_direta("transformer", "attention_mechanism")',
            'comparavel("bert", "gpt")',
        ],
        "conflicts_detected": [],
        "papers": [{"id": f"p{i}", "title": f"Paper {i}", "abstract": "...", "authors": [], "year": 2024, "citation_count": 0, "doi": ""} for i in range(5)],
        "answer_sets": [],
    }


# ── Schema tests ───────────────────────────────────────────────────────────────

class TestSchemas:
    def test_query_request_valid(self):
        req = QueryRequest(user_query="How do transformers work?")
        assert req.max_papers == 30

    def test_query_request_too_short(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            QueryRequest(user_query="AB")  # < 3 chars

    def test_query_response(self):
        resp = QueryResponse(
            final_response="Some response.",
            inferred_facts=["fact1"],
            conflicts_detected=[],
            papers_analyzed=5,
        )
        assert resp.papers_analyzed == 5
        assert resp.answer_sets == []


# ── Health endpoint ────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "neo4j" in data


# ── Research endpoint ──────────────────────────────────────────────────────────

class TestResearchEndpoint:
    def test_post_research_success(self, client, mock_graph_result):
        with patch("api.main.asyncio") as mock_asyncio:
            loop = MagicMock()
            loop.run_in_executor = AsyncMock(return_value=mock_graph_result)
            mock_asyncio.get_event_loop.return_value = loop
            # Also patch build_graph
            with patch("graph.graph.build_graph") as mock_build:
                mock_graph = MagicMock()
                mock_graph.invoke.return_value = mock_graph_result
                mock_build.return_value = mock_graph

                resp = client.post(
                    "/research",
                    json={"user_query": "How do transformers outperform RNNs?"},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert "final_response" in data
        assert "inferred_facts" in data
        assert "conflicts_detected" in data
        assert "papers_analyzed" in data

    def test_post_research_query_too_short(self, client):
        resp = client.post("/research", json={"user_query": "AB"})
        assert resp.status_code == 422  # Validation error

    def test_post_research_empty_query(self, client):
        resp = client.post("/research", json={"user_query": ""})
        assert resp.status_code == 422


# ── Graph query endpoint ───────────────────────────────────────────────────────

class TestGraphQuery:
    def test_graph_query_without_neo4j_returns_503(self, client):
        """Without Neo4j connected, should return 503."""
        resp = client.get("/graph/query", params={"cypher": "MATCH (n) RETURN n LIMIT 5"})
        assert resp.status_code == 503

    def test_graph_query_rejects_write_operations(self, client):
        """Write queries should be rejected with 400."""
        from unittest.mock import AsyncMock
        import api.main as main_module

        # Mock the Neo4j client
        mock_client = AsyncMock()
        main_module._neo4j_client = mock_client

        try:
            for write_op in ["CREATE (n) RETURN n", "MERGE (n:Test)", "DELETE n", "SET n.x = 1"]:
                resp = client.get("/graph/query", params={"cypher": write_op})
                assert resp.status_code == 400, f"Expected 400 for: {write_op}"
        finally:
            main_module._neo4j_client = None

    def test_graph_query_with_mock_client(self, client):
        """Valid MATCH query should return results."""
        import api.main as main_module

        mock_client = AsyncMock()
        mock_client.run_cypher = AsyncMock(return_value=[{"title": "BERT Paper"}, {"title": "GPT Paper"}])
        main_module._neo4j_client = mock_client

        try:
            resp = client.get(
                "/graph/query",
                params={"cypher": "MATCH (p:Paper) RETURN p.title AS title LIMIT 5"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 2
            assert data[0]["title"] == "BERT Paper"
        finally:
            main_module._neo4j_client = None


# ── SSE streaming endpoint ────────────────────────────────────────────────────

class TestStreamEndpoint:
    def test_stream_returns_sse_content_type(self, client):
        """SSE endpoint should return text/event-stream."""
        with patch("graph.nodes.elicit_domain", return_value={"domain": "NLP", "intent": "comparison", "keywords": [], "research_question": "test"}), \
             patch("graph.nodes.research_papers", return_value={"papers": []}), \
             patch("graph.nodes.extract_triples", return_value={"extracted_triples": []}), \
             patch("graph.nodes.persist_to_graph", return_value={}), \
             patch("graph.nodes.reason_datalog", return_value={"inferred_facts": [], "conflicts_detected": []}), \
             patch("graph.nodes.synthesize_response", return_value={"final_response": "Test response " * 20}):

            resp = client.get(
                "/research/stream",
                params={"user_query": "How do transformers work?"},
            )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_stream_emits_expected_events(self, client):
        """Stream should emit at least 4 distinct event types."""
        with patch("graph.nodes.elicit_domain", return_value={"domain": "NLP", "intent": "survey", "keywords": ["transformer"], "research_question": "How do transformers work?"}), \
             patch("graph.nodes.research_papers", return_value={"papers": [{"id": "p1", "title": "T", "abstract": "A", "authors": [], "year": 2024, "citation_count": 0, "doi": ""}]}), \
             patch("graph.nodes.extract_triples", return_value={"extracted_triples": []}), \
             patch("graph.nodes.persist_to_graph", return_value={}), \
             patch("graph.nodes.reason_datalog", return_value={"inferred_facts": ["fact1"], "conflicts_detected": []}), \
             patch("graph.nodes.synthesize_response", return_value={"final_response": "Transformers use attention. " * 20}):

            resp = client.get(
                "/research/stream",
                params={"user_query": "How do transformers work?"},
            )

        # Parse SSE events
        content = resp.text
        events = []
        for line in content.split("\n"):
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())

        assert len(events) >= 4, f"Expected ≥4 events, got: {events}"
        assert "elicit_done" in events
        assert "papers_found" in events
        assert "reasoning_done" in events
        assert "complete" in events
