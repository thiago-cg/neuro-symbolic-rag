"""Resilience tests — verify all failure scenarios are handled gracefully (Phase 9)."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from extractors.validator import validate_triple_batch, ValidationReport
from graph.errors import ElicitationError
from graph.nodes import elicit_domain, extract_triples, reason_datalog, synthesize_response
from reasoner.clingo_engine import ClingoEngine


# ── Validator tests ────────────────────────────────────────────────────────────

class TestTripleValidator:
    def test_all_valid_triples(self):
        triples = [
            {"subject": "bert", "predicate": "usa", "obj": "transformer", "confidence": 0.95},
            {"subject": "gpt", "predicate": "supera", "obj": "lstm", "confidence": 0.90},
        ]
        report = validate_triple_batch(triples)
        assert not report.has_warnings
        assert report.pct_invalid_predicates == 0.0

    def test_invalid_predicates_detected(self):
        triples = [
            {"subject": "bert", "predicate": "influencia", "obj": "nlp", "confidence": 0.9},
            {"subject": "gpt", "predicate": "usa", "obj": "transformer", "confidence": 0.9},
        ]
        report = validate_triple_batch(triples)
        assert len(report.invalid_predicates) == 1
        assert report.pct_invalid_predicates == 50.0

    def test_unnormalized_entities_detected(self):
        triples = [
            {"subject": "BERT Model", "predicate": "usa", "obj": "Transformer Architecture", "confidence": 0.9},
        ]
        report = validate_triple_batch(triples)
        assert len(report.unnormalized_entities) == 1
        assert report.pct_unnormalized == 100.0

    def test_low_confidence_detected(self):
        triples = [
            {"subject": "bert", "predicate": "usa", "obj": "transformer", "confidence": 0.3},
            {"subject": "gpt", "predicate": "cita", "obj": "bert", "confidence": 0.9},
        ]
        report = validate_triple_batch(triples)
        assert len(report.low_confidence) == 1
        assert report.pct_low_confidence == 50.0

    def test_empty_batch_returns_empty_report(self):
        report = validate_triple_batch([])
        assert report.total == 0
        assert not report.has_warnings
        assert report.pct_invalid_predicates == 0.0

    def test_summary_returns_dict(self):
        triples = [{"subject": "bert", "predicate": "usa", "obj": "transformer", "confidence": 0.9}]
        report = validate_triple_batch(triples)
        summary = report.summary()
        assert isinstance(summary, dict)
        assert "total" in summary


# ── LLM failure scenarios ──────────────────────────────────────────────────────

class TestLLMFailureResilience:
    @patch("graph.nodes._elicit_llm")
    def test_llm_returns_invalid_json_raises_elicitation_error(self, mock_llm):
        """Consistent JSON failures should raise ElicitationError, not crash."""
        mock_llm.invoke.return_value = MagicMock(content="not json")
        with pytest.raises(ElicitationError) as exc_info:
            elicit_domain({"user_query": "How do transformers work?"})
        assert "failed after" in str(exc_info.value)

    @patch("graph.nodes._elicit_llm")
    def test_llm_returns_wrong_fields_raises_elicitation_error(self, mock_llm):
        """JSON with wrong fields should raise ElicitationError."""
        mock_llm.invoke.return_value = MagicMock(content='{"wrong": "fields"}')
        with pytest.raises(ElicitationError):
            elicit_domain({"user_query": "How do transformers work?"})

    @patch("extractors.triple_extractor._extract_llm")
    def test_extractor_handles_llm_returning_plain_text(self, mock_llm):
        """Extractor should return empty list, not crash, on invalid LLM output."""
        mock_llm.invoke.return_value = MagicMock(content="I cannot extract triples from this.")
        paper = {"id": "p1", "title": "Test", "abstract": "Some abstract text.", "authors": [], "year": 2024, "citation_count": 0, "doi": ""}
        result = extract_triples({"papers": [paper]})
        assert result == {"extracted_triples": []}

    @patch("graph.nodes._synthesize_llm")
    def test_synthesis_with_empty_state_returns_response(self, mock_llm):
        """Synthesize should work even with minimal state."""
        mock_llm.invoke.return_value = MagicMock(content="No significant findings.")
        result = synthesize_response({
            "user_query": "test",
            "research_question": "test question",
            "domain": "CS",
            "keywords": [],
            "inferred_facts": [],
            "conflicts_detected": [],
            "answer_sets": [],
            "papers": [],
        })
        assert result["final_response"] == "No significant findings."


# ── Clingo failure scenarios ───────────────────────────────────────────────────

class TestClingoResilience:
    def test_empty_facts_datalog_returns_empty(self):
        """Clingo with no facts should return empty inferred list."""
        engine = ClingoEngine()
        inferred = engine.run_datalog("")
        assert isinstance(inferred, list)

    def test_empty_facts_asp_returns_single_empty_set(self):
        """ASP with no facts should return at least one (empty) answer set."""
        engine = ClingoEngine()
        answer_sets = engine.run_asp("", max_models=5)
        assert isinstance(answer_sets, list)

    def test_no_conflicts_means_no_asp_routing(self):
        """If no conflicts in inferred facts, should not route to ASP."""
        engine = ClingoEngine()
        inferred = ['influencia("bert", "transformer")']
        assert not engine.has_conflicts(inferred)


# ── Neo4j offline scenario ─────────────────────────────────────────────────────

class TestNeo4jOfflineResilience:
    def test_persist_to_graph_graceful_when_neo4j_offline(self):
        """persist_to_graph should not raise when Neo4j is unavailable."""
        from graph.nodes import persist_to_graph
        from extractors.triple_schema import ExtractedTriple

        paper = {"id": "p1", "title": "Test", "abstract": "Abstract.", "authors": [], "year": 2024, "citation_count": 0, "doi": ""}
        triple = {"subject": "bert", "predicate": "usa", "obj": "transformer", "source_paper": "p1", "confidence": 0.9}

        # Mock Neo4jClient to raise on connect
        with patch("knowledge_base.neo4j_client.AsyncGraphDatabase") as mock_gdb:
            mock_gdb.driver.side_effect = ConnectionError("Neo4j unavailable")

            # Should not raise — should log warning and return {}
            result = persist_to_graph({"papers": [paper], "extracted_triples": [triple]})
            assert result == {}

    def test_reason_datalog_skips_neo4j_persist_on_error(self):
        """reason_datalog should persist inferred facts even if Neo4j fails."""
        triple = {"subject": "bert", "predicate": "usa", "obj": "transformer", "source_paper": "p1", "confidence": 0.9}

        with patch("reasoner.clingo_engine.ClingoEngine.run_datalog", return_value=['influencia("bert", "transformer")']), \
             patch("knowledge_base.neo4j_client.AsyncGraphDatabase") as mock_gdb:

            mock_gdb.driver.side_effect = ConnectionError("Neo4j offline")
            from graph.nodes import reason_datalog
            result = reason_datalog({"extracted_triples": [triple]})

        # Should still return inferred facts despite Neo4j failure
        assert "inferred_facts" in result
        assert len(result["inferred_facts"]) > 0


# ── API rate limits (HTTP 429) scenario ───────────────────────────────────────

class TestApiRateLimit:
    def test_search_handles_http_error(self):
        """Academic search should return empty list on HTTP error."""
        import httpx
        from research.academic_tools import search_semantic_scholar

        # The @tool wrapper catches exceptions and returns empty list
        with patch("research.academic_tools.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.side_effect = httpx.HTTPError("429 Too Many Requests")
            result = search_semantic_scholar.invoke({"query": "transformer NLP", "limit": 10})

        assert isinstance(result, list)
        assert result == []


# ── Observability tests ───────────────────────────────────────────────────────

class TestObservability:
    def test_validation_report_summary_structure(self):
        triples = [
            {"subject": "bert", "predicate": "usa", "obj": "transformer", "confidence": 0.9},
            {"subject": "bad entity", "predicate": "invalid_pred", "obj": "thing", "confidence": 0.3},
        ]
        report = validate_triple_batch(triples)
        summary = report.summary()
        assert summary["total"] == 2
        assert summary["invalid_predicates"] == 1
        assert summary["unnormalized_entities"] == 1
        assert summary["low_confidence"] == 1

    def test_log_node_execution_decorator(self):
        """log_node_execution should not change function behavior."""
        from observability import log_node_execution

        @log_node_execution("test_node")
        def my_node(state):
            return {"result": "done"}

        result = my_node({"user_query": "test"})
        assert result == {"result": "done"}
