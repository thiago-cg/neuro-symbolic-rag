"""Tests for the elicitation agent (Phase 2)."""

import json
import pytest
from unittest.mock import MagicMock, patch

from graph.errors import ElicitationError
from graph.nodes import elicit_domain
from graph.schemas import ElicitationOutput
from pydantic import ValidationError


# --- Helpers ---

def _make_llm_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.content = content
    return resp


def _valid_json(domain, keywords, question, intent) -> str:
    return json.dumps({
        "domain": domain,
        "keywords": keywords,
        "research_question": question,
        "intent": intent,
    })


# --- Unit tests ---

class TestElicitationOutput:
    def test_valid_output_parses(self):
        out = ElicitationOutput(
            domain="NLP",
            keywords=["transformer", "rnn"],
            research_question="How do transformers outperform RNNs?",
            intent="comparison",
        )
        assert out.domain == "NLP"
        assert "transformer" in out.keywords

    def test_empty_keywords_raises(self):
        with pytest.raises(ValidationError):
            ElicitationOutput(
                domain="NLP",
                keywords=[],
                research_question="test",
                intent="survey",
            )

    def test_invalid_intent_raises(self):
        with pytest.raises(ValidationError):
            ElicitationOutput(
                domain="NLP",
                keywords=["test"],
                research_question="test",
                intent="unknown",  # not in Literal
            )


class TestElicitDomainNode:
    def _state(self, query: str) -> dict:
        return {"user_query": query}

    @patch("graph.nodes._elicit_llm")
    def test_nlp_query(self, mock_llm):
        mock_llm.invoke.return_value = _make_llm_response(
            _valid_json(
                "NLP",
                ["transformer", "rnn", "sequence-to-sequence"],
                "How do transformers outperform RNNs in NLP tasks?",
                "comparison",
            )
        )
        result = elicit_domain(self._state("How do transformers outperform RNNs in NLP?"))
        assert result["domain"] == "NLP"
        assert "transformer" in result["keywords"]
        assert result["intent"] == "comparison"

    @patch("graph.nodes._elicit_llm")
    def test_bioinformatics_query(self, mock_llm):
        mock_llm.invoke.return_value = _make_llm_response(
            _valid_json(
                "Bioinformatics",
                ["protein folding", "alphafold", "deep learning"],
                "What is the impact of AlphaFold on protein structure prediction?",
                "specific_question",
            )
        )
        result = elicit_domain(self._state("What did AlphaFold change in protein structure prediction?"))
        assert result["domain"] == "Bioinformatics"
        assert result["intent"] == "specific_question"

    @patch("graph.nodes._elicit_llm")
    def test_survey_query(self, mock_llm):
        mock_llm.invoke.return_value = _make_llm_response(
            _valid_json(
                "Machine Learning",
                ["graph neural network", "gnn", "graph learning"],
                "What are the main approaches in graph neural network research?",
                "survey",
            )
        )
        result = elicit_domain(self._state("Give me an overview of graph neural networks"))
        assert result["intent"] == "survey"
        assert len(result["keywords"]) >= 2

    @patch("graph.nodes._elicit_llm")
    def test_invalid_json_raises_elicitation_error(self, mock_llm):
        # LLM returns plain text both times (exhaust retries)
        mock_llm.invoke.return_value = _make_llm_response("This is not JSON at all.")
        with pytest.raises(ElicitationError):
            elicit_domain(self._state("Some ambiguous query"))

    @patch("graph.nodes._elicit_llm")
    def test_recovers_on_second_attempt(self, mock_llm):
        """First attempt returns invalid JSON, second returns valid."""
        valid = _valid_json("CV", ["cnn", "image classification"], "How do CNNs classify images?", "specific_question")
        mock_llm.invoke.side_effect = [
            _make_llm_response("not json"),
            _make_llm_response(valid),
        ]
        result = elicit_domain(self._state("How does CNN do image classification?"))
        assert result["domain"] == "CV"
