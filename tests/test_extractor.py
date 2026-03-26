"""Tests for the triple extractor (Phase 4)."""

import json
import pytest
from unittest.mock import MagicMock, patch

from extractors.normalizer import normalize_entity
from extractors.triple_extractor import extract_from_paper, extract_all
from extractors.triple_schema import ExtractedTriple
from pydantic import ValidationError


# --- Fixed test abstracts ---

BERT_ABSTRACT = (
    "We introduce a new language representation model called BERT. "
    "BERT is designed to pre-train deep bidirectional representations from unlabeled text. "
    "BERT outperforms previous state-of-the-art models on eleven NLP tasks. "
    "Our model uses the Transformer architecture and extends ELMo by training bidirectionally."
)

ATTENTION_ABSTRACT = (
    "We propose the Transformer, a novel architecture based entirely on attention mechanisms. "
    "The Transformer defines multi-head attention which uses scaled dot-product attention. "
    "Our model surpasses recurrent neural networks on machine translation tasks. "
    "This work cites Bahdanau et al. attention paper and contradicts the assumption that "
    "recurrence is necessary for sequence modeling."
)

SURVEY_ABSTRACT = (
    "This survey covers graph neural networks (GNNs). "
    "We define message passing as the core operation in GNNs. "
    "Modern GNNs extend spectral graph convolutions. "
    "GNNs treat_conceito node classification and link prediction tasks."
)


def _make_paper(abstract: str, paper_id: str = "test_paper") -> dict:
    return {"id": paper_id, "title": "Test Paper", "abstract": abstract, "authors": [], "year": 2024, "citation_count": 0, "doi": ""}


def _mock_llm_response(triples: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.content = json.dumps(triples)
    return resp


# --- Normalizer tests ---

class TestNormalizer:
    def test_lowercase(self):
        assert normalize_entity("BERT") == "bert"

    def test_spaces_to_underscore(self):
        assert normalize_entity("BERT Model") == "bert_model"

    def test_alias_resolution(self):
        assert normalize_entity("large language model") == "llm"
        assert normalize_entity("natural language processing") == "nlp"

    def test_special_chars_removed(self):
        result = normalize_entity("GPT-3.5!")
        assert "!" not in result
        assert result  # not empty

    def test_empty_returns_unknown(self):
        assert normalize_entity("") == ""

    def test_hyphen_to_underscore(self):
        result = normalize_entity("self-attention")
        assert " " not in result


# --- Schema validation tests ---

class TestExtractedTripleSchema:
    def test_valid_triple(self):
        t = ExtractedTriple(
            subject="bert",
            predicate="usa",
            obj="transformer_architecture",
            source_paper="devlin2018",
            confidence=0.95,
        )
        assert t.predicate == "usa"

    def test_invalid_predicate_raises(self):
        with pytest.raises(ValidationError):
            ExtractedTriple(
                subject="bert",
                predicate="influencia",  # not in vocabulary
                obj="transformer",
                source_paper="p1",
                confidence=0.9,
            )

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            ExtractedTriple(
                subject="bert",
                predicate="usa",
                obj="transformer",
                source_paper="p1",
                confidence=1.5,  # > 1.0
            )

    def test_empty_subject_raises(self):
        with pytest.raises(ValidationError):
            ExtractedTriple(
                subject="",
                predicate="usa",
                obj="transformer",
                source_paper="p1",
                confidence=0.9,
            )


# --- Extractor tests ---

class TestExtractFromPaper:
    @patch("extractors.triple_extractor._extract_llm")
    def test_bert_abstract_returns_triples(self, mock_llm):
        mock_llm.invoke.return_value = _mock_llm_response([
            {"subject": "bert", "predicate": "supera", "obj": "previous_models", "confidence": 0.95},
            {"subject": "bert", "predicate": "usa", "obj": "transformer_architecture", "confidence": 0.98},
            {"subject": "bert", "predicate": "estende", "obj": "elmo", "confidence": 0.90},
            {"subject": "bert", "predicate": "trata_conceito", "obj": "language_representation", "confidence": 0.92},
            {"subject": "bert", "predicate": "define", "obj": "masked_language_modeling", "confidence": 0.85},
        ])
        triples = extract_from_paper(_make_paper(BERT_ABSTRACT, "bert2018"))
        assert len(triples) >= 5
        predicates = {t.predicate for t in triples}
        assert predicates.issubset({"cita", "supera", "usa", "trata_conceito", "define", "contradiz", "estende"})

    @patch("extractors.triple_extractor._extract_llm")
    def test_all_triples_above_confidence_threshold(self, mock_llm):
        mock_llm.invoke.return_value = _mock_llm_response([
            {"subject": "bert", "predicate": "usa", "obj": "transformer", "confidence": 0.95},
            {"subject": "bert", "predicate": "cita", "obj": "attention_paper", "confidence": 0.50},  # below threshold
        ])
        triples = extract_from_paper(_make_paper(BERT_ABSTRACT))
        assert all(t.confidence >= 0.7 for t in triples)
        assert len(triples) == 1  # only the high-confidence one

    @patch("extractors.triple_extractor._extract_llm")
    def test_invalid_json_returns_empty(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content="not json at all")
        triples = extract_from_paper(_make_paper(BERT_ABSTRACT))
        assert triples == []

    @patch("extractors.triple_extractor._extract_llm")
    def test_entities_are_normalized(self, mock_llm):
        mock_llm.invoke.return_value = _mock_llm_response([
            {"subject": "BERT Model", "predicate": "usa", "obj": "Transformer Architecture", "confidence": 0.95},
        ])
        triples = extract_from_paper(_make_paper(BERT_ABSTRACT))
        assert len(triples) == 1
        assert triples[0].subject == "bert_model"
        assert triples[0].obj == "transformer_architecture"

    def test_empty_abstract_returns_empty(self):
        paper = _make_paper("")
        triples = extract_from_paper(paper)
        assert triples == []

    @patch("extractors.triple_extractor._extract_llm")
    def test_invalid_predicates_are_filtered(self, mock_llm):
        mock_llm.invoke.return_value = _mock_llm_response([
            {"subject": "bert", "predicate": "influencia", "obj": "nlp", "confidence": 0.95},
            {"subject": "bert", "predicate": "usa", "obj": "transformer", "confidence": 0.95},
        ])
        triples = extract_from_paper(_make_paper(BERT_ABSTRACT))
        # "influencia" is not valid — only "usa" should pass
        assert len(triples) == 1
        assert triples[0].predicate == "usa"


# --- Parallel extraction test ---

class TestExtractAll:
    @pytest.mark.asyncio
    async def test_extracts_from_multiple_papers(self):
        papers = [_make_paper(BERT_ABSTRACT, f"p{i}") for i in range(5)]

        with patch("extractors.triple_extractor._extract_llm") as mock_llm:
            mock_llm.invoke.return_value = _mock_llm_response([
                {"subject": "bert", "predicate": "usa", "obj": "transformer", "confidence": 0.95},
                {"subject": "bert", "predicate": "trata_conceito", "obj": "nlp", "confidence": 0.90},
            ])
            triples = await extract_all(papers, max_concurrency=3)

        assert len(triples) == 10  # 2 per paper × 5 papers
        assert all(isinstance(t, ExtractedTriple) for t in triples)
