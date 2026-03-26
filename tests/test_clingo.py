"""Tests for the Clingo reasoner (Phase 5)."""

import pytest

from extractors.triple_schema import ExtractedTriple
from reasoner.clingo_engine import ClingoEngine, triples_to_asp_facts


def _make_triple(subject: str, predicate: str, obj: str, confidence: float = 0.95) -> ExtractedTriple:
    return ExtractedTriple(
        subject=subject,
        predicate=predicate,
        obj=obj,
        source_paper="test_paper",
        confidence=confidence,
    )


class TestTriplesToAspFacts:
    def test_converts_triple_to_asp_fact(self):
        triple = _make_triple("bert", "usa", "transformer")
        facts = triples_to_asp_facts([triple])
        assert 'usa("bert", "transformer")' in facts

    def test_multiple_triples(self):
        triples = [
            _make_triple("bert", "usa", "transformer"),
            _make_triple("bert", "supera", "lstm"),
        ]
        facts = triples_to_asp_facts(triples)
        assert 'usa("bert", "transformer")' in facts
        assert 'supera("bert", "lstm")' in facts

    def test_empty_triples_returns_empty_string(self):
        facts = triples_to_asp_facts([])
        assert facts == ""


class TestClingoEngineDatalog:
    def setup_method(self):
        self.engine = ClingoEngine()

    def test_direct_influence_inferred(self):
        facts = 'usa("bert", "transformer").'
        inferred = self.engine.run_datalog(facts)
        assert any("influencia_direta" in atom and "bert" in atom and "transformer" in atom
                   for atom in inferred)

    def test_transitive_influence_inferred(self):
        # bert usa transformer, transformer usa attention_mechanism
        # → bert should transitively influence attention_mechanism
        facts = 'usa("bert", "transformer").\nusa("transformer", "attention_mechanism").'
        inferred = self.engine.run_datalog(facts)
        # bert directly influences transformer
        assert any("influencia_direta" in a and "bert" in a and "transformer" in a for a in inferred)
        # bert transitively influences attention_mechanism
        assert any("influencia" in a and "bert" in a and "attention_mechanism" in a for a in inferred)

    def test_comparavel_inferred(self):
        facts = 'usa("bert", "transformer").\nusa("gpt", "transformer").'
        inferred = self.engine.run_datalog(facts)
        assert any("comparavel" in a and "bert" in a and "gpt" in a for a in inferred)

    def test_conflict_detected(self):
        facts = 'contradiz("bert", "gpt").'
        inferred = self.engine.run_datalog(facts)
        assert any("conflito_potencial" in a for a in inferred)
        assert self.engine.has_conflicts(inferred)

    def test_no_conflict_when_no_contradiz(self):
        facts = 'usa("bert", "transformer").'
        inferred = self.engine.run_datalog(facts)
        assert not self.engine.has_conflicts(inferred)

    def test_from_triples(self):
        triples = [
            _make_triple("bert", "usa", "transformer"),
            _make_triple("gpt", "usa", "transformer"),
        ]
        facts = triples_to_asp_facts(triples)
        inferred = self.engine.run_datalog(facts)
        assert any("influencia" in a for a in inferred)
        assert any("comparavel" in a for a in inferred)


class TestClingoEngineASP:
    def setup_method(self):
        self.engine = ClingoEngine()

    def test_asp_returns_multiple_answer_sets(self):
        # Two conflicting positions → 2 answer sets (each picks one)
        facts = 'contradiz("bert_is_better", "gpt_is_better").\nconflito_potencial("bert_is_better", "gpt_is_better").'
        answer_sets = self.engine.run_asp(facts, max_models=10)
        # Should have at least 1 answer set
        assert len(answer_sets) >= 1
        # Each answer set should contain posicao_aceita
        for answer_set in answer_sets:
            assert any("posicao_aceita" in atom for atom in answer_set)

    def test_asp_empty_facts(self):
        answer_sets = self.engine.run_asp("", max_models=5)
        # No conflicts → single empty answer set
        assert isinstance(answer_sets, list)

    def test_has_conflicts_true(self):
        inferred = ["conflito_potencial(bert,gpt)", "influencia(bert,transformer)"]
        assert self.engine.has_conflicts(inferred)

    def test_has_conflicts_false(self):
        inferred = ["influencia(bert,transformer)", "comparavel(bert,gpt)"]
        assert not self.engine.has_conflicts(inferred)

    def test_extract_conflicts(self):
        inferred = [
            "conflito_potencial(bert,gpt)",
            "influencia(bert,transformer)",
            "conflito_potencial(method_a,method_b)",
        ]
        conflicts = self.engine.extract_conflicts(inferred)
        assert len(conflicts) == 2
        assert all("conflito_potencial" in c for c in conflicts)
