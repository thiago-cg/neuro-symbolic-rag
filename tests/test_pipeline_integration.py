"""Integration tests for the full LangGraph pipeline (Phase 7).

Mocks all external services at the dependency level (not node level).
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from graph.graph import build_graph
from research.schemas import PaperRef, ResearchBrief, SubAgentFindings

# --- Fixed test data ---

PAPERS = [
    {
        "id": "devlin2018",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "abstract": "We introduce BERT. BERT uses the Transformer architecture and outperforms previous RNN-based models on NLP tasks.",
        "authors": ["Devlin, J."], "year": 2018, "citation_count": 50000, "doi": "10.18653/v1/N19-1423",
    },
    {
        "id": "vaswani2017",
        "title": "Attention Is All You Need",
        "abstract": "We propose the Transformer, a model based entirely on attention mechanisms.",
        "authors": ["Vaswani, A."], "year": 2017, "citation_count": 80000, "doi": "",
    },
]

TRIPLES = [
    {"subject": "bert", "predicate": "usa", "obj": "transformer_architecture", "source_paper": "devlin2018", "confidence": 0.98},
    {"subject": "transformer", "predicate": "supera", "obj": "rnn", "source_paper": "vaswani2017", "confidence": 0.97},
]

INFERRED = [
    'influencia_direta("bert", "transformer_architecture")',
    'influencia("bert", "transformer_architecture")',
    'comparavel("bert", "transformer")',
]

FINAL_RESPONSE = (
    "The symbolic analysis reveals that BERT and the Transformer architecture represent "
    "a paradigm shift in NLP, systematically outperforming recurrent architectures. "
    "Inferred transitive influences show that BERT's usage of the Transformer mechanism "
    "connects it to attention-based computation. The knowledge graph reveals comparability "
    "between BERT and LSTM. No significant conflicts were detected — the literature converges "
    "on the superiority of attention-based models for most NLP benchmarks."
)


def _mock_response(content: str) -> MagicMock:
    m = MagicMock()
    m.content = content
    m.tool_calls = []
    return m


def _elicit_json() -> str:
    return json.dumps({
        "domain": "NLP",
        "keywords": ["transformer", "rnn", "bert", "attention"],
        "research_question": "How do transformers outperform RNNs in NLP?",
        "intent": "comparison",
    })


class TestFullPipeline:
    def _make_brief(self) -> ResearchBrief:
        return ResearchBrief(
            main_question="How do transformers outperform RNNs?",
            sub_topics=["attention mechanisms", "transformer vs rnn"],
            inclusion_criteria="peer-reviewed papers",
            exclusion_criteria="non-NLP",
            target_domains=["NLP"],
        )

    def _make_findings(self) -> list[SubAgentFindings]:
        papers = [PaperRef(**{**p, "abstract": p["abstract"]}) for p in PAPERS]
        return [
            SubAgentFindings(subtopic="attention", papers_found=papers[:1], key_insights="BERT uses attention.", sources_used=[]),
            SubAgentFindings(subtopic="transformers", papers_found=papers[1:], key_insights="Transformer outperforms RNNs.", sources_used=[]),
        ]

    def test_pipeline_produces_final_response(self):
        """Full graph invocation with all external services mocked at dependency level."""
        graph = build_graph()

        with patch("graph.nodes._elicit_llm") as mock_elicit_llm, \
             patch("research.scope._scope_llm") as mock_scope_llm, \
             patch("research.supervisor.run_sub_agent", new_callable=AsyncMock) as mock_sub_agent, \
             patch("research.supervisor._supervisor_llm") as mock_supervisor_llm, \
             patch("research.harvester._fetch_abstract", return_value=""), \
             patch("extractors.triple_extractor._extract_llm") as mock_extract_llm, \
             patch("graph.nodes.persist_to_graph") as mock_persist, \
             patch("reasoner.clingo_engine.ClingoEngine.run_datalog", return_value=INFERRED), \
             patch("knowledge_base.neo4j_client.Neo4jClient.connect", new_callable=AsyncMock), \
             patch("knowledge_base.neo4j_client.Neo4jClient.upsert_inferred_facts", new_callable=AsyncMock), \
             patch("graph.nodes._synthesize_llm") as mock_synth_llm:

            # Elicitation
            mock_elicit_llm.invoke.return_value = _mock_response(_elicit_json())

            # Scope
            mock_scope_llm.invoke.side_effect = [
                _mock_response('{"needs_clarification": false}'),  # clarification check
                _mock_response(json.dumps({                         # brief generator
                    "main_question": "How do transformers outperform RNNs?",
                    "sub_topics": ["attention", "transformers"],
                    "inclusion_criteria": "NLP papers",
                    "exclusion_criteria": "non-NLP",
                    "target_domains": ["NLP"],
                })),
            ]

            # Sub-agents
            mock_sub_agent.side_effect = self._make_findings()

            # Supervisor evaluation
            mock_supervisor_llm.invoke.return_value = _mock_response('{"adequate": true, "gaps": []}')

            # Triple extraction
            mock_extract_llm.invoke.return_value = _mock_response(json.dumps([
                {"subject": "bert", "predicate": "usa", "obj": "transformer_architecture", "confidence": 0.95},
                {"subject": "transformer", "predicate": "supera", "obj": "rnn", "confidence": 0.95},
            ]))

            # Persist (no-op)
            mock_persist.return_value = {}

            # Synthesis
            mock_synth_llm.invoke.return_value = _mock_response(FINAL_RESPONSE)

            result = graph.invoke({"user_query": "How do transformers outperform RNNs?"})

        assert "final_response" in result
        assert isinstance(result["final_response"], str)
        assert len(result["final_response"]) >= 200

    def test_pipeline_with_conflicts_routes_to_asp(self):
        """Verify that detected conflicts trigger ASP conflict resolution."""
        graph = build_graph()

        inferred_with_conflicts = [
            'conflito_potencial("bert", "gpt")',
            'influencia("bert", "transformer")',
        ]
        asp_answer_sets = [
            ['posicao_aceita("bert")', 'evidencia("bert", 5)'],
            ['posicao_aceita("gpt")', 'evidencia("gpt", 3)'],
        ]

        with patch("graph.nodes._elicit_llm") as mock_elicit_llm, \
             patch("research.scope._scope_llm") as mock_scope_llm, \
             patch("research.supervisor.run_sub_agent", new_callable=AsyncMock) as mock_sub_agent, \
             patch("research.supervisor._supervisor_llm") as mock_supervisor_llm, \
             patch("research.harvester._fetch_abstract", return_value=""), \
             patch("extractors.triple_extractor._extract_llm") as mock_extract_llm, \
             patch("graph.nodes.persist_to_graph") as mock_persist, \
             patch("reasoner.clingo_engine.ClingoEngine.run_datalog", return_value=inferred_with_conflicts), \
             patch("reasoner.clingo_engine.ClingoEngine.run_asp", return_value=asp_answer_sets), \
             patch("knowledge_base.neo4j_client.Neo4jClient.connect", new_callable=AsyncMock), \
             patch("knowledge_base.neo4j_client.Neo4jClient.upsert_inferred_facts", new_callable=AsyncMock), \
             patch("graph.nodes._synthesize_llm") as mock_synth_llm:

            mock_elicit_llm.invoke.return_value = _mock_response(_elicit_json())
            mock_scope_llm.invoke.side_effect = [
                _mock_response('{"needs_clarification": false}'),
                _mock_response(json.dumps({
                    "main_question": "BERT vs GPT?",
                    "sub_topics": ["bert", "gpt"],
                    "inclusion_criteria": "any",
                    "exclusion_criteria": "none",
                    "target_domains": ["NLP"],
                })),
            ]
            mock_sub_agent.side_effect = self._make_findings()
            mock_supervisor_llm.invoke.return_value = _mock_response('{"adequate": true, "gaps": []}')
            mock_extract_llm.invoke.return_value = _mock_response(json.dumps([
                {"subject": "bert", "predicate": "contradiz", "obj": "gpt", "confidence": 0.8},
            ]))
            mock_persist.return_value = {}
            mock_synth_llm.invoke.return_value = _mock_response(FINAL_RESPONSE)

            result = graph.invoke({"user_query": "Is BERT or GPT better?"})

        assert "answer_sets" in result
        assert len(result["answer_sets"]) == 2
        assert "final_response" in result

    def test_graph_has_7_nodes(self):
        graph = build_graph()
        nodes = set(graph.get_graph().nodes.keys())
        expected = {"elicit", "research", "extract_triples", "persist", "reason_datalog", "reason_asp_conflicts", "synthesize"}
        assert expected.issubset(nodes)
