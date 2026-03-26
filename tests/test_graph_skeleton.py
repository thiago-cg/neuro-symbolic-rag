"""Tests for the LangGraph skeleton (Phase 1)."""

import pytest
from graph.graph import build_graph
from graph.router import should_run_asp


def test_build_graph_returns_compiled_graph():
    graph = build_graph()
    assert graph is not None


def test_graph_has_correct_nodes():
    graph = build_graph()
    node_names = set(graph.get_graph().nodes.keys())
    expected = {"elicit", "extract_triples", "reason_datalog", "reason_asp_conflicts", "synthesize"}
    assert expected.issubset(node_names)


def test_router_no_conflicts_goes_to_synthesize():
    state = {"conflicts_detected": [], "user_query": "test"}
    assert should_run_asp(state) == "synthesize"


def test_router_with_conflicts_goes_to_asp():
    state = {"conflicts_detected": ["bert contradiz gpt"], "user_query": "test"}
    assert should_run_asp(state) == "reason_asp_conflicts"


def test_graph_invoke_with_minimal_state():
    """Verify graph can be invoked with all nodes mocked."""
    import json
    from unittest.mock import MagicMock, AsyncMock, patch
    from research.schemas import PaperRef, SubAgentFindings

    graph = build_graph()

    mock_llm_resp = MagicMock()
    mock_llm_resp.content = json.dumps({
        "domain": "NLP", "keywords": ["transformer"], "research_question": "What is BERT?", "intent": "specific_question"
    })
    mock_llm_resp.tool_calls = []

    findings = [SubAgentFindings(subtopic="bert", papers_found=[], key_insights="BERT.", sources_used=[])]

    with patch("graph.nodes._elicit_llm") as m_elicit, \
         patch("research.scope._scope_llm") as m_scope, \
         patch("research.supervisor.run_sub_agent", new_callable=AsyncMock, return_value=findings[0]), \
         patch("research.supervisor._supervisor_llm") as m_sup, \
         patch("research.harvester._fetch_abstract", return_value=""), \
         patch("extractors.triple_extractor._extract_llm") as m_extract, \
         patch("graph.nodes.persist_to_graph", return_value={}), \
         patch("reasoner.clingo_engine.ClingoEngine.run_datalog", return_value=[]), \
         patch("knowledge_base.neo4j_client.Neo4jClient.connect", new_callable=AsyncMock), \
         patch("knowledge_base.neo4j_client.Neo4jClient.upsert_inferred_facts", new_callable=AsyncMock), \
         patch("graph.nodes._synthesize_llm") as m_synth:

        m_elicit.invoke.return_value = mock_llm_resp
        m_scope.invoke.side_effect = [
            MagicMock(content='{"needs_clarification": false}'),
            MagicMock(content=json.dumps({
                "main_question": "What is BERT?", "sub_topics": ["bert overview", "bert applications"],
                "inclusion_criteria": "NLP", "exclusion_criteria": "none", "target_domains": ["NLP"],
            })),
        ]
        m_sup.invoke.return_value = MagicMock(content='{"adequate": true, "gaps": []}')
        m_extract.invoke.return_value = MagicMock(content="[]", tool_calls=[])
        m_synth.invoke.return_value = MagicMock(content="BERT is a language model.")

        result = graph.invoke({"user_query": "What is BERT?"})

    assert "user_query" in result
    assert result["user_query"] == "What is BERT?"
    assert "final_response" in result
