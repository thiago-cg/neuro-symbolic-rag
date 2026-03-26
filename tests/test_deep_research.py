"""Tests for the deep research pipeline (Phase 3)."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from research.schemas import PaperRef, ResearchBrief, SubAgentFindings
from research.harvester import harvest_papers, _are_duplicate
from research.scope import clarification_agent, brief_generator


# --- Helpers ---

def _make_paper(title: str, abstract: str = "Some abstract.", doi: str = "") -> PaperRef:
    return PaperRef(id=title[:10], title=title, abstract=abstract, doi=doi)


def _make_findings(subtopic: str, papers: list[PaperRef]) -> SubAgentFindings:
    return SubAgentFindings(
        subtopic=subtopic,
        papers_found=papers,
        key_insights="Key insight.",
        sources_used=["search_semantic_scholar"],
    )


# --- Deduplication tests ---

class TestDeduplication:
    def test_exact_doi_match_is_duplicate(self):
        a = _make_paper("BERT paper", doi="10.1234/bert")
        b = _make_paper("BERT: Pre-training", doi="10.1234/bert")
        assert _are_duplicate(a, b)

    def test_fuzzy_title_match_is_duplicate(self):
        a = _make_paper("Attention Is All You Need")
        b = _make_paper("Attention is all you need")
        assert _are_duplicate(a, b)

    def test_different_papers_not_duplicate(self):
        a = _make_paper("BERT: Pre-training of Deep Bidirectional Transformers")
        b = _make_paper("GPT: Improving Language Understanding by Generative Pre-training")
        assert not _are_duplicate(a, b)

    def test_harvest_removes_duplicates(self):
        papers = [
            _make_paper("Attention Is All You Need", "Abstract A"),
            _make_paper("attention is all you need", "Abstract B"),  # duplicate (fuzzy)
            _make_paper("BERT: Pre-training", "Abstract C"),
        ]
        findings = [_make_findings("topic", papers)]
        result = harvest_papers(findings)
        titles = [p["title"] for p in result]
        # Should have 2 unique papers
        assert len(result) == 2

    def test_harvest_filters_no_abstract(self):
        papers = [
            _make_paper("Paper with abstract", "This paper presents..."),
            PaperRef(id="no-abs", title="Paper without abstract", abstract=""),
        ]
        findings = [_make_findings("topic", papers)]
        # Mock _fetch_abstract to return empty (can't reach network)
        with patch("research.harvester._fetch_abstract", return_value=""):
            result = harvest_papers(findings)
        assert all(p["abstract"] for p in result)
        assert len(result) == 1


# --- Scope tests ---

class TestScope:
    @patch("research.scope._scope_llm")
    def test_clear_query_no_clarification(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(
            content='{"needs_clarification": false}'
        )
        needs, question = clarification_agent("How do transformers outperform RNNs?")
        assert not needs
        assert question == ""

    @patch("research.scope._scope_llm")
    def test_ambiguous_query_returns_question(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(
            content='{"needs_clarification": true, "question": "Do you mean transformer in NLP or electrical engineering?"}'
        )
        needs, question = clarification_agent("Tell me about transformers")
        assert needs
        assert "transformer" in question.lower() or question

    @patch("research.scope._scope_llm")
    def test_brief_generator_returns_valid_brief(self, mock_llm):
        import json
        mock_llm.invoke.return_value = MagicMock(content=json.dumps({
            "main_question": "How do transformers outperform RNNs?",
            "sub_topics": ["transformer architecture", "RNN limitations", "benchmark comparisons"],
            "inclusion_criteria": "peer-reviewed papers from 2017-2024",
            "exclusion_criteria": "non-NLP applications",
            "target_domains": ["NLP", "Deep Learning"],
        }))
        brief = brief_generator("How do transformers outperform RNNs in NLP?")
        assert isinstance(brief, ResearchBrief)
        assert 2 <= len(brief.sub_topics) <= 6
        assert brief.main_question

    @patch("research.scope._scope_llm")
    def test_brief_generator_falls_back_on_error(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content="not json at all")
        brief = brief_generator("some query")
        assert isinstance(brief, ResearchBrief)
        assert len(brief.sub_topics) >= 2


# --- Sub-agent tests ---

class TestSubAgent:
    @pytest.mark.asyncio
    async def test_sub_agent_returns_findings(self):
        import json
        from research.sub_agent import run_sub_agent

        tool_result = json.dumps([
            {"id": "p1", "title": "BERT Paper", "abstract": "BERT is...", "authors": [], "year": 2018, "citation_count": 1000, "doi": ""}
        ])
        compress_result = json.dumps({
            "papers_found": [
                {"id": "p1", "title": "BERT Paper", "abstract": "BERT is...", "authors": [], "year": 2018, "citation_count": 1000, "doi": ""}
            ],
            "key_insights": "BERT revolutionized NLP.",
            "sources_used": ["search_semantic_scholar"],
        })

        # Mock: first LLM call makes a tool call, second returns nothing (done), third is compress
        tool_call_response = MagicMock()
        tool_call_response.tool_calls = [
            {"name": "search_semantic_scholar", "args": {"query": "BERT transformer", "limit": 10}, "id": "tc1"}
        ]
        done_response = MagicMock()
        done_response.tool_calls = []
        done_response.content = "Done searching."
        compress_response = MagicMock()
        compress_response.content = compress_result

        with patch("research.sub_agent._research_llm") as mock_research, \
             patch("research.sub_agent._compress_llm") as mock_compress, \
             patch("research.sub_agent._execute_tool", return_value=tool_result):
            mock_research.invoke.side_effect = [tool_call_response, done_response]
            mock_compress.invoke.return_value = compress_response

            findings = await run_sub_agent("BERT and transformers")

        assert findings.subtopic == "BERT and transformers"
        assert len(findings.papers_found) >= 1


# --- Supervisor tests ---

class TestSupervisor:
    @pytest.mark.asyncio
    async def test_supervisor_runs_parallel_agents(self):
        from research.supervisor import run_supervisor

        brief = ResearchBrief(
            main_question="How do transformers work?",
            sub_topics=["attention mechanism", "positional encoding", "transformer variants"],
            inclusion_criteria="peer-reviewed",
            exclusion_criteria="non-NLP",
            target_domains=["NLP"],
        )

        mock_findings = [
            _make_findings("attention mechanism", [_make_paper("Paper A", "Abstract A")]),
            _make_findings("positional encoding", [_make_paper("Paper B", "Abstract B")]),
            _make_findings("transformer variants", [_make_paper("Paper C", "Abstract C")]),
        ]

        # Mock run_sub_agent and supervisor evaluation (adequate=True immediately)
        with patch("research.supervisor.run_sub_agent", new_callable=AsyncMock) as mock_agent, \
             patch("research.supervisor._supervisor_llm") as mock_llm:
            mock_agent.side_effect = mock_findings
            mock_llm.invoke.return_value = MagicMock(
                content='{"adequate": true, "gaps": []}'
            )
            results = await run_supervisor(brief)

        assert len(results) == 3
        assert mock_agent.call_count == 3

    @pytest.mark.asyncio
    async def test_supervisor_runs_extra_round_for_gaps(self):
        from research.supervisor import run_supervisor

        brief = ResearchBrief(
            main_question="Transformer research",
            sub_topics=["topic A", "topic B"],
            inclusion_criteria="any",
            exclusion_criteria="none",
            target_domains=["CS"],
        )

        batch1 = [_make_findings("topic A", []), _make_findings("topic B", [])]
        batch2 = [_make_findings("gap topic", [_make_paper("Gap Paper", "Abstract")])]

        call_count = {"n": 0}
        async def mock_agent_side_effect(subtopic):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return batch1[call_count["n"] - 1]
            return batch2[0]

        eval_responses = [
            MagicMock(content='{"adequate": false, "gaps": ["gap topic"]}'),
            MagicMock(content='{"adequate": true, "gaps": []}'),
        ]
        eval_count = {"n": 0}
        def mock_eval(*args, **kwargs):
            idx = eval_count["n"]
            eval_count["n"] += 1
            return eval_responses[min(idx, len(eval_responses) - 1)]

        with patch("research.supervisor.run_sub_agent", new_callable=AsyncMock) as mock_agent, \
             patch("research.supervisor._supervisor_llm") as mock_llm:
            mock_agent.side_effect = mock_agent_side_effect
            mock_llm.invoke.side_effect = mock_eval
            results = await run_supervisor(brief)

        # Should have called 3 agents (2 initial + 1 gap)
        assert mock_agent.call_count == 3
        assert len(results) == 3
