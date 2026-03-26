"""Supervisor agent: orchestrates parallel sub-agents and evaluates coverage."""

import asyncio
import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from research.schemas import ResearchBrief, SubAgentFindings
from research.sub_agent import run_sub_agent

logger = logging.getLogger(__name__)

MAX_ROUNDS = 3
MAX_CONCURRENT = 4

from config import settings

_supervisor_llm = ChatOpenAI(
    model=settings.openrouter_model,
    openai_api_key=settings.openrouter_api_key,
    openai_api_base=settings.openrouter_base_url,
    temperature=0,
)

_EVAL_PROMPT = """You are a research supervisor evaluating if the findings adequately cover the research brief.

RESEARCH BRIEF:
{brief}

FINDINGS SUMMARY:
{findings_summary}

Return JSON:
{{
  "adequate": true/false,
  "gaps": ["<gap1>", "<gap2>"]  // empty list if adequate
}}

Return ONLY valid JSON."""


def _summarize_findings(findings: list[SubAgentFindings]) -> str:
    lines = []
    for f in findings:
        lines.append(f"Subtopic: {f.subtopic}")
        lines.append(f"  Papers: {len(f.papers_found)}")
        lines.append(f"  Insights: {f.key_insights[:200]}")
    return "\n".join(lines)


async def _run_batch(subtopics: list[str], semaphore: asyncio.Semaphore) -> list[SubAgentFindings]:
    """Run sub-agents in parallel with concurrency limit."""
    async def bounded(topic: str) -> SubAgentFindings:
        async with semaphore:
            return await run_sub_agent(topic)

    return list(await asyncio.gather(*[bounded(t) for t in subtopics]))


async def run_supervisor(brief: ResearchBrief) -> list[SubAgentFindings]:
    """Orchestrate parallel sub-agents across multiple rounds if needed."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    all_findings: list[SubAgentFindings] = []
    remaining_topics = list(brief.sub_topics)

    for round_num in range(1, MAX_ROUNDS + 1):
        logger.info("supervisor round %d: %d subtopics", round_num, len(remaining_topics))
        batch = await _run_batch(remaining_topics, semaphore)
        all_findings.extend(batch)

        # Evaluate coverage after first round (subsequent rounds fill gaps)
        if round_num < MAX_ROUNDS:
            eval_messages = [
                SystemMessage(content=_EVAL_PROMPT.format(
                    brief=brief.model_dump_json(indent=2),
                    findings_summary=_summarize_findings(all_findings),
                )),
                HumanMessage(content="Evaluate coverage."),
            ]
            eval_response = _supervisor_llm.invoke(eval_messages)
            raw = eval_response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()

            try:
                evaluation = json.loads(raw)
                if evaluation.get("adequate", True):
                    logger.info("supervisor: coverage adequate after round %d", round_num)
                    break
                gaps = evaluation.get("gaps", [])
                if not gaps:
                    break
                remaining_topics = gaps[:MAX_CONCURRENT]  # limit new topics
                logger.info("supervisor: %d gaps identified, running extra round", len(gaps))
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("supervisor evaluation failed: %s", exc)
                break

    return all_findings
