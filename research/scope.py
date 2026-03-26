"""Scope phase: clarification + research brief generation."""

import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from research.schemas import ResearchBrief

logger = logging.getLogger(__name__)

from config import settings

_scope_llm = ChatOpenAI(
    model=settings.openrouter_model,
    openai_api_key=settings.openrouter_api_key,
    openai_api_base=settings.openrouter_base_url,
    temperature=0,
)

_CLARIFICATION_PROMPT = """You are an academic research assistant. Analyze the query and determine if clarification is needed.

If the query is clear enough for academic research (domain is obvious, scope is reasonable), respond with exactly:
{"needs_clarification": false}

If clarification would significantly improve the research (ambiguous domain, multiple unrelated interpretations), respond with:
{"needs_clarification": true, "question": "<one specific question>"}

Return ONLY valid JSON. Do not ask more than one question."""

_BRIEF_PROMPT = """You are an expert research strategist. Generate a structured research brief from the query.

Return a JSON object with these EXACT fields:
{
  "main_question": "<the core research question>",
  "sub_topics": ["<subtopic1>", "<subtopic2>", "<subtopic3>"],
  "inclusion_criteria": "<what papers to include>",
  "exclusion_criteria": "<what papers to exclude>",
  "target_domains": ["<domain1>", "<domain2>"]
}

Rules:
- sub_topics: 2-6 items, each is a specific angle on the main question
- Return ONLY valid JSON"""


def clarification_agent(query: str) -> tuple[bool, str]:
    """Check if query needs clarification.

    Returns:
        (needs_clarification, question_if_needed)
    """
    messages = [
        SystemMessage(content=_CLARIFICATION_PROMPT),
        HumanMessage(content=f"Query: {query}"),
    ]
    response = _scope_llm.invoke(messages)
    raw = response.content.strip()

    try:
        data = json.loads(raw)
        if data.get("needs_clarification"):
            return True, data.get("question", "")
        return False, ""
    except (json.JSONDecodeError, KeyError):
        logger.warning("clarification_agent returned invalid JSON: %s", raw)
        return False, ""


def brief_generator(query: str, clarification_answer: str = "") -> ResearchBrief:
    """Generate a structured research brief from query (+ optional clarification)."""
    context = query
    if clarification_answer:
        context = f"{query}\n\nAdditional context: {clarification_answer}"

    messages = [
        SystemMessage(content=_BRIEF_PROMPT),
        HumanMessage(content=context),
    ]

    for attempt in range(2):
        response = _scope_llm.invoke(messages)
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            data = json.loads(raw)
            brief = ResearchBrief(**data)
            logger.info("brief_generator: %d sub_topics generated", len(brief.sub_topics))
            return brief
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("brief_generator attempt %d failed: %s", attempt + 1, exc)
            messages.append(HumanMessage(content=f"Invalid response. Error: {exc}. Return only valid JSON."))

    # Fallback: minimal brief
    logger.error("brief_generator fell back to minimal brief")
    return ResearchBrief(
        main_question=query,
        sub_topics=[query, "related work and background"],
        inclusion_criteria="peer-reviewed academic papers",
        exclusion_criteria="non-academic sources",
        target_domains=["Computer Science"],
    )
