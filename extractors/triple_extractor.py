"""Triple extractor: LLM → formal logical triples."""

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from extractors.normalizer import normalize_entity
from extractors.prompts.triple_prompt import TRIPLE_SYSTEM_PROMPT, TRIPLE_USER_TEMPLATE
from extractors.triple_schema import ExtractedTriple

if TYPE_CHECKING:
    from graph.state import Paper

logger = logging.getLogger(__name__)

from config import settings

_extract_llm = ChatOpenAI(
    model=settings.openrouter_model,
    openai_api_key=settings.openrouter_api_key,
    openai_api_base=settings.openrouter_base_url,
    temperature=0,
)
MIN_CONFIDENCE = 0.7


def extract_from_paper(paper: "Paper") -> list[ExtractedTriple]:
    """Extract triples from a single paper's abstract using LLM.

    Returns:
        List of validated, normalized ExtractedTriple objects
    """
    abstract = paper.get("abstract", "").strip()
    if not abstract:
        return []

    paper_id = paper.get("id", "unknown")
    messages = [
        SystemMessage(content=TRIPLE_SYSTEM_PROMPT),
        HumanMessage(content=TRIPLE_USER_TEMPLATE.format(
            paper_id=paper_id,
            abstract=abstract[:3000],  # limit to avoid token overflow
        )),
    ]

    response = _extract_llm.invoke(messages)
    raw = response.content.strip()

    # Strip markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            logger.warning("extract_from_paper: expected list, got %s for paper %s", type(items), paper_id)
            return []
    except json.JSONDecodeError as exc:
        logger.warning("extract_from_paper: JSON parse failed for %s: %s", paper_id, exc)
        return []

    triples: list[ExtractedTriple] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            # Normalize entities
            item["subject"] = normalize_entity(item.get("subject", ""))
            item["obj"] = normalize_entity(item.get("obj", ""))
            item["source_paper"] = paper_id

            triple = ExtractedTriple(**item)

            # Filter by confidence threshold
            if triple.confidence >= MIN_CONFIDENCE:
                triples.append(triple)
        except (ValidationError, KeyError, TypeError) as exc:
            logger.debug("invalid triple skipped: %s | error: %s", item, exc)

    logger.info("extract_from_paper: %d valid triples from paper %s", len(triples), paper_id)
    return triples


async def _extract_one(paper: "Paper", semaphore: asyncio.Semaphore) -> list[ExtractedTriple]:
    async with semaphore:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, extract_from_paper, paper)


async def extract_all(papers: list["Paper"], max_concurrency: int = 5) -> list[ExtractedTriple]:
    """Extract triples from all papers in parallel.

    Args:
        papers: List of Paper dicts with abstract field
        max_concurrency: Maximum parallel LLM calls

    Returns:
        Combined list of all extracted triples
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    results = await asyncio.gather(*[_extract_one(p, semaphore) for p in papers])

    all_triples: list[ExtractedTriple] = []
    for batch in results:
        all_triples.extend(batch)

    logger.info("extract_all: %d total triples from %d papers", len(all_triples), len(papers))
    return all_triples
