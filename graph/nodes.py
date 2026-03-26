"""LangGraph nodes."""

import asyncio
import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from graph.errors import ElicitationError
from graph.prompts.elicit_prompt import ELICIT_SYSTEM_PROMPT, ELICIT_USER_TEMPLATE
from graph.schemas import ElicitationOutput
from graph.state import ResearchState

logger = logging.getLogger(__name__)

from config import settings

_elicit_llm = ChatOpenAI(
    model=settings.openrouter_model,
    openai_api_key=settings.openrouter_api_key,
    openai_api_base=settings.openrouter_base_url,
    temperature=0,
)

MAX_ELICIT_RETRIES = 2


def elicit_domain(state: ResearchState) -> dict:
    """Extract domain, keywords, research_question and intent from the user query.

    Retries up to MAX_ELICIT_RETRIES times if the LLM returns invalid JSON.
    Raises ElicitationError after exhausting retries.
    """
    query = state["user_query"]
    user_message = ELICIT_USER_TEMPLATE.format(query=query)
    messages = [SystemMessage(content=ELICIT_SYSTEM_PROMPT), HumanMessage(content=user_message)]

    last_error: Exception | None = None
    for attempt in range(1, MAX_ELICIT_RETRIES + 1):
        response = _elicit_llm.invoke(messages)
        raw = response.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            data = json.loads(raw)
            output = ElicitationOutput(**data)
            logger.info("elicit_domain succeeded on attempt %d", attempt)
            return {
                "domain": output.domain,
                "keywords": output.keywords,
                "research_question": output.research_question,
                "intent": output.intent,
            }
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            last_error = exc
            logger.warning("elicit_domain attempt %d failed: %s", attempt, exc)
            # Add the failed response + error feedback for next attempt
            messages.append(HumanMessage(content=raw))
            messages.append(
                HumanMessage(
                    content=(
                        f"Your response was not valid JSON or had wrong fields. Error: {exc}. "
                        "Return ONLY the JSON object with fields: domain, keywords, research_question, intent."
                    )
                )
            )

    raise ElicitationError(
        f"elicit_domain failed after {MAX_ELICIT_RETRIES} attempts. Last error: {last_error}"
    )


def research_papers(state: ResearchState) -> dict:
    """Orchestrate the deep research pipeline: scope → multi-agent research → harvest."""
    from research.scope import brief_generator, clarification_agent
    from research.supervisor import run_supervisor
    from research.harvester import harvest_papers
    from config import settings

    query = state["user_query"]

    # Phase 1: Scope
    needs_clarification, _ = clarification_agent(query)
    # Note: in a real HITL flow, we'd interrupt() here to get user answer.
    # For now, proceed without clarification.
    brief = brief_generator(query)
    logger.info("research_papers: brief generated with %d sub_topics", len(brief.sub_topics))

    # Phase 2: Parallel multi-agent research
    all_findings = asyncio.run(run_supervisor(brief))
    logger.info("research_papers: %d findings collected", len(all_findings))

    # Phase 3: Harvest and deduplicate
    papers = harvest_papers(all_findings, max_papers=settings.max_papers)
    logger.info("research_papers: %d papers ready", len(papers))

    return {"papers": papers}


def extract_triples(state: ResearchState) -> dict:
    """Convert paper abstracts into formal logical triples."""
    from extractors.triple_extractor import extract_all

    papers = state.get("papers", [])
    if not papers:
        logger.warning("extract_triples: no papers in state")
        return {"extracted_triples": []}

    triples = asyncio.run(extract_all(papers))
    return {
        "extracted_triples": [t.model_dump() for t in triples],
    }


def persist_to_graph(state: ResearchState) -> dict:
    """Persist papers and triples to Neo4j knowledge graph."""
    from knowledge_base.neo4j_client import Neo4jClient
    from extractors.triple_schema import ExtractedTriple

    papers = state.get("papers", [])
    raw_triples = state.get("extracted_triples", [])

    if not papers and not raw_triples:
        logger.info("persist_to_graph: nothing to persist")
        return {}

    triples = [ExtractedTriple(**t) for t in raw_triples]

    async def _persist():
        async with Neo4jClient() as client:
            await client.setup_schema()
            if papers:
                await client.upsert_papers(papers)
            if triples:
                await client.upsert_triples(triples)

    try:
        asyncio.run(_persist())
        logger.info("persist_to_graph: persisted %d papers, %d triples", len(papers), len(triples))
    except Exception as exc:
        logger.warning("persist_to_graph: Neo4j unavailable, skipping — %s", exc)

    return {}


def reason_datalog(state: ResearchState) -> dict:
    """Run deterministic Datalog inference on extracted triples."""
    from reasoner.clingo_engine import ClingoEngine, triples_to_asp_facts
    from extractors.triple_schema import ExtractedTriple
    from knowledge_base.neo4j_client import Neo4jClient

    raw_triples = state.get("extracted_triples", [])
    if not raw_triples:
        logger.warning("reason_datalog: no triples to reason over")
        return {"inferred_facts": [], "conflicts_detected": []}

    triples = [ExtractedTriple(**t) for t in raw_triples]
    facts = triples_to_asp_facts(triples)

    engine = ClingoEngine()
    inferred = engine.run_datalog(facts)
    conflicts = engine.extract_conflicts(inferred)

    # Persist inferred facts to Neo4j (best-effort)
    async def _persist_inferred():
        async with Neo4jClient() as client:
            await client.upsert_inferred_facts(inferred, rule_source="datalog")

    try:
        asyncio.run(_persist_inferred())
    except Exception as exc:
        logger.warning("reason_datalog: failed to persist to Neo4j — %s", exc)

    logger.info("reason_datalog: %d facts inferred, %d conflicts", len(inferred), len(conflicts))
    return {
        "inferred_facts": inferred,
        "conflicts_detected": conflicts,
    }


def reason_asp_conflicts(state: ResearchState) -> dict:
    """Run non-deterministic ASP solving when conflicts are detected."""
    from reasoner.clingo_engine import ClingoEngine, triples_to_asp_facts
    from extractors.triple_schema import ExtractedTriple
    from config import settings

    raw_triples = state.get("extracted_triples", [])
    triples = [ExtractedTriple(**t) for t in raw_triples]
    facts = triples_to_asp_facts(triples)

    engine = ClingoEngine()
    answer_sets = engine.run_asp(facts, max_models=settings.clingo_max_models)

    # Collect all unique conflict atoms across answer sets
    all_conflicts: set[str] = set()
    for answer_set in answer_sets:
        for atom in answer_set:
            if "conflito_potencial" in atom or "contradiz" in atom:
                all_conflicts.add(atom)

    logger.info("reason_asp_conflicts: %d answer sets, %d conflict atoms", len(answer_sets), len(all_conflicts))
    return {
        "answer_sets": answer_sets,
        "conflicts_detected": list(all_conflicts),
    }


_synthesize_llm = ChatOpenAI(
    model=settings.openrouter_model,
    openai_api_key=settings.openrouter_api_key,
    openai_api_base=settings.openrouter_base_url,
    temperature=0.3,
)


def synthesize_response(state: ResearchState) -> dict:
    """Generate final response grounded in symbolic reasoning results."""
    from graph.prompts.synthesize_prompt import SYNTHESIZE_SYSTEM_PROMPT, SYNTHESIZE_USER_TEMPLATE

    inferred = state.get("inferred_facts", [])
    conflicts = state.get("conflicts_detected", [])
    answer_sets = state.get("answer_sets", [])

    # Build symbolic context (truncated for token efficiency)
    inferred_summary = "\n".join(inferred[:50]) if inferred else "No inferred facts available."
    conflicts_summary = "\n".join(conflicts[:20]) if conflicts else "No conflicts detected — literature appears consistent."
    answer_sets_summary = (
        "\n".join(f"Interpretation {i+1}: {', '.join(atoms[:10])}" for i, atoms in enumerate(answer_sets[:3]))
        if answer_sets else "N/A"
    )

    messages = [
        SystemMessage(content=SYNTHESIZE_SYSTEM_PROMPT),
        HumanMessage(content=SYNTHESIZE_USER_TEMPLATE.format(
            research_question=state.get("research_question", state.get("user_query", "")),
            domain=state.get("domain", "Unknown"),
            keywords=", ".join(state.get("keywords", [])),
            inferred_facts=inferred_summary,
            conflicts=conflicts_summary,
            answer_sets=answer_sets_summary,
            paper_count=len(state.get("papers", [])),
        )),
    ]

    response = _synthesize_llm.invoke(messages)
    logger.info("synthesize_response: %d chars generated", len(response.content))
    return {"final_response": response.content}
