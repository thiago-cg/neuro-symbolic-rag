"""Sub-agent: focused academic search for a single subtopic."""

import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from research.academic_tools import ACADEMIC_TOOLS
from research.schemas import PaperRef, SubAgentFindings

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 10
from config import settings

_research_llm = ChatOpenAI(
    model=settings.openrouter_model,
    openai_api_key=settings.openrouter_api_key,
    openai_api_base=settings.openrouter_base_url,
    temperature=0,
).bind_tools(ACADEMIC_TOOLS)
_compress_llm = ChatOpenAI(
    model=settings.openrouter_model,
    openai_api_key=settings.openrouter_api_key,
    openai_api_base=settings.openrouter_base_url,
    temperature=0,
)

_RESEARCH_PROMPT = """You are a focused academic researcher. Search for papers on this specific subtopic:

SUBTOPIC: {subtopic}

Use the available search tools to find relevant academic papers. Search multiple times with different queries to maximize coverage.
Stop when you have found 5-15 relevant papers or have done 10 searches."""

_COMPRESS_PROMPT = """Summarize the academic research findings for this subtopic into a structured JSON:

SUBTOPIC: {subtopic}

FINDINGS:
{findings_text}

Return JSON:
{{
  "papers_found": [
    {{"id": "...", "title": "...", "abstract": "...", "authors": [], "year": 0, "citation_count": 0, "doi": ""}}
  ],
  "key_insights": "<2-3 sentence summary of main findings>",
  "sources_used": ["<tool names used>"]
}}

Return ONLY valid JSON."""


def _execute_tool(tool_call: dict) -> str:
    """Execute a tool call and return its result as string."""
    name = tool_call["name"]
    args = tool_call["args"]

    tool_map = {t.name: t for t in ACADEMIC_TOOLS}
    if name not in tool_map:
        return f"Unknown tool: {name}"

    try:
        result = tool_map[name].invoke(args)
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.warning("tool %s failed: %s", name, exc)
        return f"Tool error: {exc}"


async def run_sub_agent(subtopic: str) -> SubAgentFindings:
    """Run a focused academic search for a single subtopic."""
    messages = [
        SystemMessage(content=_RESEARCH_PROMPT.format(subtopic=subtopic)),
        HumanMessage(content=f"Start searching for papers on: {subtopic}"),
    ]
    sources_used: set[str] = set()
    raw_findings: list[str] = []

    for iteration in range(_MAX_ITERATIONS):
        response = _research_llm.invoke(messages)
        messages.append(response)

        # If no tool calls, the agent is done
        if not response.tool_calls:
            logger.info("sub_agent '%s' finished after %d iterations", subtopic, iteration + 1)
            break

        # Execute all tool calls
        for tool_call in response.tool_calls:
            sources_used.add(tool_call["name"])
            result = _execute_tool(tool_call)
            raw_findings.append(result)
            messages.append(
                ToolMessage(content=result, tool_call_id=tool_call["id"])
            )

    # Compress findings
    findings_text = "\n---\n".join(raw_findings) if raw_findings else "No papers found."
    compress_messages = [
        SystemMessage(content=_COMPRESS_PROMPT.format(
            subtopic=subtopic,
            findings_text=findings_text[:8000],  # limit context
        )),
        HumanMessage(content="Summarize the findings."),
    ]

    compress_response = _compress_llm.invoke(compress_messages)
    raw = compress_response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        data = json.loads(raw)
        papers = [PaperRef(**p) for p in data.get("papers_found", [])]
        return SubAgentFindings(
            subtopic=subtopic,
            papers_found=papers,
            key_insights=data.get("key_insights", ""),
            sources_used=list(sources_used) or data.get("sources_used", []),
        )
    except Exception as exc:
        logger.warning("sub_agent compress failed for '%s': %s", subtopic, exc)
        return SubAgentFindings(
            subtopic=subtopic,
            papers_found=[],
            key_insights=findings_text[:500],
            sources_used=list(sources_used),
        )
