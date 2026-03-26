"""Academic search tools using Semantic Scholar and ArXiv HTTP APIs."""

import asyncio
import logging
import urllib.parse

import httpx
from langchain_core.tools import tool

from research.schemas import PaperRef

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
ARXIV_BASE = "https://export.arxiv.org/api"

_FIELDS = "paperId,title,abstract,authors,year,citationCount,externalIds"


def _to_paper_ref(item: dict) -> PaperRef:
    """Convert a Semantic Scholar API response item to PaperRef."""
    doi = ""
    ext = item.get("externalIds") or {}
    if isinstance(ext, dict):
        doi = ext.get("DOI", "")

    return PaperRef(
        id=item.get("paperId", ""),
        title=item.get("title", ""),
        abstract=item.get("abstract", "") or "",
        authors=[a.get("name", "") for a in (item.get("authors") or [])],
        year=item.get("year") or 0,
        citation_count=item.get("citationCount") or 0,
        doi=doi,
    )


@tool
def search_semantic_scholar(query: str, limit: int = 10) -> list[dict]:
    """Search Semantic Scholar for academic papers.

    Args:
        query: Search query string
        limit: Maximum number of results (default 10, max 100)

    Returns:
        List of paper dicts with id, title, abstract, authors, year, citation_count
    """
    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": _FIELDS,
    }
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(f"{SEMANTIC_SCHOLAR_BASE}/paper/search", params=params)
            resp.raise_for_status()
            data = resp.json()
            papers = [_to_paper_ref(item).model_dump() for item in data.get("data", [])]
            logger.info("semantic_scholar: %d results for '%s'", len(papers), query)
            return papers
    except httpx.HTTPError as exc:
        logger.warning("semantic_scholar search failed: %s", exc)
        return []


@tool
def search_arxiv(query: str, limit: int = 10) -> list[dict]:
    """Search ArXiv for academic preprints.

    Args:
        query: Search query (supports ArXiv query syntax)
        limit: Maximum number of results

    Returns:
        List of paper dicts with id, title, abstract, authors, year
    """
    import xml.etree.ElementTree as ET

    params = {
        "search_query": f"all:{urllib.parse.quote(query)}",
        "max_results": min(limit, 100),
        "sortBy": "relevance",
    }
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(f"{ARXIV_BASE}/query", params=params)
            resp.raise_for_status()

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)
        papers = []
        for entry in root.findall("atom:entry", ns):
            title = (entry.findtext("atom:title", "", ns) or "").strip()
            abstract = (entry.findtext("atom:summary", "", ns) or "").strip()
            arxiv_id = (entry.findtext("atom:id", "", ns) or "").split("/abs/")[-1]
            authors = [
                a.findtext("atom:name", "", ns) or ""
                for a in entry.findall("atom:author", ns)
            ]
            published = entry.findtext("atom:published", "", ns) or ""
            year = int(published[:4]) if published else 0

            papers.append(PaperRef(
                id=f"arxiv:{arxiv_id}",
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
            ).model_dump())

        logger.info("arxiv: %d results for '%s'", len(papers), query)
        return papers
    except Exception as exc:
        logger.warning("arxiv search failed: %s", exc)
        return []


ACADEMIC_TOOLS = [search_semantic_scholar, search_arxiv]
