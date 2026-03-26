"""Paper harvester: deduplicate and enrich papers from sub-agent findings."""

import difflib
import logging

import httpx

from graph.state import Paper
from research.schemas import PaperRef, SubAgentFindings

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "paperId,title,abstract,authors,year,citationCount,externalIds"
FUZZY_THRESHOLD = 0.85


def _fetch_abstract(paper_id: str) -> str:
    """Fetch paper abstract from Semantic Scholar by ID."""
    if not paper_id or paper_id.startswith("arxiv:"):
        return ""
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(
                f"{SEMANTIC_SCHOLAR_BASE}/paper/{paper_id}",
                params={"fields": "abstract"},
            )
            resp.raise_for_status()
            return resp.json().get("abstract", "") or ""
    except httpx.HTTPError as exc:
        logger.debug("failed to fetch abstract for %s: %s", paper_id, exc)
        return ""


def _normalize_title(title: str) -> str:
    return " ".join(title.lower().split())


def _are_duplicate(a: PaperRef, b: PaperRef) -> bool:
    """Check if two papers are duplicates (by DOI or fuzzy title match)."""
    if a.doi and b.doi and a.doi == b.doi:
        return True
    ratio = difflib.SequenceMatcher(
        None,
        _normalize_title(a.title),
        _normalize_title(b.title),
    ).ratio()
    return ratio > FUZZY_THRESHOLD


def _paper_ref_to_paper(ref: PaperRef) -> Paper:
    return Paper(
        id=ref.id or ref.doi or _normalize_title(ref.title)[:50],
        title=ref.title,
        abstract=ref.abstract,
        authors=ref.authors,
        year=ref.year,
        citation_count=ref.citation_count,
        doi=ref.doi,
    )


def harvest_papers(all_findings: list[SubAgentFindings], max_papers: int = 50) -> list[Paper]:
    """Extract, deduplicate and enrich papers from all sub-agent findings.

    Args:
        all_findings: Results from all sub-agents
        max_papers: Maximum papers to return

    Returns:
        Deduplicated list of Papers with abstracts
    """
    # Collect all paper references
    all_refs: list[PaperRef] = []
    for findings in all_findings:
        all_refs.extend(findings.papers_found)

    logger.info("harvester: %d raw paper refs collected", len(all_refs))

    # Deduplicate
    unique: list[PaperRef] = []
    for candidate in all_refs:
        if not candidate.title.strip():
            continue
        is_dup = any(_are_duplicate(candidate, existing) for existing in unique)
        if not is_dup:
            unique.append(candidate)

    logger.info("harvester: %d unique papers after deduplication", len(unique))

    # Enrich abstracts and filter papers without content
    papers: list[Paper] = []
    for ref in unique[:max_papers]:
        if not ref.abstract and ref.id:
            ref.abstract = _fetch_abstract(ref.id)

        if not ref.abstract.strip():
            logger.debug("skipping paper without abstract: %s", ref.title[:60])
            continue

        papers.append(_paper_ref_to_paper(ref))

    logger.info("harvester: %d papers with abstracts ready", len(papers))
    return papers
