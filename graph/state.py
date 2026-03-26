from typing import Annotated
import operator
from typing_extensions import TypedDict


class Paper(TypedDict):
    id: str
    title: str
    abstract: str
    authors: list[str]
    year: int
    citation_count: int
    doi: str


class Triple(TypedDict):
    subject: str
    predicate: str
    object: str
    source_paper: str
    confidence: float


class ResearchState(TypedDict):
    # Input
    user_query: str

    # Elicitation output
    domain: str
    keywords: list[str]
    research_question: str
    intent: str  # "survey" | "comparison" | "specific"

    # Research output
    papers: list[Paper]

    # Extraction output
    extracted_triples: Annotated[list[Triple], operator.add]

    # Reasoner output
    asp_program: str
    inferred_facts: Annotated[list[str], operator.add]
    conflicts_detected: list[str]
    answer_sets: list[list[str]]

    # Final
    final_response: str
