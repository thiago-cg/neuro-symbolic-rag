"""Pydantic schemas for the research pipeline."""

from pydantic import BaseModel, field_validator


class ResearchBrief(BaseModel):
    main_question: str
    sub_topics: list[str]
    inclusion_criteria: str
    exclusion_criteria: str
    target_domains: list[str]

    @field_validator("sub_topics")
    @classmethod
    def validate_sub_topics(cls, v: list[str]) -> list[str]:
        if not 2 <= len(v) <= 6:
            raise ValueError(f"sub_topics must have 2-6 items, got {len(v)}")
        return v


class PaperRef(BaseModel):
    id: str = ""
    title: str
    abstract: str = ""
    authors: list[str] = []
    year: int = 0
    citation_count: int = 0
    doi: str = ""


class SubAgentFindings(BaseModel):
    subtopic: str
    papers_found: list[PaperRef]
    key_insights: str
    sources_used: list[str]
