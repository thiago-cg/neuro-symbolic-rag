"""Pydantic schemas for LLM output validation."""

from typing import Literal
from pydantic import BaseModel, field_validator


class ElicitationOutput(BaseModel):
    domain: str
    keywords: list[str]
    research_question: str
    intent: Literal["survey", "comparison", "specific_question"]

    @field_validator("keywords")
    @classmethod
    def keywords_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("keywords must not be empty")
        return [kw.strip().lower() for kw in v if kw.strip()]

    @field_validator("domain")
    @classmethod
    def domain_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("domain must not be empty")
        return v.strip()
