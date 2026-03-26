"""Schema for extracted triples."""

from typing import Literal
from pydantic import BaseModel, field_validator

VALID_PREDICATES = frozenset([
    "cita", "supera", "usa", "trata_conceito", "define", "contradiz", "estende"
])


class ExtractedTriple(BaseModel):
    subject: str
    predicate: Literal["cita", "supera", "usa", "trata_conceito", "define", "contradiz", "estende"]
    obj: str
    source_paper: str
    confidence: float

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {v}")
        return v

    @field_validator("subject", "obj")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("subject and obj must not be empty")
        return v
