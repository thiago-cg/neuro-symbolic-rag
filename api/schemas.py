"""FastAPI request/response schemas."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    user_query: str = Field(..., min_length=3, max_length=1000, description="The research question")
    max_papers: int = Field(default=30, ge=1, le=100, description="Maximum papers to analyze")


class QueryResponse(BaseModel):
    final_response: str
    inferred_facts: list[str]
    conflicts_detected: list[str]
    papers_analyzed: int
    answer_sets: list[list[str]] = []


class GraphQueryRequest(BaseModel):
    cypher: str = Field(..., description="Cypher query to execute against the knowledge graph")
    params: dict = Field(default_factory=dict, description="Optional query parameters")


class HealthResponse(BaseModel):
    status: str
    neo4j: str = "unknown"
    version: str = "0.1.0"
