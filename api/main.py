"""FastAPI application — VFS Neuro-Symbolic Research System."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Annotated, AsyncGenerator

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.schemas import GraphQueryRequest, HealthResponse, QueryRequest, QueryResponse
from config import settings

logger = logging.getLogger(__name__)

# ── Rate limiting ──────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

# ── Neo4j client (shared across requests) ─────────────────────────────────────

_neo4j_client = None


async def get_neo4j():
    """Dependency that returns the shared Neo4j client."""
    if _neo4j_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge graph not connected",
        )
    return _neo4j_client


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Connect to Neo4j on startup, disconnect on shutdown."""
    global _neo4j_client
    from knowledge_base.neo4j_client import Neo4jClient

    client = Neo4jClient()
    try:
        await client.connect()
        await client.setup_schema()
        _neo4j_client = client
        logger.info("startup: Neo4j connected")
    except Exception as exc:
        logger.warning("startup: Neo4j unavailable — graph features disabled: %s", exc)
        _neo4j_client = None

    yield

    if _neo4j_client:
        await _neo4j_client.close()
        logger.info("shutdown: Neo4j disconnected")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VFS Neuro-Symbolic Research System",
    description="Academic research pipeline combining LLMs with symbolic reasoning (Clingo/ASP) and knowledge graphs (Neo4j).",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """System health check."""
    neo4j_status = "connected" if _neo4j_client else "unavailable"
    return HealthResponse(status="ok", neo4j=neo4j_status)


# ── Research ───────────────────────────────────────────────────────────────────

@app.post(
    "/research",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    tags=["Research"],
    summary="Run the full neuro-symbolic research pipeline",
)
@limiter.limit("10/minute")
async def run_research(request: Request, payload: QueryRequest) -> QueryResponse:
    """Execute the full pipeline for a research query.

    Runs: elicit → research → extract triples → reason → synthesize
    """
    from graph.graph import build_graph
    from config import settings as cfg

    graph = build_graph()
    initial_state = {
        "user_query": payload.user_query,
        "max_papers": payload.max_papers,
    }

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: graph.invoke(initial_state),
        )
    except Exception as exc:
        logger.error("pipeline error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {exc}",
        )

    return QueryResponse(
        final_response=result.get("final_response", ""),
        inferred_facts=result.get("inferred_facts", []),
        conflicts_detected=result.get("conflicts_detected", []),
        papers_analyzed=len(result.get("papers", [])),
        answer_sets=result.get("answer_sets", []),
    )


# ── SSE Streaming ─────────────────────────────────────────────────────────────

@app.get(
    "/research/stream",
    tags=["Research"],
    summary="Stream research pipeline progress via Server-Sent Events",
)
@limiter.limit("5/minute")
async def stream_research(
    request: Request,
    user_query: str = Query(..., min_length=3, description="The research question"),
    max_papers: int = Query(default=30, ge=1, le=100),
) -> StreamingResponse:
    """Stream pipeline progress using Server-Sent Events (SSE).

    Emits events: elicit_done, papers_found, triples_extracted, reasoning_done, complete
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        from graph.state import ResearchState
        from graph.nodes import (
            elicit_domain, research_papers, extract_triples,
            persist_to_graph, reason_datalog, reason_asp_conflicts, synthesize_response,
        )
        from graph.router import should_run_asp

        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        state: dict = {"user_query": user_query}

        try:
            # 1. Elicitation
            yield sse("progress", {"step": "elicit", "message": "Extracting research intent..."})
            update = await asyncio.get_event_loop().run_in_executor(None, elicit_domain, state)
            state.update(update)
            yield sse("elicit_done", {"domain": state.get("domain"), "intent": state.get("intent")})

            # 2. Research
            yield sse("progress", {"step": "research", "message": "Searching academic papers..."})
            state["papers"] = state.get("papers", [])
            state["extracted_triples"] = state.get("extracted_triples", [])
            state["inferred_facts"] = state.get("inferred_facts", [])
            state["conflicts_detected"] = state.get("conflicts_detected", [])
            state["answer_sets"] = state.get("answer_sets", [])
            update = await asyncio.get_event_loop().run_in_executor(None, research_papers, state)
            state.update(update)
            yield sse("papers_found", {"count": len(state.get("papers", []))})

            # 3. Extract triples
            yield sse("progress", {"step": "extract", "message": "Extracting symbolic triples..."})
            update = await asyncio.get_event_loop().run_in_executor(None, extract_triples, state)
            state.update(update)
            yield sse("triples_extracted", {"count": len(state.get("extracted_triples", []))})

            # 4. Persist
            await asyncio.get_event_loop().run_in_executor(None, persist_to_graph, state)

            # 5. Datalog reasoning
            yield sse("progress", {"step": "reason", "message": "Running symbolic inference..."})
            update = await asyncio.get_event_loop().run_in_executor(None, reason_datalog, state)
            state.update(update)

            # 6. ASP if conflicts
            if should_run_asp(state) == "reason_asp_conflicts":
                yield sse("progress", {"step": "asp", "message": "Resolving conflicts with ASP..."})
                update = await asyncio.get_event_loop().run_in_executor(None, reason_asp_conflicts, state)
                state.update(update)

            yield sse("reasoning_done", {
                "inferred_count": len(state.get("inferred_facts", [])),
                "conflicts": len(state.get("conflicts_detected", [])),
            })

            # 7. Synthesize
            yield sse("progress", {"step": "synthesize", "message": "Generating synthesis..."})
            update = await asyncio.get_event_loop().run_in_executor(None, synthesize_response, state)
            state.update(update)

            yield sse("complete", {
                "final_response": state.get("final_response", ""),
                "papers_analyzed": len(state.get("papers", [])),
            })

        except Exception as exc:
            logger.error("stream error: %s", exc, exc_info=True)
            yield sse("error", {"message": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Knowledge Graph ────────────────────────────────────────────────────────────

@app.get(
    "/graph/query",
    tags=["Knowledge Graph"],
    summary="Execute a read-only Cypher query against the knowledge graph",
)
@limiter.limit("30/minute")
async def query_graph(
    request: Request,
    cypher: str = Query(..., description="Cypher read query (no writes allowed)"),
) -> list[dict]:
    """Execute a Cypher query on the Neo4j knowledge graph.

    Example: `MATCH (p:Paper) RETURN p.title LIMIT 5`
    """
    # Reject write operations for safety
    upper = cypher.strip().upper()
    if any(upper.startswith(op) for op in ("CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only read (MATCH) queries are allowed via this endpoint",
        )

    client = await get_neo4j()
    try:
        return await client.run_cypher(cypher)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Query error: {exc}",
        )
