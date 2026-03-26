<div align="center">

# Neuro-Symbolic Research System

**A hybrid academic research pipeline combining Large Language Models with formal symbolic reasoning**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-4B8BBE?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Clingo](https://img.shields.io/badge/Clingo-5.7%2B-orange?style=flat-square)](https://potassco.org/clingo/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-008CC1?style=flat-square&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Tests](https://img.shields.io/badge/tests-97%20passed-brightgreen?style=flat-square)](./tests/)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](./LICENSE)
[![uv](https://img.shields.io/badge/uv-managed-5C4EE5?style=flat-square)](https://docs.astral.sh/uv/)

<br/>

> *"The marriage of statistical pattern recognition with formal logical inference — giving AI systems the ability not just to predict, but to reason."*

<br/>

[Getting Started](#-getting-started) •
[Architecture](#-architecture) •
[API Reference](#-api-reference) •
[Symbolic Reasoning](#-symbolic-reasoning-aspdatalog) •
[Contributing](#-contributing) •
[Citation](#-citation)

</div>

---

## Abstract

The **VFS Neuro-Symbolic Research System** is an end-to-end academic research pipeline that automates the process of literature discovery, knowledge extraction, logical inference, and evidence-grounded synthesis. Given a natural language research question, the system:

1. Elicits research intent and domain via structured LLM prompting
2. Deploys a multi-agent supervisor architecture to retrieve papers from Semantic Scholar and ArXiv
3. Extracts formal logical triples from paper abstracts using controlled predicate vocabularies
4. Persists a queryable knowledge graph in Neo4j
5. Applies **Datalog** rules for deterministic transitive inference and conflict detection
6. Invokes **Answer Set Programming (ASP)** to enumerate all consistent interpretations when contradictions arise
7. Synthesizes a grounded final response enriched with symbolic evidence

This architecture addresses a fundamental limitation of purely neural approaches: the inability to perform sound logical inference and to explicitly represent contradictions discovered across the literature.

---

## Table of Contents

- [Motivation](#-motivation)
- [Architecture](#-architecture)
- [Pipeline Execution](#-pipeline-execution)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Usage](#-usage)
  - [Start the Server](#start-the-server)
  - [REST API](#rest-api)
  - [Server-Sent Events (Streaming)](#server-sent-events-streaming)
  - [Knowledge Graph Queries](#knowledge-graph-queries)
  - [Programmatic Usage](#programmatic-usage)
- [API Reference](#-api-reference)
- [Core Modules](#-core-modules)
- [Symbolic Reasoning (ASP/Datalog)](#-symbolic-reasoning-aspdatalog)
- [Testing](#-testing)
- [Environment Variables](#-environment-variables)
- [Technology Stack](#-technology-stack)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)

---

## 💡 Motivation

Modern large language models excel at language understanding and generation but exhibit well-documented weaknesses in formal reasoning: they hallucinate facts, fail to detect logical contradictions across sources, and cannot guarantee sound inference chains. Classical symbolic AI, on the other hand, offers formal correctness guarantees but requires hand-engineered knowledge bases and cannot scale to unstructured natural language.

This project bridges that gap through a **neuro-symbolic** architecture:

| Capability | Neural (LLM) | Symbolic (Clingo) | This System |
|------------|:------------:|:-----------------:|:-----------:|
| Natural language understanding | ✅ | ❌ | ✅ |
| Structured knowledge extraction | ✅ | ❌ | ✅ |
| Sound transitive inference | ❌ | ✅ | ✅ |
| Contradiction detection | ❌ | ✅ | ✅ |
| Non-monotonic reasoning | ❌ | ✅ | ✅ |
| Evidence-grounded synthesis | Partial | ❌ | ✅ |

The approach is inspired by the **Open Deep Research** multi-agent retrieval architecture and extends it with a formal symbolic reasoning layer based on the **Potassco/Clingo** Answer Set Programming solver.

---

## 🏗 Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              FastAPI Layer                                 │
│          POST /research   GET /research/stream   GET /graph/query          │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   │
                       ┌───────────▼────────────┐
                       │    LangGraph Pipeline   │
                       │    (StateGraph · 7 nodes)│
                       └───────────┬────────────┘
                                   │
    ┌──────────┐  ┌──────────┐  ┌──▼───────┐  ┌────────────┐  ┌────────────┐
    │  Elicit  │→ │ Research │→ │ Extract  │→ │  Persist   │→ │  Datalog   │
    │  Domain  │  │ (Multi-  │  │ Triples  │  │  to Neo4j  │  │ Inference  │
    │  (LLM)   │  │  Agent)  │  │  (LLM)   │  │            │  │ (Clingo)   │
    └──────────┘  └──────────┘  └──────────┘  └────────────┘  └─────┬──────┘
                                                                      │
                                                          conflicts?  │
                                                       ┌──────────────┴──────────────┐
                                                       │ YES                         │ NO
                                                ┌──────▼──────┐               ┌─────▼──────┐
                                                │  ASP Conflict│               │ Synthesize │
                                                │  Resolution  │──────────────▶│  Response  │
                                                │  (Clingo)    │               │   (LLM)    │
                                                └─────────────┘               └────────────┘
                                                                                      │
                                              ┌───────────────────────────┐          │
                                              │      External Services     │          ▼
                                              │  Neo4j · Semantic Scholar  │       RESPONSE
                                              │  ArXiv · OpenRouter LLM   │
                                              └───────────────────────────┘
```

### Design Principles

- **Separation of concerns** — each pipeline node has a single responsibility and communicates only via the shared `ResearchState`
- **Graceful degradation** — all external services (Neo4j, academic APIs) are optional; the pipeline continues with reduced functionality if unavailable
- **Formal grounding** — the final LLM response is conditioned on symbolically verified facts, not raw retrieved text
- **Reproducibility** — the Datalog/ASP programs are deterministic given the same extracted triples; symbolic inference results are logged and persisted

---

## 🔄 Pipeline Execution

The pipeline is a compiled LangGraph `StateGraph` with **7 nodes** and one conditional branch:

```
START
  │
  ▼
[1] elicit_domain       ── Extracts domain, keywords, intent from the user query
  │
  ▼
[2] research_papers     ── Multi-agent paper retrieval (Semantic Scholar + ArXiv)
  │
  ▼
[3] extract_triples     ── LLM extracts formal (subject, predicate, object) triples
  │
  ▼
[4] persist_to_graph    ── Upserts papers, triples, and concepts to Neo4j
  │
  ▼
[5] reason_datalog      ── Clingo: transitive inference, comparability, conflict detection
  │
  ├── conflicts_detected? ──YES──▶ [6] reason_asp_conflicts  (non-deterministic ASP)
  │                                           │
  └── no conflicts ──────────────▶ [7] synthesize_response ◀─┘
                                              │
                                             END
```

### Node Descriptions

| # | Node | Model | Description |
|---|------|-------|-------------|
| 1 | `elicit_domain` | LLM | Structured JSON extraction: `domain`, `keywords`, `research_question`, `intent` |
| 2 | `research_papers` | LLM + Tools | Supervisor + up to 4 parallel sub-agents; 3 refinement rounds; DOI/fuzzy deduplication |
| 3 | `extract_triples` | LLM | Extracts triples from each abstract with 7 controlled predicates; `confidence ≥ 0.7` |
| 4 | `persist_to_graph` | — | Async upsert to Neo4j: `(:Paper)`, `(:Concept)`, `[:RELATION]`, `[:INFERRED]` |
| 5 | `reason_datalog` | Clingo | Deterministic mode: `influencia_direta`, `influencia` (transitive), `comparavel`, `conflito_potencial` |
| 6 | `reason_asp_conflicts` | Clingo | Non-deterministic mode: enumerates all consistent answer sets for detected conflicts |
| 7 | `synthesize_response` | LLM | Generates a final answer grounded in symbolic evidence |

### Triple Predicate Vocabulary

The extraction layer uses a **closed predicate set** to enable formal reasoning:

| Predicate | Semantics |
|-----------|-----------|
| `cita` | A cites B as a reference |
| `supera` | A outperforms B |
| `usa` | A uses technique/method B |
| `trata_conceito` | A addresses concept B |
| `define` | A formally defines B |
| `contradiz` | A contradicts the claim of B |
| `estende` | A extends the work of B |

---

## 📁 Repository Structure

```
neuro_symbolic/
│
├── api/                        # Delivery layer
│   ├── main.py                 # FastAPI app: lifespan, CORS, rate limiting, endpoints
│   └── schemas.py              # Pydantic V2: QueryRequest, QueryResponse, HealthResponse
│
├── extractors/                 # Triple extraction layer
│   ├── normalizer.py           # Entity normalization (lowercase, alias resolution)
│   ├── triple_extractor.py     # Async LLM extraction (max_concurrency=5)
│   ├── triple_schema.py        # ExtractedTriple (Pydantic V2, validated predicates)
│   └── validator.py            # ValidationReport for triple batches
│
├── graph/                      # LangGraph orchestration
│   ├── errors.py               # ElicitationError
│   ├── graph.py                # build_graph() → CompiledGraph
│   ├── nodes.py                # All 7 node implementations
│   ├── router.py               # should_run_asp() — conditional branch
│   ├── schemas.py              # ElicitationOutput (Pydantic V2)
│   ├── state.py                # ResearchState, Paper, Triple (TypedDict)
│   └── prompts/
│       ├── elicit_prompt.py    # System + user prompt for domain elicitation
│       └── synthesize_prompt.py
│
├── knowledge_base/             # Graph persistence layer
│   └── neo4j_client.py         # Neo4jClient: async driver, schema setup, upserts
│
├── reasoner/                   # Symbolic reasoning layer
│   ├── clingo_engine.py        # ClingoEngine: run_datalog(), run_asp()
│   ├── rules.asp               # Datalog rules: influencia, comparavel, conflito_potencial
│   └── asp_conflict.asp        # ASP rules: posicao_aceita, evidencia, consenso
│
├── research/                   # Multi-agent retrieval layer
│   ├── academic_tools.py       # LangChain @tools: Semantic Scholar, ArXiv
│   ├── harvester.py            # Paper deduplication and abstract fetching
│   ├── schemas.py              # ResearchBrief, PaperRef, SubAgentFindings
│   ├── scope.py                # clarification_agent, brief_generator
│   ├── sub_agent.py            # run_sub_agent() — focused tool-calling loop
│   └── supervisor.py           # SupervisorAgent — parallel orchestration
│
├── tests/                      # Test suite (97 tests, all mocked)
│   ├── test_api.py             # 12 API endpoint tests
│   ├── test_clingo.py          # 14 Clingo/ASP engine tests
│   ├── test_deep_research.py   # 13 multi-agent research tests
│   ├── test_elicit.py          # 9 elicitation agent tests
│   ├── test_extractor.py       # 15 triple extraction tests
│   ├── test_graph_skeleton.py  # 5 LangGraph structure tests
│   ├── test_neo4j_client.py    # 8 Neo4j client tests
│   ├── test_pipeline_integration.py  # 3 end-to-end integration tests
│   └── test_resilience.py      # 18 resilience and failure-mode tests
│
├── config.py                   # Centralized settings via pydantic-settings
├── observability.py            # structlog JSON logging, @log_node_execution decorator
├── retry.py                    # tenacity policies: llm_retry (3×), api_retry (5×)
└── pyproject.toml              # uv-managed project with all dependencies
```

---

## 🚀 Getting Started

### Prerequisites

| Dependency | Version | Required | Notes |
|------------|---------|----------|-------|
| Python | 3.11+ | Yes | |
| [uv](https://docs.astral.sh/uv/) | latest | Yes | Package manager |
| Neo4j | 5.x | No | System degrades gracefully without it |
| OpenRouter account | — | Yes | LLM access via OpenAI-compatible API |

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/vfs-neuro-symbolic.git
cd vfs-neuro-symbolic/neuro_symbolic

# Create virtual environment and install all dependencies
uv sync

# Verify the installation
uv run python -c "from graph.graph import build_graph; g = build_graph(); print('Graph OK:', g)"
uv run python -c "from config import settings; print('Config OK:', settings.openrouter_model)"
```

### Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

```env
# ── LLM Provider (required) ───────────────────────────────────────────────────
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free

# ── Knowledge Graph (optional — pipeline works without Neo4j) ─────────────────
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

# ── Academic APIs (optional — improves retrieval quality) ─────────────────────
SEMANTIC_SCHOLAR_KEY=your-key

# ── Observability (optional) ──────────────────────────────────────────────────
LANGSMITH_API_KEY=ls__...
LANGCHAIN_TRACING_V2=false

# ── Pipeline Tuning ───────────────────────────────────────────────────────────
MAX_PARALLEL_AGENTS=4       # concurrent sub-agents during retrieval
MAX_PAPERS=50               # maximum papers per query
MIN_TRIPLE_CONFIDENCE=0.7   # confidence threshold for triple acceptance
CLINGO_MAX_MODELS=10        # maximum answer sets in ASP mode
LOG_LEVEL=INFO
```

#### Running Neo4j with Docker

```bash
docker run -d \
  --name neo4j-vfs \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your-password \
  neo4j:5
```

The Neo4j Browser will be available at `http://localhost:7474`.

---

## 📖 Usage

### Start the Server

```bash
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API documentation:
- **Swagger UI** → `http://localhost:8000/docs`
- **ReDoc** → `http://localhost:8000/redoc`
- **OpenAPI JSON** → `http://localhost:8000/openapi.json`

---

### REST API

#### Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "neo4j": "connected",
  "version": "0.1.0"
}
```

#### Run Full Pipeline

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "user_query": "How do transformer architectures outperform RNNs in NLP tasks?",
    "max_papers": 20
  }'
```

**Request schema:**

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `user_query` | `string` | required | 3–1000 chars | Natural language research question |
| `max_papers` | `integer` | `30` | 1–100 | Maximum number of papers to analyze |

**Response schema:**

```json
{
  "final_response": "Transformer architectures outperform RNNs in NLP primarily due to...",
  "papers_analyzed": 18,
  "inferred_facts": [
    "influencia(transformer, bert)",
    "influencia(bert, gpt)",
    "comparavel(transformer, rnn)"
  ],
  "conflicts_detected": [
    "conflito_potencial(rnn_seq2seq, transformer_attention)"
  ],
  "answer_sets": [
    ["posicao_aceita(transformer_attention)", "consenso(transformer_attention)"]
  ]
}
```

> **Rate limit:** 10 requests/minute per IP address.

---

### Server-Sent Events (Streaming)

Track pipeline progress in real time using [SSE](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events):

```bash
curl -N "http://localhost:8000/research/stream?user_query=How+do+transformers+outperform+RNNs&max_papers=20"
```

The server emits events at each pipeline stage:

| Event | Payload | Description |
|-------|---------|-------------|
| `progress` | `{"step": "elicit", "message": "..."}` | Stage start notification |
| `elicit_done` | `{"domain": "NLP", "intent": "comparison"}` | Elicitation completed |
| `papers_found` | `{"count": 18}` | Retrieval completed |
| `triples_extracted` | `{"count": 142}` | Triple extraction completed |
| `reasoning_done` | `{"inferred_count": 87, "conflicts": 3}` | Symbolic reasoning completed |
| `complete` | `{"final_response": "...", "papers_analyzed": 18}` | Pipeline finished |
| `error` | `{"message": "..."}` | Error during execution |

**Sample stream output:**

```
event: progress
data: {"step": "elicit", "message": "Extracting research intent..."}

event: elicit_done
data: {"domain": "NLP", "intent": "comparison"}

event: papers_found
data: {"count": 18}

event: triples_extracted
data: {"count": 142}

event: reasoning_done
data: {"inferred_count": 87, "conflicts": 3}

event: complete
data: {"final_response": "Transformers outperform RNNs because...", "papers_analyzed": 18}
```

> **Rate limit:** 5 requests/minute per IP address.

#### JavaScript Client Example

```javascript
const source = new EventSource(
  'http://localhost:8000/research/stream?user_query=transformers+vs+RNNs+in+NLP'
);

source.addEventListener('elicit_done', (e) => {
  const { domain, intent } = JSON.parse(e.data);
  console.log(`Domain: ${domain} | Intent: ${intent}`);
});

source.addEventListener('papers_found', (e) => {
  console.log(`Retrieved ${JSON.parse(e.data).count} papers`);
});

source.addEventListener('reasoning_done', (e) => {
  const { inferred_count, conflicts } = JSON.parse(e.data);
  console.log(`Inferred ${inferred_count} facts, ${conflicts} conflicts detected`);
});

source.addEventListener('complete', (e) => {
  const { final_response, papers_analyzed } = JSON.parse(e.data);
  console.log(`Answer (from ${papers_analyzed} papers):`, final_response);
  source.close();
});

source.addEventListener('error', (e) => {
  console.error('Pipeline error:', JSON.parse(e.data).message);
  source.close();
});
```

---

### Knowledge Graph Queries

After executing queries, all knowledge is persisted in Neo4j and can be queried via Cypher:

```bash
# List indexed papers
curl "http://localhost:8000/graph/query?cypher=MATCH%20(p%3APaper)%20RETURN%20p.title%2C%20p.year%20ORDER%20BY%20p.year%20DESC%20LIMIT%2010"

# List inferred facts with their rule source
curl "http://localhost:8000/graph/query?cypher=MATCH%20(a)-[r%3AINFERRED]->(b)%20RETURN%20a.name%2C%20r.rule_source%2C%20b.name%20LIMIT%2020"

# Find all concepts that influence a target concept
curl "http://localhost:8000/graph/query?cypher=MATCH%20(a)-[:RELATION%20%7Btype%3A'influencia'%7D]->(b%20%7Bname%3A'transformer'%7D)%20RETURN%20a.name"
```

> **Security:** Only `MATCH` (read) queries are permitted. Write operations (`CREATE`, `MERGE`, `DELETE`, `SET`, `REMOVE`, `DROP`) are blocked at the API level.

> **Rate limit:** 30 requests/minute per IP address.

#### Neo4j Graph Schema

```
(:Paper {id, title, abstract, authors, year, citation_count, doi})
(:Concept {name})

(:Paper)-[:AUTHORED_BY]->(:Author)
(:Concept)-[:RELATION {type, confidence, source_paper}]->(:Concept)
(:Concept)-[:INFERRED {rule_source, timestamp}]->(:Concept)
```

---

### Programmatic Usage

Run the pipeline directly without the API layer:

```python
from graph.graph import build_graph

graph = build_graph()

result = graph.invoke({
    "user_query": "What neural network architectures are most effective for NLP?"
})

print(result["final_response"])
print(f"Papers analyzed: {len(result['papers'])}")
print(f"Logical facts inferred: {result['inferred_facts']}")
print(f"Conflicts detected: {result['conflicts_detected']}")
print(f"ASP answer sets: {result['answer_sets']}")
```

---

## 📡 API Reference

| Method | Endpoint | Rate Limit | Description |
|--------|----------|:----------:|-------------|
| `GET` | `/health` | — | System health check and Neo4j status |
| `POST` | `/research` | 10/min | Execute full pipeline, returns JSON |
| `GET` | `/research/stream` | 5/min | Execute pipeline with SSE progress stream |
| `GET` | `/graph/query` | 30/min | Execute read-only Cypher query |
| `GET` | `/docs` | — | Swagger UI interactive documentation |
| `GET` | `/redoc` | — | ReDoc documentation |
| `GET` | `/openapi.json` | — | OpenAPI 3.0 specification |

---

## 🧩 Core Modules

### `config.py` — Centralized Settings

All configuration is managed via `pydantic-settings`, reading from the `.env` file with type validation:

```python
from config import settings

print(settings.openrouter_model)       # "nvidia/nemotron-3-super-120b-a12b:free"
print(settings.max_parallel_agents)    # 4
print(settings.min_triple_confidence)  # 0.7
print(settings.clingo_max_models)      # 10
```

### `graph/state.py` — Shared Pipeline State

The `ResearchState` TypedDict is the single source of truth passed between all nodes:

```python
class ResearchState(TypedDict):
    # Input
    user_query: str

    # Elicitation outputs
    domain: str                                           # e.g. "NLP", "bioinformatics"
    keywords: list[str]                                   # search terms
    research_question: str                                # reformulated query
    intent: str                                           # survey | comparison | specific_question

    # Retrieval outputs
    papers: list[Paper]                                   # deduplicated papers with abstracts

    # Extraction outputs (reducer: accumulates across nodes)
    extracted_triples: Annotated[list[Triple], operator.add]

    # Reasoning outputs
    asp_program: str                                      # generated ASP program text
    inferred_facts: Annotated[list[str], operator.add]    # Datalog inferred atoms
    conflicts_detected: list[str]                         # conflito_potencial atoms
    answer_sets: list[list[str]]                          # ASP answer sets

    # Final output
    final_response: str
```

### `reasoner/clingo_engine.py` — Symbolic Reasoning Engine

```python
from reasoner.clingo_engine import ClingoEngine, triples_to_asp_facts

engine = ClingoEngine()

# Convert extracted triples to ASP facts
facts = triples_to_asp_facts(extracted_triples)
# e.g.: usa("bert", "transformer").  % src=paper_01 conf=0.92

# Datalog mode — deterministic, single answer set
inferred = engine.run_datalog(facts)
# → ["influencia(bert,transformer)", "comparavel(rnn,lstm)", ...]

# Check for conflicts
if engine.has_conflicts(inferred):
    # ASP mode — non-deterministic, multiple answer sets
    answer_sets = engine.run_asp(facts, max_models=10)
    # → [["posicao_aceita(transformer)", "consenso(transformer)"], ...]
```

### `extractors/validator.py` — Triple Validation

```python
from extractors.validator import validate_triple_batch

report = validate_triple_batch(triples)
print(report.summary())
# {
#   "total": 142,
#   "invalid_predicates": [],
#   "unnormalized_entities": ["BERT", "GPT-3"],
#   "low_confidence": ["bert_uses_transformer_0.61"],
#   "valid_rate": 0.97
# }
```

### `observability.py` — Structured Logging

```python
from observability import configure_logging, log_node_execution

configure_logging(log_level="INFO")  # emits JSON via structlog

@log_node_execution("my_node")
def my_node(state: ResearchState) -> dict:
    ...
    # Automatically logs: node entry, exit, duration, errors
```

### `retry.py` — Resilience Policies

```python
from retry import llm_retry, api_retry

@llm_retry    # 3 attempts, exponential backoff 1s → 10s
def call_llm(prompt):
    ...

@api_retry    # 5 attempts, exponential backoff 1s → 30s
def call_external_api(url):
    ...
```

---

## 🔬 Symbolic Reasoning (ASP/Datalog)

The distinguishing feature of this system is the integration of **Answer Set Programming** (via the [Potassco Clingo](https://potassco.org/clingo/) solver) for sound logical inference over extracted triples.

### Datalog Mode — Deterministic Inference

Given extracted triples such as:

```prolog
usa("bert", "transformer").
supera("bert", "lstm").
supera("gpt", "bert").
contradiz("rnn_seq2seq", "transformer_attention").
```

The Clingo solver applies the base rules (`rules.asp`) and infers:

```prolog
% Direct influence (from any action predicate)
influencia_direta("bert", "transformer")

% Transitive influence (multi-hop)
influencia("gpt", "transformer")    % gpt → bert → transformer

% Comparability (two concepts influence the same target)
comparavel("bert", "gpt")           % both influence "transformer"

% Potential conflict (from contradiz predicate)
conflito_potencial("rnn_seq2seq", "transformer_attention")
```

The full Datalog ruleset:

```prolog
% rules.asp
influencia_direta(X, Y) :- usa(X, Y).
influencia_direta(X, Y) :- cita(X, Y).
influencia_direta(X, Y) :- estende(X, Y).
influencia_direta(X, Y) :- supera(X, Y).
influencia_direta(X, Y) :- define(X, Y).

influencia(X, Y) :- influencia_direta(X, Y).
influencia(X, Z) :- influencia(X, Y), influencia_direta(Y, Z), X != Z.

comparavel(X, Y) :- influencia(X, Z), influencia(Y, Z), X != Y.
comparavel(X, Y) :- comparavel(Y, X).

conflito_potencial(X, Y) :- contradiz(X, Y).
conflito_potencial(X, Y) :- contradiz(Y, X).
```

### ASP Mode — Non-Deterministic Conflict Resolution

When `conflito_potencial` atoms are present, the system switches to **non-deterministic ASP mode**, which enumerates all *stable models* (consistent interpretations):

```prolog
% asp_conflict.asp

% Choose exactly one position per conflicting pair
1 { posicao_aceita(X) ; posicao_aceita(Y) } 1 :- conflito_potencial(X, Y).

% Hard constraint: cannot accept both sides of a direct contradiction
:- contradiz(X, Y), posicao_aceita(X), posicao_aceita(Y).

% Evidence: count supporting papers per accepted position
evidencia(X, N) :- posicao_aceita(X), N = #count { P : cita(P, X) }.

% Consensus: positions supported by 3 or more papers
consenso(X) :- posicao_aceita(X), evidencia(X, N), N >= 3.
```

For the example conflict above, Clingo generates two distinct answer sets:

- **Answer Set 1:** `posicao_aceita("transformer_attention")` — transformer_attention prevails
- **Answer Set 2:** `posicao_aceita("rnn_seq2seq")` — rnn_seq2seq prevails

The `consenso/1` predicate guides the synthesis node toward the position backed by stronger empirical evidence, producing a logically grounded, transparent final answer.

---

## 🧪 Testing

The test suite covers all modules with **97 tests**, none of which require external network access (all APIs are mocked).

```bash
# Run the full test suite
uv run pytest tests/ -v

# Run a specific test module
uv run pytest tests/test_clingo.py -v
uv run pytest tests/test_resilience.py -v
uv run pytest tests/test_pipeline_integration.py -v

# Run with coverage report
uv run pytest tests/ --cov=. --cov-report=term-missing --cov-report=html
```

### Test Coverage by Module

| Test Module | Count | What It Covers |
|-------------|:-----:|----------------|
| `test_graph_skeleton.py` | 5 | LangGraph structure, conditional router |
| `test_elicit.py` | 9 | Elicitation agent, JSON retry, `ElicitationError` |
| `test_deep_research.py` | 13 | DOI/fuzzy deduplication, scope, sub-agents, supervisor |
| `test_extractor.py` | 15 | Entity normalization, triple schema, async extraction |
| `test_clingo.py` | 14 | Datalog inference, ASP answer sets, conflict detection |
| `test_neo4j_client.py` | 8 | Paper/triple/inferred-fact upserts, context manager |
| `test_pipeline_integration.py` | 3 | End-to-end with conflict routing |
| `test_api.py` | 12 | All endpoints, schemas, SSE events, Cypher write guard |
| `test_resilience.py` | 18 | LLM failures, Neo4j offline, HTTP errors, rate limits |
| **Total** | **97** | |

---

## ⚙️ Environment Variables

| Variable | Default | Required | Description |
|----------|---------|:--------:|-------------|
| `OPENROUTER_API_KEY` | — | **Yes** | OpenRouter API key |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | No | LLM API base URL |
| `OPENROUTER_MODEL` | `nvidia/nemotron-3-super-120b-a12b:free` | No | LLM model identifier |
| `NEO4J_URI` | `bolt://localhost:7687` | No | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | No | Neo4j username |
| `NEO4J_PASSWORD` | — | No | Neo4j password |
| `SEMANTIC_SCHOLAR_KEY` | — | No | Semantic Scholar API key (increases rate limits) |
| `LANGSMITH_API_KEY` | — | No | LangSmith pipeline tracing |
| `LANGCHAIN_TRACING_V2` | `false` | No | Enable LangSmith tracing |
| `LOG_LEVEL` | `INFO` | No | Log verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `MAX_PARALLEL_AGENTS` | `4` | No | Maximum concurrent retrieval sub-agents |
| `MAX_PAPERS` | `50` | No | Maximum papers per query |
| `MIN_TRIPLE_CONFIDENCE` | `0.7` | No | Minimum confidence threshold for triple acceptance |
| `CLINGO_MAX_MODELS` | `10` | No | Maximum answer sets in ASP non-deterministic mode |

---

## 🛠 Technology Stack

| Component | Technology | Version | Role |
|-----------|-----------|:-------:|------|
| LLM Provider | OpenRouter | — | API gateway for model access |
| Language Model | nvidia/nemotron-3-super-120b-a12b | free | Elicitation, extraction, synthesis |
| Orchestration | LangGraph | ≥0.2 | Stateful multi-node execution graph |
| LLM Integration | LangChain OpenAI | ≥0.2 | OpenAI-compatible LLM client |
| ASP Solver | Clingo (Potassco) | ≥5.7 | Formal Datalog and ASP reasoning |
| Knowledge Graph | Neo4j | 5.x | Persistent graph of papers and concepts |
| Graph Driver | neo4j-python-driver | ≥5.0 | Async Neo4j access (`AsyncGraphDatabase`) |
| API Framework | FastAPI | ≥0.115 | REST + SSE endpoints |
| ASGI Server | uvicorn | ≥0.30 | Async HTTP server |
| Data Validation | Pydantic V2 | ≥2.0 | Typed schemas across all layers |
| Settings | pydantic-settings | ≥2.0 | `.env`-based configuration |
| Rate Limiting | slowapi | ≥0.1.9 | Per-IP request throttling |
| SSE | sse-starlette | ≥2.0 | Server-Sent Events streaming |
| HTTP Client | httpx | ≥0.27 | Academic API requests |
| Resilience | tenacity | ≥9.0 | Retry policies with exponential backoff |
| Observability | structlog | ≥24.0 | Structured JSON logging |
| Testing | pytest + pytest-asyncio | ≥8.0 | Test suite (all external calls mocked) |
| Package Manager | uv | latest | Fast Python package and environment manager |

---

## 🤝 Contributing

Contributions are welcome. Please follow these guidelines:

1. **Fork** the repository and create a feature branch from `main`
2. **Install development dependencies:** `uv sync`
3. **Write tests** for new functionality — all tests must pass (`uv run pytest tests/`)
4. **Follow the existing code style** — functions are typed, nodes return partial state dicts, no direct state mutation
5. **Keep nodes single-responsibility** — each LangGraph node should do exactly one thing
6. **Do not commit secrets** — use `.env.example` for new configuration keys
7. Open a **Pull Request** with a clear description of the change and its motivation

### Adding New Predicates

To extend the triple vocabulary, update in order:
1. `extractors/triple_schema.py` — add to the `Literal` type and `VALID_PREDICATES`
2. `reasoner/rules.asp` — add inference rules for the new predicate
3. `extractors/prompts/triple_prompt.py` — update the extraction prompt
4. `tests/test_extractor.py` — add test cases

### Adding New Reasoning Rules

Add new Datalog or ASP rules to `reasoner/rules.asp` or `reasoner/asp_conflict.asp`. Rules must include `#show` directives for any new predicates to appear in the output. Run `uv run pytest tests/test_clingo.py` to validate.

---

## 📄 Citation

If you use this system in academic research, please cite:

```bibtex
@software{vfs_neurosymbolic_2026,
  title        = {{VFS Neuro-Symbolic Research System}: A Hybrid Pipeline for
                  Academic Literature Analysis with LLMs and Answer Set Programming},
  author       = {VFS Research},
  year         = {2026},
  url          = {https://github.com/your-org/vfs-neuro-symbolic},
  note         = {Python 3.11+, LangGraph, Clingo 5.x, Neo4j 5.x},
  version      = {0.1.0}
}
```

### Related Work

This system builds upon and is inspired by the following research:

- **LangGraph** — Pregel-inspired stateful agent orchestration: [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)
- **Open Deep Research** — Multi-agent academic retrieval architecture: [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research)
- **Potassco/Clingo** — Answer Set Programming solver: [Gebser et al., 2014](https://potassco.org/)
- **Neuro-Symbolic AI Survey** — Kautz, H. (2022). *The third AI summer*. AI Magazine, 43(1), 93–104.
- **Knowledge Graph Embeddings** — Wang et al. (2017). *Knowledge graph embedding: A survey of approaches and applications*. TKDE.

---

## 🙏 Acknowledgements

- [Potassco Group](https://potassco.org/) (University of Potsdam) for the Clingo ASP system
- [LangChain / LangGraph](https://github.com/langchain-ai) for the agent orchestration framework
- [Semantic Scholar](https://www.semanticscholar.org/) and [ArXiv](https://arxiv.org/) for open academic data access
- [OpenRouter](https://openrouter.ai/) for unified LLM API access
- [Neo4j](https://neo4j.com/) for the graph database platform

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.

```
MIT License — Copyright (c) 2026 VFS Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions: [...]
```

---

<div align="center">

**Built with rigor. Reasoned with logic. Grounded in evidence.**

<br/>

[Back to top](#vfs-neuro-symbolic-research-system)

</div>
