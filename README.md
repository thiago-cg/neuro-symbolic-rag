<div align="center">

# VFS Neuro-Symbolic Research System

**A hybrid academic research pipeline combining Large Language Models with formal symbolic reasoning**

[![Node.js](https://img.shields.io/badge/Node.js-20%2B-339933?style=flat-square&logo=node.js&logoColor=white)](https://nodejs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.5%2B-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-4B8BBE?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![Fastify](https://img.shields.io/badge/Fastify-5.x-000000?style=flat-square&logo=fastify&logoColor=white)](https://fastify.dev/)
[![Clingo WASM](https://img.shields.io/badge/Clingo-WASM-orange?style=flat-square)](https://potassco.org/clingo/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-008CC1?style=flat-square&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](./LICENSE)

> **Branch:** `ambiente-nodejs` — Node.js/TypeScript rewrite with CLI-first experience and local file analysis mode.
> The original Python version is on `main`.

<br/>

> *"The marriage of statistical pattern recognition with formal logical inference — giving AI systems the ability not just to predict, but to reason."*

<br/>

[Getting Started](#-getting-started) •
[Architecture](#-architecture) •
[CLI Reference](#-cli-reference) •
[API Reference](#-api-reference) •
[Symbolic Reasoning](#-symbolic-reasoning-aspdatalog)

</div>

---

## Abstract

The **VFS Neuro-Symbolic Research System** is an end-to-end pipeline that combines LLMs with formal symbolic reasoning for two primary use cases:

**Mode 1 — Research:** Given a natural language query, the system fetches academic papers, extracts knowledge triples, applies Datalog/ASP reasoning, and synthesizes a grounded answer.

**Mode 2 — Data Analysis:** Given local files (Excel, CSV, SQLite, PDF) and reference methodology articles, the system ingests structured data, extracts relational triples, reasons over them symbolically, and produces an analysis report — optionally written back into the original spreadsheet.

---

## Table of Contents

- [What's New in this Branch](#-whats-new-in-this-branch)
- [Architecture](#-architecture)
- [Getting Started](#-getting-started)
- [CLI Reference](#-cli-reference)
- [API Reference](#-api-reference)
- [Core Modules](#-core-modules)
- [Symbolic Reasoning](#-symbolic-reasoning-aspdatalog)
- [Environment Variables](#-environment-variables)
- [Technology Stack](#-technology-stack)

---

## 🆕 What's New in this Branch

This branch replaces the Python stack with Node.js/TypeScript and adds a full CLI experience:

| Python (`main`)       | Node.js (`ambiente-nodejs`) |
|-----------------------|-----------------------------|
| Python 3.11 + uv      | Node.js 20 LTS + npm        |
| FastAPI + Uvicorn     | Fastify 5 + SSE             |
| LangGraph (Python)    | @langchain/langgraph        |
| Pydantic v2           | Zod                         |
| Clingo (Python)       | clingo-wasm (WebAssembly)   |
| neo4j (Python async)  | neo4j-driver (official JS)  |
| structlog             | pino                        |
| tenacity              | p-retry                     |
| pytest                | vitest                      |
| API only              | **CLI + API**               |
| Research only         | **Research + File Analysis**|

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI / API                           │
│         @clack/prompts  ·  Commander.js  ·  Fastify         │
└───────────────────┬─────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
┌───────▼──────────┐   ┌────────▼──────────┐
│  Research Graph  │   │  Analyze Graph    │
│  (LangGraph JS)  │   │  (LangGraph JS)   │
│                  │   │                   │
│ elicit           │   │ ingest_files      │
│ research         │   │ extract_triples   │
│ extract_triples  │   │ persist           │
│ persist          │   │ reason            │
│ reason_datalog   │   │ synthesize_report │
│ [reason_asp?]    │   │ export_result     │
│ synthesize       │   └───────────────────┘
└──────────────────┘
        │                       │
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │   Symbolic Reasoner   │
        │   clingo-wasm (WASM)  │
        │   Datalog + ASP rules │
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │   Neo4j Knowledge     │
        │   Graph (neo4j-driver)│
        └───────────────────────┘
```

### Research Pipeline (7 nodes)

```
START → elicit → research → extract_triples → persist
      → reason_datalog → [reason_asp_conflicts?] → synthesize → END
```

### Data Analysis Pipeline (6 nodes)

```
START → ingest_files → extract_data_triples → persist
      → reason → synthesize_report → export_result → END
```

---

## 🚀 Getting Started

### Prerequisites

- Node.js 20 LTS or higher
- Neo4j 5.x (local or cloud)
- npm or pnpm

### Installation

```bash
git clone https://github.com/thiago-cg/neuro-symbolic-rag.git
cd neuro-symbolic-rag
git checkout ambiente-nodejs

npm install
```

### Configuration

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

```env
# LLM (OpenRouter)
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

# Optional
SEMANTIC_SCHOLAR_KEY=your-key
LOG_LEVEL=info
```

---

## 🖥 CLI Reference

The CLI supports two modes:

**Interactive** (no arguments — guided menus):
```bash
npm run cli
# or
npx tsx src/cli.ts
```

**Command mode** (with subcommands):
```bash
npx tsx src/cli.ts <command> [options]
```

### `research <query>`

Fetch academic papers, extract triples, reason symbolically, and synthesize an answer.

```bash
npx tsx src/cli.ts research "neural networks and symbolic reasoning"
npx tsx src/cli.ts research "climate change impacts" --stream
```

| Option | Description |
|--------|-------------|
| `--stream` | Stream progress events to stdout |

### `analyze`

Analyze local data files guided by methodology from reference PDFs.

```bash
npx tsx src/cli.ts analyze \
  --data dados.xlsx \
  --refs metodologia.pdf artigo.pdf \
  --prompt "Evaluate the data according to methodology X and identify patterns" \
  --output report.md
```

| Option | Description |
|--------|-------------|
| `--data <files...>` | Data files: `.xlsx`, `.csv`, `.sqlite`, `.db`, `.txt` |
| `--refs <files...>` | Reference PDFs (methodology / bibliography) |
| `--prompt <text>` | Analysis instruction (asked interactively if omitted) |
| `--output <file>` | Output file (default: `analysis_report.md`). Use `.xlsx` to write back into the spreadsheet |

**Output to Excel:** if `--output` ends in `.xlsx`, the report is written as a new sheet named `VFS Analysis` inside the original spreadsheet.

### `query <cypher>`

Run a read-only Cypher query against the Neo4j knowledge graph.

```bash
npx tsx src/cli.ts query "MATCH (p:Paper) RETURN p.title LIMIT 10"
npx tsx src/cli.ts query "MATCH (a)-[r:TRIPLE]->(b) RETURN a.name, r.predicate, b.name LIMIT 20"
```

### `serve`

Start the Fastify REST API server.

```bash
npx tsx src/cli.ts serve
npx tsx src/cli.ts serve --port 3000 --host localhost
```

### `reset-db`

Drop and recreate Neo4j schema (constraints + indexes).

```bash
npx tsx src/cli.ts reset-db
```

### Makefile shortcuts

```bash
make install    # npm install
make cli        # interactive CLI
make run        # start API server
make dev        # server with hot reload
make test       # vitest
make lint       # tsc --noEmit
make build      # tsc → dist/
```

---

## 🌐 API Reference

Start the server: `make run` (default: `http://localhost:8000`)

### `GET /health`

```json
{
  "status": "ok",
  "version": "0.1.0",
  "timestamp": "2026-03-27T10:00:00.000Z",
  "neo4j": "connected"
}
```

### `POST /research`

Rate limit: 10 req/min

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "transformers in NLP"}'
```

```json
{
  "answer": "...",
  "events": ["elicit_done", "papers_found:42", "..."],
  "durationMs": 12400
}
```

### `GET /research/stream`

Rate limit: 5 req/min — Server-Sent Events

```bash
curl "http://localhost:8000/research/stream?query=transformers"
```

Events: `elicit_done`, `papers_found`, `triples_extracted`, `reasoning_done`, `complete`, `error`

### `GET /graph/query?cypher=...`

Rate limit: 30 req/min — read-only Cypher queries.

```bash
curl "http://localhost:8000/graph/query?cypher=MATCH+(p%3APaper)+RETURN+p.title+LIMIT+5"
```

---

## 📦 Core Modules

```
src/
├── cli.ts                        # CLI entry point
├── config.ts                     # Zod-based environment config
├── observability.ts              # pino structured logger + withTiming()
├── retry.ts                      # p-retry policies (LLM: 3x, API: 5x)
│
├── api/
│   ├── main.ts                   # Fastify app (CORS, rate-limit, SSE)
│   └── schemas.ts                # Zod request/response schemas
│
├── graph/
│   ├── graph.ts                  # Research StateGraph (7 nodes)
│   ├── analyzeGraph.ts           # Analysis StateGraph (6 nodes)
│   ├── state.ts                  # ResearchState (Annotation.Root)
│   ├── nodes.ts                  # Node implementations
│   ├── router.ts                 # shouldRunAsp() conditional edge
│   └── prompts/                  # LLM prompt templates
│
├── extractors/
│   ├── tripleExtractor.ts        # LLM → validated triples
│   ├── tripleSchema.ts           # Zod + 7 predicates
│   ├── normalizer.ts             # Entity normalization
│   └── validator.ts              # Triple validation
│
├── ingest/
│   ├── fileIngestor.ts           # Route file by extension
│   ├── excelParser.ts            # xlsx → text + writeAnalysisSheet()
│   ├── pdfParser.ts              # pdf-parse → text
│   ├── csvParser.ts              # csv-parse → text
│   ├── sqliteParser.ts           # sql.js (WASM) → text
│   └── dataTripleExtractor.ts    # LLM triple extraction from tabular data
│
├── reasoner/
│   ├── clingoEngine.ts           # clingo-wasm wrapper
│   ├── rules.lp                  # Datalog: transitive closure, conflict detection
│   └── aspConflict.lp            # ASP: choice rules, evidence counting
│
├── knowledge_base/
│   └── neo4jClient.ts            # neo4j-driver async client
│
└── research/
    ├── supervisor.ts             # Multi-agent orchestrator
    ├── subAgent.ts               # Individual research agent
    ├── harvester.ts              # Deduplication + harvesting
    └── academicTools.ts          # Semantic Scholar + ArXiv APIs
```

---

## 🧠 Symbolic Reasoning (ASP/Datalog)

The system uses [Clingo](https://potassco.org/clingo/) compiled to **WebAssembly** (`clingo-wasm`) — no native binary or Python required.

### Datalog Rules (`rules.lp`)

Deterministic, single-model inference:

```prolog
% Transitive closure
cita_indireta(X, Z) :- cita(X, Y), cita(Y, Z).
estende_indiretamente(X, Z) :- estende(X, Y), estende(Y, Z).

% Comparability
comparavel(X, Y) :- trata_conceito(X, C), trata_conceito(Y, C), X != Y.

% Conflict detection
conflict(X, Y) :- contradiz(X, Y).
conflict(X, Y) :- supera(X, Y), supera(Y, X).
```

### ASP Conflict Resolution (`aspConflict.lp`)

Non-deterministic, enumerates all consistent answer sets:

```prolog
{ accept(X) ; accept(Y) } = 1 :- conflict(X, Y).
:- contradiz(X, Y), accept(X), accept(Y).
#maximize { N,X : evidence_count(X, N) }.
```

### Triple Predicates

| Predicate | Meaning |
|-----------|---------|
| `cita` | Paper A cites paper B |
| `supera` | Work A surpasses/outperforms B |
| `usa` | Work A uses method/tool B |
| `trata_conceito` | Work A addresses concept B |
| `define` | Work A defines concept B |
| `contradiz` | Work A contradicts B |
| `estende` | Work A extends B |

---

## ⚙️ Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | ✅ | — | OpenRouter API key |
| `OPENROUTER_BASE_URL` | | `https://openrouter.ai/api/v1` | LLM base URL |
| `OPENROUTER_MODEL` | | `nvidia/nemotron-3-super-120b-a12b:free` | Model ID |
| `NEO4J_URI` | ✅ | `bolt://localhost:7687` | Neo4j URI |
| `NEO4J_USER` | ✅ | `neo4j` | Neo4j user |
| `NEO4J_PASSWORD` | ✅ | — | Neo4j password |
| `SEMANTIC_SCHOLAR_KEY` | | — | Semantic Scholar API key |
| `LOG_LEVEL` | | `info` | `trace`/`debug`/`info`/`warn`/`error` |
| `MAX_PARALLEL_AGENTS` | | `4` | Parallel research sub-agents |
| `MAX_PAPERS` | | `50` | Max papers per research run |
| `MIN_TRIPLE_CONFIDENCE` | | `0.7` | Min confidence threshold for triples |
| `CLINGO_MAX_MODELS` | | `10` | Max ASP answer sets to enumerate |
| `PORT` | | `8000` | API server port |
| `HOST` | | `0.0.0.0` | API server host |

---

## 🛠 Technology Stack

| Layer | Technology |
|-------|-----------|
| Runtime | Node.js 20 LTS |
| Language | TypeScript 5.5 |
| LLM Orchestration | @langchain/langgraph, @langchain/openai |
| REST API | Fastify 5 + @fastify/cors + @fastify/rate-limit |
| CLI | Commander.js + @clack/prompts |
| Symbolic Reasoning | clingo-wasm (Clingo 5.8 compiled to WASM) |
| Knowledge Graph | Neo4j 5 + neo4j-driver |
| Schema Validation | Zod |
| File Ingestion | xlsx, pdf-parse, csv-parse, sql.js |
| Logging | pino + pino-pretty |
| Retry | p-retry |
| Testing | vitest |
| Build | tsx (dev), tsc (prod) |

---

## 📄 License

MIT © VFS Research Team
