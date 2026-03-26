"""Neo4j async client for the knowledge graph.

Schema:
  (:Paper {id, title, abstract, authors, year, citation_count, doi})
  (:Concept {name})
  (:Paper)-[:RELATION {type, confidence, source_paper}]->(:Concept)
  (:Concept)-[:RELATION {type, confidence, source_paper}]->(:Concept)
  (:Concept)-[:INFERRED {rule_source}]->(:Concept)
"""

import logging
from typing import TYPE_CHECKING

from neo4j import AsyncGraphDatabase, AsyncDriver

if TYPE_CHECKING:
    from extractors.triple_schema import ExtractedTriple
    from graph.state import Paper

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Async Neo4j driver with schema management and bulk insert operations."""

    def __init__(self, uri: str = "", user: str = "neo4j", password: str = "") -> None:
        from config import settings
        self._uri = uri or settings.neo4j_uri
        self._user = user or settings.neo4j_user
        self._password = password or settings.neo4j_password
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Initialize the async driver and verify connectivity."""
        self._driver = AsyncGraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password),
        )
        await self._driver.verify_connectivity()
        logger.info("neo4j connected: %s", self._uri)

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *_):
        await self.close()

    def _check_driver(self) -> AsyncDriver:
        if not self._driver:
            raise RuntimeError("Neo4j driver not connected. Call connect() first.")
        return self._driver

    # ── Schema ───────────────────────────────────────────────────────────────

    async def setup_schema(self) -> None:
        """Create UNIQUE constraints and indexes."""
        driver = self._check_driver()
        async with driver.session(database="neo4j") as session:
            await session.run(
                "CREATE CONSTRAINT paper_id IF NOT EXISTS "
                "FOR (p:Paper) REQUIRE p.id IS UNIQUE"
            )
            await session.run(
                "CREATE CONSTRAINT concept_name IF NOT EXISTS "
                "FOR (c:Concept) REQUIRE c.name IS UNIQUE"
            )
            await session.run(
                "CREATE INDEX paper_domain IF NOT EXISTS "
                "FOR (p:Paper) ON (p.domain)"
            )
            await session.run(
                "CREATE INDEX paper_year IF NOT EXISTS "
                "FOR (p:Paper) ON (p.year)"
            )
        logger.info("neo4j schema setup complete")

    async def reset_schema(self) -> None:
        """Drop all data and recreate schema (for development/testing)."""
        driver = self._check_driver()
        async with driver.session(database="neo4j") as session:
            await session.run("MATCH (n) DETACH DELETE n")
        await self.setup_schema()
        logger.info("neo4j schema reset")

    # ── Paper operations ──────────────────────────────────────────────────────

    async def upsert_papers(self, papers: list["Paper"]) -> None:
        """Insert or update Paper nodes."""
        driver = self._check_driver()

        async def _insert(tx, batch: list["Paper"]) -> None:
            await tx.run(
                """
                UNWIND $batch AS p
                MERGE (n:Paper {id: p.id})
                SET n.title = p.title,
                    n.abstract = p.abstract,
                    n.authors = p.authors,
                    n.year = p.year,
                    n.citation_count = p.citation_count,
                    n.doi = p.doi
                """,
                batch=[dict(p) for p in batch],
            )

        async with driver.session(database="neo4j") as session:
            # Process in batches of 100
            for i in range(0, len(papers), 100):
                batch = papers[i:i + 100]
                await session.execute_write(_insert, batch)

        logger.info("neo4j upserted %d papers", len(papers))

    # ── Triple operations ─────────────────────────────────────────────────────

    async def upsert_triples(self, triples: list["ExtractedTriple"]) -> None:
        """Insert extracted triples as concept-concept or paper-concept relationships."""
        driver = self._check_driver()

        async def _insert(tx, batch: list[dict]) -> None:
            await tx.run(
                """
                UNWIND $batch AS t
                MERGE (a:Concept {name: t.subject})
                MERGE (b:Concept {name: t.obj})
                MERGE (a)-[r:RELATION {type: t.predicate, source_paper: t.source_paper}]->(b)
                SET r.confidence = t.confidence
                """,
                batch=batch,
            )

        data = [t.model_dump(mode="json") for t in triples]
        async with driver.session(database="neo4j") as session:
            for i in range(0, len(data), 200):
                batch = data[i:i + 200]
                await session.execute_write(_insert, batch)

        logger.info("neo4j upserted %d triples", len(triples))

    # ── Inferred facts ────────────────────────────────────────────────────────

    async def upsert_inferred_facts(self, facts: list[str], rule_source: str = "datalog") -> None:
        """Persist Clingo-inferred atoms as :INFERRED relationships.

        Supported atom patterns:
          influencia(X, Y)
          influencia_direta(X, Y)
          comparavel(X, Y)
          conflito_potencial(X, Y)
        """
        import re
        driver = self._check_driver()

        parsed: list[dict] = []
        for atom in facts:
            # Match predicate("X", "Y") or predicate(X, Y)
            m = re.match(r'(\w+)\("?([^",)]+)"?,\s*"?([^",)]+)"?\)', atom)
            if m:
                parsed.append({
                    "relation": m.group(1),
                    "subject": m.group(2),
                    "obj": m.group(3),
                    "rule_source": rule_source,
                })

        if not parsed:
            return

        async def _insert(tx, batch: list[dict]) -> None:
            await tx.run(
                """
                UNWIND $batch AS f
                MERGE (a:Concept {name: f.subject})
                MERGE (b:Concept {name: f.obj})
                MERGE (a)-[r:INFERRED {relation: f.relation, rule_source: f.rule_source}]->(b)
                """,
                batch=batch,
            )

        async with driver.session(database="neo4j") as session:
            for i in range(0, len(parsed), 200):
                await session.execute_write(_insert, parsed[i:i + 200])

        logger.info("neo4j upserted %d inferred facts", len(parsed))

    # ── Query ─────────────────────────────────────────────────────────────────

    async def run_cypher(self, query: str, params: dict | None = None) -> list[dict]:
        """Execute a Cypher read query and return results as list of dicts."""
        driver = self._check_driver()
        async with driver.session(database="neo4j") as session:
            result = await session.run(query, parameters=params or {})
            records = await result.data()
        return records
