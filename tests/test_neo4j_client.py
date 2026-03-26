"""Tests for the Neo4j client (Phase 6) — using mocks (no real Neo4j required)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from knowledge_base.neo4j_client import Neo4jClient
from extractors.triple_schema import ExtractedTriple


def _make_paper(paper_id: str = "p1", title: str = "Test Paper") -> dict:
    return {
        "id": paper_id,
        "title": title,
        "abstract": "Test abstract.",
        "authors": ["Author A"],
        "year": 2024,
        "citation_count": 100,
        "doi": f"10.1234/{paper_id}",
    }


def _make_triple(subject: str, predicate: str, obj: str) -> ExtractedTriple:
    return ExtractedTriple(
        subject=subject,
        predicate=predicate,
        obj=obj,
        source_paper="p1",
        confidence=0.9,
    )


class TestNeo4jClientMocked:
    """Test Neo4j client with mocked driver (no real connection needed)."""

    def _make_client(self) -> Neo4jClient:
        client = Neo4jClient(uri="bolt://localhost:7687", user="neo4j", password="test")
        return client

    @pytest.mark.asyncio
    async def test_connect_and_close(self):
        client = self._make_client()
        mock_driver = AsyncMock()
        mock_driver.verify_connectivity = AsyncMock()
        mock_driver.close = AsyncMock()

        with patch("knowledge_base.neo4j_client.AsyncGraphDatabase") as mock_gdb:
            mock_gdb.driver.return_value = mock_driver
            await client.connect()
            assert client._driver is not None
            await client.close()
            assert client._driver is None

    @pytest.mark.asyncio
    async def test_check_driver_raises_when_not_connected(self):
        client = self._make_client()
        with pytest.raises(RuntimeError, match="not connected"):
            client._check_driver()

    @pytest.mark.asyncio
    async def test_upsert_papers(self):
        client = self._make_client()
        mock_session = AsyncMock()
        mock_session.execute_write = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_driver = AsyncMock()
        mock_driver.verify_connectivity = AsyncMock()
        mock_driver.session = MagicMock(return_value=mock_session)
        client._driver = mock_driver

        papers = [_make_paper("p1"), _make_paper("p2"), _make_paper("p3")]
        await client.upsert_papers(papers)
        # Should call execute_write at least once (batch of 100)
        assert mock_session.execute_write.call_count >= 1

    @pytest.mark.asyncio
    async def test_upsert_triples(self):
        client = self._make_client()
        mock_session = AsyncMock()
        mock_session.execute_write = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_driver = AsyncMock()
        mock_driver.session = MagicMock(return_value=mock_session)
        client._driver = mock_driver

        triples = [
            _make_triple("bert", "usa", "transformer"),
            _make_triple("gpt", "usa", "transformer"),
            _make_triple("bert", "supera", "lstm"),
        ]
        await client.upsert_triples(triples)
        assert mock_session.execute_write.call_count >= 1

    @pytest.mark.asyncio
    async def test_upsert_inferred_facts(self):
        client = self._make_client()
        mock_session = AsyncMock()
        mock_session.execute_write = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_driver = AsyncMock()
        mock_driver.session = MagicMock(return_value=mock_session)
        client._driver = mock_driver

        inferred = [
            'influencia("bert", "transformer")',
            'comparavel("bert", "gpt")',
        ]
        await client.upsert_inferred_facts(inferred)
        assert mock_session.execute_write.call_count >= 1

    @pytest.mark.asyncio
    async def test_upsert_inferred_facts_empty_list(self):
        """Empty inferred list should not call driver at all."""
        client = self._make_client()
        mock_driver = AsyncMock()
        client._driver = mock_driver
        # Should complete without error and without calling driver
        await client.upsert_inferred_facts([])
        mock_driver.session.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_cypher(self):
        client = self._make_client()
        mock_result = AsyncMock()
        mock_result.data = AsyncMock(return_value=[{"name": "bert"}, {"name": "gpt"}])

        mock_session = AsyncMock()
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_driver = AsyncMock()
        mock_driver.session = MagicMock(return_value=mock_session)
        client._driver = mock_driver

        records = await client.run_cypher("MATCH (c:Concept) RETURN c.name AS name")
        assert len(records) == 2
        assert records[0]["name"] == "bert"

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager protocol."""
        with patch("knowledge_base.neo4j_client.AsyncGraphDatabase") as mock_gdb:
            mock_driver = AsyncMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.close = AsyncMock()
            mock_gdb.driver.return_value = mock_driver

            async with Neo4jClient(uri="bolt://localhost:7687", user="neo4j", password="test") as client:
                assert client._driver is not None
            # After exiting, driver should be closed
            mock_driver.close.assert_called_once()
