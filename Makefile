.PHONY: help run test lint reset-db install

help:
	@echo "Available targets:"
	@echo "  install    Install dependencies with uv"
	@echo "  run        Start FastAPI server (http://localhost:8000)"
	@echo "  test       Run all tests"
	@echo "  lint       Run ruff linter and format check"
	@echo "  reset-db   Drop and recreate Neo4j schema"

install:
	uv sync --all-extras

run:
	uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check . && uv run ruff format --check .

reset-db:
	uv run python -c "from knowledge_base.neo4j_client import Neo4jClient; import asyncio; asyncio.run(Neo4jClient().reset_schema())"
