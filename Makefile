.PHONY: help install run dev cli test test-watch lint build reset-db

help:
	@echo "Available targets:"
	@echo "  install     Install dependencies with pnpm"
	@echo "  run         Start Fastify server (http://localhost:8000)"
	@echo "  dev         Start server with hot reload"
	@echo "  cli         Launch interactive CLI"
	@echo "  test        Run all tests"
	@echo "  lint        Run TypeScript type check"
	@echo "  build       Compile TypeScript to dist/"
	@echo "  reset-db    Drop and recreate Neo4j schema"

install:
	pnpm install

run:
	pnpm tsx src/cli.ts serve

dev:
	pnpm tsx watch src/cli.ts serve

cli:
	pnpm tsx src/cli.ts

test:
	pnpm vitest run

test-watch:
	pnpm vitest

lint:
	pnpm tsc --noEmit

build:
	pnpm tsc

reset-db:
	pnpm tsx src/cli.ts reset-db
