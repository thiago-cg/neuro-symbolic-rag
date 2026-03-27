import Fastify from "fastify";
import cors from "@fastify/cors";
import rateLimit from "@fastify/rate-limit";
import { getConfig } from "../config.js";
import { getLogger } from "../observability.js";
import { Neo4jClient } from "../knowledge_base/neo4jClient.js";
import { runResearchGraph } from "../graph/graph.js";
import {
  ResearchRequestSchema,
  GraphQueryRequestSchema,
} from "./schemas.js";

const log = getLogger("api");

export async function startServer(opts?: { port?: number; host?: string }) {
  const cfg = getConfig();
  const port = opts?.port ?? cfg.port;
  const host = opts?.host ?? cfg.host;

  const app = Fastify({ logger: false });

  // ─── Plugins ──────────────────────────────────────────────────────────────
  await app.register(cors, { origin: true });
  await app.register(rateLimit, { max: 100, timeWindow: "1 minute" });

  // ─── Neo4j lifecycle ──────────────────────────────────────────────────────
  const neo4j = new Neo4jClient(cfg);
  let neo4jConnected = false;

  app.addHook("onReady", async () => {
    try {
      await neo4j.connect();
      neo4jConnected = true;
      log.info("Neo4j ready");
    } catch (err) {
      log.warn({ err }, "Neo4j unavailable — continuing without it");
    }
  });

  app.addHook("onClose", async () => {
    if (neo4jConnected) await neo4j.disconnect();
  });

  // ─── Routes ───────────────────────────────────────────────────────────────

  /** GET /health */
  app.get("/health", async () => ({
    status: "ok",
    version: "0.1.0",
    timestamp: new Date().toISOString(),
    neo4j: neo4jConnected ? "connected" : "disconnected",
  }));

  /** POST /research — synchronous pipeline */
  app.post(
    "/research",
    {
      config: { rateLimit: { max: 10, timeWindow: "1 minute" } },
    },
    async (request, reply) => {
      const body = ResearchRequestSchema.safeParse(request.body);
      if (!body.success) {
        return reply.status(400).send({ error: body.error.flatten() });
      }

      const start = Date.now();
      const result = await runResearchGraph(body.data.query);
      return {
        answer: result.answer,
        events: result.events,
        durationMs: Date.now() - start,
      };
    },
  );

  /** GET /research/stream — SSE streaming */
  app.get(
    "/research/stream",
    {
      config: { rateLimit: { max: 5, timeWindow: "1 minute" } },
    },
    async (request, reply) => {
      const query = (request.query as Record<string, string>)["query"];
      if (!query) {
        return reply.status(400).send({ error: "Missing ?query= parameter" });
      }

      reply.raw.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
      });

      const sendEvent = (event: string, data: string) => {
        reply.raw.write(`event: ${event}\ndata: ${data}\n\n`);
      };

      try {
        await runResearchGraph(query, (event) => {
          const [name, detail] = event.split(":");
          sendEvent(name ?? event, detail ?? "");
        });
        sendEvent("complete", JSON.stringify({ done: true }));
      } catch (err) {
        sendEvent("error", JSON.stringify({ message: String(err) }));
      } finally {
        reply.raw.end();
      }
    },
  );

  /** GET /graph/query — read-only Cypher */
  app.get(
    "/graph/query",
    {
      config: { rateLimit: { max: 30, timeWindow: "1 minute" } },
    },
    async (request, reply) => {
      const body = GraphQueryRequestSchema.safeParse(request.query);
      if (!body.success) {
        return reply.status(400).send({ error: body.error.flatten() });
      }

      if (!neo4jConnected) {
        return reply.status(503).send({ error: "Neo4j not available" });
      }

      try {
        const rows = await neo4j.query(body.data.cypher);
        return { rows };
      } catch (err) {
        return reply.status(400).send({ error: String(err) });
      }
    },
  );

  // ─── Start ────────────────────────────────────────────────────────────────
  await app.listen({ port, host });
  log.info({ port, host }, `VFS API server listening`);
  console.log(`\nServer running at http://${host === "0.0.0.0" ? "localhost" : host}:${port}`);
  return app;
}
