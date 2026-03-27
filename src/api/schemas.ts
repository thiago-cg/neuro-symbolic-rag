import { z } from "zod";

export const ResearchRequestSchema = z.object({
  query: z.string().min(1).max(1000),
});

export const ResearchResponseSchema = z.object({
  answer: z.string(),
  events: z.array(z.string()),
  durationMs: z.number(),
});

export const GraphQueryRequestSchema = z.object({
  cypher: z.string().min(1),
});

export const HealthResponseSchema = z.object({
  status: z.enum(["ok", "degraded"]),
  version: z.string(),
  timestamp: z.string(),
  neo4j: z.enum(["connected", "disconnected"]),
});

export type ResearchRequest = z.infer<typeof ResearchRequestSchema>;
export type ResearchResponse = z.infer<typeof ResearchResponseSchema>;
export type GraphQueryRequest = z.infer<typeof GraphQueryRequestSchema>;
export type HealthResponse = z.infer<typeof HealthResponseSchema>;
