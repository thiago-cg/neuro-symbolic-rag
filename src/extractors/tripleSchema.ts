import { z } from "zod";

export const VALID_PREDICATES = [
  "cita",
  "supera",
  "usa",
  "trata_conceito",
  "define",
  "contradiz",
  "estende",
] as const;

export const ExtractedTripleSchema = z.object({
  subject: z.string().min(1),
  predicate: z.enum(VALID_PREDICATES),
  object: z.string().min(1),
  confidence: z.number().min(0).max(1).default(0.8),
  source: z.string().optional(),
});

export type ExtractedTriple = z.infer<typeof ExtractedTripleSchema>;
