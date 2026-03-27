import { Annotation } from "@langchain/langgraph";
import { z } from "zod";

export const TripleSchema = z.object({
  subject: z.string(),
  predicate: z.enum([
    "cita",
    "supera",
    "usa",
    "trata_conceito",
    "define",
    "contradiz",
    "estende",
  ]),
  object: z.string(),
  confidence: z.number().min(0).max(1),
  source: z.string().optional(),
});

export type Triple = z.infer<typeof TripleSchema>;

export const PaperSchema = z.object({
  paperId: z.string(),
  title: z.string(),
  abstract: z.string().optional(),
  authors: z.array(z.string()).default([]),
  year: z.number().optional(),
  url: z.string().optional(),
  fullText: z.string().optional(),
  score: z.number().optional(),
});

export type Paper = z.infer<typeof PaperSchema>;

export const ResearchState = Annotation.Root({
  // Input
  query: Annotation<string>(),

  // Elicitation output
  domain: Annotation<string>({ default: () => "", reducer: (_, v) => v }),
  intent: Annotation<string>({ default: () => "", reducer: (_, v) => v }),
  subTopics: Annotation<string[]>({ default: () => [], reducer: (_, v) => v }),

  // Research output (accumulates across parallel agents)
  papers: Annotation<Paper[]>({
    default: () => [],
    reducer: (curr, upd) => [...curr, ...upd],
  }),

  // Extraction output (accumulates)
  extractedTriples: Annotation<Triple[]>({
    default: () => [],
    reducer: (curr, upd) => [...curr, ...upd],
  }),

  // Symbolic reasoning output (accumulates)
  inferredFacts: Annotation<string[]>({
    default: () => [],
    reducer: (curr, upd) => [...curr, ...upd],
  }),

  // ASP conflict resolution
  conflicts: Annotation<string[]>({ default: () => [], reducer: (_, v) => v }),
  answerSets: Annotation<string[][]>({
    default: () => [],
    reducer: (_, v) => v,
  }),

  // Final synthesis
  answer: Annotation<string>({ default: () => "", reducer: (_, v) => v }),

  // Progress events for streaming
  events: Annotation<string[]>({
    default: () => [],
    reducer: (curr, upd) => [...curr, ...upd],
  }),
});

export type ResearchStateType = typeof ResearchState.State;
