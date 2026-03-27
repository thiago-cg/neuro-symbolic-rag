import { z } from "zod";

const configSchema = z.object({
  // LLM
  openrouterApiKey: z.string().min(1),
  openrouterBaseUrl: z.string().url().default("https://openrouter.ai/api/v1"),
  openrouterModel: z
    .string()
    .default("nvidia/nemotron-3-super-120b-a12b:free"),

  // Neo4j
  neo4jUri: z.string().default("bolt://localhost:7687"),
  neo4jUser: z.string().default("neo4j"),
  neo4jPassword: z.string().min(1),

  // Academic APIs
  semanticScholarKey: z.string().optional(),

  // LangSmith
  langsmithApiKey: z.string().optional(),
  langsmithProject: z.string().default("vfs-neuro-symbolic"),
  langchainTracingV2: z
    .string()
    .transform((v) => v === "true")
    .default("false"),

  // App
  logLevel: z.enum(["trace", "debug", "info", "warn", "error"]).default("info"),
  maxParallelAgents: z.coerce.number().int().min(1).default(4),
  maxPapers: z.coerce.number().int().min(1).default(50),
  minTripleConfidence: z.coerce.number().min(0).max(1).default(0.7),
  clingoMaxModels: z.coerce.number().int().min(1).default(10),

  // Server
  port: z.coerce.number().int().default(8000),
  host: z.string().default("0.0.0.0"),
});

function loadConfig() {
  const raw = {
    openrouterApiKey: process.env["OPENROUTER_API_KEY"],
    openrouterBaseUrl: process.env["OPENROUTER_BASE_URL"],
    openrouterModel: process.env["OPENROUTER_MODEL"],
    neo4jUri: process.env["NEO4J_URI"],
    neo4jUser: process.env["NEO4J_USER"],
    neo4jPassword: process.env["NEO4J_PASSWORD"],
    semanticScholarKey: process.env["SEMANTIC_SCHOLAR_KEY"],
    langsmithApiKey: process.env["LANGSMITH_API_KEY"],
    langsmithProject: process.env["LANGSMITH_PROJECT"],
    langchainTracingV2: process.env["LANGCHAIN_TRACING_V2"],
    logLevel: process.env["LOG_LEVEL"],
    maxParallelAgents: process.env["MAX_PARALLEL_AGENTS"],
    maxPapers: process.env["MAX_PAPERS"],
    minTripleConfidence: process.env["MIN_TRIPLE_CONFIDENCE"],
    clingoMaxModels: process.env["CLINGO_MAX_MODELS"],
    port: process.env["PORT"],
    host: process.env["HOST"],
  };

  const result = configSchema.safeParse(raw);
  if (!result.success) {
    const missing = result.error.issues
      .map((i) => `${i.path.join(".")}: ${i.message}`)
      .join("\n  ");
    throw new Error(`Invalid configuration:\n  ${missing}`);
  }
  return result.data;
}

export type Config = z.infer<typeof configSchema>;

let _config: Config | undefined;

export function getConfig(): Config {
  if (!_config) {
    _config = loadConfig();
  }
  return _config;
}
