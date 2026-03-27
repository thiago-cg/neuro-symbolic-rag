import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { MemorySaver } from "@langchain/langgraph";
import fs from "fs";
import path from "path";

import { getConfig } from "../config.js";
import { getLogger } from "../observability.js";
import { withLlmRetry } from "../retry.js";
import { validateTriples } from "./validator.js";
import type { Paper, Triple } from "../graph/state.js";

const log = getLogger("extractors:triple");
const TMP_DIR = path.join(process.cwd(), ".tmp_extractors");

if (!fs.existsSync(TMP_DIR)) {
  fs.mkdirSync(TMP_DIR, { recursive: true });
}

function getAgentLlm() {
  const cfg = getConfig();
  return new ChatOpenAI({
    openAIApiKey: cfg.openrouterApiKey,
    configuration: { baseURL: cfg.openrouterBaseUrl },
    model: cfg.openrouterModel,
    temperature: 0,
  });
}

function getTempFilePath(paperId: string) {
  return path.join(TMP_DIR, `${paperId}_triples.json`);
}

// ==========================================
// TOOLS FOR THE AGENT
// ==========================================

const identifyEntitiesTool = tool(
  async ({ entities }) => {
    return `Entities identified: ${entities.join(", ")}. Now proceed to extract triples connecting these entities using the allowed predicates.`;
  },
  {
    name: "identify_entities",
    description: "Use this first to list the core entities (methods, problems, concepts, prior works) found in the text. This list will be your target to ensure 100% Entity-Relation Completeness.",
    schema: z.object({
      entities: z.array(z.string()).describe("List of core entities found in the text"),
    }),
  }
);

const saveTriplesTool = tool(
  async ({ paperId, triplesRaw }) => {
    try {
      const file = getTempFilePath(paperId);

      let existing: unknown[] = [];
      if (fs.existsSync(file)) {
        existing = JSON.parse(fs.readFileSync(file, "utf-8"));
      }

      let newTriples: unknown[] = [];
      try {
         newTriples = JSON.parse(triplesRaw);
         if (!Array.isArray(newTriples)) newTriples = [newTriples];
      } catch(e) {
         return "Error: triplesRaw must be a valid JSON array string. Do not include markdown formatting like ```json. Example: [{\"subject\": \"A\", \"predicate\": \"usa\", \"object\": \"B\", \"confidence\": 0.9}]";
      }

      const combined = [...existing, ...newTriples];
      fs.writeFileSync(file, JSON.stringify(combined, null, 2));
      return `Successfully saved ${newTriples.length} new triples. Total saved so far: ${combined.length}.`;
    } catch (err) {
      return `Failed to save triples: ${err}`;
    }
  },
  {
    name: "save_triples",
    description: "Saves a batch of extracted triples to a temporary file. Always use this when you find new relations. Requires a valid JSON array string.",
    schema: z.object({
      paperId: z.string().describe("The ID of the current paper"),
      triplesRaw: z.string().describe("A valid JSON array string of triples. Valid predicates: 'cita', 'supera', 'usa', 'trata_conceito', 'define', 'contradiz', 'estende'."),
    }),
  }
);

const evaluateCoverageTool = tool(
  async ({ paperId, coreEntities }) => {
    const file = getTempFilePath(paperId);
    if (!fs.existsSync(file)) return "0% coverage. No triples saved yet. You must use 'save_triples' first.";

    try {
      const saved = JSON.parse(fs.readFileSync(file, "utf-8")) as any[];
      const coveredEntities = new Set<string>();

      saved.forEach(t => {
        if(t.subject) coveredEntities.add(t.subject.toLowerCase());
        if(t.object) coveredEntities.add(t.object.toLowerCase());
      });

      const missing = coreEntities.filter(e => !coveredEntities.has(e.toLowerCase()));
      const coverage = ((coreEntities.length - missing.length) / coreEntities.length) * 100;

      if (missing.length === 0) {
         return "100% coverage achieved. All core entities are part of at least one triple. You may finish by outputting a final message.";
      }

      return `Coverage: ${coverage.toFixed(1)}%. Missing entities: ${missing.join(", ")}. Please search the text again to find relations for these missing entities. If no relation exists, explain why.`;
    } catch (e) {
      return "Error reading saved triples for evaluation.";
    }
  },
  {
    name: "evaluate_coverage",
    description: "Evaluates if all identified core entities are present in the saved triples. Use this to check your progress towards 100% Entity-Relation completeness.",
    schema: z.object({
      paperId: z.string().describe("The ID of the current paper"),
      coreEntities: z.array(z.string()).describe("The list of entities you identified initially."),
    }),
  }
);

// ==========================================
// AGENTIC EXTRACTION FUNCTION
// ==========================================

export async function extractFromPaper(paper: Paper): Promise<Triple[]> {
  if (!paper.abstract) return [];

  const file = getTempFilePath(paper.paperId);
  if (fs.existsSync(file)) fs.unlinkSync(file);

  log.info({ paperId: paper.paperId }, "Starting iterative agentic extraction for paper");

  const llm = getAgentLlm();
  const tools = [identifyEntitiesTool, saveTriplesTool, evaluateCoverageTool];

  const checkpointer = new MemorySaver();
  const agent = createReactAgent({
    llm,
    tools,
    checkpointSaver: checkpointer,
  });

  const SYSTEM_PROMPT = `You are an iterative Knowledge Extraction Agent working on a single academic paper.
Your GOAL is to achieve 100% Entity-Relation Completeness on the provided text.

RULES:
1. ONLY use the following predicates for relations: "cita", "supera", "usa", "trata_conceito", "define", "contradiz", "estende".
2. Process Workflow:
   a) Call 'identify_entities' tool to list all significant concepts, methods, problems, and datasets.
   b) Search the text and extract triples connecting these entities.
   c) Call 'save_triples' tool with a JSON string of the discovered triples.
   d) Call 'evaluate_coverage' tool using the core entities from step (a).
   e) Repeat (b), (c), (d) until coverage is 100% OR you logically justify why a missing entity has no valid relations in the text.
3. Once satisfied, write a final summary message and STOP.

Paper Title: ${paper.title}
Paper Abstract:
${paper.abstract}

Begin by identifying entities!`;

  try {
    const threadId = `extract-${paper.paperId}-${Date.now()}`;

    // We wrap the agent invocation in a basic retry to handle network errors,
    // though the agent itself handles tool invocation errors internally.
    await withLlmRetry(async () => {
      // Clear file before each retry attempt to avoid duplicate appending
      if (fs.existsSync(file)) fs.unlinkSync(file);

      return agent.invoke(
        {
          messages: [
            new SystemMessage(SYSTEM_PROMPT),
            new HumanMessage(`Start extraction for paper ID: ${paper.paperId}`)
          ]
        },
        { configurable: { thread_id: threadId } }
      );
    });

    // Process finished. Read the final file produced by the agent.
    if (fs.existsSync(file)) {
      const raw = JSON.parse(fs.readFileSync(file, "utf-8")) as unknown[];
      const validated = validateTriples(Array.isArray(raw) ? raw : []);

      // Cleanup
      fs.unlinkSync(file);

      const triples = validated.map((t) => ({ ...t, source: paper.paperId }));
      log.info({ paperId: paper.paperId, count: triples.length }, "Agent finished extraction successfully");
      return triples;
    }

    log.warn({ paperId: paper.paperId }, "Agent finished but no triples were saved to disk.");
    return [];

  } catch (err) {
    log.error({ paperId: paper.paperId, err }, "Agentic extraction failed completely");
    if (fs.existsSync(file)) fs.unlinkSync(file); // cleanup on error
    return [];
  }
}

export async function extractAll(papers: Paper[]): Promise<Triple[]> {
  // Concurrency will be controlled by the caller (nodes.ts)
  const results = [];
  for (const p of papers) {
     results.push(await extractFromPaper(p));
  }
  return results.flat();
}
