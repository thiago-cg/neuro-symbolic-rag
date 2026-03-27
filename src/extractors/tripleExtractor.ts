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
  const safeId = paperId.replace(/[^a-zA-Z0-9_-]/g, '_');
  return path.join(TMP_DIR, `${safeId}_triples.json`);
}

function getEntitiesFilePath(paperId: string) {
  const safeId = paperId.replace(/[^a-zA-Z0-9_-]/g, '_');
  return path.join(TMP_DIR, `${safeId}_entities.json`);
}

// Global dictionary to hold texts for the active agents
const TEXT_CACHE = new Map<string, string>();

// ==========================================
// TOOLS FOR THE AGENT
// ==========================================

const readTextChunkTool = tool(
  async ({ paperId, startChar, length }) => {
    const text = TEXT_CACHE.get(paperId);
    if (!text) return `Error: No text found for paper ID ${paperId}`;

    if (startChar >= text.length) {
      return `End of Document Reached. The text has only ${text.length} characters.`;
    }

    const chunk = text.slice(startChar, startChar + length);
    const progress = Math.min(((startChar + chunk.length) / text.length) * 100, 100).toFixed(1);

    return `--- CHUNK START (Progress: ${progress}%) ---\n${chunk}\n--- CHUNK END ---`;
  },
  {
    name: "read_text_chunk",
    description: "Read a specific segment of the document by character index. Use this iteratively to explore the whole document.",
    schema: z.object({
      paperId: z.string().describe("The ID of the paper to read"),
      startChar: z.number().describe("The character index to start reading from. Starts at 0."),
      length: z.number().describe("How many characters to read. Recommend between 500 and 1500."),
    }),
  }
);

const saveTargetEntitiesTool = tool(
  async ({ paperId, entities }) => {
    try {
      const file = getEntitiesFilePath(paperId);

      let existing: string[] = [];
      if (fs.existsSync(file)) {
        existing = JSON.parse(fs.readFileSync(file, "utf-8"));
      }

      const newEntities = entities.filter(e => !existing.includes(e));
      const combined = [...existing, ...newEntities];

      fs.writeFileSync(file, JSON.stringify(combined, null, 2));
      return `Added ${newEntities.length} new entities. Total tracking entities: ${combined.length}. Currently tracking: ${combined.join(", ")}`;
    } catch (err) {
      return `Failed to save entities: ${err}`;
    }
  },
  {
    name: "save_target_entities",
    description: "As you read chunks of text, use this tool to persist new concepts, methods, and works you discover into your Target Entity Database.",
    schema: z.object({
      paperId: z.string().describe("The ID of the current paper"),
      entities: z.array(z.string()).describe("List of new core entities found in the current text chunk"),
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
    description: "Saves a batch of extracted triples to a temporary file. Requires a valid JSON array string.",
    schema: z.object({
      paperId: z.string().describe("The ID of the current paper"),
      triplesRaw: z.string().describe("A valid JSON array string of triples. Valid predicates: 'cita', 'supera', 'usa', 'trata_conceito', 'define', 'contradiz', 'estende'."),
    }),
  }
);

const evaluateCoverageTool = tool(
  async ({ paperId }) => {
    const entFile = getEntitiesFilePath(paperId);
    if (!fs.existsSync(entFile)) return "0% coverage. No target entities registered yet. Use 'save_target_entities' while reading.";

    const tripFile = getTempFilePath(paperId);
    if (!fs.existsSync(tripFile)) return "0% coverage. No triples saved yet. Use 'save_triples' to map relations.";

    try {
      const coreEntities = JSON.parse(fs.readFileSync(entFile, "utf-8")) as string[];
      const saved = JSON.parse(fs.readFileSync(tripFile, "utf-8")) as any[];
      const coveredEntities = new Set<string>();

      saved.forEach(t => {
        if(t.subject) coveredEntities.add(t.subject.toLowerCase());
        if(t.object) coveredEntities.add(t.object.toLowerCase());
      });

      const missing = coreEntities.filter(e => !coveredEntities.has(e.toLowerCase()));
      const coverage = ((coreEntities.length - missing.length) / coreEntities.length) * 100;

      if (missing.length === 0) {
         return "100% Entity-Relation Coverage achieved. All registered entities are part of at least one triple. You may finish by outputting a final message starting with DONE.";
      }

      return `Coverage: ${coverage.toFixed(1)}%. Missing entities: ${missing.join(", ")}. Please re-read the text or extract relations for these missing entities. If no relation exists, explain why in your final output.`;
    } catch (e) {
      return "Error reading saved files for evaluation.";
    }
  },
  {
    name: "evaluate_coverage",
    description: "Evaluates if all registered target entities are present in the saved triples. Use this to check your progress towards 100% Entity-Relation completeness.",
    schema: z.object({
      paperId: z.string().describe("The ID of the current paper")
    }),
  }
);

// ==========================================
// AGENTIC EXTRACTION FUNCTION
// ==========================================

export async function extractFromPaper(paper: Paper): Promise<Triple[]> {
  if (!paper.abstract) return [];

  // Initialize temporary storage
  const tripFile = getTempFilePath(paper.paperId);
  const entFile = getEntitiesFilePath(paper.paperId);
  if (fs.existsSync(tripFile)) fs.unlinkSync(tripFile);
  if (fs.existsSync(entFile)) fs.unlinkSync(entFile);

  TEXT_CACHE.set(paper.paperId, paper.abstract);

  log.info({ paperId: paper.paperId }, "Starting iterative agentic extraction for paper");

  const llm = getAgentLlm();
  const tools = [readTextChunkTool, saveTargetEntitiesTool, saveTriplesTool, evaluateCoverageTool];

  const checkpointer = new MemorySaver();
  const agent = createReactAgent({
    llm,
    tools,
    checkpointSaver: checkpointer,
  });

  const SYSTEM_PROMPT = `You are an iterative Knowledge Extraction Agent working on a single academic paper.
Your GOAL is to thoroughly map the document, identify key entities, and achieve 100% Entity-Relation Completeness.

RULES:
1. ONLY use the following predicates for relations: "cita", "supera", "usa", "trata_conceito", "define", "contradiz", "estende".
2. You do NOT have the document text yet. The document has ${paper.abstract.length} characters.
3. Process Workflow:
   a) Phase 1: EXPLORATION. Use 'read_text_chunk' to page through the text. As you read, aggressively register important concepts, datasets, models, and works using 'save_target_entities'.
   b) Phase 2: EXTRACTION. You can extract triples as you read or re-read later. Save relations using 'save_triples'.
   c) Phase 3: VERIFICATION. Call 'evaluate_coverage'. It will cross-check your saved entities against your saved triples.
   d) If coverage < 100%, re-read specific parts or extract the missing relations.
   e) Repeat until coverage is 100% OR you logically justify why a missing entity has no valid relations in the text.
4. Once satisfied, write a final summary message starting with "DONE" and STOP.

Paper Title: ${paper.title}
Paper ID: ${paper.paperId}

Begin by reading the first chunk of text!`;

  try {
    const threadId = `extract-${paper.paperId}-${Date.now()}`;

    await withLlmRetry(async () => {
      // Clear file before each retry attempt to avoid duplicate appending
      if (fs.existsSync(tripFile)) fs.unlinkSync(tripFile);
      if (fs.existsSync(entFile)) fs.unlinkSync(entFile);

      return agent.invoke(
        {
          messages: [
            new SystemMessage(SYSTEM_PROMPT),
            new HumanMessage(`Start exploration phase for paper ID: ${paper.paperId}`)
          ]
        },
        { configurable: { thread_id: threadId } }
      );
    });

    // Process finished. Read the final file produced by the agent.
    if (fs.existsSync(tripFile)) {
      const raw = JSON.parse(fs.readFileSync(tripFile, "utf-8")) as unknown[];
      const validated = validateTriples(Array.isArray(raw) ? raw : []);

      // Cleanup
      fs.unlinkSync(tripFile);
      if (fs.existsSync(entFile)) fs.unlinkSync(entFile);
      TEXT_CACHE.delete(paper.paperId);

      const triples = validated.map((t) => ({ ...t, source: paper.paperId }));
      log.info({ paperId: paper.paperId, count: triples.length }, "Agent finished extraction successfully");
      return triples;
    }

    log.warn({ paperId: paper.paperId }, "Agent finished but no triples were saved to disk.");
    TEXT_CACHE.delete(paper.paperId);
    return [];

  } catch (err) {
    log.error({ paperId: paper.paperId, err }, "Agentic extraction failed completely");
    if (fs.existsSync(tripFile)) fs.unlinkSync(tripFile); // cleanup on error
    if (fs.existsSync(entFile)) fs.unlinkSync(entFile);
    TEXT_CACHE.delete(paper.paperId);
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
