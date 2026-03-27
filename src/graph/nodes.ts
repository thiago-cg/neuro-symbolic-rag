import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { getConfig } from "../config.js";
import { getLogger } from "../observability.js";
import { withLlmRetry } from "../retry.js";
import { Neo4jClient } from "../knowledge_base/neo4jClient.js";
import { ClingoEngine } from "../reasoner/clingoEngine.js";
import { extractFromPaper } from "../extractors/tripleExtractor.js";
import { runSupervisor } from "../research/supervisor.js";
import { ELICIT_SYSTEM, elicitUserPrompt } from "./prompts/elicit_prompt.js";
import {
  SYNTHESIZE_SYSTEM,
  synthesizeUserPrompt,
} from "./prompts/synthesize_prompt.js";
import type { ResearchStateType, Triple } from "./state.js";

const log = getLogger("graph:nodes");

function getLlm() {
  const cfg = getConfig();
  return new ChatOpenAI({
    openAIApiKey: cfg.openrouterApiKey,
    configuration: { baseURL: cfg.openrouterBaseUrl },
    model: cfg.openrouterModel,
    temperature: 0.1,
  });
}

// ─── Node 1: elicit ───────────────────────────────────────────────────────────
export async function elicitNode(state: ResearchStateType) {
  log.info({ query: state.query }, "Eliciting domain and intent");

  const llm = getLlm();
  const result = await withLlmRetry(() =>
    llm.invoke([
      new SystemMessage(ELICIT_SYSTEM),
      new HumanMessage(elicitUserPrompt(state.query)),
    ]),
  );

  const parsed = JSON.parse(result.content as string) as {
    domain: string;
    intent: string;
    subTopics: string[];
  };

  return {
    domain: parsed.domain,
    intent: parsed.intent,
    subTopics: parsed.subTopics,
    events: ["elicit_done"],
  };
}

// ─── Node 2: research ─────────────────────────────────────────────────────────
export async function researchNode(state: ResearchStateType) {
  log.info({ domain: state.domain, subTopics: state.subTopics }, "Researching papers");

  const papers = await runSupervisor({
    domain: state.domain,
    intent: state.intent,
    subTopics: state.subTopics,
  });

  return {
    papers,
    events: [`papers_found:${papers.length}`],
  };
}

// ─── Node 3: extractTriples ───────────────────────────────────────────────────
export async function extractTriplesNode(state: ResearchStateType) {
  log.info({ papers: state.papers.length }, "Extracting triples via iterative agents");

  const cfg = getConfig();
  const minConf = cfg.minTripleConfidence;

  // Concurrency control to avoid OpenRouter 429 Too Many Requests
  // Since each paper triggers a multi-turn agent, we limit active papers to 2.
  const MAX_CONCURRENT_PAPERS = 2;
  const papersToProcess = state.papers.filter((p) => p.abstract);
  const results: Triple[][] = [];

  for (let i = 0; i < papersToProcess.length; i += MAX_CONCURRENT_PAPERS) {
     const batch = papersToProcess.slice(i, i + MAX_CONCURRENT_PAPERS);
     log.info({ batch: i/MAX_CONCURRENT_PAPERS + 1, total_papers: papersToProcess.length }, "Processing batch of papers");

     const batchResults = await Promise.all(
       batch.map((p) => extractFromPaper(p))
     );

     results.push(...batchResults);
  }

  const triples = results
    .flat()
    .filter((t: Triple) => t.confidence >= minConf);

  return {
    extractedTriples: triples,
    events: [`triples_extracted:${triples.length}`],
  };
}

// ─── Node 4: persist ──────────────────────────────────────────────────────────
export async function persistNode(state: ResearchStateType) {
  log.info("Persisting to Neo4j");

  const cfg = getConfig();
  const client = new Neo4jClient(cfg);
  await client.connect();
  try {
    await client.insertPapers(state.papers);
    await client.insertTriples(state.extractedTriples);
  } finally {
    await client.disconnect();
  }

  return { events: ["persist_done"] };
}

// ─── Node 5: reasonDatalog ────────────────────────────────────────────────────
export async function reasonDatalogNode(state: ResearchStateType) {
  log.info({ triples: state.extractedTriples.length }, "Running Datalog inference");

  const engine = new ClingoEngine();
  const { inferredFacts, conflicts } = await engine.runDatalog(state.extractedTriples);

  return {
    inferredFacts,
    conflicts,
    events: [`reasoning_done:${inferredFacts.length}_facts,${conflicts.length}_conflicts`],
  };
}

// ─── Node 6: reasonAspConflicts ───────────────────────────────────────────────
export async function reasonAspConflictsNode(state: ResearchStateType) {
  log.info({ conflicts: state.conflicts.length }, "Running ASP conflict resolution");

  const cfg = getConfig();
  const engine = new ClingoEngine();
  const answerSets = await engine.runAsp(
    state.extractedTriples,
    cfg.clingoMaxModels,
  );

  return {
    answerSets,
    events: [`asp_done:${answerSets.length}_answer_sets`],
  };
}

// ─── Node 7: synthesize ───────────────────────────────────────────────────────
export async function synthesizeNode(state: ResearchStateType) {
  log.info("Synthesizing final answer");

  const llm = getLlm();
  const prompt = synthesizeUserPrompt({
    query: state.query,
    intent: state.intent,
    triples: state.extractedTriples,
    inferredFacts: state.inferredFacts,
    answerSets: state.answerSets,
  });

  const result = await withLlmRetry(() =>
    llm.invoke([
      new SystemMessage(SYNTHESIZE_SYSTEM),
      new HumanMessage(prompt),
    ]),
  );

  return {
    answer: result.content as string,
    events: ["complete"],
  };
}
