import { getConfig } from "../config.js";
import { runSubAgent } from "./subAgent.js";
import { harvestPapers } from "./harvester.js";
import { rerankPapers } from "./evaluator.js";
import type { Paper } from "../graph/state.js";
import { getLogger } from "../observability.js";

const log = getLogger("research:supervisor");

export async function runSupervisor(params: {
  domain: string;
  intent: string;
  subTopics: string[];
}): Promise<Paper[]> {
  const cfg = getConfig();
  const { domain, intent, subTopics } = params;

  // Limit parallelism
  const maxParallel = Math.min(cfg.maxParallelAgents, subTopics.length);
  const activeSubs = subTopics.slice(0, maxParallel);

  log.info({ domain, subTopics: activeSubs.length }, "Supervisor launching sub-agents");

  // Run sub-agents in parallel
  const subResults = await Promise.all(
    activeSubs.map((subTopic) => runSubAgent({ domain, subTopic, intent })),
  );

  // Harvest and deduplicate
  const papers = harvestPapers(subResults, cfg.maxPapers);

  log.info({ total: papers.length }, "Supervisor applying semantic reranking");

  // Rerank via LLM (Avaliação Semântica)
  const rerankedPapers = await rerankPapers(intent, papers);

  log.info({ totalFiltered: rerankedPapers.length }, "Supervisor finished processing papers");

  return rerankedPapers;
}
