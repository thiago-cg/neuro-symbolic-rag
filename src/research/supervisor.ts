import { getConfig } from "../config.js";
import { runSubAgent } from "./subAgent.js";
import { harvestPapers } from "./harvester.js";
import type { Paper } from "../graph/state.js";
import { getLogger } from "../observability.js";

const log = getLogger("research:supervisor");

export async function runSupervisor(params: {
  domain: string;
  intent: string;
  subTopics: string[];
}): Promise<Paper[]> {
  const cfg = getConfig();
  const { domain, subTopics } = params;

  // Limit parallelism
  const maxParallel = Math.min(cfg.maxParallelAgents, subTopics.length);
  const activeSubs = subTopics.slice(0, maxParallel);

  log.info({ domain, subTopics: activeSubs.length }, "Supervisor launching sub-agents");

  // Run sub-agents in parallel
  const subResults = await Promise.all(
    activeSubs.map((subTopic) => runSubAgent({ domain, subTopic })),
  );

  // Harvest and deduplicate
  const papers = harvestPapers(subResults, cfg.maxPapers);
  log.info({ total: papers.length }, "Supervisor harvested papers");
  return papers;
}
