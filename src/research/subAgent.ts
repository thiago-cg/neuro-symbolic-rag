import { getConfig } from "../config.js";
import { searchSemanticScholar, searchArXiv } from "./academicTools.js";
import { deduplicatePapers } from "./harvester.js";
import type { Paper } from "../graph/state.js";
import { getLogger } from "../observability.js";

const log = getLogger("research:sub-agent");

export async function runSubAgent(params: {
  domain: string;
  subTopic: string;
}): Promise<Paper[]> {
  const cfg = getConfig();
  const query = `${params.domain} ${params.subTopic}`;
  const perSource = Math.ceil(cfg.maxPapers / 4);

  log.debug({ subTopic: params.subTopic }, "Sub-agent starting");

  const [ssResults, arxivResults] = await Promise.allSettled([
    searchSemanticScholar(query, perSource),
    searchArXiv(query, perSource),
  ]);

  const all: Paper[] = [
    ...(ssResults.status === "fulfilled" ? ssResults.value : []),
    ...(arxivResults.status === "fulfilled" ? arxivResults.value : []),
  ];

  const deduplicated = deduplicatePapers(all);
  log.debug(
    { subTopic: params.subTopic, found: deduplicated.length },
    "Sub-agent done",
  );
  return deduplicated;
}
