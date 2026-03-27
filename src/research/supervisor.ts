import { getConfig } from "../config.js";
import { runSubAgent } from "./subAgent.js";
import { harvestPapers } from "./harvester.js";
import { rerankPapers } from "./evaluator.js";
import { fetchAndParsePdf } from "./academicTools.js";
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


  log.info({ count: rerankedPapers.length }, "Supervisor fetching full texts for top ranked papers");

  // Apenas baixar PDF dos artigos aprovados e limitar a top 3 para não travar o processo
  const TOP_PDF_LIMIT = 3;
  for (let i = 0; i < Math.min(rerankedPapers.length, TOP_PDF_LIMIT); i++) {
    const p = rerankedPapers[i];
    if (p && p.url && (p.url.endsWith(".pdf") || p.url.includes("pdf"))) {
      try {
        log.debug({ paperId: p.paperId }, "Fetching PDF for highly ranked paper");
        const text = await fetchAndParsePdf(p.url);
        if (text) {
          p.fullText = text;
          log.debug({ paperId: p.paperId, textLength: text.length }, "Successfully extracted full text from PDF");
        }
      } catch (e) {
        log.warn({ paperId: p.paperId }, "Silent fail on fetching PDF for ranked paper");
      }
    }
  }

  return rerankedPapers;
}
