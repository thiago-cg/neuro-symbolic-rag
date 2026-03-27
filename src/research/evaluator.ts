import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { getConfig } from "../config.js";
import { withLlmRetry } from "../retry.js";
import type { Paper } from "../graph/state.js";
import { getLogger } from "../observability.js";

const log = getLogger("research:evaluator");

function getLlm() {
  const cfg = getConfig();
  return new ChatOpenAI({
    openAIApiKey: cfg.openrouterApiKey,
    configuration: { baseURL: cfg.openrouterBaseUrl },
    model: cfg.openrouterModel, // Or a faster model if configured
    temperature: 0.1,
  });
}

const RERANK_SYSTEM = `You are a strict research evaluation assistant.
Given a user's research intent and an academic paper's abstract, score the paper's relevance to the intent from 0 to 10.
0 = Completely irrelevant
10 = Perfect match, highly crucial for the intent
Respond ONLY with a JSON object: { "score": number, "reason": "brief string" }`;

export async function rerankPapers(intent: string, papers: Paper[]): Promise<Paper[]> {
  const llm = getLlm();
  const scoredPapers: Paper[] = [];
  const MAX_CONCURRENT = 5;

  log.info({ papersCount: papers.length }, "Starting semantic reranking");

  for (let i = 0; i < papers.length; i += MAX_CONCURRENT) {
    const batch = papers.slice(i, i + MAX_CONCURRENT);

    const batchResults = await Promise.all(
      batch.map(async (paper) => {
        if (!paper.abstract) {
          paper.score = 0;
          return paper;
        }

        const prompt = `Intent: ${intent}\n\nTitle: ${paper.title}\nAbstract: ${paper.abstract}`;

        try {
          const result = await withLlmRetry(() =>
            llm.invoke([
              new SystemMessage(RERANK_SYSTEM),
              new HumanMessage(prompt),
            ])
          );

          const parsed = JSON.parse(result.content as string) as { score: number, reason: string };
          paper.score = parsed.score;
        } catch (e) {
          log.warn({ paperId: paper.paperId }, "Failed to rerank paper, defaulting score to 5");
          paper.score = 5;
        }

        return paper;
      })
    );

    scoredPapers.push(...batchResults);
  }

  // Filter papers with score >= 6
  const MIN_SCORE = 6;
  const filtered = scoredPapers
     .filter(p => p.score !== undefined && p.score >= MIN_SCORE)
     .sort((a, b) => (b.score || 0) - (a.score || 0));

  log.info({ originalCount: papers.length, filteredCount: filtered.length }, "Finished reranking");
  return filtered;
}
