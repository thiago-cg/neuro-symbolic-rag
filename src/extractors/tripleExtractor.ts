import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { getConfig } from "../config.js";
import { withLlmRetry } from "../retry.js";
import { validateTriples } from "./validator.js";
import { TRIPLE_SYSTEM, tripleUserPrompt } from "./prompts/triple_prompt.js";
import type { Paper, Triple } from "../graph/state.js";
import { getLogger } from "../observability.js";

const log = getLogger("extractors:triple");

function getLlm() {
  const cfg = getConfig();
  return new ChatOpenAI({
    openAIApiKey: cfg.openrouterApiKey,
    configuration: { baseURL: cfg.openrouterBaseUrl },
    model: cfg.openrouterModel,
    temperature: 0,
  });
}

export async function extractFromPaper(paper: Paper): Promise<Triple[]> {
  if (!paper.abstract) return [];

  const llm = getLlm();
  try {
    const result = await withLlmRetry(() =>
      llm.invoke([
        new SystemMessage(TRIPLE_SYSTEM),
        new HumanMessage(tripleUserPrompt(paper.abstract!, paper.title)),
      ]),
    );

    const raw = JSON.parse(result.content as string) as unknown[];
    const triples = validateTriples(Array.isArray(raw) ? raw : []);
    return triples.map((t) => ({ ...t, source: paper.paperId }));
  } catch (err) {
    log.warn({ paperId: paper.paperId, err }, "Triple extraction failed");
    return [];
  }
}

export async function extractAll(papers: Paper[]): Promise<Triple[]> {
  const results = await Promise.all(papers.map(extractFromPaper));
  return results.flat();
}
