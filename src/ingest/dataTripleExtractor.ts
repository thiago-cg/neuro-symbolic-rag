import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { getConfig } from "../config.js";
import { withLlmRetry } from "../retry.js";
import { validateTriples } from "../extractors/validator.js";
import type { Triple } from "../graph/state.js";
import { getLogger } from "../observability.js";

const log = getLogger("ingest:triple-extractor");

const DATA_TRIPLE_SYSTEM = `You are a data analysis knowledge extractor.
Given structured data (table/spreadsheet/database rows) and a methodology description,
extract semantic triples that represent relationships found in the data.

Use EXACTLY one of these predicates:
- cita, supera, usa, trata_conceito, define, contradiz, estende

Focus on relationships between entities, variables, indicators, and findings in the data.
Output ONLY valid JSON array: [{ "subject": "...", "predicate": "...", "object": "...", "confidence": 0.0-1.0 }]`;

function getLlm() {
  const cfg = getConfig();
  return new ChatOpenAI({
    openAIApiKey: cfg.openrouterApiKey,
    configuration: { baseURL: cfg.openrouterBaseUrl },
    model: cfg.openrouterModel,
    temperature: 0,
  });
}

export async function extractTriplesFromData(params: {
  dataText: string;
  methodology: string;
  source: string;
}): Promise<Triple[]> {
  const llm = getLlm();

  const userPrompt = `Methodology context:
${params.methodology.slice(0, 2000)}

Data to analyze:
${params.dataText.slice(0, 3000)}

Extract triples representing relationships in this data according to the methodology.`;

  try {
    const result = await withLlmRetry(() =>
      llm.invoke([
        new SystemMessage(DATA_TRIPLE_SYSTEM),
        new HumanMessage(userPrompt),
      ]),
    );

    const raw = JSON.parse(result.content as string) as unknown[];
    const triples = validateTriples(Array.isArray(raw) ? raw : []);
    return triples.map((t) => ({ ...t, source: params.source }));
  } catch (err) {
    log.warn({ source: params.source, err }, "Data triple extraction failed");
    return [];
  }
}
