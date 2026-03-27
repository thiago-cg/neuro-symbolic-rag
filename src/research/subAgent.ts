import { getConfig } from "../config.js";
import { searchSemanticScholar, searchArXiv } from "./academicTools.js";
import { deduplicatePapers } from "./harvester.js";
import type { Paper } from "../graph/state.js";
import { getLogger } from "../observability.js";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { withLlmRetry } from "../retry.js";

const log = getLogger("research:sub-agent");

function getLlm() {
  const cfg = getConfig();
  return new ChatOpenAI({
    openAIApiKey: cfg.openrouterApiKey,
    configuration: { baseURL: cfg.openrouterBaseUrl },
    model: cfg.openrouterModel,
    temperature: 0.3,
  });
}

const EXPAND_QUERY_SYSTEM = `You are a deep research AI.
Based on the user's intent and the abstracts of the top initial papers found, generate 2 specific search queries to dive deeper into the topic.
These queries should use jargon or specific concepts mentioned in the abstracts.
Respond ONLY with a JSON array of strings: ["query1", "query2"]`;

export async function runSubAgent(params: {
  domain: string;
  subTopic: string;
  intent: string;
}): Promise<Paper[]> {
  const cfg = getConfig();
  const initialQuery = `${params.domain} ${params.subTopic}`;
  const perSource = Math.ceil(cfg.maxPapers / 4);

  log.debug({ subTopic: params.subTopic }, "Sub-agent starting PASS 1");

  // PASS 1
  const [ssResults1, arxivResults1] = await Promise.allSettled([
    searchSemanticScholar(initialQuery, perSource),
    searchArXiv(initialQuery, perSource),
  ]);

  let pass1Papers: Paper[] = [
    ...(ssResults1.status === "fulfilled" ? ssResults1.value : []),
    ...(arxivResults1.status === "fulfilled" ? arxivResults1.value : []),
  ];

  pass1Papers = deduplicatePapers(pass1Papers);

  if (pass1Papers.length === 0) {
    return [];
  }

  // PASS 2: Query Expansion
  const topAbstracts = pass1Papers
     .filter(p => p.abstract)
     .slice(0, 2)
     .map(p => `Title: ${p.title}\nAbstract: ${p.abstract}`)
     .join("\n\n");

  let newQueries: string[] = [];
  if (topAbstracts) {
    const prompt = `Intent: ${params.intent}\n\nTop Papers:\n${topAbstracts}`;
    try {
       const llm = getLlm();
       const result = await withLlmRetry(() =>
         llm.invoke([
           new SystemMessage(EXPAND_QUERY_SYSTEM),
           new HumanMessage(prompt),
         ])
       );
       const cleanContent = (result.content as string).replace(/```json/g, '').replace(/```/g, '').trim();
       newQueries = JSON.parse(cleanContent) as string[];
    } catch (e) {
       log.warn({ subTopic: params.subTopic }, "Failed to expand queries, continuing with PASS 1 results only");
    }
  }

  let pass2Papers: Paper[] = [];
  if (newQueries.length > 0) {
     log.debug({ subTopic: params.subTopic, newQueries }, "Sub-agent starting PASS 2");
     const pass2Promises = newQueries.map(async (q) => {
        const [ss, ax] = await Promise.allSettled([
           searchSemanticScholar(q, perSource),
           searchArXiv(q, perSource),
        ]);
        return [
           ...(ss.status === "fulfilled" ? ss.value : []),
           ...(ax.status === "fulfilled" ? ax.value : []),
        ];
     });

     const pass2Results = await Promise.all(pass2Promises);
     pass2Papers = pass2Results.flat();
  }

  // Consolidação
  const allPapers = deduplicatePapers([...pass1Papers, ...pass2Papers]);



  log.debug(
    { subTopic: params.subTopic, found: allPapers.length },
    "Sub-agent done",
  );
  return allPapers;
}
