import type { Triple } from "../state.js";

export const SYNTHESIZE_SYSTEM = `You are a research synthesizer. Given:
- The user's original query and research intent
- Extracted knowledge triples from academic papers
- Inferred facts from symbolic reasoning (Datalog/ASP)
- Any conflict resolutions from Answer Set Programming

Produce a well-structured academic synthesis that:
1. Directly answers the research query
2. Grounds claims in the extracted triples and papers
3. Highlights any resolved conflicts or contradictions
4. Is written in clear, academic prose`;

export function synthesizeUserPrompt(params: {
  query: string;
  intent: string;
  triples: Triple[];
  inferredFacts: string[];
  answerSets: string[][];
}): string {
  const triplesText = params.triples
    .slice(0, 60)
    .map((t) => `  (${t.subject}) --[${t.predicate}]--> (${t.object}) [conf: ${t.confidence.toFixed(2)}]`)
    .join("\n");

  const inferredText = params.inferredFacts.slice(0, 30).join("\n  ");

  const aspText =
    params.answerSets.length > 0
      ? params.answerSets
          .slice(0, 3)
          .map((as, i) => `  AS${i + 1}: ${as.slice(0, 5).join(", ")}`)
          .join("\n")
      : "  (no conflicts detected)";

  return `Query: "${params.query}"
Intent: ${params.intent}

Extracted Triples (${params.triples.length} total, showing top 60):
${triplesText}

Inferred Facts (Datalog):
  ${inferredText}

ASP Answer Sets (conflict resolution):
${aspText}

Please synthesize a comprehensive answer to the research query based on the above evidence.`;
}
