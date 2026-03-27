import { StateGraph, START, END } from "@langchain/langgraph";
import { ResearchState } from "./state.js";
import {
  elicitNode,
  researchNode,
  extractTriplesNode,
  persistNode,
  reasonDatalogNode,
  reasonAspConflictsNode,
  synthesizeNode,
} from "./nodes.js";
import { shouldRunAsp } from "./router.js";

export function buildResearchGraph() {
  return new StateGraph(ResearchState)
    .addNode("elicit", elicitNode)
    .addNode("research", researchNode)
    .addNode("extract_triples", extractTriplesNode)
    .addNode("persist", persistNode)
    .addNode("reason_datalog", reasonDatalogNode)
    .addNode("reason_asp_conflicts", reasonAspConflictsNode)
    .addNode("synthesize", synthesizeNode)
    .addEdge(START, "elicit")
    .addEdge("elicit", "research")
    .addEdge("research", "extract_triples")
    .addEdge("extract_triples", "persist")
    .addEdge("persist", "reason_datalog")
    .addConditionalEdges("reason_datalog", shouldRunAsp, [
      "reason_asp_conflicts",
      "synthesize",
    ])
    .addEdge("reason_asp_conflicts", "synthesize")
    .addEdge("synthesize", END)
    .compile();
}

let _graph: ReturnType<typeof buildResearchGraph> | undefined;

function getGraph() {
  if (!_graph) _graph = buildResearchGraph();
  return _graph;
}

/** Run the full research pipeline, optionally streaming progress events */
export async function runResearchGraph(
  query: string,
  onEvent?: (event: string) => void,
): Promise<{ answer: string; events: string[] }> {
  const graph = getGraph();

  if (onEvent) {
    const stream = graph.streamEvents(
      { query },
      { version: "v2", streamMode: "custom" },
    );
    for await (const event of stream) {
      if (event.event === "on_custom_event") {
        onEvent(event.data as string);
      }
    }
    // After streaming, get final state
    const result = await graph.invoke({ query });
    return { answer: result.answer, events: result.events };
  }

  const result = await graph.invoke({ query });
  return { answer: result.answer, events: result.events };
}
