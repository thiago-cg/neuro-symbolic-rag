import { Annotation, StateGraph, START, END } from "@langchain/langgraph";
import { writeFileSync } from "fs";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ingestFiles } from "../ingest/fileIngestor.js";
import { extractTriplesFromData } from "../ingest/dataTripleExtractor.js";
import { Neo4jClient } from "../knowledge_base/neo4jClient.js";
import { getConfig } from "../config.js";
import { withLlmRetry } from "../retry.js";
import { getLogger } from "../observability.js";
import { TripleSchema, type Triple } from "./state.js";
import { writeAnalysisSheet } from "../ingest/excelParser.js";
import { reasonDatalogNode, reasonAspConflictsNode } from "./nodes.js";

const log = getLogger("graph:analyze");

// ─── State ────────────────────────────────────────────────────────────────────
export const AnalyzeState = Annotation.Root({
  dataFiles: Annotation<string[]>(),
  refFiles: Annotation<string[]>(),
  instruction: Annotation<string>(),
  outputFile: Annotation<string>({ default: () => "analysis_report.md", reducer: (_, v) => v }),

  dataTexts: Annotation<string[]>({ default: () => [], reducer: (_, v) => v }),
  methodology: Annotation<string>({ default: () => "", reducer: (_, v) => v }),

  extractedTriples: Annotation<Triple[]>({
    default: () => [],
    reducer: (curr, upd) => [...curr, ...upd],
  }),

  inferredFacts: Annotation<string[]>({
    default: () => [],
    reducer: (curr, upd) => [...curr, ...upd],
  }),
  conflicts: Annotation<string[]>({ default: () => [], reducer: (_, v) => v }),
  answerSets: Annotation<string[][]>({ default: () => [], reducer: (_, v) => v }),

  report: Annotation<string>({ default: () => "", reducer: (_, v) => v }),
});

export type AnalyzeStateType = typeof AnalyzeState.State;

function getLlm() {
  const cfg = getConfig();
  return new ChatOpenAI({
    openAIApiKey: cfg.openrouterApiKey,
    configuration: { baseURL: cfg.openrouterBaseUrl },
    model: cfg.openrouterModel,
    temperature: 0.2,
  });
}

// ─── Nodes ────────────────────────────────────────────────────────────────────

async function ingestFilesNode(state: AnalyzeStateType) {
  log.info({ dataFiles: state.dataFiles, refFiles: state.refFiles }, "Ingesting files");

  const [dataIngested, refIngested] = await Promise.all([
    ingestFiles(state.dataFiles),
    ingestFiles(state.refFiles),
  ]);

  const dataTexts = dataIngested.map((f) => `[File: ${f.path}]\n${f.text}`);
  const methodology = refIngested.map((f) => f.text).join("\n\n---\n\n");

  return { dataTexts, methodology };
}

async function extractDataTriplesNode(state: AnalyzeStateType) {
  log.info({ files: state.dataTexts.length }, "Extracting triples from data");
  const cfg = getConfig();

  const results = await Promise.all(
    state.dataTexts.map((text: string, i: number) =>
      extractTriplesFromData({
        dataText: text,
        methodology: state.methodology,
        source: state.dataFiles[i] ?? `data_${i}`,
      }),
    ),
  );

  const triples = results
    .flat()
    .filter((t: Triple) => t.confidence >= cfg.minTripleConfidence);

  return { extractedTriples: triples };
}

async function persistDataNode(state: AnalyzeStateType) {
  const cfg = getConfig();
  const client = new Neo4jClient(cfg);
  await client.connect();
  try {
    await client.insertTriples(state.extractedTriples);
  } finally {
    await client.disconnect();
  }
  return {};
}

// Wrapping the existing logic nodes since they expect a generic state
// but they access properties common to both state types:
// `extractedTriples`, `inferredFacts`, `conflicts`, `answerSets`.

async function wrappedReasonDatalogNode(state: AnalyzeStateType) {
  // We can pass the AnalyzeStateType into reasonDatalogNode as long as it fits the expected signature.
  // We'll safely await it. It returns inferredFacts and conflicts.
  return reasonDatalogNode(state as any);
}

async function wrappedReasonAspConflictsNode(state: AnalyzeStateType) {
  // Returns answerSets
  return reasonAspConflictsNode(state as any);
}

export function analyzeShouldRunAsp(
  state: AnalyzeStateType,
): "reason_asp_conflicts" | "synthesize_report" {
  return state.conflicts.length > 0 ? "reason_asp_conflicts" : "synthesize_report";
}

async function synthesizeReportNode(state: AnalyzeStateType) {
  log.info("Synthesizing analysis report");
  const llm = getLlm();

  const triplesText = state.extractedTriples
    .slice(0, 40)
    .map((t: Triple) => `  (${t.subject}) --[${t.predicate}]--> (${t.object})`)
    .join("\n");

  const systemPrompt = `You are a data analysis expert. Using the provided data triples, inferred facts, and methodology, write a structured analysis report in Markdown.

The report must:
1. Answer the user's instruction directly
2. Present key findings grounded in the data triples
3. Highlight patterns and relationships discovered
4. Note any conflicts or contradictions found and how ASP conflict resolution models handled them
5. Include a conclusions section`;

  let answerSetsText = "";
  if (state.answerSets && state.answerSets.length > 0) {
     answerSetsText = `\nAnswer Sets (ASP resolved conflicts):\n${state.answerSets.map((s,i) => `Set ${i+1}: ${s.join(", ")}`).join("\n")}`;
  }

  const userPrompt = `Instruction: ${state.instruction}

Methodology (from reference files):
${state.methodology.slice(0, 1500)}

Extracted Data Triples (${state.extractedTriples.length} total, top 40):
${triplesText}

Inferred Facts (${state.inferredFacts.length}):
${state.inferredFacts.slice(0, 20).join("\n  ")}

Conflicts detected: ${state.conflicts.length}
${state.conflicts.slice(0, 5).join("\n  ")}
${answerSetsText}

Write a comprehensive analysis report in Markdown.`;

  const result = await withLlmRetry(() =>
    llm.invoke([new SystemMessage(systemPrompt), new HumanMessage(userPrompt)]),
  );

  return { report: result.content as string };
}

async function exportResultNode(state: AnalyzeStateType) {
  const { outputFile, report } = state;

  if (outputFile.endsWith(".xlsx") || outputFile.endsWith(".xls")) {
    const excelDataFile = state.dataFiles.find(
      (f: string) => f.endsWith(".xlsx") || f.endsWith(".xls"),
    );
    const targetFile = excelDataFile ?? outputFile;
    writeAnalysisSheet(targetFile, "VFS Analysis", report);
    log.info({ file: targetFile }, "Analysis written to Excel sheet");
  } else {
    writeFileSync(outputFile, report, "utf8");
    log.info({ file: outputFile }, "Analysis report written");
  }

  return {};
}

// ─── Graph ────────────────────────────────────────────────────────────────────

function buildAnalyzeGraph() {
  const graph = new StateGraph(AnalyzeState)
    .addNode("ingest_files", ingestFilesNode)
    .addNode("extract_data_triples", extractDataTriplesNode)
    .addNode("persist", persistDataNode)
    .addNode("reason_datalog", wrappedReasonDatalogNode)
    .addNode("reason_asp_conflicts", wrappedReasonAspConflictsNode)
    .addNode("synthesize_report", synthesizeReportNode)
    .addNode("export_result", exportResultNode)

    .addEdge(START, "ingest_files")
    .addEdge("ingest_files", "extract_data_triples")
    .addEdge("extract_data_triples", "persist")
    .addEdge("persist", "reason_datalog")

    // Conditional routing identical to the Research Graph
    .addConditionalEdges("reason_datalog", analyzeShouldRunAsp, [
      "reason_asp_conflicts",
      "synthesize_report",
    ])
    .addEdge("reason_asp_conflicts", "synthesize_report")
    .addEdge("synthesize_report", "export_result")
    .addEdge("export_result", END);

  return graph.compile();
}

let _analyzeGraph: ReturnType<typeof buildAnalyzeGraph> | undefined;

export async function runAnalyzeGraph(params: {
  dataFiles: string[];
  refFiles: string[];
  instruction: string;
  outputFile: string;
}): Promise<void> {
  if (!_analyzeGraph) _analyzeGraph = buildAnalyzeGraph();
  await _analyzeGraph.invoke(params);
}
