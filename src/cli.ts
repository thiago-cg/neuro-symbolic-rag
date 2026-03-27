#!/usr/bin/env tsx
import "dotenv/config";
import { Command } from "commander";
import * as p from "@clack/prompts";
import chalk from "chalk";
import { getLogger } from "./observability.js";

const log = getLogger("cli");
const program = new Command();

program
  .name("vfs")
  .description("VFS Neuro-Symbolic Research System")
  .version("0.1.0");

// ─── research ────────────────────────────────────────────────────────────────
program
  .command("research <query>")
  .description("Research a topic: fetch papers, extract triples, reason and synthesize")
  .option("-s, --stream", "Stream progress events to stdout", false)
  .action(async (query: string, opts: { stream: boolean }) => {
    const { runResearchGraph } = await import("./graph/graph.js");
    p.intro(chalk.cyan("VFS Research"));

    const spin = p.spinner();
    spin.start("Running research pipeline…");
    try {
      const result = await runResearchGraph(query, opts.stream ? (event: string) => {
        p.log.step(event);
      } : undefined);
      spin.stop("Done");
      p.note(result.answer ?? "No answer produced", "Result");
    } catch (err) {
      spin.stop("Failed");
      p.log.error(String(err));
      process.exitCode = 1;
    }
    p.outro("Research complete");
  });

// ─── analyze ─────────────────────────────────────────────────────────────────
program
  .command("analyze")
  .description("Analyze local data files guided by research methodology")
  .option("-d, --data <files...>", "Data files (Excel, CSV, SQLite)")
  .option("-r, --refs <files...>", "Reference PDFs (methodology / bibliography)")
  .option("-p, --prompt <instruction>", "Analysis instruction")
  .option("-o, --output <file>", "Output file (default: analysis_report.md)")
  .action(
    async (opts: {
      data?: string[];
      refs?: string[];
      prompt?: string;
      output?: string;
    }) => {
      const { runAnalyzeGraph } = await import("./graph/analyzeGraph.js");
      p.intro(chalk.magenta("VFS Data Analysis"));

      // Fill missing options interactively
      if (!opts.data) {
        const answer = await p.text({
          message: "Path(s) to data file(s) (comma-separated):",
          placeholder: "dados.xlsx, base.sqlite",
        });
        if (p.isCancel(answer)) { p.cancel("Aborted"); process.exit(0); }
        opts.data = (answer as string).split(",").map((s) => s.trim());
      }

      if (!opts.prompt) {
        const answer = await p.text({
          message: "What should be done with the data?",
          placeholder: "Evaluate according to methodology X and generate a report",
        });
        if (p.isCancel(answer)) { p.cancel("Aborted"); process.exit(0); }
        opts.prompt = answer as string;
      }

      const outputFile = opts.output ?? "analysis_report.md";
      const spin = p.spinner();
      spin.start("Analyzing data…");
      try {
        await runAnalyzeGraph({
          dataFiles: opts.data,
          refFiles: opts.refs ?? [],
          instruction: opts.prompt,
          outputFile,
        });
        spin.stop(`Report written to ${chalk.green(outputFile)}`);
      } catch (err) {
        spin.stop("Failed");
        p.log.error(String(err));
        process.exitCode = 1;
      }
      p.outro("Analysis complete");
    },
  );

// ─── query ────────────────────────────────────────────────────────────────────
program
  .command("query <cypher>")
  .description("Run a read-only Cypher query against the Neo4j knowledge graph")
  .action(async (cypher: string) => {
    const { Neo4jClient } = await import("./knowledge_base/neo4jClient.js");
    const { getConfig } = await import("./config.js");
    const cfg = getConfig();
    const client = new Neo4jClient(cfg);
    await client.connect();
    try {
      const rows = await client.query(cypher);
      console.log(JSON.stringify(rows, null, 2));
    } finally {
      await client.disconnect();
    }
  });

// ─── serve ────────────────────────────────────────────────────────────────────
program
  .command("serve")
  .description("Start the Fastify REST API server")
  .option("-p, --port <number>", "Port to listen on")
  .option("-h, --host <host>", "Host to bind to")
  .action(async (opts: { port?: string; host?: string }) => {
    const { startServer } = await import("./api/main.js");
    const { getConfig } = await import("./config.js");
    const cfg = getConfig();
    await startServer({
      port: opts.port ? parseInt(opts.port, 10) : cfg.port,
      host: opts.host ?? cfg.host,
    });
  });

// ─── reset-db ─────────────────────────────────────────────────────────────────
program
  .command("reset-db")
  .description("Drop and recreate the Neo4j schema (constraints + indexes)")
  .action(async () => {
    const { Neo4jClient } = await import("./knowledge_base/neo4jClient.js");
    const { getConfig } = await import("./config.js");
    const cfg = getConfig();
    const client = new Neo4jClient(cfg);
    await client.connect();
    await client.resetSchema();
    await client.disconnect();
    console.log("Schema reset complete.");
  });

// ─── Interactive mode (no args) ───────────────────────────────────────────────
async function interactiveMode() {
  p.intro(chalk.bold.cyan("VFS Neuro-Symbolic System"));

  const action = await p.select({
    message: "What would you like to do?",
    options: [
      { value: "research", label: "Research a topic (online papers)" },
      { value: "analyze",  label: "Analyze local data files" },
      { value: "query",    label: "Query the knowledge graph (Cypher)" },
      { value: "serve",    label: "Start API server" },
    ],
  });

  if (p.isCancel(action)) { p.cancel("Bye!"); process.exit(0); }

  if (action === "research") {
    const query = await p.text({ message: "Research query:" });
    if (p.isCancel(query)) { p.cancel("Bye!"); process.exit(0); }
    process.argv.push("research", query as string);
  } else if (action === "analyze") {
    process.argv.push("analyze");
  } else if (action === "query") {
    const cypher = await p.text({
      message: "Cypher query:",
      placeholder: "MATCH (n:Paper) RETURN n LIMIT 10",
    });
    if (p.isCancel(cypher)) { p.cancel("Bye!"); process.exit(0); }
    process.argv.push("query", cypher as string);
  } else if (action === "serve") {
    process.argv.push("serve");
  }

  await program.parseAsync(process.argv);
}

// ─── Entry ────────────────────────────────────────────────────────────────────
const args = process.argv.slice(2);
if (args.length === 0) {
  interactiveMode().catch((err) => {
    log.error(err);
    process.exit(1);
  });
} else {
  program.parseAsync(process.argv).catch((err) => {
    log.error(err);
    process.exit(1);
  });
}
