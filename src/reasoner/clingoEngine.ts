import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import type { Triple } from "../graph/state.js";
import { getLogger } from "../observability.js";

const log = getLogger("reasoner:clingo");

const __dirname = dirname(fileURLToPath(import.meta.url));

// clingo-wasm module reference — initialized once
let clingoReady = false;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type ClingoModule = { run: (program: string, models: number) => Promise<ClingoResult> };

interface ClingoWitness {
  Value: string[];
  Time?: number;
}

interface ClingoCall {
  Witnesses: ClingoWitness[];
}

interface ClingoResult {
  Result: "SATISFIABLE" | "UNSATISFIABLE" | "UNKNOWN" | "ERROR";
  Call?: ClingoCall[];
  Error?: string;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let clingoMod: any;

async function getClingo(): Promise<ClingoModule> {
  if (!clingoReady) {
    // clingo-wasm exports: { init, run, Runner }
    const mod = await import("clingo-wasm");
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    clingoMod = mod as any;
    await clingoMod.init();
    clingoReady = true;
  }
  return clingoMod as ClingoModule;
}

/** Convert triples to ASP facts */
function triplesToFacts(triples: Triple[]): string {
  return triples
    .map((t) => {
      const subj = t.subject.toLowerCase().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "");
      const obj = t.object.toLowerCase().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "");
      if (!subj || !obj) return null;
      return `${t.predicate}(${subj}, ${obj}).`;
    })
    .filter(Boolean)
    .join("\n");
}

function loadRules(filename: string): string {
  try {
    return readFileSync(join(__dirname, filename), "utf8");
  } catch {
    log.warn({ filename }, "Rules file not found, using empty rules");
    return "";
  }
}

function extractAtoms(result: ClingoResult): string[] {
  return result.Call?.[0]?.Witnesses?.[0]?.Value ?? [];
}

export class ClingoEngine {
  /** Deterministic Datalog inference — single stable model */
  async runDatalog(
    triples: Triple[],
  ): Promise<{ inferredFacts: string[]; conflicts: string[] }> {
    const clingo = await getClingo();
    const facts = triplesToFacts(triples);
    const rules = loadRules("rules.lp");
    const program = `${facts}\n${rules}`;

    log.debug({ facts: triples.length }, "Running Datalog inference");

    const result = await clingo.run(program, 1);

    if (result.Result === "UNSATISFIABLE" || result.Result === "ERROR") {
      log.warn({ result: result.Result, error: result.Error }, "Datalog issue");
      return { inferredFacts: [], conflicts: [] };
    }

    const atoms = extractAtoms(result);
    const inferredFacts = atoms.filter((a) => a.startsWith("inferred("));
    const conflicts = atoms.filter((a) => a.startsWith("conflict("));

    return { inferredFacts, conflicts };
  }

  /** Non-deterministic ASP — enumerate answer sets for conflict resolution */
  async runAsp(triples: Triple[], maxModels = 10): Promise<string[][]> {
    const clingo = await getClingo();
    const facts = triplesToFacts(triples);
    const rules = loadRules("rules.lp");
    const conflictRules = loadRules("aspConflict.lp");
    const program = `${facts}\n${rules}\n${conflictRules}`;

    log.debug({ maxModels }, "Running ASP conflict resolution");

    const result = await clingo.run(program, maxModels);

    if (result.Result === "ERROR") {
      log.warn({ error: result.Error }, "ASP error");
      return [];
    }

    return (result.Call?.[0]?.Witnesses ?? []).map((w: ClingoWitness) => w.Value ?? []);
  }
}
