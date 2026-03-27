import type { Paper } from "../graph/state.js";

/** Remove duplicate papers by paperId, keeping the one with more metadata */
export function deduplicatePapers(papers: Paper[]): Paper[] {
  const seen = new Map<string, Paper>();
  for (const paper of papers) {
    const existing = seen.get(paper.paperId);
    if (!existing || (!existing.abstract && paper.abstract)) {
      seen.set(paper.paperId, paper);
    }
  }
  return [...seen.values()];
}

/** Merge and deduplicate results from multiple sub-agents */
export function harvestPapers(allPapers: Paper[][], maxPapers: number): Paper[] {
  const merged = deduplicatePapers(allPapers.flat());
  // Prioritize papers with abstracts
  return merged
    .sort((a, b) => (b.abstract ? 1 : 0) - (a.abstract ? 1 : 0))
    .slice(0, maxPapers);
}
