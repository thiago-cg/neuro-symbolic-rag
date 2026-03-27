import type { Paper } from "../graph/state.js";

/** Remove duplicate papers by paperId, keeping the one with more metadata */
export function deduplicatePapers(papers: Paper[]): Paper[] {
  const seenIds = new Map<string, Paper>();
  const seenTitles = new Map<string, Paper>();

  for (const paper of papers) {
    const cleanTitle = paper.title.toLowerCase().replace(/[^a-z0-9]/g, '');

    // Check if we already have this paper via ID or Title
    const existingById = seenIds.get(paper.paperId);
    const existingByTitle = seenTitles.get(cleanTitle);

    const existing = existingById || existingByTitle;

    if (!existing) {
      seenIds.set(paper.paperId, paper);
      seenTitles.set(cleanTitle, paper);
    } else {
      // Merge metadata (keep the one with abstract, or merge fields)
      if (!existing.abstract && paper.abstract) {
        existing.abstract = paper.abstract;
      }
      if (!existing.url && paper.url) {
        existing.url = paper.url;
      }
      if (!existing.year && paper.year) {
         existing.year = paper.year;
      }
      // If the new one has more authors
      if (paper.authors.length > existing.authors.length) {
         existing.authors = paper.authors;
      }
    }
  }

  // Return unique papers from seenIds
  // It's safe to use either Map's values since they reference the same merged objects
  return Array.from(new Set(seenIds.values()));
}

/** Merge and deduplicate results from multiple sub-agents */
export function harvestPapers(allPapers: Paper[][], maxPapers: number): Paper[] {
  const merged = deduplicatePapers(allPapers.flat());
  // Prioritize papers with abstracts
  return merged
    .sort((a, b) => (b.abstract ? 1 : 0) - (a.abstract ? 1 : 0))
    .slice(0, maxPapers);
}
