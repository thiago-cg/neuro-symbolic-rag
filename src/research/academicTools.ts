import { getConfig } from "../config.js";
import { withApiRetry } from "../retry.js";
import type { Paper } from "../graph/state.js";
import { getLogger } from "../observability.js";

const log = getLogger("research:academic");

const SS_BASE = "https://api.semanticscholar.org/graph/v1";
const ARXIV_BASE = "https://export.arxiv.org/api/query";

interface SsRecord {
  paperId: string;
  title: string;
  abstract?: string;
  authors?: Array<{ name: string }>;
  year?: number;
  url?: string;
}

export async function searchSemanticScholar(
  query: string,
  limit = 10,
): Promise<Paper[]> {
  const cfg = getConfig();
  const headers: Record<string, string> = {
    "User-Agent": "VFS-Research/0.1",
  };
  if (cfg.semanticScholarKey) {
    headers["x-api-key"] = cfg.semanticScholarKey;
  }

  const url = new URL(`${SS_BASE}/paper/search`);
  url.searchParams.set("query", query);
  url.searchParams.set("limit", String(limit));
  url.searchParams.set("fields", "paperId,title,abstract,authors,year,url");

  return withApiRetry(async () => {
    const res = await fetch(url.toString(), { headers });
    if (!res.ok) throw new Error(`Semantic Scholar error: ${res.status}`);
    const data = (await res.json()) as { data?: SsRecord[] };
    return (data.data ?? []).map(
      (p): Paper => ({
        paperId: p.paperId,
        title: p.title,
        abstract: p.abstract,
        authors: (p.authors ?? []).map((a) => a.name),
        year: p.year,
        url: p.url,
      }),
    );
  });
}

export async function searchArXiv(query: string, maxResults = 10): Promise<Paper[]> {
  const url = new URL(ARXIV_BASE);
  url.searchParams.set("search_query", `all:${encodeURIComponent(query)}`);
  url.searchParams.set("max_results", String(maxResults));

  return withApiRetry(async () => {
    const res = await fetch(url.toString());
    if (!res.ok) throw new Error(`ArXiv error: ${res.status}`);
    const xml = await res.text();
    return parseArXivXml(xml);
  });
}

function parseArXivXml(xml: string): Paper[] {
  const papers: Paper[] = [];
  const entries = xml.match(/<entry>([\s\S]*?)<\/entry>/g) ?? [];

  for (const entry of entries) {
    const id = entry.match(/<id>(.*?)<\/id>/)?.[1] ?? "";
    const title = entry.match(/<title>([\s\S]*?)<\/title>/)?.[1]?.trim() ?? "";
    const abstract = entry.match(/<summary>([\s\S]*?)<\/summary>/)?.[1]?.trim();
    const yearMatch = entry.match(/<published>(\d{4})/);
    const year = yearMatch ? parseInt(yearMatch[1]!, 10) : undefined;

    const authorMatches = [...entry.matchAll(/<name>(.*?)<\/name>/g)];
    const authors = authorMatches.map((m) => m[1] ?? "");

    if (id && title) {
      papers.push({ paperId: id, title, abstract, authors, year, url: id });
    }
  }

  log.debug({ count: papers.length }, "ArXiv results parsed");
  return papers;
}
