import { getConfig } from "../config.js";
import { withApiRetry } from "../retry.js";
import type { Paper } from "../graph/state.js";
import { getLogger } from "../observability.js";
import { XMLParser } from "fast-xml-parser";
import pdfParse from "pdf-parse";
import { fetch } from "undici";

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
  const parser = new XMLParser({
    ignoreAttributes: false,
    attributeNamePrefix: "@_",
    textNodeName: "#text"
  });
  const parsed = parser.parse(xml);
  const papers: Paper[] = [];

  if (!parsed.feed || !parsed.feed.entry) return papers;

  const entries = Array.isArray(parsed.feed.entry) ? parsed.feed.entry : [parsed.feed.entry];

  for (const entry of entries) {
    const id = entry.id || "";
    const title = entry.title ? entry.title.replace(/\n/g, " ").trim() : "";
    const abstract = entry.summary ? entry.summary.replace(/\n/g, " ").trim() : undefined;

    let year: number | undefined;
    if (entry.published) {
      const yearMatch = entry.published.match(/^(\d{4})/);
      if (yearMatch) {
        year = parseInt(yearMatch[1], 10);
      }
    }

    let authors: string[] = [];
    if (entry.author) {
      const authorList = Array.isArray(entry.author) ? entry.author : [entry.author];
      authors = authorList.map((a: any) => a.name || "");
    }

    // Try to find a PDF link
    let pdfUrl = id; // Fallback to id url
    if (entry.link) {
      const links = Array.isArray(entry.link) ? entry.link : [entry.link];
      const pdfLink = links.find((l: any) => l["@_title"] === "pdf" || (l["@_type"] && l["@_type"].includes("pdf")));
      if (pdfLink && pdfLink["@_href"]) {
        pdfUrl = pdfLink["@_href"];
      } else {
         // Fallback heuristic for arXiv: replace /abs/ with /pdf/
         if (id.includes("/abs/")) {
            pdfUrl = id.replace("/abs/", "/pdf/");
         }
      }
    }

    if (id && title) {
      papers.push({ paperId: id, title, abstract, authors, year, url: pdfUrl });
    }
  }

  log.debug({ count: papers.length }, "ArXiv results parsed via fast-xml-parser");
  return papers;
}

export async function fetchAndParsePdf(url: string): Promise<string | undefined> {
    try {
        log.debug({ url }, "Attempting to fetch and parse PDF");
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 15000); // 15s timeout

        let res;
        try {
            res = await fetch(url, {
                headers: {
                    "User-Agent": "VFS-Research/0.1"
                },
                signal: controller.signal
            });
        } finally {
            clearTimeout(timeout);
        }
        if (!res.ok) {
           log.warn({ url, status: res.status }, "Failed to fetch PDF");
           return undefined;
        }
        const buffer = await res.arrayBuffer();
        const data = await pdfParse(Buffer.from(buffer));
        return data.text;
    } catch (error) {
        log.error({ url, error }, "Error parsing PDF");
        return undefined;
    }
}
