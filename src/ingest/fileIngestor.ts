import { extname } from "path";
import { parseExcel } from "./excelParser.js";
import { parsePdf } from "./pdfParser.js";
import { parseCsv } from "./csvParser.js";
import { parseSqlite } from "./sqliteParser.js";
import { getLogger } from "../observability.js";

const log = getLogger("ingest");

export interface IngestedFile {
  path: string;
  type: "excel" | "pdf" | "csv" | "sqlite" | "text";
  text: string;
  /** Raw structured data (for Excel/CSV/SQLite output writing) */
  raw?: unknown;
}

export async function ingestFile(filePath: string): Promise<IngestedFile> {
  const ext = extname(filePath).toLowerCase();
  log.debug({ filePath, ext }, "Ingesting file");

  switch (ext) {
    case ".xlsx":
    case ".xls": {
      const sheets = parseExcel(filePath);
      const text = sheets.map((s) => s.text).join("\n\n---\n\n");
      return { path: filePath, type: "excel", text, raw: sheets };
    }
    case ".pdf": {
      const data = await parsePdf(filePath);
      return { path: filePath, type: "pdf", text: data.text };
    }
    case ".csv": {
      const data = parseCsv(filePath);
      return { path: filePath, type: "csv", text: data.text, raw: data };
    }
    case ".sqlite":
    case ".db": {
      const data = await parseSqlite(filePath);
      return { path: filePath, type: "sqlite", text: data.text, raw: data };
    }
    default: {
      const { readFileSync } = await import("fs");
      const text = readFileSync(filePath, "utf8");
      return { path: filePath, type: "text", text };
    }
  }
}

export async function ingestFiles(filePaths: string[]): Promise<IngestedFile[]> {
  return Promise.all(filePaths.map(ingestFile));
}
