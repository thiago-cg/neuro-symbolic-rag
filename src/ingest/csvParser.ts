import { readFileSync } from "fs";
import { parse } from "csv-parse/sync";

export interface CsvData {
  headers: string[];
  rows: Record<string, string>[];
  text: string;
}

export function parseCsv(filePath: string): CsvData {
  const content = readFileSync(filePath, "utf8");
  const rows = parse(content, {
    columns: true,
    skip_empty_lines: true,
    trim: true,
  }) as Record<string, string>[];

  const headers = rows.length > 0 ? Object.keys(rows[0]!) : [];
  const textLines = [
    `Columns: ${headers.join(", ")}`,
    ...rows.slice(0, 100).map((r) =>
      headers.map((h) => `${h}=${r[h]}`).join(" | "),
    ),
  ];

  return { headers, rows, text: textLines.join("\n") };
}
