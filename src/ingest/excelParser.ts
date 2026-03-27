import * as XLSX from "xlsx";

export interface SheetData {
  name: string;
  headers: string[];
  rows: Record<string, unknown>[];
  /** Plain text representation for LLM ingestion */
  text: string;
}

export function parseExcel(filePath: string): SheetData[] {
  const workbook = XLSX.readFile(filePath);
  const results: SheetData[] = [];

  for (const sheetName of workbook.SheetNames) {
    const sheet = workbook.Sheets[sheetName];
    if (!sheet) continue;

    const rows = XLSX.utils.sheet_to_json<Record<string, unknown>>(sheet, {
      defval: "",
    });
    const headers = rows.length > 0 ? Object.keys(rows[0]!) : [];

    const textLines = [
      `Sheet: ${sheetName}`,
      `Columns: ${headers.join(", ")}`,
      ...rows.slice(0, 100).map((r) =>
        headers.map((h) => `${h}=${r[h]}`).join(" | "),
      ),
    ];

    results.push({ name: sheetName, headers, rows, text: textLines.join("\n") });
  }

  return results;
}

/** Write a new sheet into an existing Excel file (or create it) */
export function writeAnalysisSheet(
  filePath: string,
  sheetName: string,
  content: string,
): void {
  let workbook: XLSX.WorkBook;
  try {
    workbook = XLSX.readFile(filePath);
  } catch {
    workbook = XLSX.utils.book_new();
  }

  const lines = content.split("\n").map((line) => [line]);
  const sheet = XLSX.utils.aoa_to_sheet(lines);
  XLSX.utils.book_append_sheet(workbook, sheet, sheetName);
  XLSX.writeFile(workbook, filePath);
}
