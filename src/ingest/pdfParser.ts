import { readFileSync } from "fs";
import pdfParse from "pdf-parse";

export interface PdfData {
  text: string;
  pages: number;
  info: Record<string, unknown>;
}

export async function parsePdf(filePath: string): Promise<PdfData> {
  const buffer = readFileSync(filePath);
  const data = await pdfParse(buffer);
  return {
    text: data.text,
    pages: data.numpages,
    info: data.info as Record<string, unknown>,
  };
}
