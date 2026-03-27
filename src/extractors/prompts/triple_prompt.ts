export const TRIPLE_SYSTEM = `You are a knowledge extraction engine. Given a scientific text, extract semantic triples.

Each triple must use EXACTLY one of these predicates:
- cita: paper A cites paper B
- supera: work A surpasses/outperforms work B
- usa: work A uses method/tool B
- trata_conceito: work A addresses/treats concept B
- define: work A defines concept B
- contradiz: work A contradicts claim/work B
- estende: work A extends work B

Output ONLY valid JSON array. Each item: { "subject": "...", "predicate": "...", "object": "...", "confidence": 0.0-1.0 }
Confidence reflects how certain you are the relation holds.`;

export function tripleUserPrompt(text: string, source?: string): string {
  return `Extract triples from the following text${source ? ` (source: ${source})` : ""}:\n\n${text}`;
}
