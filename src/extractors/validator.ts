import { ExtractedTripleSchema, type ExtractedTriple } from "./tripleSchema.js";
import { normalizeEntity } from "./normalizer.js";
import { getLogger } from "../observability.js";

const log = getLogger("extractors:validator");

export function validateTriple(raw: unknown): ExtractedTriple | null {
  const result = ExtractedTripleSchema.safeParse(raw);
  if (!result.success) {
    log.debug({ raw, error: result.error.message }, "Triple validation failed");
    return null;
  }
  const t = result.data;
  return {
    ...t,
    subject: normalizeEntity(t.subject),
    object: normalizeEntity(t.object),
  };
}

export function validateTriples(raws: unknown[]): ExtractedTriple[] {
  return raws.flatMap((r) => {
    const v = validateTriple(r);
    return v ? [v] : [];
  });
}
