/** Normalize entity name: lowercase, trim, collapse whitespace */
export function normalizeEntity(name: string): string {
  return name.toLowerCase().trim().replace(/\s+/g, " ");
}
