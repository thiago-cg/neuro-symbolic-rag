import type { ResearchStateType } from "./state.js";

/** Route to ASP conflict resolution if conflicts were detected, else skip to synthesize */
export function shouldRunAsp(
  state: ResearchStateType,
): "reason_asp_conflicts" | "synthesize" {
  return state.conflicts.length > 0 ? "reason_asp_conflicts" : "synthesize";
}
