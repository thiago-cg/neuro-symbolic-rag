from typing import Literal

from graph.state import ResearchState


def should_run_asp(state: ResearchState) -> Literal["reason_asp_conflicts", "synthesize"]:
    """Route to ASP conflict resolution if conflicts were detected, else synthesize directly."""
    if state.get("conflicts_detected"):
        return "reason_asp_conflicts"
    return "synthesize"
