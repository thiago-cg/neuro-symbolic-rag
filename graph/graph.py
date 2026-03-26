from langgraph.graph import StateGraph, START, END

from graph.state import ResearchState
from graph.nodes import (
    elicit_domain,
    research_papers,
    extract_triples,
    persist_to_graph,
    reason_datalog,
    reason_asp_conflicts,
    synthesize_response,
)
from graph.router import should_run_asp


def build_graph():
    """Build and compile the neuro-symbolic research graph."""
    builder = StateGraph(ResearchState)

    # Register nodes
    builder.add_node("elicit", elicit_domain)
    builder.add_node("research", research_papers)
    builder.add_node("extract_triples", extract_triples)
    builder.add_node("persist", persist_to_graph)
    builder.add_node("reason_datalog", reason_datalog)
    builder.add_node("reason_asp_conflicts", reason_asp_conflicts)
    builder.add_node("synthesize", synthesize_response)

    # Static edges
    builder.add_edge(START, "elicit")
    builder.add_edge("elicit", "research")
    builder.add_edge("research", "extract_triples")
    builder.add_edge("extract_triples", "persist")
    builder.add_edge("persist", "reason_datalog")
    # (persist → reason_datalog already added above)

    # Conditional: after Datalog, go to ASP conflict resolution OR direct synthesis
    builder.add_conditional_edges(
        "reason_datalog",
        should_run_asp,
        ["reason_asp_conflicts", "synthesize"],
    )

    builder.add_edge("reason_asp_conflicts", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()
