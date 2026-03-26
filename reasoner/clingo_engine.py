"""Clingo ASP engine wrapper.

Provides two modes:
- Datalog (deterministic): single answer set for transitive inference
- ASP (non-deterministic): multiple answer sets for conflict resolution
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from clingo import Control

if TYPE_CHECKING:
    from extractors.triple_schema import ExtractedTriple

logger = logging.getLogger(__name__)

RULES_DIR = Path(__file__).parent
RULES_ASP = str(RULES_DIR / "rules.asp")
CONFLICT_ASP = str(RULES_DIR / "asp_conflict.asp")


def triples_to_asp_facts(triples: list["ExtractedTriple"]) -> str:
    """Convert extracted triples into Clingo ASP fact strings.

    Args:
        triples: List of ExtractedTriple objects

    Returns:
        Multi-line string of ASP facts
    """
    lines: list[str] = []
    for t in triples:
        s = t.subject.replace('"', "'")
        o = t.obj.replace('"', "'")
        src = t.source_paper.replace('"', "'")
        # e.g.: usa("bert", "transformer").
        lines.append(f'{t.predicate}("{s}", "{o}").  % src={src} conf={t.confidence:.2f}')
    return "\n".join(lines)


class ClingoEngine:
    """Wrapper around Clingo for Datalog and ASP reasoning."""

    def run_datalog(self, facts: str) -> list[str]:
        """Run deterministic Datalog inference.

        Returns a single answer set's atoms as strings.
        Raises ValueError if the program is unsatisfiable.
        """
        ctl = Control()
        ctl.add("base", [], facts)
        ctl.load(RULES_ASP)
        ctl.ground([("base", [])])

        inferred: list[str] = []

        def collect_model(model) -> None:
            inferred.extend(str(sym) for sym in model.symbols(shown=True))

        result = ctl.solve(on_model=collect_model)

        if result.unsatisfiable:
            raise ValueError("Datalog program is unsatisfiable — check for contradictory facts")

        logger.info("clingo datalog: %d atoms inferred", len(inferred))
        ctl.cleanup()
        return inferred

    def run_asp(self, facts: str, max_models: int = 10) -> list[list[str]]:
        """Run non-deterministic ASP conflict resolution.

        Returns multiple answer sets — each represents a valid interpretation.
        """
        ctl = Control(["0"])
        ctl.configuration.solve.models = max_models
        ctl.add("base", [], facts)
        ctl.load(RULES_ASP)
        ctl.load(CONFLICT_ASP)
        ctl.ground([("base", [])])

        answer_sets: list[list[str]] = []

        with ctl.solve(yield_=True) as handle:
            for model in handle:
                answer_sets.append([str(sym) for sym in model.symbols(shown=True)])

        logger.info("clingo asp: %d answer sets found", len(answer_sets))
        ctl.cleanup()
        return answer_sets

    def has_conflicts(self, inferred: list[str]) -> bool:
        """Check if any conflict_potencial atoms exist in the inferred facts."""
        return any("conflito_potencial" in atom for atom in inferred)

    def extract_conflicts(self, inferred: list[str]) -> list[str]:
        """Return only the conflict atoms from inferred facts."""
        return [atom for atom in inferred if "conflito_potencial" in atom]
