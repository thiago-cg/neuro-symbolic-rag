"""Triple batch validation — quality checks before symbolic reasoning."""

from dataclasses import dataclass, field

from extractors.triple_schema import VALID_PREDICATES

CONFIDENCE_WARNING_THRESHOLD = 0.5


@dataclass
class ValidationReport:
    total: int
    invalid_predicates: list[str] = field(default_factory=list)
    unnormalized_entities: list[str] = field(default_factory=list)
    low_confidence: list[str] = field(default_factory=list)

    @property
    def pct_invalid_predicates(self) -> float:
        return len(self.invalid_predicates) / self.total * 100 if self.total else 0.0

    @property
    def pct_unnormalized(self) -> float:
        return len(self.unnormalized_entities) / self.total * 100 if self.total else 0.0

    @property
    def pct_low_confidence(self) -> float:
        return len(self.low_confidence) / self.total * 100 if self.total else 0.0

    @property
    def has_warnings(self) -> bool:
        return bool(self.invalid_predicates or self.unnormalized_entities or self.low_confidence)

    def summary(self) -> dict:
        return {
            "total": self.total,
            "invalid_predicates": len(self.invalid_predicates),
            "pct_invalid_predicates": round(self.pct_invalid_predicates, 1),
            "unnormalized_entities": len(self.unnormalized_entities),
            "pct_unnormalized": round(self.pct_unnormalized, 1),
            "low_confidence": len(self.low_confidence),
            "pct_low_confidence": round(self.pct_low_confidence, 1),
        }


def validate_triple_batch(triples: list[dict]) -> ValidationReport:
    """Validate a batch of triple dicts for quality issues.

    Checks:
    - Predicates outside the allowed vocabulary
    - Entities with spaces (not normalized)
    - Triples with confidence below warning threshold

    Args:
        triples: List of triple dicts (from model_dump() or raw dicts)

    Returns:
        ValidationReport with categorized issues
    """
    report = ValidationReport(total=len(triples))

    for t in triples:
        predicate = t.get("predicate", "")
        subject = t.get("subject", "")
        obj = t.get("obj", "")
        confidence = t.get("confidence", 1.0)
        triple_id = f"{subject}:{predicate}:{obj}"

        # Check predicate vocabulary
        if predicate not in VALID_PREDICATES:
            report.invalid_predicates.append(triple_id)

        # Check entity normalization (should have no spaces)
        if " " in subject or " " in obj:
            report.unnormalized_entities.append(triple_id)

        # Check confidence threshold
        if confidence < CONFIDENCE_WARNING_THRESHOLD:
            report.low_confidence.append(triple_id)

    return report
