from alrf.classifier.base import ClassifierResult, QueryComplexity
from alrf.routing.decision import RoutingDecision

_TIER_MAP: dict[QueryComplexity, str] = {
    QueryComplexity.simple:   "fast",
    QueryComplexity.moderate: "fast",
    QueryComplexity.complex:  "reasoning",
}

_PROVIDER_MODEL: dict[str, tuple[str, str]] = {
    "fast":      ("openai",    "gpt-4o-mini"),
    "reasoning": ("anthropic", "claude-sonnet-4-6"),
}


class RoutingEngine:
    def decide(self, result: ClassifierResult) -> RoutingDecision:
        tier = _TIER_MAP[result.complexity]
        provider, model = _PROVIDER_MODEL[tier]
        return RoutingDecision(
            provider=provider,
            model=model,
            tier=tier,
            policy="default",
            use_rag=result.needs_retrieval,
            reason=f"complexity={result.complexity.value} → {tier}",
            classifier_result=result,
        )
