from typing import Protocol, runtime_checkable

from alrf.classifier.base import ClassifierResult, QueryComplexity
from alrf.models.config import RouterConfig
from alrf.routing.decision import RoutingDecision


@runtime_checkable
class RoutingPolicy(Protocol):
    def decide(self, result: ClassifierResult, config: RouterConfig) -> RoutingDecision: ...


def _make(tier: str, result: ClassifierResult, config: RouterConfig, policy: str) -> RoutingDecision:
    if tier == "local":
        provider, model = "ollama", config.local_model
    elif tier == "fast":
        provider, model = "openai", config.fast_model
    else:
        provider, model = "anthropic", config.reasoning_model
    return RoutingDecision(
        provider=provider,
        model=model,
        tier=tier,
        policy=policy,
        use_rag=result.needs_retrieval,
        reason=f"policy={policy} → {tier}",
        classifier_result=result,
    )


class CostAwarePolicy:
    def decide(self, result: ClassifierResult, config: RouterConfig) -> RoutingDecision:
        tier_map = {
            QueryComplexity.simple:   "local",
            QueryComplexity.moderate: "fast",
            QueryComplexity.complex:  "reasoning",
        }
        return _make(tier_map[result.complexity], result, config, "cost_aware")


class QualityFirstPolicy:
    def decide(self, result: ClassifierResult, config: RouterConfig) -> RoutingDecision:
        tier = "fast" if result.complexity == QueryComplexity.simple else "reasoning"
        return _make(tier, result, config, "quality_first")


class LatencyFirstPolicy:
    def decide(self, result: ClassifierResult, config: RouterConfig) -> RoutingDecision:
        tier_map = {
            QueryComplexity.simple:   "local",
            QueryComplexity.moderate: "fast",
            QueryComplexity.complex:  "fast",   # skip reasoning to keep latency low
        }
        return _make(tier_map[result.complexity], result, config, "latency_first")
