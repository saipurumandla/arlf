import json
from dataclasses import dataclass
from pathlib import Path

from alrf.classifier.base import BaseClassifier
from alrf.classifier.heuristic import HeuristicClassifier
from alrf.models.config import RouterConfig
from alrf.router import Router
from alrf.routing.policies import CostAwarePolicy, LatencyFirstPolicy, QualityFirstPolicy

# flat per-query estimate, enough to compare routes against each other
_TIER_COST = {"local": 0.0, "fast": 0.0006, "reasoning": 0.009}

_POLICIES = {
    "cost_aware": CostAwarePolicy(),
    "quality_first": QualityFirstPolicy(),
    "latency_first": LatencyFirstPolicy(),
}


@dataclass(frozen=True)
class EvalCase:
    query: str
    expected_tier: str
    expects_escalation: bool
    note: str


def load_cases(path: Path) -> list[EvalCase]:
    with path.open(encoding="utf-8") as fh:
        return [EvalCase(**json.loads(line)) for line in fh if line.strip()]


def _ratio(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


class RoutingEvaluator:
    def __init__(
        self,
        case_path: Path = Path("eval/routing_set.jsonl"),
        config: RouterConfig | None = None,
        classifier: BaseClassifier | None = None,
    ) -> None:
        self._cases = load_cases(case_path)
        self._config = config or RouterConfig()
        self._classifier = classifier or HeuristicClassifier()
        self._policy = _POLICIES[self._config.policy]

    async def run(self, router: Router | None = None) -> dict[str, object]:
        correct = routed_cost = cached = 0.0
        true_pos = false_pos = false_neg = 0

        for case in self._cases:
            clf_result = await self._classifier.classify(case.query)
            decision = self._policy.decide(clf_result, self._config)
            correct += decision.tier == case.expected_tier
            routed_cost += _TIER_COST[decision.tier]

            if router is None:
                continue
            result = await router.run(case.query)
            cached += result.cached
            if result.escalated and case.expects_escalation:
                true_pos += 1
            elif result.escalated:
                false_pos += 1
            elif case.expects_escalation:
                false_neg += 1

        total = len(self._cases)
        reasoning_cost = _TIER_COST["reasoning"] * total
        return {
            "cases": total,
            "policy": self._config.policy,
            "tier_accuracy": round(correct / total, 4),
            "cost_usd_routed": round(routed_cost, 4),
            "cost_usd_always_reasoning": round(reasoning_cost, 4),
            "cost_delta_pct": round(100 * (routed_cost - reasoning_cost) / reasoning_cost, 1),
            "escalation_precision": _ratio(true_pos, true_pos + false_pos),
            "escalation_recall": _ratio(true_pos, true_pos + false_neg),
            "cache_hit_rate": _ratio(int(cached), total) if router is not None else None,
        }

    def save(self, report: dict[str, object], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
