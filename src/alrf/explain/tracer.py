from dataclasses import dataclass

from alrf.classifier.base import ClassifierResult
from alrf.models.config import RouterConfig
from alrf.models.response import RouterResult
from alrf.routing.decision import RoutingDecision


@dataclass(frozen=True)
class TraceStep:
    step: str
    label: str
    details: dict[str, object]

    def __str__(self) -> str:
        parts = "  ".join(f"{k}={v}" for k, v in self.details.items())
        return f"{self.step}. {self.label:<14} {parts}"


@dataclass(frozen=True)
class DecisionTrace:
    steps: list[TraceStep]

    def __str__(self) -> str:
        return "\n".join(str(s) for s in self.steps)


class RoutingExplainer:
    def explain(
        self,
        query: str,
        clf: ClassifierResult,
        initial_decision: RoutingDecision,
        result: RouterResult,
        config: RouterConfig,
    ) -> DecisionTrace:
        steps: list[TraceStep] = []

        steps.append(TraceStep(
            step="1",
            label="classifier",
            details={
                "complexity": clf.complexity.value,
                "intent": clf.intent.value,
                "token_est": clf.token_estimate,
                "signals": "[" + ", ".join(clf.signals) + "]",
            },
        ))

        steps.append(TraceStep(
            step="2",
            label="policy",
            details={
                "policy": config.policy,
                "tier": initial_decision.tier,
                "provider": initial_decision.provider,
                "model": initial_decision.model,
            },
        ))

        steps.append(TraceStep(
            step="3",
            label="confidence",
            details={
                "score": result.confidence,
                "threshold": config.escalation_threshold,
                "escalate": result.escalated,
            },
        ))

        steps.append(TraceStep(
            step="4",
            label="escalation",
            details={
                "escalated": result.escalated,
                "final_tier": result.route.lstrip("rag+"),
                "final_provider": result.provider,
                "final_model": result.model,
            },
        ))

        steps.append(TraceStep(
            step="5",
            label="final",
            details={
                "route": result.route,
                "latency_ms": result.latency_ms,
                "rag_used": result.rag_used,
            },
        ))

        return DecisionTrace(steps=steps)
