from dataclasses import dataclass

from alrf.classifier.base import ClassifierResult


@dataclass(frozen=True)
class RoutingDecision:
    provider: str           # "openai" | "anthropic" | "ollama" | "gemini"
    model: str
    tier: str               # "local" | "fast" | "reasoning"
    policy: str
    use_rag: bool
    reason: str
    classifier_result: ClassifierResult
