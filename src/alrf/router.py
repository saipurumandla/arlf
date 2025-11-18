import os

import structlog
from dotenv import load_dotenv

from alrf.classifier.heuristic import HeuristicClassifier
from alrf.fallback.chain import FallbackChain
from alrf.providers.anthropic import AnthropicProvider
from alrf.providers.openai import OpenAIProvider
from alrf.routing.engine import RoutingEngine

load_dotenv()

_log_format = os.getenv("LOG_FORMAT", "console")
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        (
            structlog.dev.ConsoleRenderer()
            if _log_format == "console"
            else structlog.processors.JSONRenderer()
        ),
    ]
)

logger = structlog.get_logger()


class Router:
    def __init__(self) -> None:
        self._classifier = HeuristicClassifier()
        self._engine = RoutingEngine()
        self._fast_chain = FallbackChain([
            (OpenAIProvider(), "gpt-4o-mini"),
            (AnthropicProvider(), "claude-haiku-4-5-20251001"),
        ])
        self._reasoning_chain = FallbackChain([
            (AnthropicProvider(), "claude-sonnet-4-6"),
            (OpenAIProvider(), "gpt-4o"),
        ])

    async def run(self, query: str) -> str:
        clf_result = self._classifier.classify(query)
        decision = self._engine.decide(clf_result)

        chain = self._reasoning_chain if decision.tier == "reasoning" else self._fast_chain
        response = await chain.run(query)

        await logger.ainfo(
            "routing_decision",
            tier=decision.tier,
            provider=response.provider,
            model=response.model,
            complexity=clf_result.complexity.value,
            signals=clf_result.signals,
        )
        return response.text
