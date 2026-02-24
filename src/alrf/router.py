import os
import time

import structlog
from dotenv import load_dotenv

from alrf.classifier.heuristic import HeuristicClassifier
from alrf.evaluation.confidence import ConfidenceScorer
from alrf.evaluation.escalation import EscalationHandler
from alrf.fallback.chain import FallbackChain
from alrf.models.config import RouterConfig
from alrf.models.response import RouterResult
from alrf.providers.anthropic import AnthropicProvider
from alrf.providers.ollama import OllamaProvider
from alrf.providers.openai import OpenAIProvider
from alrf.rag.hook import RAGHook
from alrf.routing.decision import RoutingDecision
from alrf.routing.policies import CostAwarePolicy, LatencyFirstPolicy, QualityFirstPolicy

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

_POLICIES = {
    "cost_aware":    CostAwarePolicy(),
    "quality_first": QualityFirstPolicy(),
    "latency_first": LatencyFirstPolicy(),
}


class Router:
    def __init__(
        self,
        config: RouterConfig | None = None,
        rag_hook: RAGHook | None = None,
    ) -> None:
        self._config = config or RouterConfig()
        self._classifier = HeuristicClassifier()
        self._scorer = ConfidenceScorer()
        self._escalation = EscalationHandler()
        self._rag_hook = rag_hook

    def _chain_for(self, decision: RoutingDecision) -> FallbackChain:
        cfg = self._config
        if decision.tier == "local":
            local_provider = (
                OllamaProvider(base_url=cfg.local_base_url)
                if cfg.local_provider == "ollama"
                else OpenAIProvider(base_url=cfg.local_base_url, api_key="local")
            )
            return FallbackChain([(local_provider, cfg.local_model)])
        if decision.tier == "fast":
            return FallbackChain([
                (OpenAIProvider(), cfg.fast_model),
                (AnthropicProvider(), "claude-haiku-4-5-20251001"),
            ])
        return FallbackChain([
            (AnthropicProvider(), cfg.reasoning_model),
            (OpenAIProvider(), "gpt-4o"),
        ])

    async def run(self, query: str) -> RouterResult:
        cfg = self._config
        clf_result = self._classifier.classify(query)
        policy = _POLICIES.get(cfg.policy, CostAwarePolicy())
        decision = policy.decide(clf_result, cfg)

        prompt = query
        rag_used = False
        if decision.use_rag and self._rag_hook is not None:
            try:
                chunks = await self._rag_hook.retrieve(query)
                prompt = "[CONTEXT]\n" + "\n".join(chunks) + "\n\n" + query
                rag_used = True
            except Exception as exc:
                await logger.awarning("rag_hook_failed", error=str(exc))

        attempts = 0
        escalated = False
        trace: list[dict] = []
        t0 = time.monotonic()

        while attempts < cfg.retry_budget:
            response = await self._chain_for(decision).run(prompt)
            attempts += 1
            confidence = self._scorer.score(response, query)
            trace.append({"tier": decision.tier, "provider": response.provider,
                          "model": response.model, "confidence": confidence})

            if self._escalation.should_escalate(confidence, cfg, attempts):
                next_decision = self._escalation.escalate(decision, cfg)
                if next_decision is not None:
                    decision = next_decision
                    escalated = True
                    continue
            break

        latency_ms = int((time.monotonic() - t0) * 1000)
        route = ("rag+" if rag_used else "") + decision.tier

        await logger.ainfo(
            "routing_decision",
            route=route, provider=response.provider, model=response.model,
            confidence=confidence, escalated=escalated, attempts=attempts,
        )

        return RouterResult(
            answer=response.text,
            route=route,
            provider=response.provider,
            model=response.model,
            cost_usd=None,
            latency_ms=latency_ms,
            confidence=confidence,
            escalated=escalated,
            rag_used=rag_used,
            decision_trace=trace,
        )
