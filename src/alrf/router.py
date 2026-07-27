import os
import time
from collections.abc import AsyncIterator

import structlog
from dotenv import load_dotenv

from alrf.cache.semantic import SemanticCache
from alrf.classifier.heuristic import HeuristicClassifier
from alrf.classifier.llm import LLMClassifier
from alrf.evaluation.confidence import ConfidenceScorer
from alrf.evaluation.escalation import EscalationHandler
from alrf.fallback.chain import FallbackChain
from alrf.models.config import RouterConfig
from alrf.models.response import RouterResult
from alrf.observability.otel import configure_tracing, get_tracer, set_result_attributes
from alrf.observability.store import ObservabilityStore
from alrf.providers.anthropic import AnthropicProvider
from alrf.providers.ollama import OllamaProvider
from alrf.providers.base import ProviderResponse
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
        self._classifier = (
            LLMClassifier(
                base_url=self._config.llm_classifier_url,
                model=self._config.llm_classifier_model,
            )
            if self._config.classifier == "llm"
            else HeuristicClassifier()
        )
        self._scorer = ConfidenceScorer()
        self._escalation = EscalationHandler()
        self._rag_hook = rag_hook
        self._store = ObservabilityStore(self._config.observability_db_path)
        self._cache = (
            SemanticCache(
                db_path=self._config.observability_db_path,
                threshold=self._config.cache_threshold,
                ttl_seconds=self._config.cache_ttl_seconds,
                max_entries=self._config.cache_max_entries,
            )
            if self._config.cache_enabled
            else None
        )
        configure_tracing(self._config.otel_endpoint)
        self._tracer = get_tracer()
        self._last_result: RouterResult | None = None

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

    async def _build_prompt(self, query: str, decision: RoutingDecision) -> tuple[str, bool]:
        if not decision.use_rag or self._rag_hook is None:
            return query, False
        try:
            chunks = await self._rag_hook.retrieve(query)
        except Exception as exc:
            await logger.awarning("rag_hook_failed", error=str(exc))
            return query, False
        return "[CONTEXT]\n" + "\n".join(chunks) + "\n\n" + query, True

    async def run(self, query: str) -> RouterResult:
        cfg = self._config

        with self._tracer.start_as_current_span("router.run") as span:
            if self._cache is not None:
                hit = await self._cache.lookup(query)
                if hit is not None:
                    set_result_attributes(span, hit)
                    await logger.ainfo("cache_hit", route=hit.route, model=hit.model)
                    return hit

            with self._tracer.start_as_current_span("classify") as clf_span:
                clf_result = await self._classifier.classify(query)
                clf_span.set_attribute("complexity", clf_result.complexity.value)
                clf_span.set_attribute("intent", clf_result.intent.value)

            with self._tracer.start_as_current_span("policy") as policy_span:
                policy = _POLICIES.get(cfg.policy, CostAwarePolicy())
                decision = policy.decide(clf_result, cfg)
                policy_span.set_attribute("policy", cfg.policy)
                policy_span.set_attribute("tier", decision.tier)

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
                with self._tracer.start_as_current_span("provider.call") as call_span:
                    response = await self._chain_for(decision).run(prompt)
                    call_span.set_attribute("provider", response.provider)
                    call_span.set_attribute("model", response.model)
                attempts += 1
                confidence = self._scorer.score(response, query)
                trace.append({"tier": decision.tier, "provider": response.provider,
                              "model": response.model, "confidence": confidence})

                if self._escalation.should_escalate(confidence, cfg, attempts):
                    next_decision = self._escalation.escalate(decision, cfg)
                    if next_decision is not None:
                        with self._tracer.start_as_current_span("escalation") as esc_span:
                            esc_span.set_attribute("from_tier", decision.tier)
                            esc_span.set_attribute("to_tier", next_decision.tier)
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

            result = RouterResult(
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
            set_result_attributes(span, result)

        await self._store.record(query, result, self._config.policy)
        if self._cache is not None:
            await self._cache.store(query, result)
        return result

    async def stream(self, query: str) -> AsyncIterator[str]:
        cfg = self._config
        clf_result = await self._classifier.classify(query)
        policy = _POLICIES.get(cfg.policy, CostAwarePolicy())
        decision = policy.decide(clf_result, cfg)
        prompt, rag_used = await self._build_prompt(query, decision)
        provider, model = self._chain_for(decision).primary

        buffer: list[str] = []
        t0 = time.monotonic()
        async for chunk in provider.stream(prompt, model):
            buffer.append(chunk)
            yield chunk

        answer = "".join(buffer)
        latency_ms = int((time.monotonic() - t0) * 1000)
        # the tier is fixed once the first token is out, so a weak answer is flagged, not retried
        confidence = self._scorer.score(
            ProviderResponse(
                text=answer,
                model=model,
                provider=decision.provider,
                stop_reason="stop",
                input_tokens=0,
                output_tokens=len(buffer),
            ),
            query,
        )
        if confidence < cfg.escalation_threshold:
            await logger.awarning("low_confidence_stream", confidence=confidence, route=decision.tier)

        route = ("rag+" if rag_used else "") + decision.tier
        self._last_result = RouterResult(
            answer=answer,
            route=route,
            provider=decision.provider,
            model=model,
            cost_usd=None,
            latency_ms=latency_ms,
            confidence=confidence,
            escalated=False,
            rag_used=rag_used,
            decision_trace=[{"tier": decision.tier, "provider": decision.provider,
                             "model": model, "confidence": confidence}],
        )
        await self._store.record(query, self._last_result, cfg.policy)

    @property
    def last_result(self) -> RouterResult | None:
        return self._last_result
