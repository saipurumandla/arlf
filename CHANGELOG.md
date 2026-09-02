# Changelog

## v0.2.0 — August 2026

### Phase 4 — Caching, Tracing, Streaming, Evals (Jun–Aug 2026)
- Semantic cache: embedding similarity lookup before the classifier, `cached=True` and zero cost on a hit
- Cache TTL and LRU eviction at `cache_max_entries`, hit rate recorded per route
- OpenTelemetry spans per `Router.run()` with child spans for classify, policy, provider call, escalation
- `Router.stream()` yielding tokens, confidence scored on the completed buffer, no mid-stream escalation
- Streaming adapters for OpenAI, Anthropic and Ollama; Gemini yields a single chunk
- Routing eval harness over a 20-case labelled set: tier accuracy 55%, cost 96% below always-reasoning

## v0.1.0 — April 2026

### Phase 3 — Observability, Explainability, CLI (Mar–Apr 2026)
- SQLite observability store recording every routing decision
- Query history API: cost by route, p50/p95 latency, escalation rate, RAG hit rate
- Routing explainer: 5-step human-readable decision trace
- Rich CLI dashboard with live routing table (`alrf dashboard`)
- `alrf history` command for cost/latency summaries

### Phase 2 — Policies, RAG Hook, Multi-Provider (Dec 2025–Feb 2026)
- Policy engine: `cost_aware`, `quality_first`, `latency_first`
- Confidence scorer: hedge detection + length ratio + stop reason
- Escalation: auto-upgrade tier when confidence below threshold
- Ollama adapter for local models; OpenAI-compat adapter for LM Studio/vLLM
- RAG hook Protocol: inject retrieved context when classifier signals knowledge query
- Gemini adapter + normalized RouterResult schema
- Retry budget capping total attempts across fallback and escalation

### Phase 1 — Core Routing Engine (Oct–Nov 2025)
- Heuristic classifier: token count + keyword patterns -> complexity/intent
- OpenAI and Anthropic async adapters
- RoutingEngine mapping classifier output to provider decision
- FallbackChain: retry with alternate provider on timeout or 5xx
- structlog routing trace
