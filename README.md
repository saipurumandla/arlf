# alrf — Adaptive LLM Routing Framework

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)

Routes queries to the right LLM provider based on complexity, confidence, and cost policy.
Stop paying gpt-4o prices for "what is Redis?" lookups.

## Architecture

```
Query
  |
  v
HeuristicClassifier      token count + keyword signals -> complexity / intent / needs_retrieval
  |
  v
RoutingPolicy            cost_aware | quality_first | latency_first
  |
  v
FallbackChain            primary provider -> fallback on timeout/5xx
  |
  v
ConfidenceScorer         hedge phrases + length ratio + stop reason -> 0.0-1.0
  |
  v
EscalationHandler        if confidence < threshold AND budget remains -> upgrade tier
  |
  v
ObservabilityStore       aiosqlite routing_events table (non-blocking)
  |
  v
RouterResult             answer + route + provider + model + latency + confidence + trace
```

## Providers

| Tier      | Provider  | Model                    |
|-----------|-----------|--------------------------|
| local     | Ollama    | llama3.2 (configurable)  |
| local*    | LM Studio / vLLM / llama.cpp | any (OpenAI-compat via base_url) |
| fast      | OpenAI    | gpt-4o-mini              |
| fast      | Anthropic | claude-haiku-4-5-20251001|
| reasoning | Anthropic | claude-sonnet-4-6        |
| reasoning | OpenAI    | gpt-4o                   |
| reasoning | Gemini    | gemini-1.5-pro           |

*Set `local_provider="lm_studio"` and `local_base_url="http://localhost:1234/v1"` in RouterConfig.

## Install

    pip install alrf

## Quick start

```python
from alrf import Router, RouterConfig

router = Router(config=RouterConfig(policy="cost_aware"))
result = await router.run("Why is my auth endpoint returning 500s under load?")

print(result.answer)
print(f"routed to {result.provider} via {result.route} in {result.latency_ms}ms")
print(f"confidence={result.confidence:.2f}  escalated={result.escalated}")
```

## Policies

| Policy        | Routing                                      |
|---------------|----------------------------------------------|
| cost_aware    | simple->local, moderate->fast, complex->reasoning |
| quality_first | simple->fast, moderate/complex->reasoning    |
| latency_first | simple->local, moderate/complex->fast (no escalation) |

## RAG hook

```python
class MyRetriever:
    async def retrieve(self, query: str) -> list[str]:
        return search_your_index(query)

router = Router(rag_hook=MyRetriever())
```

Retrieved chunks are prepended as `[CONTEXT]` when the classifier signals `needs_retrieval`.

## CLI

```bash
alrf dashboard          # live Rich table refreshing every 2s
alrf history --days 7   # cost by route + latency percentiles
```

## Configuration

```python
RouterConfig(
    policy="cost_aware",
    escalation_threshold=0.70,   # escalate when confidence below this
    retry_budget=3,              # max total attempts per request
    local_provider="ollama",     # "ollama" | "lm_studio" | "vllm"
    local_base_url="http://localhost:11434",
    local_model="llama3.2",
    fast_model="gpt-4o-mini",
    reasoning_model="claude-sonnet-4-6",
)
```

Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` in `.env` (see `.env.example`).

## Requirements

Python 3.12+. See `pyproject.toml`.
