# alrf — Adaptive LLM Routing Framework

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)

Routes queries to the right LLM provider based on complexity, confidence, and cost policy.
Stop paying gpt-4o prices for "what is Redis?" lookups.

## Architecture

```
Query
  |
  v
SemanticCache            embedding similarity >= threshold -> stored answer, zero cost
  |  (miss)
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
  |                      + OpenTelemetry spans when otel_endpoint is set
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

## Classifiers

By default ALRF uses a fast keyword heuristic to classify queries. Switch to the LLM classifier
for better accuracy — it sends the query to a local Ollama model before routing:

```python
RouterConfig(
    classifier="llm",
    llm_classifier_model="llama3.2",   # any model in your Ollama library
    llm_classifier_url="http://localhost:11434",
)
```

The LLM classifier understands the *meaning* of the question, not just its surface keywords.
If Ollama is unavailable it falls back to the heuristic automatically.

| Classifier  | Latency overhead | Accuracy | Requires |
|-------------|-----------------|----------|----------|
| heuristic   | ~0ms            | Good for obvious cases, fragile on edge cases | nothing |
| llm         | ~200-500ms      | Understands intent, handles ambiguous phrasing | Ollama running locally |

## Eval results

Routing accuracy is measured, not asserted. `eval/routing_set.jsonl` holds 20 labelled
queries with the tier a human would have picked; `RoutingEvaluator` classifies and routes
each one and scores the result.

| Metric                          | Value                       |
|---------------------------------|-----------------------------|
| Cases                           | 20                          |
| Tier accuracy                   | 55%                         |
| Cost (routed)                   | $0.0072                     |
| Cost (always reasoning)         | $0.18                       |
| Cost delta                      | -96%                        |

Run: heuristic classifier, `cost_aware` policy. Full report in `reports/eval/routing_eval.json`.

The cost number is the easy half. The accuracy number is the interesting one: every miss is a
query that should have gone to `reasoning` and went to `fast` instead. Token count is doing most
of the work in the heuristic, and real questions are short — "why is this test flaky?" is seven
tokens and needs the strongest model available. The LLM classifier exists for exactly this case.

```python
from alrf.eval.harness import RoutingEvaluator

evaluator = RoutingEvaluator()
report = await evaluator.run()          # pass router=... to also score escalation and cache
evaluator.save(report, Path("reports/eval/routing_eval.json"))
```

## Semantic cache

Repeated questions cost nothing. The cache embeds the query and compares it against stored
queries by cosine similarity — a hit above `cache_threshold` returns the previous answer with
`cached=True` and `cost_usd=0.0`, and skips classification and routing entirely.

Entries expire after `cache_ttl_seconds` and the oldest are evicted at `cache_max_entries`.
Hit rate is tracked per route.

```python
router = Router(config=RouterConfig(cache_threshold=0.95, cache_ttl_seconds=3600))
await router.run("What is Redis?")     # routed
await router.run("what is redis")      # cached
```

## Streaming

```python
async for chunk in router.stream("Explain write-ahead logging"):
    print(chunk, end="")

print(router.last_result.confidence)
```

The tier is chosen before the first token and stays fixed — confidence is scored on the
completed buffer, so a weak streamed answer is logged as `low_confidence_stream`, never retried.

## Tracing

Set `otel_endpoint` and every `Router.run()` emits a span with child spans for classify, policy,
the provider call, and escalation. Leave it unset and spans go nowhere.

```python
RouterConfig(otel_endpoint="http://localhost:4317")
```

The SQLite store stays either way — it is what the CLI dashboard reads.

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
    cache_enabled=True,
    cache_threshold=0.95,        # cosine similarity required for a hit
    cache_ttl_seconds=3600,
    cache_max_entries=1000,
    otel_endpoint=None,          # e.g. "http://localhost:4317"
)
```

Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` in `.env` (see `.env.example`).

## Requirements

Python 3.12+. See `pyproject.toml`.
