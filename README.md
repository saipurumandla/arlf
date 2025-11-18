# alrf — Adaptive LLM Routing Framework

Routes queries to the right LLM provider based on complexity.
Stop paying gpt-4o prices for "what is Redis?" lookups.

## How it works

Classifies each query using token count and keyword signals, then routes:

- Simple/moderate → `gpt-4o-mini` (fast, cheap)
- Complex → `claude-sonnet-4-6` (reasoning)
- Falls back to the other provider if the first times out or returns a 5xx.

Every routing decision is logged with structlog.

## Install

    pip install -e ".[dev]"

Copy `.env.example` to `.env` and fill in your API keys.

## Usage

```python
from alrf import Router

router = Router()
answer = await router.run("Why is my auth endpoint returning 500s under load?")
```

## Requirements

Python 3.12+
See `pyproject.toml` for full dependency list.
