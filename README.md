# alrf — Adaptive LLM Routing Framework

Routes queries to the right LLM provider based on complexity.
Stop paying gpt-4o prices for "what is Redis?" lookups.

## Install

    pip install -e ".[dev]"

## Usage

```python
from alrf import Router

router = Router()
result = await router.run("Why is my login endpoint returning 500?")
```

## Requirements

Python 3.12+
See pyproject.toml for dependencies.
