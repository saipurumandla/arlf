import pytest
from alrf.classifier.heuristic import HeuristicClassifier
from alrf.classifier.base import QueryComplexity, QueryIntent


@pytest.fixture
def clf() -> HeuristicClassifier:
    return HeuristicClassifier()


# complexity

async def test_simple_short_factual(clf: HeuristicClassifier) -> None:
    assert (await clf.classify("What is Redis?")).complexity == QueryComplexity.simple


async def test_simple_no_signals(clf: HeuristicClassifier) -> None:
    assert (await clf.classify("List users")).complexity == QueryComplexity.simple


async def test_moderate_one_keyword_signal(clf: HeuristicClassifier) -> None:
    assert (await clf.classify("Why is this slow?")).complexity == QueryComplexity.moderate


async def test_moderate_medium_token_count(clf: HeuristicClassifier) -> None:
    query = "Can you tell me what the default timeout setting is for an HTTP connection pool and how it interacts with retries when a downstream service is slow to respond?"
    assert (await clf.classify(query)).complexity == QueryComplexity.moderate


async def test_complex_error_causal_and_detail(clf: HeuristicClassifier) -> None:
    result = await clf.classify("explain step by step why I'm getting a 500 exception in my auth service")
    assert result.complexity == QueryComplexity.complex


async def test_complex_three_signals(clf: HeuristicClassifier) -> None:
    query = "explain the difference between async and sync, step by step, and compare vs threads"
    assert (await clf.classify(query)).complexity == QueryComplexity.complex


async def test_complex_long_query(clf: HeuristicClassifier) -> None:
    result = await clf.classify("word " * 90)
    assert result.complexity == QueryComplexity.complex
    assert "long_query" in result.signals


async def test_moderate_with_error_keyword(clf: HeuristicClassifier) -> None:
    assert (await clf.classify("getting a 500 error")).complexity == QueryComplexity.moderate


async def test_simple_very_short(clf: HeuristicClassifier) -> None:
    result = await clf.classify("ping")
    assert result.complexity == QueryComplexity.simple
    assert "short_query" in result.signals


async def test_complex_code_request_with_detail(clf: HeuristicClassifier) -> None:
    query = "implement a retry decorator step by step and explain how it compares vs a simple loop"
    assert (await clf.classify(query)).complexity == QueryComplexity.complex


# intent

async def test_intent_debug(clf: HeuristicClassifier) -> None:
    assert (await clf.classify("getting a 500 error on the payment endpoint")).intent == QueryIntent.debug


async def test_intent_explanation(clf: HeuristicClassifier) -> None:
    assert (await clf.classify("how does connection pooling work?")).intent == QueryIntent.explanation


async def test_intent_code(clf: HeuristicClassifier) -> None:
    assert (await clf.classify("implement a retry decorator")).intent == QueryIntent.code


async def test_intent_qa(clf: HeuristicClassifier) -> None:
    assert (await clf.classify("What is the capital of France?")).intent == QueryIntent.qa


# retrieval

async def test_needs_retrieval_causal_long_enough(clf: HeuristicClassifier) -> None:
    query = (
        "how does the garbage collector work in CPython and why does it sometimes cause "
        "unexpected latency spikes in long-running async services that hold lots of live object references in memory"
    )
    result = await clf.classify(query)
    assert result.needs_retrieval is True


async def test_no_retrieval_causal_too_short(clf: HeuristicClassifier) -> None:
    assert (await clf.classify("why?")).needs_retrieval is False


async def test_no_retrieval_no_signals(clf: HeuristicClassifier) -> None:
    assert (await clf.classify("ping")).needs_retrieval is False


async def test_needs_retrieval_error_keyword_long(clf: HeuristicClassifier) -> None:
    query = (
        "I keep getting a traceback when calling the database layer under heavy concurrent load "
        "and the exception message says the connection pool was exhausted after waiting 30 seconds to acquire"
    )
    assert (await clf.classify(query)).needs_retrieval is True


# validation

async def test_empty_query_raises(clf: HeuristicClassifier) -> None:
    with pytest.raises(ValueError):
        await clf.classify("")


async def test_whitespace_only_raises(clf: HeuristicClassifier) -> None:
    with pytest.raises(ValueError):
        await clf.classify("   ")


async def test_too_long_raises(clf: HeuristicClassifier) -> None:
    with pytest.raises(ValueError):
        await clf.classify("x" * 2001)


async def test_signals_populated(clf: HeuristicClassifier) -> None:
    result = await clf.classify("getting an exception in production")
    assert "has_error_keyword" in result.signals


async def test_token_estimate_positive(clf: HeuristicClassifier) -> None:
    assert (await clf.classify("What is the meaning of life?")).token_estimate > 0
