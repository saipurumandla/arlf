import pytest
from alrf.classifier.heuristic import HeuristicClassifier
from alrf.classifier.base import QueryComplexity, QueryIntent


@pytest.fixture
def clf() -> HeuristicClassifier:
    return HeuristicClassifier()


# complexity

def test_simple_short_factual(clf: HeuristicClassifier) -> None:
    assert clf.classify("What is Redis?").complexity == QueryComplexity.simple


def test_simple_no_signals(clf: HeuristicClassifier) -> None:
    assert clf.classify("List users").complexity == QueryComplexity.simple


def test_moderate_one_keyword_signal(clf: HeuristicClassifier) -> None:
    assert clf.classify("Why is this slow?").complexity == QueryComplexity.moderate


def test_moderate_medium_token_count(clf: HeuristicClassifier) -> None:
    # needs enough tokens to land in the medium_query band (20–80)
    query = "Can you tell me what the default timeout setting is for an HTTP connection pool and how it interacts with retries when a downstream service is slow to respond?"
    assert clf.classify(query).complexity == QueryComplexity.moderate


def test_complex_error_causal_and_detail(clf: HeuristicClassifier) -> None:
    # 3 signals: has_error_keyword + causal_question + detailed_request → complex
    result = clf.classify("explain step by step why I'm getting a 500 exception in my auth service")
    assert result.complexity == QueryComplexity.complex


def test_complex_three_signals(clf: HeuristicClassifier) -> None:
    query = "explain the difference between async and sync, step by step, and compare vs threads"
    assert clf.classify(query).complexity == QueryComplexity.complex


def test_complex_long_query(clf: HeuristicClassifier) -> None:
    result = clf.classify("word " * 90)
    assert result.complexity == QueryComplexity.complex
    assert "long_query" in result.signals


def test_moderate_with_error_keyword(clf: HeuristicClassifier) -> None:
    assert clf.classify("getting a 500 error").complexity == QueryComplexity.moderate


def test_simple_very_short(clf: HeuristicClassifier) -> None:
    result = clf.classify("ping")
    assert result.complexity == QueryComplexity.simple
    assert "short_query" in result.signals


def test_complex_code_request_with_detail(clf: HeuristicClassifier) -> None:
    query = "implement a retry decorator step by step and explain how it compares vs a simple loop"
    assert clf.classify(query).complexity == QueryComplexity.complex


# intent

def test_intent_debug(clf: HeuristicClassifier) -> None:
    assert clf.classify("getting a 500 error on the payment endpoint").intent == QueryIntent.debug


def test_intent_explanation(clf: HeuristicClassifier) -> None:
    assert clf.classify("how does connection pooling work?").intent == QueryIntent.explanation


def test_intent_code(clf: HeuristicClassifier) -> None:
    assert clf.classify("implement a retry decorator").intent == QueryIntent.code


def test_intent_qa(clf: HeuristicClassifier) -> None:
    assert clf.classify("What is the capital of France?").intent == QueryIntent.qa


# retrieval

def test_needs_retrieval_causal_long_enough(clf: HeuristicClassifier) -> None:
    # needs causal signal AND token_est > 30 — use a genuinely long question
    query = (
        "how does the garbage collector work in CPython and why does it sometimes cause "
        "unexpected latency spikes in long-running async services that hold lots of live object references in memory"
    )
    result = clf.classify(query)
    assert result.needs_retrieval is True


def test_no_retrieval_causal_too_short(clf: HeuristicClassifier) -> None:
    assert clf.classify("why?").needs_retrieval is False


def test_no_retrieval_no_signals(clf: HeuristicClassifier) -> None:
    assert clf.classify("ping").needs_retrieval is False


def test_needs_retrieval_error_keyword_long(clf: HeuristicClassifier) -> None:
    # needs error signal AND token_est > 30
    query = (
        "I keep getting a traceback when calling the database layer under heavy concurrent load "
        "and the exception message says the connection pool was exhausted after waiting 30 seconds to acquire"
    )
    assert clf.classify(query).needs_retrieval is True


# validation

def test_empty_query_raises(clf: HeuristicClassifier) -> None:
    with pytest.raises(ValueError):
        clf.classify("")


def test_whitespace_only_raises(clf: HeuristicClassifier) -> None:
    with pytest.raises(ValueError):
        clf.classify("   ")


def test_too_long_raises(clf: HeuristicClassifier) -> None:
    with pytest.raises(ValueError):
        clf.classify("x" * 2001)


def test_signals_populated(clf: HeuristicClassifier) -> None:
    result = clf.classify("getting an exception in production")
    assert "has_error_keyword" in result.signals


def test_token_estimate_positive(clf: HeuristicClassifier) -> None:
    assert clf.classify("What is the meaning of life?").token_estimate > 0
