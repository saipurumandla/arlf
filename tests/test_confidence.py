from alrf.evaluation.confidence import ConfidenceScorer
from alrf.providers.base import ProviderResponse


def _resp(text: str, stop_reason: str = "stop") -> ProviderResponse:
    return ProviderResponse(
        text=text,
        model="gpt-4o-mini",
        provider="openai",
        stop_reason=stop_reason,
        input_tokens=10,
        output_tokens=len(text.split()),
    )


def test_clean_detailed_answer_scores_high() -> None:
    query = "how does connection pooling work"
    text = (
        "Connection pooling maintains a cache of database connections so applications "
        "can reuse them rather than opening a new connection for every request. "
        "This reduces overhead and improves throughput significantly under load."
    )
    score = ConfidenceScorer().score(_resp(text), query)
    assert score >= 0.7


def test_hedge_heavy_response_scores_low() -> None:
    query = "what causes the 500 error"
    text = "I'm not sure, it depends on the situation. I don't know exactly, it's hard to say."
    score = ConfidenceScorer().score(_resp(text), query)
    assert score < 0.5


def test_truncated_response_penalised() -> None:
    query = "explain async await"
    text = "Async await is a pattern that"
    score_truncated = ConfidenceScorer().score(_resp(text, stop_reason="length"), query)
    score_clean = ConfidenceScorer().score(_resp(text, stop_reason="stop"), query)
    assert score_truncated < score_clean


def test_score_bounded_0_to_1() -> None:
    scorer = ConfidenceScorer()
    for text, stop in [("yes", "stop"), ("i'm not sure i don't know i cannot", "length")]:
        s = scorer.score(_resp(text), "test query")
        assert 0.0 <= s <= 1.0


def test_single_hedge_reduces_score() -> None:
    query = "what is redis"
    no_hedge = ConfidenceScorer().score(_resp("Redis is an in-memory data store."), query)
    with_hedge = ConfidenceScorer().score(_resp("I'm not sure but Redis is an in-memory data store."), query)
    assert with_hedge < no_hedge
