from alrf.providers.base import ProviderResponse

_HEDGE_PHRASES = (
    "i'm not sure",
    "i don't know",
    "it depends",
    "i cannot",
    "i can't",
    "not certain",
    "unclear",
    "possibly",
    "it's hard to say",
)


class ConfidenceScorer:
    """Score a provider response 0.0–1.0 based on quality signals."""

    def score(self, response: ProviderResponse, query: str) -> float:
        text = response.text.lower()

        # length ratio: longer answer relative to query is a good sign
        query_len = max(len(query.split()), 1)
        answer_len = len(response.text.split())
        length_signal = min((answer_len / query_len) / 5.0, 1.0)

        # stop reason: truncated responses are less reliable
        stop_signal = 1.0 if response.stop_reason == "stop" else 0.4

        base = (length_signal + stop_signal) / 2.0

        # hedge phrases scale confidence down multiplicatively
        hedge_count = sum(1 for phrase in _HEDGE_PHRASES if phrase in text)
        hedge_factor = max(0.1, 1.0 - hedge_count * 0.2)

        return round(base * hedge_factor, 3)
