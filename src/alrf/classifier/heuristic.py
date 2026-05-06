import tiktoken
from typing import override

from alrf.classifier.base import BaseClassifier, ClassifierResult, QueryComplexity, QueryIntent

_enc = tiktoken.get_encoding("cl100k_base")

_ERROR_KW   = ("error", "exception", "traceback", "failed", "crash", "500")
_CAUSAL_KW  = ("why", "how does", "explain", "what causes")
_COMPARE_KW = ("compare", "difference", "tradeoff", "vs", "versus")
_CODE_KW    = ("implement", "write code", "create a function", "fix this")
_DETAIL_KW  = ("step by step", "walk me through", "in detail")


class HeuristicClassifier(BaseClassifier):

    @override
    async def classify(self, query: str) -> ClassifierResult:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if len(query) > 2000:
            raise ValueError("query exceeds 2000 characters")

        q = query.lower()
        token_est = len(_enc.encode(query))
        signals: list[str] = []

        if token_est < 20:
            signals.append("short_query")
        elif token_est <= 80:
            signals.append("medium_query")
        else:
            signals.append("long_query")

        if any(kw in q for kw in _ERROR_KW):
            signals.append("has_error_keyword")
        if any(kw in q for kw in _CAUSAL_KW):
            signals.append("causal_question")
        if any(kw in q for kw in _COMPARE_KW):
            signals.append("comparative_question")
        if any(kw in q for kw in _CODE_KW):
            signals.append("code_request")
        if any(kw in q for kw in _DETAIL_KW):
            signals.append("detailed_request")

        complexity_signals = [s for s in signals if s not in ("short_query", "medium_query", "long_query")]

        if "long_query" in signals or len(complexity_signals) >= 3:
            complexity = QueryComplexity.complex
        elif "short_query" in signals and len(complexity_signals) == 0:
            complexity = QueryComplexity.simple
        else:
            complexity = QueryComplexity.moderate

        if "code_request" in signals:
            intent = QueryIntent.code
        elif "has_error_keyword" in signals:
            intent = QueryIntent.debug
        elif "causal_question" in signals:
            intent = QueryIntent.explanation
        else:
            intent = QueryIntent.qa

        needs_retrieval = (
            ("causal_question" in signals or "has_error_keyword" in signals)
            and token_est > 30
        )

        confidence = 0.9 if complexity == QueryComplexity.simple else 0.75

        return ClassifierResult(
            complexity=complexity,
            intent=intent,
            token_estimate=token_est,
            confidence=confidence,
            needs_retrieval=needs_retrieval,
            signals=signals,
        )
