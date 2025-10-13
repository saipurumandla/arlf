from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class QueryComplexity(str, Enum):
    simple   = "simple"
    moderate = "moderate"
    complex  = "complex"


class QueryIntent(str, Enum):
    qa          = "qa"
    debug       = "debug"
    explanation = "explanation"
    code        = "code"


@dataclass(frozen=True)
class ClassifierResult:
    complexity:      QueryComplexity
    intent:          QueryIntent
    token_estimate:  int
    confidence:      float
    needs_retrieval: bool
    signals:         list[str]


class BaseClassifier(ABC):
    @abstractmethod
    def classify(self, query: str) -> ClassifierResult: ...
