from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    model: str
    provider: str
    stop_reason: str    # "stop" | "length" | "error"
    input_tokens: int
    output_tokens: int


class BaseProvider(ABC):
    @abstractmethod
    async def complete(self, prompt: str, model: str) -> ProviderResponse: ...
