from typing import Protocol, runtime_checkable


@runtime_checkable
class RAGHook(Protocol):
    async def retrieve(self, query: str) -> list[str]: ...
