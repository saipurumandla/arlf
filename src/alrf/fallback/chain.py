import httpx

from alrf.providers.base import BaseProvider, ProviderResponse

type ProviderSlot = tuple[BaseProvider, str]  # (provider_instance, model_name)


class FallbackChain:
    def __init__(self, slots: list[ProviderSlot]) -> None:
        if not slots:
            raise ValueError("FallbackChain requires at least one slot")
        self._slots = slots

    @property
    def primary(self) -> ProviderSlot:
        return self._slots[0]

    async def run(self, prompt: str) -> ProviderResponse:
        failures: list[Exception] = []
        for provider, model in self._slots:
            try:
                return await provider.complete(prompt, model)
            except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                failures.append(exc)
        raise failures[-1]
