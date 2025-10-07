import structlog

logger = structlog.get_logger()


class Router:
    """Entry point — classify a query and route it to the right provider."""

    async def run(self, query: str) -> str:
        raise NotImplementedError
