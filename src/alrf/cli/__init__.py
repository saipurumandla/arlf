import click

from alrf.cli.dashboard import run_dashboard
from alrf.observability.history import QueryHistory


@click.group()
def app() -> None:
    """ALRF — Adaptive LLM Routing Framework CLI."""


@app.command()
@click.option("--db", default=".alrf/routing.db", show_default=True)
def dashboard(db: str) -> None:
    """Live routing dashboard (refreshes every 2s)."""
    import asyncio
    asyncio.run(run_dashboard(db))


@app.command()
@click.option("--db", default=".alrf/routing.db", show_default=True)
@click.option("--days", default=7, show_default=True)
def history(db: str, days: int) -> None:
    """Show cost and latency summary."""
    import asyncio
    asyncio.run(_print_history(db, days))


async def _print_history(db: str, days: int) -> None:
    h = QueryHistory(db)
    costs = await h.cost_by_route(last_days=days)
    rates = await h.latency_percentiles()
    click.echo(f"\nCost by route (last {days}d):")
    for route, cost in costs.items():
        click.echo(f"  {route:<20} ${cost:.4f}")
    click.echo("\nLatency (p50 / p95) by provider:")
    for provider, stats in rates.items():
        click.echo(f"  {provider:<20} {stats['p50']:.0f}ms / {stats['p95']:.0f}ms")
