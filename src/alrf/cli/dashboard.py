import asyncio
from datetime import datetime
from pathlib import Path

import aiosqlite
from rich.live import Live
from rich.table import Table
from rich.text import Text


async def _fetch_recent(db_path: str, limit: int = 20) -> list[tuple]:
    if not Path(db_path).exists():
        return []
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT created_at, query_hash, route, provider, latency_ms, "
            "cost_usd, confidence, escalated "
            "FROM routing_events ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ) as cur:
            return await cur.fetchall()


def _build_table(rows: list[tuple]) -> Table:
    table = Table(title="ALRF — Live Routing", expand=True)
    table.add_column("Time",       style="dim",    width=10)
    table.add_column("Query",      style="cyan",   width=12)
    table.add_column("Route",      style="green")
    table.add_column("Provider",   style="yellow")
    table.add_column("Latency",    justify="right")
    table.add_column("Cost",       justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Escalated",  justify="center")

    for row in rows:
        created_at, query_hash, route, provider, latency_ms, cost_usd, confidence, escalated = row
        time_str = created_at[11:19] if len(created_at) > 10 else created_at
        cost_str = f"${cost_usd:.4f}" if cost_usd is not None else "free"
        esc_str = Text("YES", style="red bold") if escalated else Text("no", style="dim")
        table.add_row(
            time_str,
            query_hash[:10],
            route,
            provider,
            f"{latency_ms}ms",
            cost_str,
            f"{confidence:.2f}",
            esc_str,
        )
    return table


async def run_dashboard(db_path: str) -> None:
    with Live(refresh_per_second=0.5) as live:
        while True:
            rows = await _fetch_recent(db_path, limit=20)
            live.update(_build_table(rows))
            await asyncio.sleep(2)
