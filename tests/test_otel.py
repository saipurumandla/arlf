import pytest
import respx
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from alrf import Router, RouterConfig

_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)


@pytest.fixture(autouse=True)
def _clear_spans() -> None:
    _exporter.clear()


def _span_names() -> list[str]:
    return [span.name for span in _exporter.get_finished_spans()]


async def test_run_emits_span_per_stage(openai_mock: respx.MockRouter) -> None:
    config = RouterConfig(policy="quality_first", escalation_threshold=0.0)
    await Router(config=config).run("What is Redis?")

    assert _span_names() == ["classify", "policy", "provider.call", "router.run"]


async def test_run_span_carries_result_attributes(openai_mock: respx.MockRouter) -> None:
    config = RouterConfig(policy="quality_first", escalation_threshold=0.0)
    result = await Router(config=config).run("What is Redis?")

    span = next(s for s in _exporter.get_finished_spans() if s.name == "router.run")
    attrs = dict(span.attributes or {})
    assert attrs["route"] == result.route
    assert attrs["provider"] == "openai"
    assert attrs["model"] == result.model
    assert attrs["escalated"] is False
    assert attrs["cached"] is False


async def test_escalation_emits_span(openai_mock: respx.MockRouter, anthropic_mock: respx.MockRouter) -> None:
    config = RouterConfig(policy="quality_first", escalation_threshold=1.0, retry_budget=2)
    await Router(config=config).run("What is Redis?")

    assert "escalation" in _span_names()


async def test_cache_hit_marks_span_cached(openai_mock: respx.MockRouter) -> None:
    config = RouterConfig(policy="quality_first", escalation_threshold=0.0)
    router = Router(config=config)
    await router.run("What is Redis?")
    _exporter.clear()
    await router.run("What is Redis?")

    span = next(s for s in _exporter.get_finished_spans() if s.name == "router.run")
    assert dict(span.attributes or {})["cached"] is True
    assert "classify" not in _span_names()
