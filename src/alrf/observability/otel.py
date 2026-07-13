from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from alrf.models.response import RouterResult

_configured = False


def configure_tracing(endpoint: str | None) -> None:
    global _configured
    if endpoint is None or _configured:
        return

    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    provider = TracerProvider(resource=Resource.create({"service.name": "alrf"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    _configured = True


def get_tracer() -> trace.Tracer:
    return trace.get_tracer("alrf")


def set_result_attributes(span: trace.Span, result: RouterResult) -> None:
    span.set_attribute("route", result.route)
    span.set_attribute("provider", result.provider)
    span.set_attribute("model", result.model)
    span.set_attribute("confidence", result.confidence)
    span.set_attribute("escalated", result.escalated)
    span.set_attribute("cached", result.cached)
    if result.cost_usd is not None:
        span.set_attribute("cost_usd", result.cost_usd)
