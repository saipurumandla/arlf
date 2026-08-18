import json
from pathlib import Path
from unittest.mock import patch

import pytest

from alrf import Router, RouterConfig
from alrf.eval.harness import RoutingEvaluator, load_cases
from alrf.providers.base import ProviderResponse

CASE_PATH = Path(__file__).resolve().parents[1] / "eval" / "routing_set.jsonl"


@pytest.fixture
def evaluator() -> RoutingEvaluator:
    return RoutingEvaluator(case_path=CASE_PATH)


async def _hedge(prompt: str, model: str) -> ProviderResponse:
    return ProviderResponse(
        text="I'm not sure, it depends, I don't know.",
        model="gpt-4o-mini",
        provider="openai",
        stop_reason="stop",
        input_tokens=10,
        output_tokens=9,
    )


def test_case_set_is_fully_labelled() -> None:
    cases = load_cases(CASE_PATH)
    assert len(cases) == 20
    assert all(case.expected_tier in ("local", "fast", "reasoning") for case in cases)
    assert all(case.note for case in cases)


async def test_report_scores_tier_accuracy(evaluator: RoutingEvaluator) -> None:
    report = await evaluator.run()
    assert report["cases"] == 20
    assert isinstance(report["tier_accuracy"], float)
    assert 0.0 <= report["tier_accuracy"] <= 1.0


async def test_cost_delta_is_negative_against_always_reasoning(evaluator: RoutingEvaluator) -> None:
    report = await evaluator.run()
    assert report["cost_usd_routed"] < report["cost_usd_always_reasoning"]
    assert report["cost_delta_pct"] < 0


async def test_escalation_metrics_need_a_router(evaluator: RoutingEvaluator) -> None:
    report = await evaluator.run()
    assert report["escalation_precision"] is None
    assert report["escalation_recall"] is None
    assert report["cache_hit_rate"] is None


async def test_live_run_scores_escalation_and_cache(evaluator: RoutingEvaluator) -> None:
    config = RouterConfig(policy="quality_first", escalation_threshold=0.99, retry_budget=2)
    with patch("alrf.router.OpenAIProvider") as MockOpenAI, \
         patch("alrf.router.AnthropicProvider") as MockAnthropic:
        MockOpenAI.return_value.complete = _hedge
        MockAnthropic.return_value.complete = _hedge
        report = await evaluator.run(router=Router(config=config))

    assert 0.0 <= report["escalation_precision"] <= 1.0
    assert 0.0 <= report["escalation_recall"] <= 1.0
    assert 0.0 <= report["cache_hit_rate"] <= 1.0


async def test_report_is_saved_as_json(evaluator: RoutingEvaluator, tmp_path: Path) -> None:
    report = await evaluator.run()
    out = tmp_path / "eval" / "routing_eval.json"
    evaluator.save(report, out)

    assert json.loads(out.read_text())["cases"] == 20
