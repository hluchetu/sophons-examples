"""Sweeping an agent over a test set with EvalRunner — and measuring pass^k.

A small support agent answers order-status questions with a lookup tool.
The runner executes every case three times, judges each trial with three
evaluators (trajectory, tool parameters, output), and reports two rates:

- pass rate  — fraction of individual trials that passed (the optimistic number)
- pass^k     — fraction of cases that passed *every* trial (the honest number)

pass^k is Sierra's tau-bench metric: agents are non-deterministic, and
passing once is weak evidence. Users do not experience your agent once.

Run:
    uv run evaluation/pass_k.py
"""

from __future__ import annotations

import asyncio

from pydantic_settings import BaseSettings, SettingsConfigDict

from sophons.agents import Agent
from sophons.evals import (
    EvalCase,
    EvalDataset,
    EvalRunner,
    OutputEvaluator,
    ToolParameterEvaluator,
    TrajectoryEvaluator,
    render_report,
)
from sophons.integrations.models import DeepSeekModel
from sophons.tools import tool


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deepseek_api_key: str
    deepseek_model: str = "deepseek-chat"


settings = Settings()

ORDERS = {
    "ord_123": "in transit, expected to arrive Thursday",
    "ord_777": "delivered on Monday and signed for at the front desk",
}


@tool
def lookup_order(order_id: str) -> str:
    """Look up the shipping status of an order by its order id."""
    return ORDERS.get(order_id, "no order found with that id")


DATASET = EvalDataset(
    name="order-status",
    version="v1",
    cases=[
        EvalCase(
            id="in-transit",
            question="Where is order ord_123 right now?",
            reference="Order ord_123 is in transit and expected to arrive Thursday.",
            expected_tools=["lookup_order"],
            expected_tool_calls=[
                {"name": "lookup_order", "args": {"order_id": "ord_123"}}
            ],
        ),
        EvalCase(
            id="delivered",
            question="Has order ord_777 arrived yet?",
            reference="Yes — order ord_777 was delivered on Monday and signed for at the front desk.",
            expected_tools=["lookup_order"],
            expected_tool_calls=[
                {"name": "lookup_order", "args": {"order_id": "ord_777"}}
            ],
        ),
    ],
)


async def main() -> None:
    model = DeepSeekModel(
        model=settings.deepseek_model, api_key=settings.deepseek_api_key
    )
    agent = Agent(
        model=model,
        tools=[lookup_order],
        system_prompt=(
            "You answer order-status questions. Always use the lookup_order "
            "tool; never guess a status."
        ),
    )

    runner = EvalRunner(
        agent,
        evaluators=[
            TrajectoryEvaluator(mode="exact"),
            ToolParameterEvaluator(),
            OutputEvaluator(model),
        ],
        num_trials=3,
    )

    run = await runner.run(DATASET)
    print(render_report(run))


if __name__ == "__main__":
    asyncio.run(main())
