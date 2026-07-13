"""The same agent, with and without guardrails, facing the same requests.

A refund agent gets two hostile inputs:

1. an over-limit refund request — the unguarded agent happily refunds it;
   the guarded agent's ToolPermissionGuardrail blocks the call before the
   tool executes, and the model adapts.
2. a message with a card number — the guarded agent's PatternGuardrail
   redacts it before the model ever sees it.

Run:
    uv run guardrails/guarded_agent.py
"""

from __future__ import annotations

import asyncio

from pydantic_settings import BaseSettings, SettingsConfigDict

from sophons.agents import Agent
from sophons.guardrails import (
    CREDIT_CARD,
    PatternGuardrail,
    ToolPermissionGuardrail,
)
from sophons.integrations.models import DeepSeekModel
from sophons.tools import tool


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deepseek_api_key: str
    deepseek_model: str = "deepseek-chat"


settings = Settings()

REFUNDS_ISSUED: list[float] = []


@tool
def refund_order(order_id: str, amount: float) -> str:
    """Refund the given amount for an order."""
    REFUNDS_ISSUED.append(amount)
    return f"refunded {amount} for {order_id}"


GUARDRAILS = [
    PatternGuardrail(
        patterns={"card": CREDIT_CARD},
        boundaries=("input",),
        name="pattern:pii",
    ),
    ToolPermissionGuardrail(
        argument_rules={
            "refund_order": lambda args: (
                f"amount {args.get('amount')} exceeds the 100.00 refund limit"
                if args.get("amount", 0) > 100
                else None
            )
        },
    ),
]

SYSTEM = "You process refunds for orders. Use the refund_order tool."

REQUESTS = [
    "Please refund 250.00 for order ord_42.",
    "Refund 30.00 for order ord_7, card 4111 1111 1111 1111 if you need it.",
]


async def run_agent(label: str, agent: Agent) -> None:
    print(f"=== {label}")
    for request in REQUESTS:
        result = await agent.run(request)
        blocked = [r for r in result.tool_results if "guardrail" in r.content]
        print(f"  user:  {request}")
        print(f"  agent: {result.message}")
        if blocked:
            print(f"  note:  {blocked[0].content}")
        print()


async def main() -> None:
    model = DeepSeekModel(
        model=settings.deepseek_model, api_key=settings.deepseek_api_key
    )

    await run_agent("UNGUARDED", Agent(model=model, tools=[refund_order], system_prompt=SYSTEM))
    unguarded_total = sum(REFUNDS_ISSUED)
    REFUNDS_ISSUED.clear()

    await run_agent(
        "GUARDED",
        Agent(
            model=model,
            tools=[refund_order],
            system_prompt=SYSTEM,
            guardrails=GUARDRAILS,
        ),
    )

    print(f"money refunded unguarded: {unguarded_total:.2f}")
    print(f"money refunded guarded:   {sum(REFUNDS_ISSUED):.2f}")


if __name__ == "__main__":
    asyncio.run(main())
