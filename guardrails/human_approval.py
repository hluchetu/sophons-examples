"""An agent that asks a human before acting — approval at the terminal.

The refund agent auto-approves refunds up to 100.00. Anything larger
triggers a confirm decision: the run pauses, the request is printed, and
the person at the terminal decides. Approved -> the tool executes.
Denied -> the model is told who said no (and why) and explains gracefully.

This is the Claude Code pattern: the permission prompt before a risky
action, with a human as the last gate.

Run (and answer the prompt yourself):
    uv run guardrails/human_approval.py
"""

from __future__ import annotations

import asyncio

from pydantic_settings import BaseSettings, SettingsConfigDict

from sophons.agents import Agent
from sophons.guardrails import ApprovalGuardrail, ConsoleApprover
from sophons.integrations.models import DeepSeekModel
from sophons.tools import tool


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deepseek_api_key: str
    deepseek_model: str = "deepseek-chat"


settings = Settings()


@tool
def refund_order(order_id: str, amount: float) -> str:
    """Refund the given amount for an order."""
    return f"refunded {amount} for {order_id}"


AUTO_APPROVE_LIMIT = 100.0

agent_kwargs = dict(
    system_prompt="You process refunds for orders. Use the refund_order tool.",
    guardrails=[
        ApprovalGuardrail(
            rules={
                "refund_order": lambda args: (
                    f"refund of {args.get('amount')} exceeds the "
                    f"auto-approve limit of {AUTO_APPROVE_LIMIT}"
                    if args.get("amount", 0) > AUTO_APPROVE_LIMIT
                    else None
                )
            },
        )
    ],
    approver=ConsoleApprover(),
)


async def main() -> None:
    model = DeepSeekModel(
        model=settings.deepseek_model, api_key=settings.deepseek_api_key
    )
    agent = Agent(model=model, tools=[refund_order], **agent_kwargs)

    for request in [
        "Refund 30.00 for order ord_7.",          # under the limit — no prompt
        "Please refund 250.00 for order ord_42.",  # over — the terminal asks YOU
    ]:
        print(f"\nuser:  {request}")
        result = await agent.run(request)
        print(f"agent: {result.message}")


if __name__ == "__main__":
    asyncio.run(main())
