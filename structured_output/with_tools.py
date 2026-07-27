"""Structured output with a tool in the way.

Triage needs a fact the model cannot know: whether the customer is on a
premium plan. So the run has two beats — call account_tier, then fill in
the Ticket — and both travel over the same tool-calling channel.

Run:
    uv run structured_output/with_tools.py
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from sophons.agents import Agent
from sophons.cli import ui
from sophons.integrations.models import DeepSeekModel
from sophons.tools import tool


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deepseek_api_key: str
    deepseek_model: str = "deepseek-chat"


def load_settings() -> Settings:
    # Pydantic Settings loads required values from the environment at runtime.
    return Settings()  # pyright: ignore[reportCallIssue]


settings = load_settings()

TIERS = {
    "0114455": "premium",
    "0119912": "standard",
}


@tool
def account_tier(account_number: str) -> str:
    """Look up whether a Luche Bank account is premium or standard."""
    return TIERS.get(account_number, "unknown")


class Ticket(BaseModel):
    """A triaged Luche Bank support ticket."""

    account_number: str
    tier: str = Field(description="premium or standard, from the account_tier tool")
    category: str = Field(description="one of: cards, transfers, app, loans, other")
    urgency: int = Field(
        description="1 to 5; premium customers start one level higher", ge=1, le=5
    )
    summary: str = Field(description="one sentence, third person, no greeting")


MESSAGE = (
    "Account 0114455 here — the ATM at Westlands swallowed my card "
    "and I fly out tomorrow!"
)


def main() -> None:
    agent = Agent(
        model=DeepSeekModel(
            model=settings.deepseek_model, api_key=settings.deepseek_api_key
        ),
        tools=[account_tier],
        output_type=Ticket,
        system_prompt=(
            "You triage Luche Bank support messages. Always look up the "
            "account tier before deciding urgency; never guess it."
        ),
    )

    ui.header("structured_output/with_tools.py", subtitle="look it up · then fill the shape")
    ui.user(MESSAGE)

    result = agent.run_sync(MESSAGE)

    # Both beats of the run, in order. The last one is structured_response:
    # not a tool that did work, but the answer taking its shape.
    for use in result.tool_uses:
        ui.tool(f"{use.name}({use.input})")

    ticket = result.output
    ui.agent(
        ticket.summary,
        footer=(
            f"{ticket.account_number} · {ticket.tier} · {ticket.category} · "
            f"urgency {ticket.urgency}/5 · steps={result.metrics.steps}"
        ),
    )


if __name__ == "__main__":
    main()
