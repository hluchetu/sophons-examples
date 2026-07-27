"""Structured output: a record instead of a sentence.

The agent triages a support message. Rather than prose it must return a
Ticket — category, urgency, summary — validated by Pydantic before the run
is allowed to end.

Run:
    uv run structured_output/basic.py
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from sophons.agents import Agent
from sophons.cli import ui
from sophons.integrations.models import DeepSeekModel


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deepseek_api_key: str
    deepseek_model: str = "deepseek-chat"


def load_settings() -> Settings:
    # Pydantic Settings loads required values from the environment at runtime.
    return Settings()  # pyright: ignore[reportCallIssue]


settings = load_settings()


class Ticket(BaseModel):
    """A triaged Luche Bank support ticket."""

    category: str = Field(description="one of: cards, transfers, app, loans, other")
    urgency: int = Field(description="1 (routine) to 5 (drop everything)", ge=1, le=5)
    summary: str = Field(description="one sentence, third person, no greeting")


MESSAGES = [
    "The ATM at Westlands swallowed my card and I fly out tomorrow!",
    "hi, just wondering what the fee code FEE-WDR-021 is about",
]


def main() -> None:
    agent = Agent(
        model=DeepSeekModel(
            model=settings.deepseek_model, api_key=settings.deepseek_api_key
        ),
        output_type=Ticket,
        system_prompt="You triage incoming Luche Bank support messages.",
    )

    ui.header("structured_output/basic.py", subtitle="prose in · Ticket out")

    for message in MESSAGES:
        ui.user(message)

        result = agent.run_sync(message)
        ticket = result.output

        # A real Ticket instance, not a string that happens to look like one.
        ui.tool(
            f"{type(ticket).__name__}("
            f"category={ticket.category!r}, urgency={ticket.urgency})"
        )
        ui.agent(
            ticket.summary,
            footer=(
                f"{ticket.category} · urgency {ticket.urgency}/5 · "
                f"steps={result.metrics.steps}"
            ),
        )


if __name__ == "__main__":
    main()
