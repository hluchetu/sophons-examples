"""Context window: history grows, the window does not.

session.py gave the agent a memory with no ceiling — every turn is replayed
on every later call, so each request is larger than the last until it exceeds
the model's context window.

A conversation manager decides what the model sees. There are three, because
there are only three answers to "what do we keep":

    NullConversationManager   everything
    SlidingWindowManager      the recent part
    SummarizingManager        the recent part, plus a summary of the rest

The same five-turn conversation runs through each. Input tokens are reported
by the provider, so the difference is measured rather than claimed.

Run:
    uv run context/context_window.py
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from sophons.agents import (
    Agent,
    ConversationManager,
    NullConversationManager,
    SlidingWindowManager,
    SummarizingManager,
)
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

BRANCHES = {
    "westlands": "open until 17:00 on weekdays, 12:00 on Saturdays",
    "kilimani": "closed for refurbishment until March",
    "yaya": "open until 18:00 on weekdays, closed Sundays",
}


@tool
def branch_hours(branch: str) -> str:
    """Look up the opening hours of a Luche Bank branch by name."""
    return BRANCHES.get(branch.lower(), "no branch by that name")


SYSTEM = "You answer questions about Luche Bank branches. Use branch_hours."

CONVERSATION = [
    "What are the Westlands branch hours?",
    "And Kilimani?",
    "What about Yaya?",
    "Which of those is open latest on a weekday?",
    "So where should I go on a Saturday afternoon?",
]


def converse(label: str, manager: ConversationManager) -> list[int]:
    """Hold the same conversation, returning input tokens used per turn."""
    agent = Agent(
        model=DeepSeekModel(
            model=settings.deepseek_model, api_key=settings.deepseek_api_key
        ),
        tools=[branch_hours],
        system_prompt=SYSTEM,
        conversation_manager=manager,
    )

    used:list[int] = []
    for turn in CONVERSATION:
        result = agent.run_sync(turn, session_id=label)
        used.append(result.metrics.input_tokens)

    ui.note(label)
    ui.tool("input tokens per turn: " + "  ".join(f"{n:>5}" for n in used))
    return used


def main() -> None:
    ui.header("context/context_window.py", subtitle="five turns, three strategies")

    unmanaged = converse("no manager", NullConversationManager())
    windowed = converse("sliding window", SlidingWindowManager(max_messages=6))
    summarized = converse(
        "summarizing",
        SummarizingManager(
            model=DeepSeekModel(
                model=settings.deepseek_model, api_key=settings.deepseek_api_key
            ),
            keep_recent_messages=4,
            trigger_message_count=8,
        ),
    )

    ui.agent(
        "Unmanaged history grows every turn and never stops. A window holds "
        "steady by forgetting. Summarizing holds steady by compressing, at "
        "the cost of one extra model call each time it fires.",
        footer=(
            f"final turn — none: {unmanaged[-1]} · "
            f"window: {windowed[-1]} · "
            f"summarizing: {summarized[-1]} tokens"
        ),
    )


if __name__ == "__main__":
    main()
