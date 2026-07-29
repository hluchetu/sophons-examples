"""Customer support memory: the agent learns customer context.

This example shows the ergonomic Sophons path for a real app:

    memory = MemoryManager(..., namespace=("customer", "C-1042"))
    agent = Agent(model=model, memory_manager=memory)

The first support turn teaches the agent durable customer context. Sophons
extracts and stores that memory after the run. The next turn recalls it and
injects it before the model call.

Run:
    uv run memory/customer_support.py
"""

from __future__ import annotations

import asyncio

from pydantic_settings import BaseSettings, SettingsConfigDict

from sophons.agents import Agent
from sophons.cli import ui
from sophons.integrations.models import DeepSeekModel
from sophons.memory import (
    InMemoryStorage,
    LexicalRetriever,
    LLMMemoryExtractor,
    MemoryManager,
    MemoryStore,
    MemoryStoreConfig,
)
from sophons.tools import tool


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deepseek_api_key: str
    deepseek_model: str = "deepseek-chat"


def load_settings() -> Settings:
    # Pydantic Settings loads required values from the environment at runtime.
    return Settings()  # pyright: ignore[reportCallIssue]


settings = load_settings()

CUSTOMER = ("customer", "C-1042")


@tool
def card_status(card_last4: str) -> str:
    """Look up the status of a customer's card by last four digits."""
    if card_last4 == "4481":
        return "card 4481 is active; last declined transaction was due to daily limit"
    return "no card found with those last four digits"


def build_model() -> DeepSeekModel:
    return DeepSeekModel(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
    )


def build_memory(model: DeepSeekModel) -> MemoryManager:
    store = MemoryStore(
        storage=InMemoryStorage(),
        retrievers=[LexicalRetriever()],
    )
    return MemoryManager(
        stores=[
            MemoryStoreConfig(
                name="main",
                description="customer profile, preferences, and support history",
                store=store,
            )
        ],
        extractor=LLMMemoryExtractor(model=model),
        namespace=CUSTOMER,
        inject_limit=5,
    )


def build_agent(model: DeepSeekModel, memory: MemoryManager) -> Agent:
    return Agent(
        model=model,
        tools=[card_status],
        memory_manager=memory,
        system_prompt=(
            "You are a Luche Bank support agent. Use tools for current account "
            "facts only when the customer asks you to check something now. "
            "When the customer only asks you to remember context, acknowledge "
            "the context and do not call tools. "
            "Use long-term memory for customer context, preferences, and prior "
            "support history. Be concise and give clear next steps."
        ),
    )


async def saved_context(memory: MemoryManager) -> str:
    entries = []
    seen: set[str] = set()
    for query in [
        "communication preference short answers clear next steps",
        "account plan Luche Business",
        "supplier payment card ending 4481",
    ]:
        for entry in await memory.search(query=query, namespace=CUSTOMER, limit=3):
            if entry.id in seen:
                continue
            seen.add(entry.id)
            entries.append(entry)
    return "\n".join(
        f"- [{entry.memory_type}] {entry.key}: {entry.content}" for entry in entries
    )


def main() -> None:
    ui.header("memory/customer_support.py", subtitle="extract, store, recall")

    model = build_model()
    memory = build_memory(model)
    agent = build_agent(model, memory)

    first = (
        "Please remember this for customer C-1042 for future conversations. "
        "Do not check anything yet: my account plan is Luche Business, my "
        "communication preference is short answers with clear next steps, and "
        "my supplier payment on card ending 4481 failed yesterday."
    )
    ui.note("TURN 1 — customer gives durable context")
    ui.user(first)
    result = agent.run_sync(first)
    ui.agent(result.message, footer="Sophons extracts memory after this run")
    ui.note("Memory extracted by Sophons")
    ui.agent(asyncio.run(saved_context(memory)) or "No memory extracted.")

    second = (
        "The supplier payment failed again. Can you check the card ending 4481 "
        "and tell me what to do next?"
    )
    ui.note("TURN 2 — customer asks a follow-up later")
    ui.user(second)

    result = agent.run_sync(second)

    for use, outcome in zip(result.tool_uses, result.tool_results):
        ui.tool(f"{use.name}({use.input}) -> [{outcome.status}] {outcome.content}")

    ui.agent(result.message, footer="customer memory injected automatically by Sophons")


if __name__ == "__main__":
    main()
