"""Long-term memory: learn once, recall on a later turn.

This example is intentionally local and deterministic. No API key is needed.
The fake model only reports whether long-term memory reached its prompt, while
the rule-based extractor turns "Remember: ..." user messages into durable
memory entries.

Run:
    uv run memory/agent_memory.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from sophons.agents import Agent
from sophons.cli import ui
from sophons.memory import (
    InMemoryStorage,
    MemoryEntry,
    MemoryExtractionRequest,
    MemoryExtractionResult,
    MemoryManager,
    MemoryStore,
    MemoryStoreConfig,
)
from sophons.models import Message


NAMESPACE = ("user", "alice")


class InspectingModel:
    """A tiny model that makes memory injection visible."""

    def invoke(self, messages: list[Message], tools=None) -> Message:
        latest = messages[-1].content
        marker = "Relevant long-term memory:\n"
        if marker not in latest:
            return Message(
                role="assistant",
                content="No long-term memory was injected yet.",
            )

        memory_block = latest.split(marker, 1)[1].split("\n\nCurrent user message:", 1)[0]
        return Message(
            role="assistant",
            content=f"I received long-term memory:\n{memory_block}",
        )


@dataclass
class RememberExtractor:
    """Extracts durable memories from user messages that start with Remember."""

    async def extract(
        self,
        request: MemoryExtractionRequest,
    ) -> MemoryExtractionResult:
        entries: list[MemoryEntry] = []
        existing_keys = {entry.key for entry in request.existing_memories}

        for message in request.messages:
            if message.role != "user":
                continue
            content = message.content.strip()
            prefix = "Remember:"
            if not content.lower().startswith(prefix.lower()):
                continue

            remembered = content[len(prefix):].strip()
            if not remembered:
                continue

            key = "user.preference"
            if key in existing_keys:
                continue

            entries.append(
                MemoryEntry(
                    memory_type="preference",
                    namespace=request.namespace,
                    key=key,
                    content=remembered,
                    importance=1.0,
                )
            )
            existing_keys.add(key)

        return MemoryExtractionResult(entries=entries)


def build_memory() -> MemoryManager:
    store = MemoryStore(storage=InMemoryStorage())
    return MemoryManager(
        stores=[
            MemoryStoreConfig(
                name="main",
                description="local learner memory",
                store=store,
            )
        ],
        extractor=RememberExtractor(),
        namespace=NAMESPACE,
    )


def print_memories(memory: MemoryManager) -> None:
    entries = asyncio.run(
        memory.search(
            query="preference",
            namespace=NAMESPACE,
            limit=10,
        )
    )
    if not entries:
        ui.note("Stored memories: none")
        return

    ui.note("Stored memories")
    for entry in entries:
        ui.agent(f"- [{entry.memory_type}] {entry.key}: {entry.content}")


def main() -> None:
    ui.header("memory/agent_memory.py", subtitle="learn once, recall later")

    memory = build_memory()
    agent = Agent(model=InspectingModel(), memory_manager=memory)

    first = (
        "Remember: I prefer first-principles explanations with concrete code "
        "examples."
    )
    ui.note("TURN 1 — teach a preference")
    ui.user(first)
    result = agent.run_sync(first)
    ui.agent(result.message)
    print_memories(memory)

    second = "Explain how agent memory works."
    ui.note("TURN 2 — ask a later question")
    ui.user(second)
    result = agent.run_sync(second)
    ui.agent(result.message)


if __name__ == "__main__":
    main()
