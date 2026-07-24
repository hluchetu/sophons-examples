from __future__ import annotations

from sophons.agents import Agent, AgentResult
from sophons.cli import ui

from .agent import build_agent
from .indexing import build_retriever
from .settings import Settings
from .tools import build_policy_search_tool


def tool_names(result: AgentResult) -> list[str]:
    return [tool_use.name for tool_use in result.tool_uses]


def run_question(agent: Agent, session_id: str, label: str, question: str) -> None:
    ui.note(label)
    ui.user(question)

    result = agent.run_sync(question, session_id=session_id)

    calls = tool_names(result)
    if calls:
        ui.tool("tool calls: " + " · ".join(calls))
    else:
        ui.tool("tool calls: none")

    ui.agent(result.message)


def main() -> None:
    ui.header(
        "agentic_RAG",
        subtitle="retrieval as a tool the agent can choose",
    )

    settings = Settings()  # pyright: ignore[reportCallIssue]
    retriever = build_retriever()
    search_tool = build_policy_search_tool(retriever)
    agent = build_agent(settings, tools=[search_tool])
    session_id = agent.new_session_id()

    run_question(
        agent,
        session_id,
        "policy question: should retrieve",
        "Can I reverse a money transfer I sent to the wrong phone number after three days?",
    )

    run_question(
        agent,
        session_id,
        "follow-up/editing question: should not retrieve",
        "Make that answer shorter.",
    )


if __name__ == "__main__":
    main()
