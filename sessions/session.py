"""Session memory: what scope does the agent remember in?

The test is a follow-up question that means nothing on its own. "Is it
still open on Saturdays?" — is *what* still open? Answering it requires
the previous turn.

The thing that switches memory on is the **session id**, not the session
manager. An Agent built without a manager still gets one
(``InMemorySessionManager``), so what changes across the three runs below
is only how far the memory reaches:

    no session id                  forgets every turn
    session id, default manager    remembers, until the process exits
    session id, FileSessionManager remembers across restarts

Run:
    uv run sessions/session.py
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from sophons.agents import Agent
from sophons.agents.session import FileSessionManager, SessionManager
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

SESSIONS_DIR = Path(__file__).parent / ".sessions"

BRANCHES = {
    "westlands": "open until 17:00 on weekdays, 12:00 on Saturdays",
    "kilimani": "closed for refurbishment until March",
}


@tool
def branch_hours(branch: str) -> str:
    """Look up the opening hours of a Luche Bank branch by name."""
    return BRANCHES.get(branch.lower(), "no branch by that name")


SYSTEM = "You answer questions about Luche Bank branches. Use branch_hours."

TURNS = [
    "What are the hours for the Westlands branch?",
    "Is it still open on Saturdays?",  # "it" only resolves with history
]


def build_agent(session_manager: SessionManager | None = None) -> Agent:
    return Agent(
        model=DeepSeekModel(
            model=settings.deepseek_model, api_key=settings.deepseek_api_key
        ),
        tools=[branch_hours],
        system_prompt=SYSTEM,
        session_manager=session_manager,
    )


def converse(label: str, agent: Agent, session_id: str | None) -> None:
    ui.note(label)
    for turn in TURNS:
        ui.user(turn)
        # Agent.run() loads history before the loop and saves after it, every
        # call — but both are no-ops without a session_id to file them under.
        result = agent.run_sync(turn, session_id=session_id)
        ui.agent(result.message, footer=f"session_id={session_id or 'None'}")


def main() -> None:
    ui.header("sessions/session.py", subtitle="the same two turns, three scopes")

    # No id: nothing to load, nothing saved. The follow-up has no referent.
    converse("FORGETFUL — no session id", build_agent(), None)

    # An id alone is enough to remember, using the default in-memory manager.
    # Note this is a fresh Agent, proving the id does the work, not the object.
    converse("IN PROCESS — session id, default manager", build_agent(), "branch-chat")

    # Same id, durable storage: this history outlives the process.
    converse(
        "ON DISK — session id + FileSessionManager",
        build_agent(FileSessionManager(SESSIONS_DIR)),
        "branch-chat",
    )

    ui.note(f"history on disk: {SESSIONS_DIR}/branch-chat.json")


if __name__ == "__main__":
    main()
