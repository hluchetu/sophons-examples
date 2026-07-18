"""Interactive refund-agent chat with human approval — the full experience.

The chat CLI from the *-from-scratch repos, wired to today's guardrails:
refunds up to 100.00 run unattended; larger ones pause the run and render
an approval panel right in the conversation. You are the approver.

Run:
    uv run guardrails/approval_chat.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from sophons.agents import Agent
from sophons.agents.responses import AgentMetrics
from sophons.guardrails import (
    ApprovalDecision,
    ApprovalGuardrail,
    ApprovalRequest,
    CallbackApprover,
)
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
console = Console()

_PROMPT_STYLE = Style.from_dict({"prompt": "bold #5f87ff"})
_HISTORY_FILE = Path.home() / ".sophons_approval_chat_history"

AUTO_APPROVE_LIMIT = 100.0


@tool
def refund_order(order_id: str, amount: float) -> str:
    """Refund the given amount for an order."""
    return f"refunded {amount} for {order_id}"


async def rich_approval(request: ApprovalRequest) -> ApprovalDecision:
    """Render the pending action as a panel and ask the human in the chat."""
    console.print()
    console.print(
        Panel(
            f"[bold]{request.tool_name}[/bold]({request.value})\n"
            f"[dim]{request.reason}[/dim]",
            title="[bold yellow]Approval needed[/bold yellow]",
            border_style="yellow",
            padding=(0, 1),
        )
    )
    answer = await asyncio.to_thread(input, "  Approve? [y/N] ")
    approved = answer.strip().lower() in ("y", "yes")
    verdict = "[green]approved[/green]" if approved else "[red]denied[/red]"
    console.print(f"  [dim]→ {verdict} by you[/dim]\n")
    return ApprovalDecision(approved=approved, approver="you")


def _print_header() -> None:
    console.print()
    console.print(
        Panel.fit(
            "[bold white]Refund Agent[/bold white]  [dim]deepseek-chat · "
            f"auto-approve ≤ {AUTO_APPROVE_LIMIT:.0f}[/dim]\n"
            "[dim]Refunds above the limit will ask YOU for approval. "
            "[bold]exit[/bold] or Ctrl+C to quit.[/dim]",
            border_style="bright_black",
            padding=(0, 2),
        )
    )
    console.print()


def _print_user(text: str) -> None:
    console.print(
        Panel(
            Text(text, style="white"),
            title="[bold #5f87ff]You[/bold #5f87ff]",
            border_style="#5f87ff",
            padding=(0, 1),
        )
    )


def _print_agent(text: str, metrics: AgentMetrics) -> None:
    subtitle = (
        f"[dim]steps={metrics.steps}  model_calls={metrics.model_calls}  "
        f"tool_calls={metrics.tool_calls}[/dim]"
    )
    console.print(
        Panel(
            Markdown(text),
            title="[bold #2dba4e]Agent[/bold #2dba4e]",
            subtitle=subtitle,
            border_style="#2dba4e",
            padding=(0, 1),
        )
    )


def main() -> None:
    _print_header()

    agent = Agent(
        model=DeepSeekModel(
            model=settings.deepseek_model, api_key=settings.deepseek_api_key
        ),
        tools=[refund_order],
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
        approver=CallbackApprover(rich_approval, name="you"),
    )

    session: PromptSession[str] = PromptSession(
        history=FileHistory(str(_HISTORY_FILE)), style=_PROMPT_STYLE
    )

    while True:
        try:
            user_input = session.prompt("  You › ", style=_PROMPT_STYLE).strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "/exit", "/quit"}:
            console.print("[dim]Goodbye.[/dim]")
            break

        console.print()
        _print_user(user_input)
        console.print()

        try:
            result = agent.run_sync(user_input, session_id="approval-chat")
            _print_agent(result.message, result.metrics)
            console.print()
        except KeyboardInterrupt:
            console.print("\n[dim]Cancelled.[/dim]\n")
        except Exception as exc:
            console.print(f"\n[bold red]Error:[/bold red] {exc}\n")


if __name__ == "__main__":
    main()
