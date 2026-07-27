"""
Tool use: one tool, one call, the whole loop visible.

the model cannot know an account balance. so it must ask. This example
prints the middle of the run - the tool the model chose, the arguments it
built, and what came back- not just the final sentece.

Run:
    uv run tool_use/basic.py
"""

from __future__ import annotations

from pydantic_settings import BaseSettings,SettingsConfigDict

from sophons.agents import Agent
from sophons.cli import ui
from sophons.integrations.models import DeepSeekModel
from sophons.tools import tool


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env",extra = "ignore")

    deepseek_api_key: str
    deepseek_model:str = "deepseek-chat"

def load_settings()->Settings:
    # Pydantic Settings loads required values from the environment at runtime.
    return Settings() # pyright: ignore[reportCallIssue]


settings = load_settings()


BALANCES = {
    "0114455": "KES 12,480.00 available, KES 500.00 on hold",
    "0119912": "KES 320.50 available",
}

@tool
def account_balance(account_number:str)->str:
    """Look up the current balance for a Luche Bank account number."""
    return BALANCES.get(account_number,"no account found with that number")


def main()->None:
    agent = Agent(
        model = DeepSeekModel(
            model = settings.deepseek_model,
            api_key = settings.deepseek_api_key
        ),
        tools = [account_balance],
        system_prompt = (
            "You answer questions about Luche Bank accounts. Always use the "
            "account_balance tool; never guess a balance"
        ),
    )

    question = "How much is left on account 0114455?"

    ui.header("tool_use/basic.py", subtitle = "one tool . one call")
    ui.user(question)

    result = agent.run_sync(question)

    # The middle of the run: what the model chose, and what came back.
    for use,outcome in zip(result.tool_uses,result.tool_results):
        ui.tool(f"{use.name}({use.input})->[{outcome.status}] {outcome.content}")

    ui.agent(
        result.message,
        footer = (
            f"steps = {result.metrics.steps} "
            f"model_calls = {result.metrics.model_calls} "
            f"tool_calls = {result.metrics.tool_calls} "
            f"stop={result.stop_reason.value}"
        )
    )

if __name__ == "__main__":
    main()
