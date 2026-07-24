from __future__ import annotations

from sophons.agents import Agent
from sophons.integrations.models import DeepSeekModel
from sophons.tools import AsyncTool, Tool

from .settings import Settings


SYSTEM_PROMPT = """\
You are a Luche Bank support assistant.

Use search_bank_policy_docs when the user asks about Luche Bank policies,
account procedures, fees, cards, loans, branches, mobile app support, or
transfer reversals.

Do not use search_bank_policy_docs for greetings, rewriting, summarizing,
or general conversation.

When you use policy documents, answer only from the retrieved tool result.
If the tool result does not contain enough evidence, say so plainly.
"""


def build_agent(settings: Settings, tools: list[Tool | AsyncTool]) -> Agent:
    model = DeepSeekModel(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
    )

    return Agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
