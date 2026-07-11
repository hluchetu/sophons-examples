"""Catching a hallucination with FaithfulnessEvaluator.

Two answers to the same question, judged against the same retrieved
context by deepseek-chat:

- a confident, invented answer -> passed=False, every unsupported claim named
- an honest "the docs don't say" -> passed=True (asserting nothing is faithful)

Run:
    uv run evaluation/faithfulness.py
"""

from __future__ import annotations

import asyncio

from pydantic_settings import BaseSettings, SettingsConfigDict

from sophons.evals import FaithfulnessEvaluator
from sophons.integrations.models import DeepSeekModel


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deepseek_api_key: str
    deepseek_model: str = "deepseek-chat"


settings = Settings()

QUESTION = "Do annual plans come with priority support?"

# What the retriever returned: nothing about priority support.
CONTEXT = """\
Annual plans include a 14-day refund window from the date of purchase.
Support tickets are answered within two business days on all plans.
The Team plan allows up to 25 seats; the Starter plan allows 3."""

ANSWERS = {
    "hallucinated": (
        "Yes — annual plans include priority support. Tickets from annual "
        "subscribers are routed to a dedicated queue with a target "
        "first-response time of 4 business hours."
    ),
    "honest": (
        "The documentation does not mention priority support for annual "
        "plans. It only states that support tickets are answered within two "
        "business days on all plans."
    ),
}


async def main() -> None:
    judge = DeepSeekModel(
        model=settings.deepseek_model, api_key=settings.deepseek_api_key
    )
    evaluator = FaithfulnessEvaluator(judge)

    for label, answer in ANSWERS.items():
        result = await evaluator.evaluate(QUESTION, answer, context=CONTEXT)
        verdict = result.scores[0]
        print(f"--- {label}")
        print(f"passed={verdict.passed}  score={verdict.score:.2f}")
        print(f"reason: {verdict.reason}")
        for cv in verdict.metadata.get("claim_verdicts", []):
            mark = "+" if cv["supported"] else "x"
            print(f"  [{mark}] {cv['claim']}")
            print(f"      evidence: {cv['evidence']}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
