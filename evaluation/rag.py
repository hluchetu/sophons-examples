"""Evaluating a completed RAG run across the four basic RAG dimensions.

This example runs a tiny RAG pipeline first, then judges what happened:

- context relevance: did retrieval bring useful evidence?
- faithfulness: is the answer grounded in that evidence?
- answer relevance: did the answer respond to the question?
- answer correctness: does the answer match a reference answer?

Run:
    uv run evaluation/rag.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from sophons.cli import ui
from sophons.documents import Document
from sophons.evals import (
    AnswerCorrectnessEvaluator,
    AnswerRelevanceEvaluator,
    ContextRelevanceEvaluator,
    Evaluator,
    FaithfulnessEvaluator,
)
from sophons.integrations.models import DeepSeekModel
from sophons.loaders import FileLoader
from sophons.models import Message
from sophons.retrieval import BM25Retriever
from sophons.splitters import RecursiveCharacterSplitter


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deepseek_api_key: str
    deepseek_model: str = "deepseek-chat"


DOCS_DIR = Path(__file__).parents[1] / "rag" / "docs"

QUESTION = (
    "Can I reverse a money transfer I sent to the wrong phone number "
    "after three days?"
)

REFERENCE = """\
Wrong-recipient transfer reversals can be requested within 48 hours.
After 48 hours, a reversal is only possible with a fraud report filed
through the app or at a branch, and the case moves to the disputes team.
"""

GROUNDED_PROMPT = """\
Answer the question using only the context below. If the context does
not contain the answer, say so plainly.

Context:
{context}

Question: {question}"""


def load_chunks() -> list[Document]:
    documents: list[Document] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        documents.extend(FileLoader(path).load())

    splitter = RecursiveCharacterSplitter(chunk_size=320, chunk_overlap=50)
    return splitter.split_documents(documents)


def source_name(document: Document) -> str:
    return str(document.metadata.get("file_name", document.id or "unknown"))


def context_from(chunks: list[Document]) -> str:
    return "\n\n".join(
        f"[{source_name(chunk)}] {chunk.content}" for chunk in chunks
    )


async def ask(model: DeepSeekModel, prompt: str) -> str:
    response = model.invoke([Message(role="user", content=prompt)])
    if asyncio.iscoroutine(response):
        response = await response
    return response.content


async def score_run(
    evaluators: list[Evaluator],
    *,
    question: str,
    answer: str,
    context: str,
    reference: str,
) -> None:
    for evaluator in evaluators:
        result = await evaluator.evaluate(
            question,
            answer,
            context=context,
            reference=reference,
        )
        score = result.scores[0]
        ui.tool(
            f"{score.dimension}: "
            f"score={score.score:.2f} "
            f"passed={score.passed} · "
            f"{score.reason}"
        )


async def main() -> None:
    settings = Settings()  # pyright: ignore[reportCallIssue]
    model = DeepSeekModel(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
    )

    chunks = load_chunks()
    retriever = BM25Retriever(chunks)

    ui.header("evaluation/rag.py", subtitle="run RAG, then score the run")
    ui.note(f"indexed {len(chunks)} chunks")
    ui.user(QUESTION)

    retrieved = retriever.retrieve(QUESTION, limit=3)
    context = context_from(retrieved)
    ui.tool(
        "retrieved: "
        + " · ".join(source_name(document) for document in retrieved)
    )

    answer = await ask(
        model,
        GROUNDED_PROMPT.format(context=context, question=QUESTION),
    )
    ui.agent(answer, footer="candidate RAG answer")
    ui.note("reference answer:\n" + REFERENCE.strip())

    evaluators: list[Evaluator] = [
        ContextRelevanceEvaluator(model),
        FaithfulnessEvaluator(model),
        AnswerRelevanceEvaluator(model),
        AnswerCorrectnessEvaluator(model),
    ]

    await score_run(
        evaluators,
        question=QUESTION,
        answer=answer,
        context=context,
        reference=REFERENCE,
    )


if __name__ == "__main__":
    asyncio.run(main())
