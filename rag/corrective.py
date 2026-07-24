"""Corrective RAG: grade retrieval before trusting it.

Naive RAG assumes the retrieved chunks are good enough. Corrective RAG
adds a small control loop: retrieve, grade the evidence, repair the query
when evidence is bad, then generate only from the corrected context.

Run:
    uv run rag/corrective.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from sophons.cli import ui
from sophons.documents import Document
from sophons.evals import ContextRelevanceEvaluator, EvalScore
from sophons.integrations.models import DeepSeekModel
from sophons.loaders import FileLoader
from sophons.models import Message
from sophons.retrieval import BM25Retriever, MultiQueryRetriever
from sophons.splitters import RecursiveCharacterSplitter

DOCS_DIR = Path(__file__).parent / "docs"

GROUNDED_PROMPT = """\
Answer the question using only the context below. If the context does
not contain the answer, say so plainly.

Context:
{context}

Question: {question}"""

REWRITE_PROMPT = """\
Rewrite the user's question into {count} short search queries for a bank
policy knowledge base.

The knowledge base uses terms like:
- wrong-recipient transfer reversals
- fraud report
- 48 hours
- app PIN reset
- card dispute
- fee code

Rules:
- Keep concrete details such as time windows, amounts, codes, and product names.
- Use likely policy terms.
- Return one query per line.
- Do not number the lines.

Question: {question}"""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deepseek_api_key: str
    deepseek_model: str = "deepseek-chat"


def empty_documents() -> list[Document]:
    return []


def empty_strings() -> list[str]:
    return []


@dataclass
class CRAGState:
    question: str
    chunks: list[Document] = field(default_factory=empty_documents)
    retrieval_score: EvalScore | None = None
    rewritten_question: str | None = None
    rewrite_queries: list[str] = field(default_factory=empty_strings)
    answer: str | None = None


def load_chunks() -> list[Document]:
    documents: list[Document] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        documents.extend(FileLoader(path).load())
    return RecursiveCharacterSplitter(chunk_size=320, chunk_overlap=50).split_documents(
        documents
    )


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


async def main() -> None:
    settings = Settings()  # pyright: ignore[reportCallIssue]
    chunks = load_chunks()
    retriever = BM25Retriever(chunks)
    model = DeepSeekModel(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
    )
    context_evaluator = ContextRelevanceEvaluator(model)
    corrective_retriever = MultiQueryRetriever(
        retriever=retriever,
        model=model,
        prompt=REWRITE_PROMPT,
        include_original=False,
    )

    state = CRAGState(
        question=(
            "I made an oopsie payment to a stranger three days ago. "
            "Can I get it back?"
        )
    )

    ui.header("corrective.py", subtitle="retrieve -> grade -> correct -> answer")
    ui.note(f"indexed {len(chunks)} chunks")
    ui.user(state.question)

    state.chunks = retriever.retrieve(state.question, limit=3)
    ui.tool(
        "initial retrieval: "
        + " · ".join(source_name(document) for document in state.chunks)
    )

    result = await context_evaluator.evaluate(
        state.question,
        context=context_from(state.chunks),
    )
    state.retrieval_score = result.scores[0]
    ui.tool(
        "retrieval score: "
        f"{state.retrieval_score.score:.2f} "
        f"passed={state.retrieval_score.passed} · "
        f"{state.retrieval_score.reason}"
    )

    if not state.retrieval_score.passed:
        queries, corrected = corrective_retriever.retrieve_with_queries(
            state.question,
            limit=3,
        )
        state.rewrite_queries = queries
        state.rewritten_question = queries[0] if queries else None
        ui.tool("rewrite queries: " + " · ".join(queries))

        state.chunks = corrected
        ui.tool(
            "corrected retrieval: "
            + " · ".join(source_name(document) for document in state.chunks)
        )

        grade_question = state.rewritten_question or state.question
        result = await context_evaluator.evaluate(
            grade_question,
            context=context_from(state.chunks),
        )
        state.retrieval_score = result.scores[0]
        ui.tool(
            "corrected score: "
            f"{state.retrieval_score.score:.2f} "
            f"passed={state.retrieval_score.passed} · "
            f"{state.retrieval_score.reason}"
        )

    if not state.retrieval_score.passed:
        state.answer = "I could not find reliable policy evidence to answer."
    else:
        state.answer = await ask(
            model,
            GROUNDED_PROMPT.format(
                context=context_from(state.chunks),
                question=state.question,
            ),
        )

    ui.agent(
        state.answer,
        footer="retrieved from: "
        + ", ".join(source_name(document) for document in state.chunks),
    )


if __name__ == "__main__":
    asyncio.run(main())
