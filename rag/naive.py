"""Naive RAG: the baseline pipeline, five stages, nothing clever.

The same question is asked twice: bare (the model alone) and grounded
(the model with retrieved context). The difference is the argument for
this entire series.

Run:
    uv run rag/naive.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from sophons.documents import Document
from sophons.cli import ui
from sophons.integrations.models import DeepSeekModel, SentenceTransformerEmbeddings
from sophons.integrations.vector_stores import InMemoryVectorStore
from sophons.loaders import FileLoader
from sophons.models import Message
from sophons.retrieval import SemanticRetriever
from sophons.splitters import RecursiveCharacterSplitter


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deepseek_api_key: str
    deepseek_model: str = "deepseek-chat"


def load_settings() -> Settings:
    # Pydantic Settings loads required values from the environment at runtime.
    return Settings()  # pyright: ignore[reportCallIssue]


settings = load_settings()

DOCS_DIR = Path(__file__).parent / "docs"

GROUNDED_PROMPT = """\
Answer the question using only the context below. If the context does
not contain the answer, say so plainly.

Context:
{context}

Question: {question}"""


def build_retriever() -> SemanticRetriever:
    # Stage 1 — load: files become Documents (metadata carries the source)
    documents: list[Document] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        documents.extend(FileLoader(path).load())

    # Stage 2 — split: documents become focused, overlapping chunks
    splitter = RecursiveCharacterSplitter(chunk_size=400, chunk_overlap=60)
    chunks = splitter.split_documents(documents)

    # Stages 3 + 4 — embed and store: chunks become searchable vectors
    retriever = SemanticRetriever(
        embedder=SentenceTransformerEmbeddings(),
        vector_store=InMemoryVectorStore(),
    )
    retriever.add(chunks)
    ui.note(f"indexed {len(documents)} documents as {len(chunks)} chunks")
    return retriever


async def ask(model: DeepSeekModel, prompt: str) -> str:
    response = model.invoke([Message(role="user", content=prompt)])
    if asyncio.iscoroutine(response):
        response = await response
    return response.content


async def main() -> None:
    model = DeepSeekModel(
        model=settings.deepseek_model, api_key=settings.deepseek_api_key
    )
    retriever = build_retriever()

    question = "I sent money to the wrong number three days ago — can the transfer still be reversed?"

    ui.header("naive.py", subtitle="bare vs grounded · Luche Bank corpus")

    # Bare: the model alone, no documents
    ui.note("bare (no retrieval)")
    ui.user(question)
    ui.agent(await ask(model, question), footer="no retrieval")

    # Stage 5 — retrieve + generate: the naive RAG answer
    chunks = retriever.retrieve(question, limit=3)
    context = "\n\n".join(
        f"[{c.metadata.get('file_name', c.id)}] {c.content}" for c in chunks
    )
    ui.note("grounded (naive RAG)")
    ui.user(question)
    ids = ", ".join(c.id.rsplit("/", 1)[-1] for c in chunks)
    ui.tool(f"retrieve(question, limit=3) → {ids}")
    ui.agent(
        await ask(model, GROUNDED_PROMPT.format(context=context, question=question)),
        footer=f"retrieved from: {chunks[0].id.rsplit('/', 1)[-1]}",
    )


if __name__ == "__main__":
    asyncio.run(main())
