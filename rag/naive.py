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


settings = Settings()

DOCS_DIR = Path(__file__).parent / "docs"

GROUNDED_PROMPT = """\
Answer the question using only the context below. If the context does
not contain the answer, say so plainly.

Context:
{context}

Question: {question}"""


def build_retriever() -> SemanticRetriever:
    # Stage 1 — load: files become Documents (metadata carries the source)
    documents = []
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
    print(f"indexed {len(documents)} documents as {len(chunks)} chunks\n")
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

    # Bare: the model alone, no documents
    print("=== BARE (no retrieval)")
    print(await ask(model, question), "\n")

    # Stage 5 — retrieve + generate: the naive RAG answer
    chunks = retriever.retrieve(question, limit=3)
    context = "\n\n".join(
        f"[{c.metadata.get('file_name', c.id)}] {c.content}" for c in chunks
    )
    print("=== GROUNDED (naive RAG)")
    print(await ask(model, GROUNDED_PROMPT.format(context=context, question=question)))
    print("\nretrieved from:", [c.id for c in chunks])


if __name__ == "__main__":
    asyncio.run(main())
