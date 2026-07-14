"""Chat with your documents — naive RAG behind the sophons terminal.

Same pipeline as naive.py; the chat loop and panels come from
sophons.cli. Every answer's footer shows which files it was grounded on.

Run:
    uv run rag/chat.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from sophons.cli import chat
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
    documents = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        documents.extend(FileLoader(path).load())
    chunks = RecursiveCharacterSplitter(
        chunk_size=400, chunk_overlap=60
    ).split_documents(documents)
    retriever = SemanticRetriever(
        embedder=SentenceTransformerEmbeddings(),
        vector_store=InMemoryVectorStore(),
    )
    retriever.add(chunks)
    return retriever


def main() -> None:
    print("indexing documents...")
    retriever = build_retriever()
    model = DeepSeekModel(
        model=settings.deepseek_model, api_key=settings.deepseek_api_key
    )

    def answer(question: str) -> tuple[str, str]:
        chunks = retriever.retrieve(question, limit=3)
        context = "\n\n".join(
            f"[{c.metadata.get('file_name', c.id)}] {c.content}" for c in chunks
        )
        response = model.invoke(
            [Message(role="user", content=GROUNDED_PROMPT.format(
                context=context, question=question))]
        )
        if asyncio.iscoroutine(response):
            response = asyncio.run(response)
        sources = " · ".join(
            dict.fromkeys(c.metadata.get("file_name", c.id) for c in chunks)
        )
        return response.content, f"sources: {sources}"

    chat(
        title="Docs Chat",
        subtitle="naive RAG · deepseek-chat · local embeddings",
        answer=answer,
        history_name="sophons_rag_chat",
    )


if __name__ == "__main__":
    main()
