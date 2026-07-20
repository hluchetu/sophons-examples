"""Reranking: retrieve broadly, then reorder with a cross-encoder.

Hybrid retrieval builds a diverse candidate set: exact keyword hits from
BM25 plus paraphrase matches from semantic search. A cross-encoder then
scores each (query, chunk) pair directly and keeps the chunks most likely
to answer the question.

Run:
    uv run rag/reranker.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from sophons.cli import ui
from sophons.documents import Document
from sophons.integrations.compressors import CrossEncoderReranker
from sophons.integrations.models import SentenceTransformerEmbeddings
from sophons.integrations.vector_stores import InMemoryVectorStore
from sophons.loaders import FileLoader
from sophons.retrieval import BM25Retriever, HybridRetriever, SemanticRetriever
from sophons.splitters import RecursiveCharacterSplitter

DOCS_DIR = Path(__file__).parent / "docs"


def load_chunks() -> list[Document]:
    documents: list[Document] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        documents.extend(FileLoader(path).load())
    return RecursiveCharacterSplitter(chunk_size=320, chunk_overlap=50).split_documents(
        documents
    )


def build_retriever(chunks: list[Document]) -> HybridRetriever:
    sparse = BM25Retriever(chunks)
    dense = SemanticRetriever(
        embedder=SentenceTransformerEmbeddings(),
        vector_store=InMemoryVectorStore(),
    )
    dense.add(chunks)
    return HybridRetriever([sparse, dense])


def source_name(document: Document) -> str:
    return str(document.metadata.get("file_name", document.id or "unknown"))


def score_text(document: Document) -> str:
    return "n/a" if document.score is None else f"{document.score:.2f}"


def brief(document: Document, *, words: int = 18) -> str:
    text = " ".join(document.content.split())
    return " ".join(text.split()[:words])


async def main() -> None:
    chunks = load_chunks()
    retriever = build_retriever(chunks)
    reranker = CrossEncoderReranker(top_n=3)

    question = (
        "Can I reverse a money transfer I sent to the wrong phone number "
        "after three days?"
    )

    ui.header("reranker.py", subtitle="hybrid candidates -> cross-encoder top 3")
    ui.note(f"indexed {len(chunks)} chunks")
    ui.user(question)

    candidates = retriever.retrieve(question, limit=8)
    before = " · ".join(source_name(document) for document in candidates[:5])
    ui.tool(f"hybrid top 5: {before}")

    reranked = await reranker.compress(candidates, question)
    after = " · ".join(
        f"{source_name(document)} ({score_text(document)})" for document in reranked
    )
    ui.tool(f"reranked top 3: {after}")

    best = reranked[0]
    ui.agent(
        brief(best, words=42),
        footer=f"best chunk: {source_name(best)}",
    )


if __name__ == "__main__":
    asyncio.run(main())
