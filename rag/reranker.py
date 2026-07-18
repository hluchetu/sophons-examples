"""Reranking: retrieve broadly, then reorder with a cross-encoder.

BM25 is good at catching exact policy terms, but it can put an
answer-bearing chunk below less useful keyword matches. A reranker reads
the query and each candidate chunk together, then returns the few chunks
that are most likely to answer the question.

Run:
    uv run rag/reranker.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

from sentence_transformers import CrossEncoder

from sophons.cli import ui
from sophons.documents import Document
from sophons.loaders import FileLoader
from sophons.retrieval import BM25Retriever
from sophons.splitters import RecursiveCharacterSplitter

DOCS_DIR = Path(__file__).parent / "docs"


class CrossEncoderReranker:
    """Sophons DocumentCompressor backed by a sentence-transformers model."""

    def __init__(
        self,
        *,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 3,
        batch_size: int = 16,
        max_chars: int = 1_200,
    ) -> None:
        self.model_name = model_name
        self.model: Any = CrossEncoder(model_name)
        self.top_n = top_n
        self.batch_size = batch_size
        self.max_chars = max_chars

    async def compress(self, documents: list[Document], query: str) -> list[Document]:
        pairs = [(query, document.content[: self.max_chars]) for document in documents]
        scores = cast(
            list[float],
            self.model.predict(
                pairs,
                batch_size=self.batch_size,
                convert_to_numpy=False,
                show_progress_bar=False,
            ),
        )

        ranked = sorted(
            zip(documents, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            document.with_score(float(score)).with_metadata(reranker=self.model_name)
            for document, score in ranked[: self.top_n]
        ]


def load_chunks() -> list[Document]:
    documents: list[Document] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        documents.extend(FileLoader(path).load())
    return RecursiveCharacterSplitter(chunk_size=320, chunk_overlap=50).split_documents(
        documents
    )


def source_name(document: Document) -> str:
    return str(document.metadata.get("file_name", document.id or "unknown"))


def score_text(document: Document) -> str:
    return "n/a" if document.score is None else f"{document.score:.2f}"


def brief(document: Document, *, words: int = 18) -> str:
    text = " ".join(document.content.split())
    return " ".join(text.split()[:words])


async def main() -> None:
    chunks = load_chunks()
    retriever = BM25Retriever(chunks)
    reranker = CrossEncoderReranker(top_n=3)

    question = (
        "Can I reverse a money transfer I sent to the wrong phone number "
        "after three days?"
    )

    ui.header("reranker.py", subtitle="BM25 candidates -> cross-encoder top 3")
    ui.note(f"indexed {len(chunks)} chunks")
    ui.user(question)

    candidates = retriever.retrieve(question, limit=8)
    before = " · ".join(source_name(document) for document in candidates[:5])
    ui.tool(f"bm25 top 5: {before}")

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
