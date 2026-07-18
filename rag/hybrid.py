"""Hybrid search: dense + sparse fused with Reciprocal Rank Fusion.

Two queries, three retrievers each. The lawyer's exact-identifier query
defeats semantic search; the paraphrase query defeats keyword search;
the hybrid wins both. No model calls — this is retrieval only, so you
can see the ranking differences without generation in the way.

Run:
    uv run rag/hybrid.py
"""

from __future__ import annotations

from pathlib import Path

from sophons.documents import Document
from sophons.integrations.models import SentenceTransformerEmbeddings
from sophons.integrations.vector_stores import InMemoryVectorStore
from sophons.loaders import FileLoader
from sophons.documents import Document
from sophons.retrieval import BM25Retriever, HybridRetriever, Retriever, SemanticRetriever
from sophons.splitters import RecursiveCharacterSplitter

DOCS_DIR = Path(__file__).parent / "docs"


def load_chunks() -> list[Document]:
    documents: list[Document] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        documents.extend(FileLoader(path).load())
    return RecursiveCharacterSplitter(
        chunk_size=400, chunk_overlap=60
    ).split_documents(documents)


def main() -> None:
    chunks = load_chunks()

    sparse = BM25Retriever(chunks)
    dense = SemanticRetriever(
        embedder=SentenceTransformerEmbeddings(),
        vector_store=InMemoryVectorStore(),
    )
    dense.add(chunks)
    hybrid = HybridRetriever([sparse, dense])

    total = len(chunks)

    # (query, a string the answer-bearing chunk must contain)
    queries = [
        ("How much is FEE-WDR-021?", "KES 110"),           # exact code — sparse territory
        ("am i charged extra when buying things overseas with my visa?",
         "3.5 percent"),                                    # paraphrase — dense territory
    ]

    def rank_of(retriever: Retriever, query: str, answer: str, limit: int) -> str:
        ranked = retriever.retrieve(query, limit=limit)
        for i, d in enumerate(ranked):
            if answer in d.content:
                return f"#{i + 1}"
        return f"not in top {limit}" if limit < total else "not returned"

    for query, answer in queries:
        print(f"\nquery: {query}")
        # engines: full-depth rank (their honest full ordering)
        print(f"  bm25      answer at rank {rank_of(sparse, query, answer, total)}")
        print(f"  semantic  answer at rank {rank_of(dense, query, answer, total)}")
        # hybrid: judged the way it runs in production — a shortlist
        print(f"  hybrid    answer at rank {rank_of(hybrid, query, answer, 3)} (of top 3)")


if __name__ == "__main__":
    main()
