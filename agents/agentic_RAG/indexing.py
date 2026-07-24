from __future__ import annotations

from pathlib import Path

from sophons import Document
from sophons.integrations.models import SentenceTransformerEmbeddings
from sophons.integrations.vector_stores import InMemoryVectorStore
from sophons.loaders import FileLoader
from sophons.retrieval import BM25Retriever, HybridRetriever, SemanticRetriever
from sophons.splitters import RecursiveCharacterSplitter


DOCS_DIR = Path(__file__).parents[2] / "rag" / "docs"


def load_chunks() -> list[Document]:
    documents: list[Document] = []

    for path in sorted(DOCS_DIR.glob("*.md")):
        documents.extend(FileLoader(path).load())

    splitter = RecursiveCharacterSplitter(
        chunk_size=320,
        chunk_overlap=50,
    )

    return splitter.split_documents(documents)


def build_retriever() -> HybridRetriever:
    chunks = load_chunks()

    sparse = BM25Retriever(chunks)

    dense = SemanticRetriever(
        embedder=SentenceTransformerEmbeddings(),
        vector_store=InMemoryVectorStore(),
    )

    dense.add(chunks)

    return HybridRetriever([sparse, dense])
