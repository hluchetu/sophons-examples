"""Query rewriting: ask once badly, retrieve with better versions.

The user's exact words are not always the best search query. A
multi-query retriever asks a model for a few precise rewrites, retrieves
for each one, then deduplicates the result set.

Run:
    uv run rag/query_rewriting.py
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from sophons.cli import ui
from sophons.documents import Document
from sophons.integrations.models import DeepSeekModel
from sophons.loaders import FileLoader
from sophons.retrieval import BM25Retriever, MultiQueryRetriever
from sophons.splitters import RecursiveCharacterSplitter

DOCS_DIR = Path(__file__).parent / "docs"

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


def load_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]


def load_chunks() -> list[Document]:
    documents: list[Document] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        documents.extend(FileLoader(path).load())
    return RecursiveCharacterSplitter(chunk_size=320, chunk_overlap=50).split_documents(
        documents
    )


def source_name(document: Document) -> str:
    return str(document.metadata.get("file_name", document.id or "unknown"))


def brief(document: Document, *, words: int = 34) -> str:
    text = " ".join(document.content.split())
    return " ".join(text.split()[:words])


def main() -> None:
    settings = load_settings()
    chunks = load_chunks()
    retriever = BM25Retriever(chunks)
    model = DeepSeekModel(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
    )
    multi_query = MultiQueryRetriever(
        retriever=retriever,
        model=model,
        prompt=REWRITE_PROMPT,
    )

    question = (
        "I made an oopsie payment to a stranger three days ago. "
        "Can I get it back?"
    )

    ui.header("query_rewriting.py", subtitle="bad phrasing -> better retrieval")
    ui.note(f"indexed {len(chunks)} chunks")
    ui.user(question)

    baseline = retriever.retrieve(question, limit=3)
    ui.tool("original query top 3: " + " · ".join(source_name(d) for d in baseline))

    queries, rewritten_results = multi_query.retrieve_with_queries(question, limit=3)
    rewrites = [query for query in queries if query != question]
    ui.tool("rewrites: " + " · ".join(rewrites))
    ui.tool(
        "rewritten query top 3: "
        + " · ".join(source_name(document) for document in rewritten_results)
    )

    best = rewritten_results[0]
    ui.agent(
        brief(best),
        footer=f"best chunk: {source_name(best)}",
    )


if __name__ == "__main__":
    main()
