from __future__ import annotations

from sophons.retrieval import Retriever
from sophons.tools import RetrieverTool


def build_policy_search_tool(retriever: Retriever) -> RetrieverTool:
    return RetrieverTool(
        name="search_bank_policy_docs",
        description=(
            "Search Luche Bank policy docs for accounts, cards, transfers, fees, "
            "loans, branches, and mobile app questions."
        ),
        retriever=retriever,
        top_k=3,
    )
