# RAG

Examples for the "RAG From the Ground Up" series — retrieval-augmented
generation built stage by stage from sophons pieces: loaders, splitters,
embeddings, vector stores, retrievers, and models.

Embeddings run locally via sentence-transformers (no API key; the model
downloads once on first run). Generation uses DeepSeek via `.env`.

## Examples

### [naive.py](naive.py)

The baseline pipeline, five stages explicit: load ([docs/](docs/)) ->
split -> embed -> store -> retrieve + generate. The same question asked
bare and grounded:

```
=== BARE (no retrieval)
Refund policies vary by company... I recommend checking the service's
official refund policy...

=== GROUNDED (naive RAG)
Based solely on the context, no. Annual plans include a 14-day refund
window... three weeks (21 days) is past the 14-day window.
```

The bare model deflects; the grounded one answers, cites, and does the
arithmetic.

```bash
uv run rag/naive.py
```
