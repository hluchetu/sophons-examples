# RAG

Examples for the "RAG From the Ground Up" series — retrieval-augmented
generation built stage by stage from sophons pieces: loaders, splitters,
embeddings, vector stores, retrievers, and models.

Embeddings run locally via sentence-transformers (no API key; the model
downloads once on first run). Generation uses DeepSeek via `.env`.

## Examples

### [naive.py](naive.py)

The baseline pipeline, five stages explicit: load ([docs/](docs/), the
Luche Bank corpus) -> split -> embed -> store -> retrieve + generate.
The same question asked bare and grounded:

```
=== BARE (no retrieval)
Whether the transfer can be reversed depends on several factors...
(a page of generic advice: contact your bank, PayPal, small claims...)

=== GROUNDED (naive RAG)
After 48 hours, a reversal is only possible with a fraud report filed
at a branch or through the app... three days ago (more than 48 hours),
it cannot be reversed through the standard wrong-recipient process.

retrieved from: reversals.md#chunk_0, branches.md#chunk_1, ...
```

The bare model produces a wall of maybe; the grounded one answers from
the actual policy and cites it.

```bash
uv run rag/naive.py
```

### [chat.py](chat.py)

The same pipeline behind the sophons terminal (`sophons.cli.chat`): ask
questions interactively; each answer's footer names the files it was
grounded on.

```bash
uv run rag/chat.py
```

### [hybrid.py](hybrid.py)

BM25 (keywords), semantic (meaning), and RRF fusion compared by the
answer's rank per retriever — including fusion's honest failure mode
on a small, topically narrow corpus:

```
query: How much is FEE-WDR-021?
  bm25      answer at rank #1
  semantic  answer at rank #3
  hybrid    answer at rank #2 (of top 3)

query: am i charged extra when buying things overseas with my visa?
  bm25      answer at rank not returned
  semantic  answer at rank #1
  hybrid    answer at rank not in top 3 (of top 3)
```

Fusion rescues the exact-identifier query and buries the paraphrase
one — on 23 chunks, every engine ranks everything, so consensus junk
outvotes a single first place. RRF needs truncated lists over a large,
diverse corpus to shine. Measuring this properly is Part 8's subject.

```bash
uv run rag/hybrid.py
```

### [reranker.py](reranker.py)

Hybrid retrieval builds a wider candidate set, then a
sentence-transformers cross-encoder reranks the `(question, chunk)` pairs
with Sophons' document-compressor shape:

```
hybrid top 5: reversals.md · reversals.md · mobile-app.md · ...
reranked top 3: reversals.md (2.10) · reversals.md (-3.47) · ...
```

The point is the production shape: cheap retriever first, expensive
reranker only on the shortlist, then pass the top chunks to generation.

```bash
uv run rag/reranker.py
```

### [query_rewriting.py](query_rewriting.py)

The next repair borrows LangChain's multi-query retriever idea: ask the
model for several better search queries, retrieve for each one, dedupe
the documents, and keep the best few candidates.

```
original query top 3: branches.md · mobile-app.md · cards.md
rewrites: wrong-recipient transfer reversal 72 hours policy · ...
rewritten query top 3: reversals.md · branches.md · reversals.md
```

The user's phrasing stays human; retrieval gets phrasing built for the
knowledge base.

```bash
uv run rag/query_rewriting.py
```

### [corrective.py](corrective.py)

Corrective RAG adds a Sophons eval judge between retrieval and
generation. The first retrieval trusts the user's messy phrasing and
pulls weak evidence; `ContextRelevanceEvaluator` returns a full
`EvalScore`, then Sophons' `MultiQueryRetriever` rewrites the question
and retrieves again. Generation only happens after the corrected
evidence passes the `context_relevance` dimension.

```
initial retrieval: branches.md · mobile-app.md · cards.md
retrieval score: 0.00 passed=False · ...
rewrite queries: wrong-recipient transfer reversal 72 hours policy · ...
corrected retrieval: reversals.md · reversals.md · branches.md
corrected score: 1.00 passed=True · ...
```

The lesson is the control loop: RAG should not blindly pass retrieved
chunks to the model just because retrieval returned something.

```bash
uv run rag/corrective.py
```
