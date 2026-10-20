# Frozen synthetic precision benchmark v1

This is fictional test material, authored before inference on 2026-10-21. It
contains no Vault text or production results. The UTF-8 JSONL bytes are the
benchmark identity; the validator records their SHA-256 before each run.

The 200 documents form 50 independent four-document scenarios. Each scenario
has two paraphrased queries. There are 100 queries: Korean semantics 30, mixed
technical terminology 20, exact identifiers 15, negation/hard negatives 15,
date/type/private filters 10, and long inputs 10. Long scenarios include a
decision at the end of a review and a rejected proposal as a hard negative.

Each query explicitly judges four documents: direct answer 2, supporting context
1, contradictory/different-condition answer 0, unrelated record 0. `rationale`
describes these judgments. Other documents are **unjudged**, not adjudicated
irrelevant; metrics may use the conventional unlisted-as-zero assumption but
must disclose it. This small synthetic set tests regression, not the quality of
a production corpus. Query paraphrases are correlated, so 100 queries do not
represent 100 independent topics.

Document metadata matches the existing search fields. Filter query judgments
are conditional on `filters`; excluded records have relevance 0 even if the
text is similar. `include_private` is false unless explicitly enabled. The
DB-free baseline records raw vectors/scores without applying retrieval filters;
the owned-Qdrant phase must exercise `SearchRequest` and the existing filter.

Do not regenerate labels from model rankings or edit this set after seeing
candidate output. Changes require a new benchmark identity and rerun of both
baseline and candidate. The baseline script scores the frozen 400 judged
reranker pairs and embeds all 300 inputs. Retrieval-candidate comparisons and
metrics are added in Task 5.

Preparation (no model import or inference):

```powershell
$env:PYTHONPATH = "$PWD/src"
.\.venv\Scripts\python.exe scripts/validate_inference_precision.py --phase baseline --prepare-only --output .knowledge/bf16-validation/prepared.json
```

Actual baseline measurement requires `KNOWLEDGE_RUN_REAL_MODELS=1`, cached pinned
revisions and an exclusive GPU window. Each component/device/default-or-fp32
combination runs sequentially in a fresh process. No package upgrades, DB
connections, Vault reads, daemon changes or indexing are part of preparation.
The `prepared_not_measured` result is not evidence of effective fp32/bf16 dtype,
output parity, quality, memory savings or performance.
