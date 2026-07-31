# ADR 0001 — Fail-closed embedding model / dimension binding

- **Status:** Accepted (implemented)
- **Date:** 2026-07-16 (landed with Wave B work)
- **Evidence:** PR [#34](https://github.com/tzervas/memory-gate-rs/pull/34), CHANGELOG “Store model binding (`mg/store-model-binding`)”, non-ignored sqlite-vec meta tests in `src/storage/sqlite_vec.rs`

## Context

Vector backends (`qdrant`, `sqlite-vec`) embed text with a catalog model (`SupportedEmbeddingModel`: MiniLM, BGE-small, BGE-base). Different models can share a dimension (e.g. 384) yet produce **incompatible** vector spaces. Reusing a Qdrant collection or SQLite file after a silent model switch yields confident, wrong retrieval.

## Decision

**Fail closed** when opening or reconciling a store if:

- stored vector dimension ≠ configured model dimension, or
- stamped embedding model metadata ≠ configured model (when present), or
- legacy SQLite data has rows without metadata that would make binding ambiguous.

Stamp metadata on create (`embedding_model`, `embedding_dim` for sqlite-vec; collection metadata for Qdrant). Operators must use **one model per collection/DB path**.

## Alternatives considered

| Alternative | Why rejected |
|-------------|--------------|
| Ignore metadata; assume operator discipline | Silent wrong answers; fleet already burned by docs that hide real behavior |
| Auto-re-embed entire store on mismatch | Expensive, surprising, needs network/models; wrong for large DBs |
| Best-effort cosine with truncated/padded vectors | Garbage similarity; length-mismatch already treated as error on search path |

## Consequences

- Opening the wrong path is a **loud error** at startup, not a subtle quality drop.
- Migration between models is an explicit re-ingest (or new path), not a config flip.
- Tests without model download still cover meta stamp/reconcile/mismatch (measured VERIFIED in CURRENT-STATE); full semantic retrieve remains behind `#[ignore]` model-download tests.

## Related

- Catalog contract name used in docs: `mg/embed-catalog@STABLE`
- Binding contract name: `mg/store-model-binding@STABLE`
- Wave C accuracy still requires the ignored golden path for end-to-end embed recall
