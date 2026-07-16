# Wave C — Acceptance criteria

**Status:** Living (Wave C delivery)  
**Canonical runtime:** **memory-gate-rs only** — the Python `memory-gate` package is a **frozen mirror** after embedding parity (catalog + fail-closed store binding). Wave C accuracy, golden recall, and further perf work belong on Rust only.

**Interface bulletin:** `mg/golden-recall@STABLE`.

---

## In scope for Wave C

| Area | Criterion | Notes |
|------|-----------|--------|
| **Accuracy** | Mean **recall@5** on the fixed golden corpus must not drop **more than 2% relative** versus `baseline_mean_recall_at_k` in the fixture | Implemented as `mean >= min_mean_recall_at_k` where `min_mean_recall_at_k >= baseline * 0.98`. Per-query recall@k = \|retrieved ∩ relevant\| / \|relevant\| (unique IDs). Fixture `model_id` is catalog stable ID `all-minilm-l6-v2`. |
| **Store binding** | Golden accuracy path uses **sqlite-vec + FastEmbed** (`open_in_memory_with_model`) | In-memory substring store is **not** used for golden accuracy. Binding follows `mg/store-model-binding@STABLE`. |
| **Metric helpers** | Pure recall@k logic is unit-tested **without** FastEmbed | Always run in default `cargo test` / `./scripts/check.sh`. |
| **Full embed golden** | Integration test loads fixture, validates hygiene, ingests, retrieve@k, asserts floor | `#[ignore]` in default CI; operators **must** run `--ignored` before merge of embed/storage changes. |

### Fixture fields (v1)

| Field | Role |
|-------|------|
| `baseline_mean_recall_at_k` | Pinned green mean from an authoritative run |
| `min_mean_recall_at_k` | Pass floor (≥ baseline × 0.98) |
| `model_id` | Catalog ID (`all-minilm-l6-v2`) |
| `k` | Top-k (5) |

### Artifacts

- `tests/fixtures/golden_corpus.json`
- `tests/golden_recall.rs` — sqlite-vec path
- `src/eval/` — pure metrics

---

## Aspirational perf goals (not Wave C ship blockers)

| Goal | Draft target | Context |
|------|--------------|---------|
| Warm single retrieve | p95 **< 50 ms** @ ~10k docs | CPU, cached query embed |
| Batch ingest | ≥ **10×** vs sequential @ 100 items | Wave B batch path |
| Bench honesty | Document **cold vs warm** retrieve | Query embed LRU |

Do not cite unpublished bench numbers as release guarantees.

---

## Local gates (required before merge)

```bash
cd memory-gate-rs
./scripts/check.sh
```

For accuracy-sensitive changes, also:

```bash
cargo test --features sqlite-vec --test golden_recall -- --ignored --nocapture
```

---

## Explicit non-goals (Wave C)

- **tero-rs** / **tero-mcp** crate wiring or MCP binary (WS-14) — see [TERO_INTEGRATION.md](TERO_INTEGRATION.md)
- Python golden ownership or new Python product features
- Shipping without a recorded baseline (baseline is in the fixture)

---

## References

- [README.md](../README.md)
- [TERO_INTEGRATION.md](TERO_INTEGRATION.md)
