# Wave C — Acceptance criteria

**Status:** Living (Wave C delivery)  
**Canonical runtime:** **memory-gate-rs only** — the Python `memory-gate` package is a **frozen mirror** after embedding parity (catalog + fail-closed store binding). Wave C accuracy, golden recall, and further perf work belong on Rust only.

**Interface bulletin:** `mg/golden-recall` (workspace handoff; DRAFT → STABLE when golden PR lands). This doc does not invent perf numbers; baselines are recorded in the golden fixture or updated here when L1-GOLDEN publishes them.

---

## In scope for Wave C

| Area | Criterion | Notes |
|------|-----------|--------|
| **Accuracy** | Mean **recall@5** on the fixed golden corpus must not drop **more than 2%** versus the baseline recorded in the fixture or in this doc | Per-query recall@k = \|retrieved ∩ relevant\| / \|relevant\|; report mean over queries. Fixture `model_id` must be a catalog stable ID (`mg/embed-catalog@STABLE`). |
| **Store binding** | Golden runs use one in-memory or sqlite-vec store per model; reopen/mismatch behavior follows `mg/store-model-binding@STABLE` | No mixed dimensions or silent model reuse on the same DB/collection path. |
| **Metric helpers** | Pure recall@k logic is unit-tested **without** FastEmbed | Always run in default `cargo test` / `./scripts/check.sh`. |
| **Full embed golden** | Integration test loads fixture, ingests documents, runs retrieve@k, asserts threshold | May be `#[ignore]` in CI (model download + CPU cost); operators run explicitly (see below). |

### Artifacts (L1-GOLDEN)

- `tests/fixtures/golden_corpus.json` (or equivalent under `tests/fixtures/`)
- `tests/golden_recall.rs` — sqlite-vec path, threshold vs baseline
- Eval helpers under `src/eval/` as needed (owned by golden-recall workstream)

---

## Aspirational perf goals (not Wave C ship blockers)

These come from the performance handoff and README (query embedding LRU). They guide follow-on tuning and benches; **failure to meet them does not block Wave C** if accuracy and local gates pass.

| Goal | Draft target | Context |
|------|--------------|---------|
| Warm single retrieve | p95 **< 50 ms** @ ~10k docs | CPU, `all-minilm-l6-v2` (or agreed golden model), **cached** query embed (repeat query text + model id) |
| Batch ingest | ≥ **10×** throughput vs sequential single upsert @ 100 items | Wave B batch path; measure locally with Criterion / custom harness |
| Bench honesty | Document **cold vs warm** retrieve when reporting latency | First query per key pays embed cost; repeats hit LRU (see `vector_storage_benchmarks` / reviews) |

Do not cite unpublished bench numbers as release guarantees.

---

## Local gates (required before merge)

From [AGENTS.md](../AGENTS.md) and [CONTRIBUTING.md](../CONTRIBUTING.md):

```bash
cd memory-gate-rs
./scripts/check.sh
```

`check.sh` runs: `fmt --check`, `clippy -D warnings`, `doc -D warnings`, `build --all-features`, `test --all-features`. This is the **required** pre-merge gate for Wave C doc + code delivery on this repo.

Optional fix mode: `./scripts/check.sh --fix` (applies `cargo fmt`).

---

## Golden recall test (embedding integration)

Default CI may skip full embed runs. After L1-GOLDEN lands, run the ignored integration test locally (downloads embedding weights on first run):

```bash
cd memory-gate-rs
cargo test --features sqlite-vec --test golden_recall -- --ignored --nocapture
```

Pin the fixture `model_id` explicitly (golden suite uses **`all-minilm-l6-v2`** for cross-port parity discipline; vector default `BgeSmallEnV15` remains the RS `new()`/`open()` default for backward compatibility).

---

## Explicit non-goals (Wave C)

- **tero-rs** / **tero-mcp** crate wiring or MCP binary (WS-14) — see [TERO_INTEGRATION.md](TERO_INTEGRATION.md)
- Python golden ownership or new Python product features
- Publishing false or placeholder recall@5 / latency SLOs without a recorded baseline

---

## References

- [README.md](../README.md) — vector backends, embedding catalog, benchmarks
- [TERO_INTEGRATION.md](TERO_INTEGRATION.md) — post-Wave C integration sketch
- Workspace: `work/bulletins/mg-golden-recall.md`, `work/memory-gate-wave-status.md`, `work/memory-gate-performance-handoff.md`