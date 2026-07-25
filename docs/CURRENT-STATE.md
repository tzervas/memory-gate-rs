# Current state — memory-gate-rs

**Measured at:** 2026-07-25 (UTC)  
**Commit:** `b9881f3d7bbea81d0e9ff98f2a9593001bcfa01d` (`main` tip at measurement)  
**Package version in tree:** `1.0.0` (see [Versioning notes](#versioning-notes))  
**Toolchain used for checks:**

| Tool | Version |
|------|---------|
| Host `rustc` (default) | `1.98.0-nightly (4c9d2bfe4 2026-07-01)` |
| Host `cargo` (default) | `1.98.0-nightly (a335d47ff 2026-06-26)` |
| `rustc` stable (fmt/clippy/doc) | `1.97.0 (2d8144b78 2026-07-07)` |
| `cargo` stable | `1.97.0 (c980f4866 2026-06-30)` |

**Resource bound:** `CARGO_BUILD_JOBS=3` for all builds/tests.

This document is the measured back-layer answer to “what works today?”  
If a capability was not exercised, it is marked **UNVERIFIED**, not assumed.

---

## Capability matrix

| Capability | Status | Evidence |
|------------|--------|----------|
| Default build (`in-memory`) | **VERIFIED** | `cargo build` → Finished in ~14s |
| All-features build (`qdrant` + `sqlite-vec` + `metrics` + `vsa-native`; note: not `vsa-accel`) | **VERIFIED** | `cargo build --all-features` → Finished in ~2m 18s |
| Default unit + integration tests | **VERIFIED** | 78 lib/integration tests + 7 doctests passed (3 doctests ignored) |
| All-features unit + integration tests (non-ignored) | **VERIFIED** | **113 passed**, **0 failed**, **26 ignored** |
| `cargo +stable fmt --check` | **VERIFIED** | exit 0 |
| `cargo +stable clippy --all-targets -- -D warnings` | **VERIFIED** | exit 0 |
| `cargo +stable clippy --all-targets --all-features -- -D warnings` | **VERIFIED** | exit 0 |
| `RUSTDOCFLAGS=-D warnings cargo +stable doc --all-features --no-deps` | **VERIFIED** | exit 0; docs generated |
| Example `basic_usage` (in-memory learn/retrieve/consolidate) | **VERIFIED** | `cargo run --example basic_usage` completed with 2 CPU hits, consolidation stats |
| Gateway learn + retrieve + domain filter (in-memory) | **VERIFIED** | gateway/storage/consolidation tests + example |
| `AgentDomain` M1 variants + prefix parse (`layer:tero`, `lang:rust`, …) | **VERIFIED** | `types::tests::test_agent_domain_parsing` passes under both feature sets |
| Pure VSA (`vsa-native` / default VSA modules) | **VERIFIED** | holographic_store / codebook / vector / ops unit tests pass |
| Eval helpers (`eval::recall_at_k`, mean recall) | **VERIFIED** | unit tests in `src/eval/mod.rs` run in default `cargo test` |
| Facade module compiles (`join/mg-facade` helpers re-exported) | **VERIFIED** | all-features build + facade code in library; thin wrappers over gateway |
| Facade end-to-end with real tero L1 index ingest | **UNVERIFIED** | no integration test exercises live tero MCP/index → `for_tero_learn` path in this run |
| sqlite-vec store open / meta fail-closed (no model download) | **VERIFIED** | several non-`#[ignore]` sqlite-vec tests pass (open, meta stamp/reconcile, dim mismatch) |
| sqlite-vec semantic store/retrieve (FastEmbed model download) | **UNVERIFIED** | 11 sqlite-vec tests `#[ignore]` — “requires embedding model download”; not run here |
| Wave C golden recall@5 (`tests/golden_recall.rs`) | **UNVERIFIED** | test is `#[ignore]`; requires model download; not run in this measurement |
| Qdrant backend against a live server | **UNVERIFIED** | 5 Qdrant tests `#[ignore]`; no Qdrant at measurement; no live connection attempt |
| Prometheus metrics export (`metrics` feature) | **UNVERIFIED** | feature compiles (all-features build); **no test exercises** scrape/export path in this run |
| `vsa-accel` (`trit-vsa` acceleration) | **UNVERIFIED** | optional feature; **not** in `full` feature set; not built or tested here; see `docs/DEBT.md` DEBT-001 |
| Batch encode / query embed cache on vector backends | **UNVERIFIED** | code present (Wave B, PR #34); exercised only if ignored embed tests/benches run with models |
| Criterion benches (actual timings) | **UNVERIFIED** | `storage_benchmarks` **compiled** with `--no-run` only; no timing numbers collected |
| crates.io publish of `memory-gate-rs` | **FALSE CLAIM** | API: `crate memory-gate-rs does not exist` (2026-07-25) |
| docs.rs pages for this crate | **FALSE CLAIM** | `https://docs.rs/memory-gate-rs` → HTTP 404 |
| Fleet CI on `main` @ measured SHA | **VERIFIED (fleet-ci success)** | run success at head `b9881f3` |
| Legacy `CI` workflow on `main` @ measured SHA | **FAILING** | `Test (beta)` job **Build** step failed (see CI section) |

---

## How this was measured

### Commands run (exact)

```bash
export CARGO_BUILD_JOBS=3

# Environment
rustc --version
cargo --version
rustup run stable rustc --version
cargo +stable --version
git rev-parse HEAD
date -u +%Y-%m-%dT%H:%M:%SZ

# Build
cargo build
cargo build --all-features

# Test
cargo test
cargo test --all-features

# Lint / docs (stable, matches scripts/check.sh toolchain preference)
cargo +stable fmt --check
cargo +stable clippy --all-targets -- -D warnings
cargo +stable clippy --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo +stable doc --all-features --no-deps

# Runnable example
cargo run --example basic_usage

# Bench compile-only
cargo +stable bench --bench storage_benchmarks --no-run

# Registry / docs claims
curl -sL -A "memory-gate-rs-docs-check/1.0" \
  "https://crates.io/api/v1/crates/memory-gate-rs"
curl -sI -A "memory-gate-rs-docs-check/1.0" "https://docs.rs/memory-gate-rs"

# CI
gh api /repos/tzervas/memory-gate-rs/actions/runs?per_page=10
gh api /repos/tzervas/memory-gate-rs/actions/runs?per_page=5\&branch=main
```

`./scripts/check.sh` was **not** invoked as a single script in this run (it would re-run the same stable fmt/clippy/doc/build/test chain). Equivalent steps above were run piecewise and all passed on the host toolchain set.

Golden / model-download paths were **not** run:

```bash
# NOT run in this measurement (model download cost + ignore flags)
cargo test --features sqlite-vec --test golden_recall -- --ignored --nocapture
cargo test --all-features -- --ignored
```

### Real output (trimmed)

**Default build**

```text
   Compiling memory-gate-rs v1.0.0 (.../memory-gate-rs)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 14.04s
```

**All-features build**

```text
   Compiling memory-gate-rs v1.0.0 (.../memory-gate-rs)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 2m 18s
```

**Default `cargo test` (suite totals)**

```text
     Running unittests src/lib.rs
test result: ok. 62 passed; 0 failed; 0 ignored; ...

     Running tests/consolidation_tests.rs
test result: ok. 4 passed; 0 failed; 0 ignored; ...

     Running tests/gateway_tests.rs
test result: ok. 5 passed; 0 failed; 0 ignored; ...

     Running tests/golden_recall.rs
test result: ok. 0 passed; 0 failed; 0 ignored; ...
  # (no tests compiled without sqlite-vec)

     Running tests/storage_tests.rs
test result: ok. 7 passed; 0 failed; 0 ignored; ...

   Doc-tests memory_gate_rs
test result: ok. 7 passed; 0 failed; 3 ignored; ...
```

**Sum (default):** 78 non-doc tests + 7 doctests passed; 3 doctests ignored.  
(62+4+5+0+7 = 78 unit/integration.)

**All-features `cargo test` (suite totals)**

```text
     Running unittests src/lib.rs
test result: ok. 90 passed; 0 failed; 16 ignored; finished in 6.76s

     Running tests/consolidation_tests.rs
test result: ok. 4 passed; 0 failed; 0 ignored; ...

     Running tests/gateway_tests.rs
test result: ok. 5 passed; 0 failed; 0 ignored; ...

     Running tests/golden_recall.rs
test result: ok. 0 passed; 0 failed; 1 ignored; ...
  # golden_corpus_mean_recall_at_k_sqlite_vec ignored (model download)

     Running tests/storage_tests.rs
test result: ok. 7 passed; 0 failed; 0 ignored; ...

   Doc-tests memory_gate_rs
test result: ok. 7 passed; 0 failed; 9 ignored; ...
```

**Sum (all-features):** **113 passed**, **0 failed**, **26 ignored** across 6 suites.

**Ignored tests (all-features) — incomplete coverage, not failures**

- Qdrant live: `storage::qdrant::tests::test_{count_and_keys,delete_and_get,domain_filtering,store_and_retrieve,upsert_overwrites}`
- sqlite-vec + FastEmbed: `storage::sqlite_vec::tests::test_{clear,delete,domain_filter,embedding_dimension,get_all_keys,get_nonexistent,metadata_preservation,open_with_minilm_model,store_and_get,store_and_retrieve,upsert_overwrites}`
- Golden: `golden_corpus_mean_recall_at_k_sqlite_vec`
- Several doctests that need network services or heavy deps (Qdrant/sqlite open examples, custom adapter sketch, VSA module example, agent example)

**Example `basic_usage`**

```text
=== Memory Gate Basic Usage Example ===

Learning from interactions...

Retrieving context for 'CPU'...

Found 2 relevant memories:
  1. [importance: 0.95] High CPU usage resolved by restarting nginx service
  2. [importance: 0.90] Horizontal pod autoscaler triggered at 80% CPU threshold

Retrieving Infrastructure-specific context...
Found 1 infrastructure memories

Running memory consolidation...
Consolidation complete: 3 processed, 0 deleted

=== Example Complete ===
```

**Fmt / clippy / doc / bench compile**

```text
# cargo +stable fmt --check
EXIT:0

# cargo +stable clippy --all-targets -- -D warnings
Finished `dev` profile ... in 49.45s
EXIT:0

# cargo +stable clippy --all-targets --all-features -- -D warnings
Finished `dev` profile ... in 59.20s
EXIT:0

# RUSTDOCFLAGS=-D warnings cargo +stable doc --all-features --no-deps
 Documenting memory-gate-rs v1.0.0 (...)
 Finished `dev` profile ... in 1m 09s
 Generated .../target/doc/memory_gate_rs/index.html
EXIT:0

# cargo +stable bench --bench storage_benchmarks --no-run
 Finished `bench` profile [optimized] target(s) in 21.48s
  Executable benches/storage_benchmarks.rs (...)
EXIT:0
```

**crates.io / docs.rs**

```text
{"errors":[{"detail":"crate `memory-gate-rs` does not exist"}]}

# docs.rs HEAD
HTTP/2 404
```

---

## CI status (GitHub Actions)

Queried with `gh api /repos/tzervas/memory-gate-rs/actions/runs?per_page=10` and branch-filtered main runs (2026-07-25).

### Recent runs (sample)

| Workflow | Branch / SHA | Conclusion | Notes |
|----------|--------------|------------|-------|
| fleet-security | `chore/semver-0x-compliance` `9711f78` | success | open PR #53 |
| fleet-ci | same | queued (at query time) | open PR |
| Commitizen | same | success | open PR |
| CI (legacy) | `ci/oss-self-hosted-tools` `6bd2225` | failure | open PR #52 |
| fleet-ci | same | success | open PR #52 |
| fleet-ci | `main` `b9881f3` | **success** | measured tip |
| fleet-security | `main` `b9881f3` | **success** | measured tip |
| Commitizen | `main` `b9881f3` | **success** | measured tip |
| **CI** (legacy `.github/workflows/ci.yml`) | `main` `b9881f3` | **failure** | `Test (beta)` → Build failed |

### Known CI defect (observed, not fixed in this docs work)

On `main` @ `b9881f3`, workflow **CI** fails:

- Job: `Test (beta)`
- Failed step: **Build**
- Run: https://github.com/tzervas/memory-gate-rs/actions/runs/29865297829  
- Other jobs on that run (Rustfmt, Clippy, Test stable, Documentation, Coverage) reported success.

**fleet-ci** / **fleet-security** on the same SHA succeeded. Treat “green on fleet” as the intended trunk gate; the legacy `CI` beta matrix is a **known red** at measurement time.

---

## Versioning notes

| Signal | Value |
|--------|--------|
| `Cargo.toml` / `.cz.toml` version | `1.0.0` |
| Git tags | `v1.0.0`, `v1.0.1` |
| GitHub Release | `v1.0.1 — Semver Baseline (M1 + W2 CommonMemory)` published 2026-07-10 |
| crates.io | **not published** |
| Fleet / branch contract | repos stay **0.x.x** until a human authorizes 1.x.x |
| Open compliance PR | [#53](https://github.com/tzervas/memory-gate-rs/pull/53) renumbers `1.0.0` → `0.2.0` (not merged at measurement) |

**Honest reading:** this tree currently **labels** itself 1.0.0 and has GitHub release tags in the 1.x line, but is **not** a registry-published 1.x product. Under the fleet branch/release contract, that is a policy gap, not proof of production maturity. Do not treat the number as a stability guarantee.

---

## What the library actually provides today (code surface)

Public modules (from `src/lib.rs`): `adapters`, `agents`, `embedding`, `eval`, `facade`, `metrics`, `storage`, `vsa`, plus core types/gateway/traits.

Feature flags (`Cargo.toml`):

| Feature | Default | In `full` | Role |
|---------|---------|-----------|------|
| `in-memory` | yes | (via default) | HashMap store |
| `qdrant` | no | yes | Qdrant + FastEmbed |
| `sqlite-vec` | no | yes | SQLite + FastEmbed |
| `metrics` | no | yes | Prometheus helpers |
| `vsa-native` | no | yes | Pure Rust VSA (modules also compile without the flag for core VSA types) |
| `vsa-accel` | no | **no** | EXPERIMENTAL `trit-vsa` 0.3.0 |

Storage behavior (from tests + source, not marketing):

- **InMemoryStore**: substring / simple ranking retrieval — good for tests and demos; **not** semantic vector search.
- **SqliteVecStore / QdrantStore**: embedding-backed vector retrieval when models download and (for Qdrant) a server is available — largely behind `#[ignore]` in CI-friendly defaults.

---

## Known defects and documentation falsehoods

| Item | Kind | Evidence |
|------|------|----------|
| README Quick Start called `retrieve_context(..., 5, ...)` with bare `usize` | **Stale API example** | Real signature is `limit: Option<usize>` (`src/gateway.rs`); lib doctest and `examples/basic_usage.rs` use `Some(5)`. Fixed in the PM-suite README pass if this doc lands with it. |
| README install `memory-gate-rs = "0.1"` | **Stale / false** | Tree is `1.0.0`; crates.io has no crate at all. |
| Crates.io + docs.rs badges | **False** | API 404 / “does not exist” |
| “Production-ready” tone in older README prose | **Overclaim** | Vector backends and golden accuracy paths are mostly ignored/unverified without model download + services |
| Legacy `CI` beta build red on `main` | **Defect** | Actions run above |
| REUSE path coverage residual | **Open issue #42** | `REUSE-DEBT.md` + fleet-gap issue |
| `vsa-accel` pin vs sibling `trit-vsa` 0.2 | **Accepted debt** | `docs/DEBT.md` DEBT-001 |
| `docs/ROADMAP.md` pre-PM-suite | **Stub** | was essentially empty after the header (pre-this update) |

---

## Open product / process work visible from GitHub

Open at measurement (non-exhaustive):

- Issue **#42** — REUSE path coverage debt  
- PR **#53** — semver 0.x renumber  
- PR **#52** — CI OSS self-hosted tools  
- Dependabot PRs **#47–#50** — dependency bumps  

---

## How to re-measure

From a clean checkout at a known SHA:

```bash
export CARGO_BUILD_JOBS=3
./scripts/check.sh          # or the piecewise commands above
cargo run --example basic_usage
# optional heavy:
cargo test --features sqlite-vec --test golden_recall -- --ignored --nocapture
```

Update this file’s date, SHA, and pasted totals whenever behavior or gates change.
