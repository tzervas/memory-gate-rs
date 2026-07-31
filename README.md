# memory-gate-rs

<!-- FLEET-BADGES:BEGIN -->
[![CI](https://github.com/tzervas/memory-gate-rs/actions/workflows/fleet-ci.yml/badge.svg?branch=main)](https://github.com/tzervas/memory-gate-rs/actions/workflows/fleet-ci.yml?query=branch%3Amain)
[![Security](https://github.com/tzervas/memory-gate-rs/actions/workflows/fleet-security.yml/badge.svg?branch=main)](https://github.com/tzervas/memory-gate-rs/actions/workflows/fleet-security.yml?query=branch%3Amain)
<!-- FLEET-BADGES:END -->

[![Crates.io](https://img.shields.io/crates/v/memory-gate-rs.svg)](https://crates.io/crates/memory-gate-rs)
[![Documentation](https://docs.rs/memory-gate-rs/badge.svg)](https://docs.rs/memory-gate-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Rust **memory learning layer** for AI agents: learn from interactions, retrieve relevant past context, consolidate low-value memories. Dual-stream design (fast store + background consolidation) inspired by Complementary Learning Systems.

> **Honesty note (measured 2026-07-25):** the crates.io and docs.rs badges above are **not** backed by a published crate yet (`memory-gate-rs` was not on crates.io; docs.rs returned 404). Prefer building from git. Fleet CI/security badges track this repository. Details: [docs/CURRENT-STATE.md](docs/CURRENT-STATE.md).

## Who it is for

- Agents that need **persistent, domain-scoped** operational memory beyond a single session.
- Integrators (e.g. tero-rs) that want a thin public facade (`join/mg-facade`) over a gateway + pluggable store.

## Status (one screen)

| | |
|--|--|
| Default path (in-memory learn/retrieve/consolidate) | Works — see Quick start |
| Vector backends (`sqlite-vec`, `qdrant`) | Compile with features; full semantic tests mostly `#[ignore]` (model download / live Qdrant) |
| Package version in tree | `1.0.0` (fleet policy prefers 0.x; see roadmap) |
| Depth docs | [docs/](docs/README.md) |

## Quick start (under one minute if Rust is installed)

**Requirements:** Rust **1.92+** (MSRV in `Cargo.toml`; measured with stable 1.97 / nightly 1.98).

```bash
git clone https://github.com/tzervas/memory-gate-rs.git
cd memory-gate-rs
export CARGO_BUILD_JOBS=3   # optional; polite on shared builders
cargo run --example basic_usage
```

Measured on 2026-07-25 at `b9881f3`: example completed (learn → retrieve “CPU” → consolidate).

### Depend from another crate (git — registry not published)

```toml
[dependencies]
memory-gate-rs = { git = "https://github.com/tzervas/memory-gate-rs", branch = "main" }
# Optional: features = ["sqlite-vec"] or ["qdrant", "metrics"]
```

### Minimal API (default `in-memory` feature)

```rust
use memory_gate_rs::{
    MemoryGateway, GatewayConfig, LearningContext, AgentDomain,
    adapters::PassthroughAdapter,
    storage::InMemoryStore,
};

#[tokio::main]
async fn main() -> memory_gate_rs::Result<()> {
    let store = InMemoryStore::new();
    let adapter = PassthroughAdapter;
    let gateway = MemoryGateway::new(adapter, store, GatewayConfig::default());

    let context = LearningContext::new(
        "Resolved high CPU by restarting nginx service",
        AgentDomain::Infrastructure,
    );
    gateway.learn_from_interaction(context, Some(0.9)).await?;

    // limit is Option<usize> — None uses GatewayConfig::retrieval_limit
    let memories = gateway
        .retrieve_context("CPU usage issues", Some(5), Some(AgentDomain::Infrastructure))
        .await?;

    for memory in memories {
        println!("Recalled: {}", memory.content);
    }
    Ok(())
}
```

This shape matches the library doctest and `examples/basic_usage.rs` (both verified in the current-state measurement). In-memory retrieval is **not** embedding-based; it is suitable for tests and demos. For semantic search enable `sqlite-vec` or `qdrant` and an embedding model (see below).

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `in-memory` | ✓ | HashMap store |
| `qdrant` | | Qdrant + FastEmbed |
| `sqlite-vec` | | SQLite + FastEmbed |
| `metrics` | | Prometheus helpers |
| `vsa-native` | | Pure Rust VSA extras in `full` |
| `vsa-accel` | | **Experimental** `trit-vsa` (not in `full`; see [docs/DEBT.md](docs/DEBT.md)) |
| `full` | | `qdrant` + `sqlite-vec` + `metrics` + `vsa-native` |

## Local checks

```bash
export CARGO_BUILD_JOBS=3
./scripts/check.sh
# fmt --check, clippy -D warnings, doc -D warnings, build, test --all-features
```

Optional golden recall (downloads embedding weights; not part of default check):

```bash
cargo test --features sqlite-vec --test golden_recall -- --ignored --nocapture
```

Criteria: [docs/WAVE_C_ACCEPTANCE.md](docs/WAVE_C_ACCEPTANCE.md).

## Core surface (map only)

| Piece | Role |
|-------|------|
| `MemoryGateway` | Learn, retrieve, consolidate |
| `LearningContext` / `AgentDomain` | Memory unit + domain filter (incl. M1: Tero, Context, Lang*, …) |
| `KnowledgeStore` / `MemoryAdapter` | Pluggable storage and transform |
| `facade` | tero-oriented helpers (`for_tero_learn`, `learn`, `retrieve`, …) |
| `SupportedEmbeddingModel` | Shared catalog for vector backends |
| `vsa::*` | Holographic / VSA memory (pure Rust) |

**Embedding models** (vector features): `all-minilm-l6-v2` (384), `bge-small-en-v1.5` (384, default), `bge-base-en-v1.5` (768). **One model per SQLite file or Qdrant collection** — stores fail closed on mismatch ([ADR 0001](docs/adr/0001-fail-closed-embedding-model-binding.md)).

## Project docs

| Doc | Contents |
|-----|----------|
| [docs/README.md](docs/README.md) | Index |
| [docs/CURRENT-STATE.md](docs/CURRENT-STATE.md) | Measured VERIFIED / UNVERIFIED matrix |
| [docs/DEVELOPMENT-PATH.md](docs/DEVELOPMENT-PATH.md) | History and decisions |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Forward work + blockers |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contributor setup |
| [AGENTS.md](AGENTS.md) | Agent / tero workflow notes |

## Versioning

Conventional Commits + Commitizen (`.cz.toml`). Until a human authorizes otherwise, fleet policy is **0.x.x** (see BRANCH-AND-RELEASE-CONTRACT). This tree may still show a `1.0.x` label historically — treat that as a labeling issue, not crates.io maturity.

## License

MIT — [LICENSE-MIT](LICENSE-MIT).
