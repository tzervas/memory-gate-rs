# memory-gate-rs — Tero Index (Layer 1)

> **Honesty:** Empirical/Declared — lite heading/line heuristic over markdown in memory-gate-rs via tero-mcp/scripts/generate_lite_index.py; source files are ground truth. Generated 2026-07-10.
> Use this index to find where to Read, not as authoritative ground truth.

- **Items:** 79
- **Flagged:** 0
- **item_tag:** `Empirical/Declared`
- **Machine index:** [`index.json`](./index.json)
- **Manifest:** [`MANIFEST.toml`](./MANIFEST.toml)

## doc (63 entries)

| Anchor | Kind | Id | Title | File:Line | Status | Summary |
|---|---|---|---|---|---|---|
| `agents` | other | — | AGENTS.md — memory-gate-rs | `AGENTS.md:2` | — | Use Tero + cabal-devmelopner for work here. |
| `agents--tero-layer-1-corpus-index` | section | — | Tero (Layer-1 corpus index) | `AGENTS.md:8` | — | Repo has docs/tero-index/index.json (generated/ refreshed via tero-mcp/scripts/generateliteindex.py). |
| `agents--agent-with-context` | other | — | agent with context: | `AGENTS.md:20` | — | uv run --project ../cabal-devmelopner cabal-devmelopner "task description here" --use-tero |
| `agents--working-with-cabal-devmelopner-agent-tool` | section | — | Working with cabal-devmelopner agent tool | `AGENTS.md:26` | — | This project is prepared for integration: |
| `agents--local-checks` | section | — | Local checks | `AGENTS.md:38` | — | Look for: |
| `agents--further-reading` | section | — | Further reading | `AGENTS.md:47` | — | - README.md |
| `agents--latest-m1-w2-facade-2026-07-09-pr-26` | section | — | Latest (M1 + W2 Facade, 2026-07-09 PR #26) | `AGENTS.md:57` | — | M1 domains + facade executed on feature/mint-m1-domain-facade. AgentDomain extended (Workspace, Tero, Context, MemoryGate, LangRust, LangPython + prefix-aware… |
| `agents--post-26-merge-propagate-2026-07-09` | section | — | Post #26 merge + propagate (2026-07-09) | `AGENTS.md:62` | — | - Merged to dev @4fb2c40 (PR#26 feature/mint-m1-domain-facade). pr-review (adapted M1/tero/W2) + gh verification comment posted. |
| `agents--w2-facade-evolution-commonmemory-mirror-stub-chore-w2-rollout-docs-wiring` | section | — | W2 Facade Evolution + CommonMemory Mirror Stub (chore/w2-rollout-docs-wiring) | `AGENTS.md:69` | — | Per plan.md w2-rollout + dev-docs/schemas/ (read: W2-STRUCTURED-SCHEMAS.md, commonmemoryfacade.rs.example, cabal schemas/agent current impl, memory types.rs M1… |
| `agents--w2-code-wiring-chore-w2-code-wiring-facade` | section | — | W2 code wiring (chore/w2-code-wiring-facade) | `AGENTS.md:82` | — | - CommonMemory trait landed in src/traits.rs (pub via lib/prelude); compiles (cargo check). |
| `contributing` | section | — | Contributing to memory-gate-rs | `CONTRIBUTING.md:1` | — | Thank you for your interest in contributing to memory-gate-rs! This document provides |
| `contributing--code-of-conduct` | section | — | Code of Conduct | `CONTRIBUTING.md:6` | — | This project follows the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct). |
| `contributing--getting-started` | section | — | Getting Started | `CONTRIBUTING.md:11` | — | - Rust 1.92 or later (check with rustc --version) |
| `contributing--prerequisites` | section | — | Prerequisites | `CONTRIBUTING.md:13` | — | - Rust 1.92 or later (check with rustc --version) |
| `contributing--development-setup` | section | — | Development Setup | `CONTRIBUTING.md:19` | — | git clone https://github.com/tzervas/memory-gate-rs.git |
| `contributing--clone-the-repository` | other | — | Clone the repository | `CONTRIBUTING.md:22` | — | git clone https://github.com/tzervas/memory-gate-rs.git |
| `contributing--build-with-all-features` | other | — | Build with all features | `CONTRIBUTING.md:26` | — | cargo build --all-features |
| `contributing--run-tests` | other | — | Run tests | `CONTRIBUTING.md:29` | — | cargo test --all-features |
| `contributing--run-clippy-we-enforce-clippypedantic` | other | — | Run clippy (we enforce clippy::pedantic) | `CONTRIBUTING.md:32` | — | cargo clippy --all-features |
| `contributing--build-documentation` | other | — | Build documentation | `CONTRIBUTING.md:35` | — | cargo doc --all-features --no-deps --open |
| `contributing--development-guidelines` | section | — | Development Guidelines | `CONTRIBUTING.md:39` | — | We use rustfmt for formatting and clippy with strict lints: |
| `contributing--code-style` | section | — | Code Style | `CONTRIBUTING.md:41` | — | We use rustfmt for formatting and clippy with strict lints: |
| `contributing--format-code` | other | — | Format code | `CONTRIBUTING.md:46` | — | cargo fmt |
| `contributing--check-lints-must-pass-with-no-warnings` | other | — | Check lints (must pass with no warnings) | `CONTRIBUTING.md:49` | — | cargo clippy --all-features |
| `contributing--documentation-requirements` | section | — | Documentation Requirements | `CONTRIBUTING.md:60` | — | Undocumented code is incomplete code. |
| `contributing--testing` | section | — | Testing | `CONTRIBUTING.md:104` | — | - Write unit tests for all new functionality |
| `contributing--commit-messages` | section | — | Commit Messages | `CONTRIBUTING.md:111` | — | Use conventional commit format: |
| `contributing--pull-requests` | section | — | Pull Requests | `CONTRIBUTING.md:130` | — | 1. Fork the repository |
| `contributing--feature-development` | section | — | Feature Development | `CONTRIBUTING.md:143` | — | 1. Create a new module in src/storage/ |
| `contributing--adding-a-new-storage-backend` | section | — | Adding a New Storage Backend | `CONTRIBUTING.md:145` | — | 1. Create a new module in src/storage/ |
| `contributing--adding-a-new-adapter` | section | — | Adding a New Adapter | `CONTRIBUTING.md:155` | — | 1. Create a new module in src/adapters/ |
| `contributing--questions` | section | — | Questions? | `CONTRIBUTING.md:162` | — | - Open an issue for bugs or feature requests |
| `contributing--license` | section | — | License | `CONTRIBUTING.md:168` | — | By contributing, you agree that your contributions will be licensed under the MIT License. |
| `readme` | other | — | memory-gate-rs | `README.md:1` | — | [![Crates.io](https://img.shields.io/crates/v/memory-gate-rs.svg)](https://crates.io/crates/memory-gate-rs) |
| `readme--overview` | section | — | Overview | `README.md:9` | — | Memory-gate provides a production-ready memory layer for AI agents, solving the fundamental problem of stateless AI by enabling: |
| `readme--architecture-complementary-learning-systems-cls` | section | — | Architecture: Complementary Learning Systems (CLS) | `README.md:18` | — | Inspired by neuroscience research on brain plasticity, memory-gate implements a dual-stream architecture: |
| `readme--installation` | section | — | Installation | `README.md:25` | — | [dependencies] |
| `readme--feature-flags` | section | — | Feature Flags | `README.md:32` | — | — |
| `readme--quick-start` | section | — | Quick Start | `README.md:42` | — | use memorygaters::{ |
| `readme--core-concepts` | section | — | Core Concepts | `README.md:80` | — | The atomic unit of memory containing learned content: |
| `readme--learningcontext` | section | — | LearningContext | `README.md:82` | — | The atomic unit of memory containing learned content: |
| `readme--agent-domains` | section | — | Agent Domains | `README.md:96` | — | Categorize memories by operational domain: |
| `readme--memory-gateway` | section | — | Memory Gateway | `README.md:122` | — | Central orchestrator for learning and memory operations: |
| `readme--consolidation` | section | — | Consolidation | `README.md:139` | — | Background process that maintains memory health: |
| `readme--storage-backends` | section | — | Storage Backends | `README.md:147` | — | Simple HashMap-based storage for testing and development: |
| `readme--in-memory-default` | section | — | In-Memory (default) | `README.md:149` | — | Simple HashMap-based storage for testing and development: |
| `readme--qdrant-feature-qdrant` | section | — | Qdrant (feature: `qdrant`) | `README.md:159` | — | Production vector database with semantic search: |
| `readme--sqlite-vector-feature-sqlite-vec` | section | — | SQLite + Vector (feature: `sqlite-vec`) | `README.md:169` | — | Embedded vector storage with SQLite: |
| `readme--metrics-feature-metrics` | section | — | Metrics (feature: `metrics`) | `README.md:179` | — | Prometheus-compatible metrics: |
| `readme--custom-adapters` | section | — | Custom Adapters | `README.md:191` | — | Implement MemoryAdapter for custom knowledge transformation: |
| `readme--custom-storage-backends` | section | — | Custom Storage Backends | `README.md:216` | — | Implement KnowledgeStore for custom storage: |
| `readme--integration-with-rust-ai-ecosystem` | section | — | Integration with rust-ai Ecosystem | `README.md:255` | — | memory-gate-rs integrates with the broader rust-ai crate ecosystem: |
| `readme--m1-domain-facade-design-mint-kickoff` | section | — | M1: Domain + Facade Design (mint kickoff) | `README.md:264` | — | Status: Executed (M1). See types::AgentDomain extensions + tero first queries. |
| `readme--license` | section | — | License | `README.md:291` | — | MIT License - see [LICENSE-MIT](LICENSE-MIT) |
| `readme--contributing` | section | — | Contributing | `README.md:295` | — | Contributions welcome! Please read CONTRIBUTING.md first. |
| `roadmap` | note | — | memory-gate-rs — Product Roadmap | `docs/ROADMAP.md:1` | Living (2026-07-09) | Status: Living (2026-07-09) |
| `roadmap--phases` | section | — | Phases | `docs/ROADMAP.md:10` | — | - Extended AgentDomain (Workspace, Tero, Context, MemoryGate, LangRust, LangPython) + prefix-aware FromStr ("layer:tero", "lang:rust", "repo:xxx"). |
| `roadmap--m1-domain-facade-design-done-2026-07-09-pr-26` | section | — | M1 — Domain + Facade Design (done, 2026-07-09 PR #26) | `docs/ROADMAP.md:12` | — | - Extended AgentDomain (Workspace, Tero, Context, MemoryGate, LangRust, LangPython) + prefix-aware FromStr ("layer:tero", "lang:rust", "repo:xxx"). |
| `roadmap--m2-pending` | section | — | M2 — (pending) | `docs/ROADMAP.md:22` | — | - TBD per cabal / wsfull waves and mint kickoffs. |
| `roadmap--hygiene-tero-facade-evolution` | section | — | Hygiene, Tero, Facade Evolution | `docs/ROADMAP.md:28` | — | - Add scripts/check.sh (fmt --check/fix, clippy -D warnings, doc -D warnings, build, test --all-features) per AGENTS local checks expectation. |
| `roadmap--w2-facade-evolution-chore-w2-rollout-docs-wiring` | section | — | W2 Facade Evolution (chore/w2-rollout-docs-wiring) | `docs/ROADMAP.md:41` | — | Append-only extension per plan.md w2-rollout (parallel safe to cabal MVP). |
| `roadmap--w2-code-wiring-chore-w2-code-wiring-facade-appended` | section | — | W2 code wiring (chore/w2-code-wiring-facade appended) | `docs/ROADMAP.md:64` | — | - CommonMemory trait (W2 mirror) implemented in src/traits.rs (pub, reexported in lib + prelude). |
| `roadmap--links` | section | — | Links | `docs/ROADMAP.md:75` | — | - AGENTS.md (Tero rule, local checks, PR flow) |

## changelog (16 entries)

| Anchor | Kind | Id | Title | File:Line | Status | Summary |
|---|---|---|---|---|---|---|
| `changelog` | entry | — | Changelog | `CHANGELOG.md:1` | — | All notable changes to this project will be documented in this file. |
| `changelog--unreleased-m1-domain-facade-mint-kickoff` | section | — | [Unreleased] - M1 domain/facade (mint kickoff) | `CHANGELOG.md:8` | — | - Extended AgentDomain with workspace integration domains: Workspace, Tero, Context, MemoryGate, LangRust, LangPython. |
| `changelog--added-m1` | section | — | Added (M1) | `CHANGELOG.md:10` | — | - Extended AgentDomain with workspace integration domains: Workspace, Tero, Context, MemoryGate, LangRust, LangPython. |
| `changelog--1.0.0-2026-01-25` | section | — | [1.0.0] - 2026-01-25 | `CHANGELOG.md:20` | — | - MemoryGateway - Central orchestrator for learning and retrieval operations |
| `changelog--added` | section | — | Added | `CHANGELOG.md:22` | — | - MemoryGateway - Central orchestrator for learning and retrieval operations |
| `changelog--core-memory-system` | section | — | Core Memory System | `CHANGELOG.md:24` | — | - MemoryGateway - Central orchestrator for learning and retrieval operations |
| `changelog--storage-backends` | section | — | Storage Backends | `CHANGELOG.md:31` | — | - InMemoryStore - HashMap-based storage for testing and development (default) |
| `changelog--trait-system` | section | — | Trait System | `CHANGELOG.md:36` | — | - KnowledgeStore - Core storage interface for any backend |
| `changelog--vector-symbolic-architecture-vsa` | section | — | Vector Symbolic Architecture (VSA) | `CHANGELOG.md:44` | — | - HolographicVector - High-dimensional vectors for compositional memory |
| `changelog--adapters` | section | — | Adapters | `CHANGELOG.md:52` | — | - PassthroughAdapter - Identity adapter for direct storage |
| `changelog--agents` | section | — | Agents | `CHANGELOG.md:55` | — | - BaseMemoryAgent - Foundation for building memory-enabled agents |
| `changelog--observability-feature-metrics` | section | — | Observability (feature: `metrics`) | `CHANGELOG.md:59` | — | - Prometheus metrics for store operations, consolidation, and retrieval |
| `changelog--changed` | section | — | Changed | `CHANGELOG.md:63` | — | - MSRV updated to Rust 1.92 |
| `changelog--experimental` | section | — | Experimental | `CHANGELOG.md:68` | — | - vsa-accel feature for rust-ai ecosystem acceleration (awaiting stable dependencies) |
| `changelog--0.1.0-2025-01-01` | section | — | [0.1.0] - 2025-01-01 | `CHANGELOG.md:72` | — | - Initial release with basic memory gateway functionality |
| `changelog--added-2` | section | — | Added | `CHANGELOG.md:74` | — | - Initial release with basic memory gateway functionality |

