# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - M1 domain/facade (mint kickoff)

### Added (docs — Wave C)
- `docs/WAVE_C_ACCEPTANCE.md` — recall@5 ≤2% relative drop vs recorded fixture baseline (`baseline_mean_recall_at_k` / `min_mean_recall_at_k`), `./scripts/check.sh` + optional `--ignored` golden, aspirational perf non-blocking.
- `docs/TERO_INTEGRATION.md` — post–Wave C follow-on: tero L1 vs memory-gate-rs; **prerequisite: purge unnecessary mycelium crates from tero-rs** before integration; tero-rs/tero-mcp named as next-wave targets (not implemented in Wave C).
- README links: Accuracy (Wave C) + Integration roadmap.

### Fixed
- **Store model binding (`mg/store-model-binding`)**: Qdrant and sqlite-vec backends fail closed on embedding dimension mismatch when reopening a collection/DB; sqlite-vec persists `store_meta` (`embedding_model`, `embedding_dim`) and rejects legacy DBs that already contain rows without metadata. Qdrant stamps collection metadata on create and validates vector size (and model metadata when present). SQLite fallback search errors on stored/query vector length mismatch instead of returning zero similarity.

### Added
- Golden recall harness: `tests/fixtures/golden_corpus.json` (baseline + min floor), `eval::recall_at_k` / `mean_recall`, ignored `golden_recall` integration test for `sqlite-vec` (`mg/golden-recall@STABLE`).
- Batch embedding (`embed_batch`) and `BatchKnowledgeStore::store_batch` on `QdrantStore` and `SqliteVecStore` (single FastEmbed lock per batch).
- Bounded LRU query embedding cache (256 entries, keyed by query text + model id) on vector backends for `retrieve_context`.
- Multi-model embedding catalog (`SupportedEmbeddingModel`) for `qdrant` and `sqlite-vec` backends: `all-minilm-l6-v2`, `bge-small-en-v1.5` (default), `bge-base-en-v1.5`; stable IDs aligned with Python `memory-gate` (`mg/embed-catalog@STABLE`).
- `QdrantStore::with_model` and `SqliteVecStore::open_with_model` / `open_in_memory_with_model`; `new()` / `open()` remain default BGE-small for backward compatibility.
- `SupportedEmbeddingModel::parse`, `dimension()`, `sentence_transformers_id()`, and feature-gated `embedding::init_text_embedding`.
- Criterion baselines for `sqlite-vec` (`vector_storage_benchmarks`); default `storage_benchmarks` unchanged.

### Added (M1)
- Extended `AgentDomain` with workspace integration domains: `Workspace`, `Tero`, `Context`, `MemoryGate`, `LangRust`, `LangPython`.
- Prefix-aware `FromStr` (e.g. "layer:tero", "lang:rust", "repo:foo") for unified scoping/facade.
- Updated `all()`, `as_str()`, VSA encoding, tests, README (tero-cited `readme--agent-domains`).
- Facade design documented: domain-filtered retrieve + ingest from tero/context for tero-first + gate scoping + no-bloat. Thin glue via existing API + MCP/scripts. Shared schemas escalated to wsfull.
- Tero-first: queries on "domain","memory","gate"; cited + read_file on paths.
- Dev-workflow: branch feature/mint-m1-..., cargo test green, append-only changelog.

See mint.md M1, memory-gate-rs README M1 section. Status: M1 complete; M2 pending.

## [1.0.0] - 2026-01-25

### Added

#### Core Memory System
- `MemoryGateway` - Central orchestrator for learning and retrieval operations
- `LearningContext` - Atomic unit of memory with domain, importance, and metadata
- `AgentDomain` - Categorical filtering for multi-domain deployments
- `GatewayConfig` - Comprehensive configuration for memory behavior
- Background consolidation worker with configurable intervals

#### Storage Backends
- `InMemoryStore` - HashMap-based storage for testing and development (default)
- `QdrantStore` - Production-grade vector database integration (feature: `qdrant`)
- `SqliteVecStore` - Embedded SQLite with vector extension (feature: `sqlite-vec`)

#### Trait System
- `KnowledgeStore` - Core storage interface for any backend
- `MemoryAdapter` - Pluggable transformation pipeline for memories
- `VectorStore` - Extended interface for vector-based retrieval
- `BatchKnowledgeStore` - Bulk operations for high-throughput scenarios
- `FilterableStore` - Domain and metadata filtering capabilities
- `MemoryEnabledAgent` - Interface for agents with memory integration

#### Vector Symbolic Architecture (VSA)
- `HolographicVector` - High-dimensional vectors for compositional memory
- `VsaCodebook` - Symbol-to-vector mapping with deterministic generation
- `HolographicStore` - VSA-based knowledge storage with holographic index
- `MapVsaOps` - Configurable VSA operations (bind, bundle, permute)
- `ResonatorNetwork` - Factorization for recovering bound components
- Support for bipolar, binary, and dense vector encodings

#### Adapters
- `PassthroughAdapter` - Identity adapter for direct storage

#### Agents
- `BaseMemoryAgent` - Foundation for building memory-enabled agents
- `EnhancedContext` - Task context enriched with retrieved memories

#### Observability (feature: `metrics`)
- Prometheus metrics for store operations, consolidation, and retrieval
- `record_store_operation`, `record_retrieval`, `record_consolidation` helpers

### Changed
- MSRV updated to Rust 1.92
- Comprehensive Google-style documentation with "why" explanations
- All public APIs are fully documented

### Experimental
- `vsa-accel` feature for rust-ai ecosystem acceleration (awaiting stable dependencies)
  - Integration with `trit-vsa` crate

## [0.1.0] - 2025-01-01

### Added
- Initial release with basic memory gateway functionality
- In-memory storage backend
- Core trait definitions

[1.0.0]: https://github.com/tzervas/memory-gate-rs/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/tzervas/memory-gate-rs/releases/tag/v0.1.0
