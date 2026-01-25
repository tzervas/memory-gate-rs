# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
