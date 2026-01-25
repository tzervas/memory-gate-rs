//! # memory-gate-rs
//!
//! Dynamic Memory Learning Layer for AI agents — persistent knowledge accumulation
//! with vector retrieval and automatic consolidation.
//!
//! ## Why Memory-Gate?
//!
//! AI agents are fundamentally stateless: each interaction starts fresh without
//! learning from past experiences. Memory-gate solves this by providing:
//!
//! - **Persistent Knowledge**: Agents build operational expertise over time rather
//!   than repeating the same mistakes or rediscovering solutions.
//! - **Context-Aware Retrieval**: When facing a new task, agents query past similar
//!   experiences to inform their decisions — like an expert recalling relevant cases.
//! - **Automatic Consolidation**: Low-value or stale memories are pruned automatically,
//!   preventing memory bloat while retaining important learnings.
//! - **Domain Isolation**: Multi-domain deployments filter memories by domain,
//!   preventing cross-contamination between different agent responsibilities.
//!
//! ## Architecture Rationale
//!
//! The design is inspired by Complementary Learning Systems (CLS) theory from
//! cognitive neuroscience, which explains how biological brains balance rapid
//! learning with stable long-term memory:
//!
//! - **Fast System (Hippocampus-like)**: Rapidly encodes new experiences without
//!   catastrophic interference. Maps to our immediate storage path.
//! - **Slow System (Neocortex-like)**: Gradually consolidates important patterns
//!   into stable representations. Maps to our background consolidation worker.
//!
//! This dual-stream approach prevents the "stability-plasticity dilemma" where
//! systems either forget old knowledge too quickly or fail to learn new patterns.
//!
//! ## Quick Start
//!
//! ```rust
//! use memory_gate_rs::{
//!     MemoryGateway, GatewayConfig, LearningContext, AgentDomain,
//!     adapters::PassthroughAdapter,
//!     storage::InMemoryStore,
//! };
//!
//! # tokio_test::block_on(async {
//! // Create storage and adapter
//! let store = InMemoryStore::new();
//! let adapter = PassthroughAdapter;
//!
//! // Initialize gateway — the central coordinator for memory operations
//! let gateway = MemoryGateway::new(adapter, store, GatewayConfig::default());
//!
//! // Learn from an interaction — this persists the knowledge
//! let context = LearningContext::new(
//!     "Resolved high CPU by restarting nginx service",
//!     AgentDomain::Infrastructure,
//! );
//! gateway.learn_from_interaction(context, Some(0.9)).await.unwrap();
//!
//! // Retrieve relevant memories for a new similar situation
//! let memories = gateway
//!     .retrieve_context("CPU issues", Some(5), Some(AgentDomain::Infrastructure))
//!     .await
//!     .unwrap();
//!
//! for memory in memories {
//!     println!("Recalled: {}", memory.content);
//! }
//! # });
//! ```
//!
//! ## Feature Flags
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `in-memory` | In-memory HashMap-based storage (default, for testing/development) |
//! | `qdrant` | Qdrant vector database backend (production-grade vector search) |
//! | `sqlite-vec` | `SQLite` with vector extension (embedded vector database) |
//! | `metrics` | Prometheus metrics export (observability) |
//! | `vsa-native` | Pure Rust VSA/holographic memory (no external dependencies) |
//! | `vsa-accel` | **EXPERIMENTAL**: Accelerated VSA via rust-ai ecosystem crates |
//! | `full` | All stable features enabled |
//!
//! ### Experimental Features
//!
//! The `vsa-accel` feature depends on external rust-ai ecosystem crates that may
//! not have stable releases. Use `vsa-native` for production deployments until
//! upstream dependencies stabilize.
//!
//! ## Modules
//!
//! - [`adapters`]: Memory adapters for knowledge transformation pipelines
//! - [`storage`]: Pluggable storage backends for persistence
//! - [`agents`]: Base agent implementations with memory integration
//! - [`metrics`]: Prometheus-compatible observability metrics
//! - [`vsa`]: Vector Symbolic Architecture for holographic/associative memory

#![doc(html_root_url = "https://docs.rs/memory-gate-rs/1.0.0")]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![forbid(unsafe_code)]
#![warn(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    missing_docs,
    rustdoc::broken_intra_doc_links
)]
#![allow(
    clippy::module_name_repetitions,
    // VSA math operations require intentional casting between numeric types.
    // These are safe within the bounds of typical hypervector dimensions (1k-100k).
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    // Some functions use unwrap() on iterator operations that cannot fail.
    clippy::missing_panics_doc,
    // Async lock guards must be held across await points in some patterns.
    // Dropping early would require restructuring that reduces readability.
    clippy::significant_drop_tightening,
    clippy::significant_drop_in_scrutinee,
    // Similar names are intentional for related variables (e.g., context/content).
    clippy::similar_names,
)]

mod error;
mod gateway;
mod traits;
mod types;

pub mod adapters;
pub mod agents;
pub mod metrics;
pub mod storage;
pub mod vsa;

// Re-export core types
pub use error::{Error, Result, StorageError};
pub use gateway::MemoryGateway;
pub use traits::{
    BatchKnowledgeStore, FilterableStore, KnowledgeStore, MemoryAdapter, MemoryEnabledAgent,
    TaskResult, VectorStore,
};
pub use types::{AgentDomain, ConsolidationStats, GatewayConfig, LearningContext};

/// Prelude module for convenient imports.
///
/// ```rust
/// use memory_gate_rs::prelude::*;
/// ```
pub mod prelude {
    pub use crate::adapters::PassthroughAdapter;
    pub use crate::agents::BaseMemoryAgent;
    pub use crate::error::{Error, Result};
    pub use crate::gateway::MemoryGateway;
    pub use crate::storage::InMemoryStore;
    pub use crate::traits::{KnowledgeStore, MemoryAdapter, MemoryEnabledAgent};
    pub use crate::types::{AgentDomain, GatewayConfig, LearningContext};
    pub use crate::vsa::{HolographicStore, HolographicVector, VsaCodebook};
}
