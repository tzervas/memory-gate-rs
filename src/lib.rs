//! # memory-gate-rs
//!
//! Dynamic Memory Learning Layer for AI agents - persistent knowledge accumulation
//! with vector retrieval and automatic consolidation.
//!
//! ## Overview
//!
//! Memory-gate provides a production-ready memory layer for AI agents, solving the
//! fundamental problem of stateless AI by enabling:
//!
//! - **Persistent Knowledge Accumulation**: Agents retain and build upon operational
//!   knowledge across sessions
//! - **Context-Aware Retrieval**: Semantic search retrieves relevant past experiences
//! - **Automatic Consolidation**: Background cleanup of low-importance/stale memories
//! - **Domain Filtering**: Categorical filtering for multi-domain agent deployments
//!
//! ## Architecture
//!
//! Inspired by neuroscience research on brain plasticity, memory-gate implements a
//! Complementary Learning Systems (CLS) dual-stream architecture:
//!
//! - **Fast System**: Rapid pattern acquisition from new experiences
//! - **Slow System**: Stable long-term retention through consolidation
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
//! // Initialize gateway
//! let gateway = MemoryGateway::new(adapter, store, GatewayConfig::default());
//!
//! // Learn from an interaction
//! let context = LearningContext::new(
//!     "Resolved high CPU by restarting nginx service",
//!     AgentDomain::Infrastructure,
//! );
//! gateway.learn_from_interaction(context, Some(0.9)).await.unwrap();
//!
//! // Retrieve relevant memories
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
//! | `in-memory` | In-memory HashMap-based storage (default) |
//! | `qdrant` | Qdrant vector database backend |
//! | `sqlite-vec` | SQLite with vector extension |
//! | `metrics` | Prometheus metrics export |
//! | `full` | All features enabled |
//!
//! ## Modules
//!
//! - [`adapters`]: Memory adapters for knowledge transformation
//! - [`storage`]: Storage backends for persistence
//! - [`agents`]: Base agent implementations
//! - [`metrics`]: Prometheus-compatible observability

#![doc(html_root_url = "https://docs.rs/memory-gate-rs/0.1.0")]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![forbid(unsafe_code)]
#![warn(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    missing_docs,
    rustdoc::broken_intra_doc_links
)]
#![allow(clippy::module_name_repetitions)]

mod error;
mod gateway;
mod traits;
mod types;

pub mod adapters;
pub mod agents;
pub mod metrics;
pub mod storage;

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
}
