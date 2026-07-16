//! Integration facade for **tero-rs** and other consumers (`join/mg-facade@STABLE`).
//!
//! This module documents the supported public surface over [`MemoryGateway`]: metadata
//! conventions for L1-sourced learns, production defaults, and thin async wrappers
//! that map to the exact gateway methods (no private API scraping).
//!
//! # Contract
//!
//! | Operation | Gateway method |
//! |-----------|----------------|
//! | Learn | [`MemoryGateway::learn_from_interaction`] |
//! | Retrieve | [`MemoryGateway::retrieve_context`] |
//! | Consolidate (once) | [`MemoryGateway::run_consolidation_once`] |
//! | Consolidate (worker) | [`MemoryGateway::start_consolidation`] / [`MemoryGateway::stop_consolidation`] |
//!
//! # Metadata (tero-sourced learns)
//!
//! | Key | Value |
//! |-----|--------|
//! | `source` | `tero-l1` |
//! | `tero_anchors` | comma-separated citation anchors |
//! | `tero_index` | optional path or content-hash of the L1 index |

use crate::{
    adapters::PassthroughAdapter, AgentDomain, ConsolidationStats, GatewayConfig, KnowledgeStore,
    LearningContext, MemoryAdapter, MemoryGateway, Result, SupportedEmbeddingModel,
};
use std::collections::HashMap;
use std::path::Path;

/// Metadata key: provenance source (value [`TERO_SOURCE_VALUE`]).
pub const TERO_META_SOURCE: &str = "source";

/// Metadata value for learns originating from tero L1.
pub const TERO_SOURCE_VALUE: &str = "tero-l1";

/// Metadata key: comma-separated citation anchors from tero L1.
pub const TERO_META_ANCHORS: &str = "tero_anchors";

/// Metadata key: optional path or content-hash of the tero index.
pub const TERO_META_INDEX: &str = "tero_index";

/// Recommended default `SQLite` filename stem for embedded prod (one file per embedding model).
pub const PROD_SQLITE_PATH_HINT: &str = "~/.tero/memory-gate/tero-memories.db";

/// Naming note for Qdrant collections when used instead of sqlite-vec.
pub const PROD_QDRANT_COLLECTION_HINT: &str =
    "Use a dedicated collection per catalog model (e.g. `tero_memories_bge_small_en_v1_5`); \
     never reuse a collection after switching `SupportedEmbeddingModel`.";

/// Helpers for tero L1 metadata on [`LearningContext`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TeroMemoryMeta {
    /// Comma-separated citation anchors.
    pub anchors: String,
    /// Optional path or content-hash of the L1 index.
    pub tero_index: Option<String>,
}

impl TeroMemoryMeta {
    /// Create metadata with required anchors (comma-separated if multiple).
    #[must_use]
    pub fn new(anchors: impl Into<String>) -> Self {
        Self {
            anchors: anchors.into(),
            tero_index: None,
        }
    }

    /// Attach optional `tero_index` (path or content-hash).
    #[must_use]
    pub fn with_tero_index(mut self, index: impl Into<String>) -> Self {
        self.tero_index = Some(index.into());
        self
    }

    /// Build the metadata map per `join/mg-facade`.
    #[must_use]
    pub fn to_metadata_map(&self) -> HashMap<String, String> {
        build_tero_metadata_map(&self.anchors, self.tero_index.as_deref())
    }

    /// Apply metadata to an existing context (replaces metadata entirely).
    #[must_use]
    pub fn apply_to(self, mut ctx: LearningContext) -> LearningContext {
        ctx.metadata = Some(self.to_metadata_map());
        ctx
    }
}

/// Build a metadata map for tero-sourced learns.
///
/// Always sets `source=tero-l1` and `tero_anchors`; includes `tero_index` when provided.
#[must_use]
pub fn build_tero_metadata_map(anchors: &str, tero_index: Option<&str>) -> HashMap<String, String> {
    let mut map = HashMap::new();
    map.insert(TERO_META_SOURCE.into(), TERO_SOURCE_VALUE.into());
    map.insert(TERO_META_ANCHORS.into(), anchors.to_string());
    if let Some(index) = tero_index {
        map.insert(TERO_META_INDEX.into(), index.to_string());
    }
    map
}

/// Join anchor tokens into the comma-separated `tero_anchors` value.
#[must_use]
pub fn join_tero_anchors<'a>(anchors: impl IntoIterator<Item = &'a str>) -> String {
    anchors.into_iter().collect::<Vec<_>>().join(",")
}

/// Build a [`LearningContext`] for tero L1 ingest (`AgentDomain::Tero` + contract metadata).
#[must_use]
pub fn for_tero_learn(
    content: impl Into<String>,
    anchors: impl Into<String>,
    importance: Option<f32>,
    tero_index: Option<impl Into<String>>,
) -> LearningContext {
    let mut meta = TeroMemoryMeta::new(anchors);
    if let Some(index) = tero_index {
        meta = meta.with_tero_index(index);
    }
    let mut ctx =
        LearningContext::new(content, AgentDomain::Tero).with_metadata(meta.to_metadata_map());
    if let Some(imp) = importance {
        ctx = ctx.with_importance(imp);
    }
    ctx
}

/// Production-oriented defaults for tero-rs wiring (`TERO_MEMORY_*` draft env names).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProdMemoryConfig {
    /// Default dense embedding catalog id (matches vector backend `open()` default).
    pub embedding_model: SupportedEmbeddingModel,
}

impl Default for ProdMemoryConfig {
    fn default() -> Self {
        Self {
            embedding_model: SupportedEmbeddingModel::DEFAULT,
        }
    }
}

impl ProdMemoryConfig {
    /// Alias for bulletin naming (`IntegrationConfig`).
    #[must_use]
    pub const fn integration() -> Self {
        Self {
            embedding_model: SupportedEmbeddingModel::DEFAULT,
        }
    }

    /// Stable catalog id string (e.g. `bge-small-en-v1.5`).
    #[must_use]
    pub const fn embedding_model_id(self) -> &'static str {
        self.embedding_model.as_str()
    }

    /// Hint for default on-disk sqlite path (consumers expand `~` themselves).
    #[must_use]
    pub const fn sqlite_path_hint(self) -> &'static str {
        PROD_SQLITE_PATH_HINT
    }

    /// Hint for Qdrant collection naming when not using sqlite-vec.
    #[must_use]
    pub const fn qdrant_collection_hint(self) -> &'static str {
        PROD_QDRANT_COLLECTION_HINT
    }
}

/// Bulletin alias: integration defaults for external crates.
pub type IntegrationConfig = ProdMemoryConfig;

/// Open a file-backed [`crate::storage::SqliteVecStore`] with production defaults.
///
/// Uses [`SqliteVecStore::open_with_model`](crate::storage::SqliteVecStore::open_with_model).
/// Prefer a **new** DB path per [`SupportedEmbeddingModel`] (see [`PROD_QDRANT_COLLECTION_HINT`]).
///
/// # Errors
///
/// Returns an error if the database cannot be opened, embedding init fails, or store metadata mismatches the model.
#[cfg(feature = "sqlite-vec")]
pub async fn open_prod_sqlite<P: AsRef<Path>>(
    path: P,
    model: SupportedEmbeddingModel,
) -> Result<crate::storage::SqliteVecStore> {
    crate::storage::SqliteVecStore::open_with_model(path, model).await
}

/// Persist an experience — delegates to [`MemoryGateway::learn_from_interaction`].
///
/// # Errors
///
/// Returns an error if adaptation or storage fails (same as the gateway method).
pub async fn learn<A, S>(
    gateway: &MemoryGateway<A, S>,
    context: LearningContext,
    feedback: Option<f32>,
) -> Result<String>
where
    A: MemoryAdapter<LearningContext> + 'static,
    S: KnowledgeStore<LearningContext> + 'static,
{
    gateway.learn_from_interaction(context, feedback).await
}

/// Ranked retrieval — delegates to [`MemoryGateway::retrieve_context`].
///
/// # Errors
///
/// Returns an error if retrieval fails (same as the gateway method).
pub async fn retrieve<A, S>(
    gateway: &MemoryGateway<A, S>,
    query: &str,
    k: Option<usize>,
    domain: Option<AgentDomain>,
) -> Result<Vec<LearningContext>>
where
    A: MemoryAdapter<LearningContext> + 'static,
    S: KnowledgeStore<LearningContext> + 'static,
{
    gateway.retrieve_context(query, k, domain).await
}

/// Run consolidation once — delegates to [`MemoryGateway::run_consolidation_once`].
///
/// # Errors
///
/// Returns an error if consolidation fails (same as the gateway method).
pub async fn consolidate_once<A, S>(gateway: &MemoryGateway<A, S>) -> Result<ConsolidationStats>
where
    A: MemoryAdapter<LearningContext> + 'static,
    S: KnowledgeStore<LearningContext> + 'static,
{
    gateway.run_consolidation_once().await
}

/// Convenience: gateway with [`PassthroughAdapter`] and caller-supplied store.
#[must_use]
pub fn gateway_with_store<S>(
    store: S,
    config: GatewayConfig,
) -> MemoryGateway<PassthroughAdapter, S>
where
    S: KnowledgeStore<LearningContext> + 'static,
{
    MemoryGateway::new(PassthroughAdapter, store, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_tero_metadata_minimal() {
        let map = build_tero_metadata_map("anchor-a,anchor-b", None);
        assert_eq!(
            map.get(TERO_META_SOURCE),
            Some(&TERO_SOURCE_VALUE.to_string())
        );
        assert_eq!(
            map.get(TERO_META_ANCHORS),
            Some(&"anchor-a,anchor-b".to_string())
        );
        assert!(!map.contains_key(TERO_META_INDEX));
    }

    #[test]
    fn build_tero_metadata_with_index() {
        let map = build_tero_metadata_map("x", Some("/data/index.json"));
        assert_eq!(
            map.get(TERO_META_INDEX),
            Some(&"/data/index.json".to_string())
        );
    }

    #[test]
    fn tero_memory_meta_struct() {
        let meta = TeroMemoryMeta::new("a,b").with_tero_index("hash:abc");
        let map = meta.to_metadata_map();
        assert_eq!(map.len(), 3);
        assert_eq!(
            map.get(TERO_META_SOURCE).map(String::as_str),
            Some(TERO_SOURCE_VALUE)
        );
    }

    #[test]
    fn test_join_tero_anchors() {
        assert_eq!(super::join_tero_anchors(["one", "two"]), "one,two");
    }

    #[test]
    fn for_tero_learn_sets_domain_and_metadata() {
        let ctx = for_tero_learn("content", "cite-1", Some(0.8), Some("idx-path"));
        assert_eq!(ctx.domain, AgentDomain::Tero);
        assert!((ctx.importance - 0.8).abs() < f32::EPSILON);
        let meta = ctx.metadata.expect("metadata");
        assert_eq!(
            meta.get(TERO_META_SOURCE).map(String::as_str),
            Some(TERO_SOURCE_VALUE)
        );
        assert_eq!(
            meta.get(TERO_META_ANCHORS).map(String::as_str),
            Some("cite-1")
        );
        assert_eq!(
            meta.get(TERO_META_INDEX).map(String::as_str),
            Some("idx-path")
        );
    }

    #[test]
    fn prod_memory_config_defaults() {
        let cfg = ProdMemoryConfig::default();
        assert_eq!(cfg.embedding_model, SupportedEmbeddingModel::BgeSmallEnV15);
        assert_eq!(cfg.embedding_model_id(), "bge-small-en-v1.5");
        assert!(!cfg.sqlite_path_hint().is_empty());
    }
}
