//! Holographic Memory Store - VSA-based knowledge storage.
//!
//! This module implements a holographic memory store that uses Vector Symbolic
//! Architecture (VSA) for compositional, distributed memory representation.
//!
//! # Holographic Memory Properties
//!
//! - **Distributed**: Information is spread across all vector dimensions
//! - **Compositional**: Complex structures from primitive operations
//! - **Fault-tolerant**: Graceful degradation with noise/damage
//! - **Content-addressable**: Retrieve by partial match
//!
//! # Memory Organization
//!
//! The store maintains several interconnected structures:
//!
//! 1. **Codebook**: Maps atomic symbols to HD vectors
//! 2. **Memory Traces**: Individual experiences encoded as HD vectors
//! 3. **Holographic Index**: Bundled summary for fast approximate search
//! 4. **Temporal Stream**: Sequence-encoded episodic memory

use crate::vsa::{HolographicVector, VsaCodebook};
use crate::{AgentDomain, KnowledgeStore, LearningContext, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Configuration for holographic memory store.
#[derive(Debug, Clone)]
pub struct HolographicConfig {
    /// Dimensionality of hypervectors (typically 1000-10000)
    pub dimensions: usize,
    /// Maximum number of memory traces to keep
    pub max_traces: usize,
    /// Similarity threshold for retrieval
    pub retrieval_threshold: f32,
    /// Number of tokens to use for content encoding
    pub content_tokens: usize,
    /// Enable temporal/episodic encoding
    pub enable_temporal: bool,
    /// Decay factor for importance over time
    pub importance_decay: f32,
    /// Random seed for reproducibility (None = random)
    pub seed: Option<u64>,
}

impl Default for HolographicConfig {
    fn default() -> Self {
        Self {
            dimensions: 10000,
            max_traces: 100_000,
            retrieval_threshold: 0.3,
            content_tokens: 50,
            enable_temporal: true,
            importance_decay: 0.99,
            seed: None,
        }
    }
}

impl HolographicConfig {
    /// Create a new configuration with custom dimensions.
    #[must_use]
    pub fn with_dimensions(dimensions: usize) -> Self {
        Self {
            dimensions,
            ..Default::default()
        }
    }

    /// Set maximum traces.
    #[must_use]
    pub const fn with_max_traces(mut self, max: usize) -> Self {
        self.max_traces = max;
        self
    }

    /// Set retrieval threshold.
    #[must_use]
    pub const fn with_threshold(mut self, threshold: f32) -> Self {
        self.retrieval_threshold = threshold;
        self
    }

    /// Enable/disable temporal encoding.
    #[must_use]
    pub const fn with_temporal(mut self, enable: bool) -> Self {
        self.enable_temporal = enable;
        self
    }

    /// Set seed for reproducibility.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// A memory trace in the holographic store.
#[derive(Debug, Clone)]
pub struct MemoryTrace {
    /// Unique identifier
    pub key: String,
    /// The encoded holographic vector
    pub vector: HolographicVector,
    /// Original learning context
    pub context: LearningContext,
    /// Importance weight (decays over time)
    pub importance: f32,
    /// Access count for popularity tracking
    pub access_count: u64,
    /// Last access time
    pub last_accessed: DateTime<Utc>,
}

impl MemoryTrace {
    /// Create a new memory trace.
    fn new(key: String, vector: HolographicVector, context: LearningContext) -> Self {
        Self {
            key,
            vector,
            importance: context.importance,
            context,
            access_count: 0,
            last_accessed: Utc::now(),
        }
    }

    /// Update access statistics.
    fn touch(&mut self) {
        self.access_count += 1;
        self.last_accessed = Utc::now();
    }
}

/// Holographic memory store using VSA.
///
/// This store provides "RAG on crack" through holographic memory:
///
/// - **Binding** associates queries with content
/// - **Bundling** creates composite memory indices
/// - **Factorization** enables analogical retrieval
/// - **Cosine similarity** for efficient search
#[derive(Debug)]
pub struct HolographicStore {
    /// Configuration
    config: HolographicConfig,
    /// Codebook for atomic symbols
    codebook: Arc<RwLock<VsaCodebook>>,
    /// Memory traces indexed by key
    traces: Arc<RwLock<HashMap<String, MemoryTrace>>>,
    /// Holographic index (bundle of all memory encodings)
    holo_index: Arc<RwLock<Option<HolographicVector>>>,
    /// Position vector for temporal encoding
    position_base: HolographicVector,
    /// Domain role vector
    domain_role: HolographicVector,
    /// Content role vector
    content_role: HolographicVector,
    /// Importance role vector
    importance_role: HolographicVector,
    /// Temporal position counter
    temporal_position: Arc<RwLock<usize>>,
}

impl HolographicStore {
    /// Create a new holographic store with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(HolographicConfig::default())
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: HolographicConfig) -> Self {
        let codebook = config.seed.map_or_else(
            || VsaCodebook::new(config.dimensions),
            |seed| VsaCodebook::with_seed(config.dimensions, seed),
        );

        // Create role vectors for structural encoding
        let mut cb = codebook;
        let position_base = cb.get_or_create_role("position");
        let domain_role = cb.get_or_create_role("domain");
        let content_role = cb.get_or_create_role("content");
        let importance_role = cb.get_or_create_role("importance");

        Self {
            config,
            codebook: Arc::new(RwLock::new(cb)),
            traces: Arc::new(RwLock::new(HashMap::new())),
            holo_index: Arc::new(RwLock::new(None)),
            position_base,
            domain_role,
            content_role,
            importance_role,
            temporal_position: Arc::new(RwLock::new(0)),
        }
    }

    /// Get the configuration.
    #[must_use]
    pub const fn config(&self) -> &HolographicConfig {
        &self.config
    }

    /// Encode a learning context into a holographic vector.
    async fn encode_context(&self, context: &LearningContext) -> HolographicVector {
        let mut codebook = self.codebook.write().await;

        // Tokenize content (simple whitespace + lowercase)
        let tokens: Vec<&str> = context.content
            .split_whitespace()
            .take(self.config.content_tokens)
            .collect();

        // Create content encoding: bundle of position-bound tokens
        let token_vecs: Vec<HolographicVector> = tokens.iter()
            .enumerate()
            .map(|(i, token)| {
                let token_lower = token.to_lowercase();
                let token_vec = codebook.get_or_create(&token_lower);
                self.position_base.permute(i as i32).bind(&token_vec)
            })
            .collect();

        let content_vec = HolographicVector::bundle_all(&token_vecs)
            .unwrap_or_else(|| HolographicVector::zeros(self.config.dimensions));

        // Encode domain
        let domain_str = match context.domain {
            AgentDomain::Infrastructure => "infrastructure",
            AgentDomain::CodeReview => "codereview",
            AgentDomain::Deployment => "deployment",
            AgentDomain::IncidentResponse => "incidentresponse",
            AgentDomain::General => "general",
        };
        let domain_vec = codebook.get_or_create(domain_str);

        // Encode importance as a position (quantized)
        let importance_idx = (context.importance * 10.0) as i32;
        let importance_vec = self.importance_role.permute(importance_idx);

        // Create record: domain_role ⊛ domain + content_role ⊛ content + importance_role ⊛ importance
        let domain_bound = self.domain_role.bind(&domain_vec);
        let content_bound = self.content_role.bind(&content_vec);
        let importance_bound = importance_vec;

        // Bundle all role-filler pairs
        domain_bound
            .bundle(&content_bound)
            .bundle(&importance_bound)
    }

    /// Decode a query string into a holographic vector.
    async fn encode_query(&self, query: &str) -> HolographicVector {
        let mut codebook = self.codebook.write().await;

        let tokens: Vec<&str> = query
            .split_whitespace()
            .take(self.config.content_tokens)
            .collect();

        let token_vecs: Vec<HolographicVector> = tokens.iter()
            .enumerate()
            .map(|(i, token)| {
                let token_lower = token.to_lowercase();
                let token_vec = codebook.get_or_create(&token_lower);
                self.position_base.permute(i as i32).bind(&token_vec)
            })
            .collect();

        // Bundle tokens and bind with content role for query
        let content_vec = HolographicVector::bundle_all(&token_vecs)
            .unwrap_or_else(|| HolographicVector::zeros(self.config.dimensions));

        self.content_role.bind(&content_vec)
    }

    /// Update the holographic index after adding a trace.
    async fn update_index(&self, trace_vec: &HolographicVector) {
        let mut index = self.holo_index.write().await;
        *index = index.take().map_or_else(
            || Some(trace_vec.clone()),
            |existing| Some(existing.bundle(trace_vec)),
        );
    }

    /// Rebuild the entire holographic index.
    async fn rebuild_index(&self) {
        let traces = self.traces.read().await;
        let vectors: Vec<HolographicVector> = traces.values()
            .map(|t| t.vector.clone())
            .collect();

        let mut index = self.holo_index.write().await;
        *index = HolographicVector::bundle_all(&vectors);
    }

    /// Apply importance decay to all traces.
    pub async fn decay_importance(&self) {
        let mut traces = self.traces.write().await;
        for trace in traces.values_mut() {
            trace.importance *= self.config.importance_decay;
        }
    }

    /// Prune low-importance traces if over capacity.
    async fn prune_if_needed(&self) {
        let mut traces = self.traces.write().await;
        if traces.len() <= self.config.max_traces {
            return;
        }

        // Sort by importance and remove lowest
        let mut items: Vec<(String, f32)> = traces.iter()
            .map(|(k, v)| (k.clone(), v.importance))
            .collect();
        items.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let to_remove = traces.len() - self.config.max_traces;
        for (key, _) in items.into_iter().take(to_remove) {
            traces.remove(&key);
        }

        // Rebuild index after pruning
        drop(traces);
        self.rebuild_index().await;
    }

    /// Find memories similar to a query using holographic operations.
    ///
    /// This uses the bound content role to query for matching content,
    /// effectively performing "unbinding" to find associated memories.
    pub async fn holographic_search(&self, query: &str, limit: usize) -> Vec<(String, f32)> {
        let query_vec = self.encode_query(query).await;
        
        let traces = self.traces.read().await;
        let mut results: Vec<(String, f32)> = traces.iter()
            .map(|(key, trace)| {
                let sim = query_vec.cosine_similarity(&trace.vector);
                (key.clone(), sim)
            })
            .filter(|(_, sim)| *sim >= self.config.retrieval_threshold)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(limit);
        results
    }

    /// Analogical retrieval: find X where A:B :: X:query
    ///
    /// Given a relational pattern (A binds to B), finds what would
    /// bind to the query in the same way.
    pub async fn analogical_search(
        &self, 
        pattern_a: &str, 
        pattern_b: &str, 
        query: &str,
        limit: usize,
    ) -> Vec<(String, f32)> {
        let mut codebook = self.codebook.write().await;
        
        let a_vec = codebook.get_or_create(pattern_a);
        let b_vec = codebook.get_or_create(pattern_b);
        let query_vec = codebook.get_or_create(query);
        
        // Compute relation: A ⊛ B^(-1) gives the "relationship"
        let relation = a_vec.bind(&b_vec);
        
        // Apply relation to query: query ⊛ relation should give analogous result
        let analogy_vec = query_vec.bind(&relation);
        
        // Find nearest in codebook
        codebook.find_nearest(&analogy_vec, limit)
    }

    /// Get statistics about the holographic store.
    #[must_use]
    pub async fn stats(&self) -> HolographicStats {
        let traces = self.traces.read().await;
        let codebook = self.codebook.read().await;
        let index = self.holo_index.read().await;

        HolographicStats {
            trace_count: traces.len(),
            codebook_size: codebook.len(),
            dimensions: self.config.dimensions,
            has_index: index.is_some(),
            avg_importance: if traces.is_empty() {
                0.0
            } else {
                traces.values().map(|t| t.importance).sum::<f32>() / traces.len() as f32
            },
        }
    }
}

impl Default for HolographicStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the holographic store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolographicStats {
    /// Number of memory traces
    pub trace_count: usize,
    /// Number of symbols in codebook
    pub codebook_size: usize,
    /// Vector dimensionality
    pub dimensions: usize,
    /// Whether holographic index exists
    pub has_index: bool,
    /// Average importance across traces
    pub avg_importance: f32,
}

#[async_trait]
impl KnowledgeStore<LearningContext> for HolographicStore {
    async fn store_experience(&self, key: &str, experience: LearningContext) -> Result<()> {
        // Encode context to holographic vector
        let vector = self.encode_context(&experience).await;
        
        // Create trace
        let trace = MemoryTrace::new(key.to_string(), vector.clone(), experience);
        
        // Store trace
        {
            let mut traces = self.traces.write().await;
            traces.insert(key.to_string(), trace);
        }
        
        // Update index
        self.update_index(&vector).await;
        
        // Prune if needed
        self.prune_if_needed().await;
        
        // Increment temporal position
        if self.config.enable_temporal {
            let mut pos = self.temporal_position.write().await;
            *pos += 1;
        }
        
        Ok(())
    }

    async fn retrieve_context(
        &self,
        query: &str,
        limit: usize,
        domain_filter: Option<AgentDomain>,
    ) -> Result<Vec<LearningContext>> {
        let query_vec = self.encode_query(query).await;
        
        let mut traces = self.traces.write().await;
        
        let mut results: Vec<(&mut MemoryTrace, f32)> = traces.values_mut()
            .filter(|trace| {
                // Apply domain filter if specified
                if let Some(ref filter) = domain_filter {
                    if trace.context.domain != *filter {
                        return false;
                    }
                }
                true
            })
            .map(|trace| {
                let sim = query_vec.cosine_similarity(&trace.vector);
                (trace, sim)
            })
            .filter(|(_, sim)| *sim >= self.config.retrieval_threshold)
            .collect();

        // Sort by combined score: similarity * importance
        results.sort_by(|a, b| {
            let score_a = a.1 * a.0.importance;
            let score_b = b.1 * b.0.importance;
            score_b.partial_cmp(&score_a).unwrap()
        });
        
        results.truncate(limit);
        
        // Touch accessed traces and return contexts
        Ok(results.into_iter()
            .map(|(trace, _)| {
                trace.touch();
                trace.context.clone()
            })
            .collect())
    }

    async fn delete_experience(&self, key: &str) -> Result<()> {
        let removed = {
            let mut traces = self.traces.write().await;
            traces.remove(key).is_some()
        };
        
        if removed {
            // Rebuild index after deletion
            self.rebuild_index().await;
        }
        
        Ok(())
    }

    async fn get_all_keys(&self) -> Result<Vec<String>> {
        let traces = self.traces.read().await;
        Ok(traces.keys().cloned().collect())
    }

    async fn get_experience(&self, key: &str) -> Result<Option<LearningContext>> {
        let mut traces = self.traces.write().await;
        Ok(traces.get_mut(key).map(|trace| {
            trace.touch();
            trace.context.clone()
        }))
    }

    async fn clear(&self) -> Result<()> {
        {
            let mut traces = self.traces.write().await;
            traces.clear();
        }
        {
            let mut index = self.holo_index.write().await;
            *index = None;
        }
        {
            let mut pos = self.temporal_position.write().await;
            *pos = 0;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AgentDomain;

    fn create_test_context(content: &str, domain: AgentDomain, importance: f32) -> LearningContext {
        LearningContext::new(content.to_string(), domain)
            .with_importance(importance)
    }

    #[tokio::test]
    async fn test_store_and_retrieve() {
        let store = HolographicStore::with_config(
            HolographicConfig::with_dimensions(5000)
                .with_threshold(0.05)  // Very low threshold for short content
        );

        let ctx = create_test_context(
            "nginx restart fixes high CPU usage problems on the server",
            AgentDomain::Infrastructure,
            0.8,
        );

        store.store_experience("key1", ctx).await.unwrap();

        let results = store.retrieve_context("nginx server CPU", 5, None).await.unwrap();
        assert!(!results.is_empty(), "Should find at least one result");
        assert!(results[0].content.contains("nginx"));
    }

    #[tokio::test]
    async fn test_domain_filter() {
        let store = HolographicStore::with_config(
            HolographicConfig::with_dimensions(1000)
                .with_threshold(0.1)
        );

        store.store_experience(
            "infra1",
            create_test_context("server restart", AgentDomain::Infrastructure, 0.7),
        ).await.unwrap();

        store.store_experience(
            "dev1",
            create_test_context("code restart process", AgentDomain::Deployment, 0.7),
        ).await.unwrap();

        // Should only find infrastructure
        let results = store.retrieve_context(
            "restart",
            10,
            Some(AgentDomain::Infrastructure),
        ).await.unwrap();

        assert!(results.iter().all(|c| c.domain == AgentDomain::Infrastructure));
    }

    #[tokio::test]
    async fn test_holographic_search() {
        let store = HolographicStore::with_config(
            HolographicConfig::with_dimensions(5000)
                .with_threshold(0.1)
                .with_seed(42)
        );

        store.store_experience(
            "k1",
            create_test_context("docker container networking issues", AgentDomain::Infrastructure, 0.8),
        ).await.unwrap();

        store.store_experience(
            "k2", 
            create_test_context("kubernetes pod networking problems", AgentDomain::Infrastructure, 0.7),
        ).await.unwrap();

        let results = store.holographic_search("network container", 5).await;
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_stats() {
        let store = HolographicStore::new();
        
        store.store_experience(
            "k1",
            create_test_context("test content", AgentDomain::General, 0.5),
        ).await.unwrap();

        let stats = store.stats().await;
        assert_eq!(stats.trace_count, 1);
        assert!(stats.codebook_size > 0);
        assert!(stats.has_index);
    }

    #[tokio::test]
    async fn test_clear() {
        let store = HolographicStore::new();
        
        store.store_experience(
            "k1",
            create_test_context("test", AgentDomain::General, 0.5),
        ).await.unwrap();

        store.clear().await.unwrap();
        
        let stats = store.stats().await;
        assert_eq!(stats.trace_count, 0);
        assert!(!stats.has_index);
    }
}
