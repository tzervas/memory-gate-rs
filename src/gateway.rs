//! Memory gateway - the central orchestrator for learning and memory operations.

use crate::{
    AgentDomain, ConsolidationStats, Error, GatewayConfig, KnowledgeStore, LearningContext,
    MemoryAdapter, Result,
};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, instrument, warn};

/// Central orchestrator for learning and memory operations.
///
/// The `MemoryGateway` coordinates adapters, stores, and learning operations.
/// It provides the primary interface for storing and retrieving memories.
///
/// # Type Parameters
///
/// * `A` - The memory adapter type
/// * `S` - The knowledge store type
///
/// # Example
///
/// ```
/// use memory_gate_rs::{
///     MemoryGateway, GatewayConfig, LearningContext, AgentDomain,
///     adapters::PassthroughAdapter,
///     storage::InMemoryStore,
/// };
///
/// # tokio_test::block_on(async {
/// let store = InMemoryStore::new();
/// let adapter = PassthroughAdapter;
/// let config = GatewayConfig::default();
///
/// let gateway = MemoryGateway::new(adapter, store, config);
///
/// // Learn from an interaction
/// let ctx = LearningContext::new("Resolved issue by restarting service", AgentDomain::Infrastructure);
/// gateway.learn_from_interaction(ctx, Some(0.9)).await.unwrap();
///
/// // Retrieve relevant memories
/// let memories = gateway.retrieve_context("restart", Some(5), None).await.unwrap();
/// assert_eq!(memories.len(), 1);
/// # });
/// ```
pub struct MemoryGateway<A, S>
where
    A: MemoryAdapter<LearningContext>,
    S: KnowledgeStore<LearningContext>,
{
    adapter: Arc<A>,
    store: Arc<S>,
    config: GatewayConfig,
    consolidation_handle: Arc<RwLock<Option<JoinHandle<()>>>>,
    shutdown_tx: Arc<RwLock<Option<mpsc::Sender<()>>>>,
}

impl<A, S> MemoryGateway<A, S>
where
    A: MemoryAdapter<LearningContext> + 'static,
    S: KnowledgeStore<LearningContext> + 'static,
{
    /// Create a new memory gateway.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The memory adapter for knowledge transformation
    /// * `store` - The knowledge store for persistence
    /// * `config` - Gateway configuration
    #[must_use]
    pub fn new(adapter: A, store: S, config: GatewayConfig) -> Self {
        Self {
            adapter: Arc::new(adapter),
            store: Arc::new(store),
            config,
            consolidation_handle: Arc::new(RwLock::new(None)),
            shutdown_tx: Arc::new(RwLock::new(None)),
        }
    }

    /// Get a reference to the adapter.
    #[must_use]
    pub fn adapter(&self) -> &A {
        &self.adapter
    }

    /// Get a reference to the store.
    #[must_use]
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Get a reference to the configuration.
    #[must_use]
    pub const fn config(&self) -> &GatewayConfig {
        &self.config
    }

    /// Learn from an interaction, storing it as a memory.
    ///
    /// # Arguments
    ///
    /// * `context` - The learning context to store
    /// * `feedback` - Optional feedback score in `[0.0, 1.0]`
    ///
    /// # Errors
    ///
    /// Returns an error if adaptation or storage fails.
    #[instrument(skip(self, context), fields(domain = %context.domain))]
    pub async fn learn_from_interaction(
        &self,
        context: LearningContext,
        feedback: Option<f32>,
    ) -> Result<String> {
        debug!(
            content_len = context.content.len(),
            importance = context.importance,
            "Learning from interaction"
        );

        // Adapt the knowledge
        let adapted = self.adapter.adapt_knowledge(context, feedback).await?;

        // Generate a deterministic key
        let key = generate_key(&adapted);

        // Store the experience
        self.store.store_experience(&key, adapted).await?;

        info!(key = %key, "Stored learning context");
        Ok(key)
    }

    /// Retrieve relevant memories based on a query.
    ///
    /// # Arguments
    ///
    /// * `query` - The search query
    /// * `limit` - Maximum number of results (uses config default if `None`)
    /// * `domain_filter` - Optional domain to filter by
    ///
    /// # Errors
    ///
    /// Returns an error if retrieval fails.
    #[instrument(skip(self))]
    pub async fn retrieve_context(
        &self,
        query: &str,
        limit: Option<usize>,
        domain_filter: Option<AgentDomain>,
    ) -> Result<Vec<LearningContext>> {
        let limit = limit.unwrap_or(self.config.retrieval_limit);
        debug!(query = %query, limit = limit, domain = ?domain_filter, "Retrieving context");

        let results = self
            .store
            .retrieve_context(query, limit, domain_filter)
            .await?;

        debug!(results_count = results.len(), "Retrieved memories");
        Ok(results)
    }

    /// Start the background consolidation worker.
    ///
    /// # Errors
    ///
    /// Returns an error if consolidation is not enabled or already running.
    pub async fn start_consolidation(&self) -> Result<()> {
        if !self.config.consolidation_enabled {
            return Err(Error::invalid_config("consolidation is not enabled"));
        }

        let mut handle_guard = self.consolidation_handle.write().await;
        if handle_guard.is_some() {
            return Err(Error::invalid_config("consolidation is already running"));
        }

        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);
        *self.shutdown_tx.write().await = Some(shutdown_tx);

        let store = Arc::clone(&self.store);
        let config = self.config.clone();

        let handle = tokio::spawn(async move {
            consolidation_loop(store, config, shutdown_rx).await;
        });

        *handle_guard = Some(handle);
        info!("Started consolidation worker");
        Ok(())
    }

    /// Stop the background consolidation worker.
    pub async fn stop_consolidation(&self) {
        // Send shutdown signal
        if let Some(tx) = self.shutdown_tx.write().await.take() {
            let _ = tx.send(()).await;
        }

        // Wait for task to complete
        if let Some(handle) = self.consolidation_handle.write().await.take() {
            let _ = handle.await;
        }

        info!("Stopped consolidation worker");
    }

    /// Check if consolidation is running.
    pub async fn is_consolidation_running(&self) -> bool {
        self.consolidation_handle.read().await.is_some()
    }

    /// Run consolidation once (manually triggered).
    ///
    /// # Errors
    ///
    /// Returns an error if consolidation fails.
    pub async fn run_consolidation_once(&self) -> Result<ConsolidationStats> {
        perform_consolidation(self.store.as_ref(), &self.config).await
    }

    /// Get the count of items in the store.
    ///
    /// # Errors
    ///
    /// Returns an error if counting fails.
    pub async fn count(&self) -> Result<usize> {
        self.store.count().await
    }

    /// Clear all items from the store.
    ///
    /// # Errors
    ///
    /// Returns an error if clearing fails.
    pub async fn clear(&self) -> Result<()> {
        warn!("Clearing all memories from store");
        self.store.clear().await
    }
}

impl<A, S> Drop for MemoryGateway<A, S>
where
    A: MemoryAdapter<LearningContext>,
    S: KnowledgeStore<LearningContext>,
{
    fn drop(&mut self) {
        // Note: We can't do async cleanup in Drop.
        // Users should call stop_consolidation() before dropping.
        debug!("MemoryGateway dropped");
    }
}

/// Generate a deterministic key from a learning context.
fn generate_key(context: &LearningContext) -> String {
    let mut hasher = Sha256::new();
    hasher.update(context.content.as_bytes());
    hasher.update(context.domain.as_str().as_bytes());
    let result = hasher.finalize();
    // Take first 16 characters of hex representation
    hex::encode(&result[..8])
}

/// Background consolidation loop.
async fn consolidation_loop<S>(
    store: Arc<S>,
    config: GatewayConfig,
    mut shutdown_rx: mpsc::Receiver<()>,
) where
    S: KnowledgeStore<LearningContext> + 'static,
{
    let mut interval = tokio::time::interval(config.consolidation_interval);

    loop {
        tokio::select! {
            _ = interval.tick() => {
                debug!("Running scheduled consolidation");
                match perform_consolidation(store.as_ref(), &config).await {
                    Ok(stats) => {
                        info!(
                            processed = stats.items_processed,
                            deleted = stats.items_deleted,
                            duration_ms = stats.duration.as_millis(),
                            "Consolidation completed"
                        );
                    }
                    Err(e) => {
                        error!(error = %e, "Consolidation failed");
                    }
                }
            }
            _ = shutdown_rx.recv() => {
                info!("Consolidation worker shutting down");
                break;
            }
        }
    }
}

/// Perform a single consolidation pass.
async fn perform_consolidation<S>(store: &S, config: &GatewayConfig) -> Result<ConsolidationStats>
where
    S: KnowledgeStore<LearningContext>,
{
    let start = std::time::Instant::now();
    let mut stats = ConsolidationStats::new();

    // Get all keys
    let keys = store.get_all_keys().await?;
    stats.items_processed = keys.len();

    for key in keys {
        // Get the experience
        if let Some(ctx) = store.get_experience(&key).await? {
            // Check if it should be consolidated
            if ctx.should_consolidate(
                config.low_importance_threshold,
                i64::from(config.age_threshold_days),
            ) {
                match store.delete_experience(&key).await {
                    Ok(()) => {
                        stats.items_deleted += 1;
                        debug!(key = %key, "Deleted low-importance memory");
                    }
                    Err(e) => {
                        stats.errors.push(format!("Failed to delete {key}: {e}"));
                    }
                }
            }
        }
    }

    stats.duration = start.elapsed();
    Ok(stats)
}

// We need hex encoding for the key generation
mod hex {
    const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";

    pub fn encode(bytes: &[u8]) -> String {
        let mut hex = String::with_capacity(bytes.len() * 2);
        for &byte in bytes {
            hex.push(HEX_CHARS[(byte >> 4) as usize] as char);
            hex.push(HEX_CHARS[(byte & 0x0f) as usize] as char);
        }
        hex
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{adapters::PassthroughAdapter, storage::InMemoryStore};

    fn create_test_gateway() -> MemoryGateway<PassthroughAdapter, InMemoryStore> {
        let store = InMemoryStore::new();
        let adapter = PassthroughAdapter;
        let config = GatewayConfig::default().with_consolidation_enabled(false);
        MemoryGateway::new(adapter, store, config)
    }

    #[tokio::test]
    async fn test_learn_and_retrieve() {
        let gateway = create_test_gateway();

        let ctx = LearningContext::new("nginx restart command", AgentDomain::Infrastructure);
        let key = gateway.learn_from_interaction(ctx, None).await.unwrap();
        assert!(!key.is_empty());

        let results = gateway
            .retrieve_context("nginx", None, None)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("nginx"));
    }

    #[tokio::test]
    async fn test_learn_with_feedback() {
        let gateway = create_test_gateway();

        let ctx =
            LearningContext::new("test content", AgentDomain::General).with_importance(1.0);

        gateway
            .learn_from_interaction(ctx, Some(0.5))
            .await
            .unwrap();

        let results = gateway
            .retrieve_context("test", None, None)
            .await
            .unwrap();

        // Importance should be blended: (1.0 + 0.5) / 2 = 0.75
        assert!((results[0].importance - 0.75).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_domain_filter() {
        let gateway = create_test_gateway();

        let ctx1 = LearningContext::new("infra content", AgentDomain::Infrastructure);
        let ctx2 = LearningContext::new("deploy content", AgentDomain::Deployment);

        gateway.learn_from_interaction(ctx1, None).await.unwrap();
        gateway.learn_from_interaction(ctx2, None).await.unwrap();

        let results = gateway
            .retrieve_context("content", None, Some(AgentDomain::Infrastructure))
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].domain, AgentDomain::Infrastructure);
    }

    #[tokio::test]
    async fn test_count_and_clear() {
        let gateway = create_test_gateway();

        gateway
            .learn_from_interaction(
                LearningContext::new("test1", AgentDomain::General),
                None,
            )
            .await
            .unwrap();
        gateway
            .learn_from_interaction(
                LearningContext::new("test2", AgentDomain::General),
                None,
            )
            .await
            .unwrap();

        assert_eq!(gateway.count().await.unwrap(), 2);

        gateway.clear().await.unwrap();
        assert_eq!(gateway.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_key_generation() {
        let ctx1 = LearningContext::new("same content", AgentDomain::Infrastructure);
        let ctx2 = LearningContext::new("same content", AgentDomain::Infrastructure);
        let ctx3 = LearningContext::new("different content", AgentDomain::Infrastructure);

        let key1 = generate_key(&ctx1);
        let key2 = generate_key(&ctx2);
        let key3 = generate_key(&ctx3);

        // Same content + domain = same key
        assert_eq!(key1, key2);
        // Different content = different key
        assert_ne!(key1, key3);
    }

    #[tokio::test]
    async fn test_manual_consolidation() {
        use chrono::{Duration, Utc};

        let store = InMemoryStore::new();
        let adapter = PassthroughAdapter;
        let config = GatewayConfig::default()
            .with_consolidation_enabled(false)
            .with_low_importance_threshold(0.3)
            .with_age_threshold_days(1);

        let gateway = MemoryGateway::new(adapter, store, config);

        // Add an old, low-importance memory
        let old_ctx = LearningContext::new("old low importance", AgentDomain::General)
            .with_importance(0.1)
            .with_timestamp(Utc::now() - Duration::days(5));

        // Add a new, high-importance memory
        let new_ctx = LearningContext::new("new high importance", AgentDomain::General)
            .with_importance(0.9);

        gateway.learn_from_interaction(old_ctx, None).await.unwrap();
        gateway.learn_from_interaction(new_ctx, None).await.unwrap();

        assert_eq!(gateway.count().await.unwrap(), 2);

        // Run consolidation
        let stats = gateway.run_consolidation_once().await.unwrap();

        // Old low-importance should be deleted
        assert_eq!(stats.items_deleted, 1);
        assert_eq!(gateway.count().await.unwrap(), 1);

        // Remaining should be the high-importance one
        let results = gateway
            .retrieve_context("importance", None, None)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].importance > 0.5);
    }
}
