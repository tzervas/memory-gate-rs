//! In-memory storage backend for testing and development.

use crate::{AgentDomain, Error, KnowledgeStore, LearningContext, Result, StorageError};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// In-memory knowledge store using a `HashMap`.
///
/// This store is primarily intended for testing and development. It provides
/// simple substring-based retrieval rather than vector similarity search.
///
/// # Example
///
/// ```
/// use memory_gate_rs::storage::InMemoryStore;
/// use memory_gate_rs::{KnowledgeStore, LearningContext, AgentDomain};
///
/// # tokio_test::block_on(async {
/// let store = InMemoryStore::new();
///
/// let ctx = LearningContext::new("nginx restart fixes CPU issues", AgentDomain::Infrastructure);
/// store.store_experience("key1", ctx).await.unwrap();
///
/// let results = store.retrieve_context("CPU", 10, None).await.unwrap();
/// assert_eq!(results.len(), 1);
/// # });
/// ```
#[derive(Debug, Clone)]
pub struct InMemoryStore {
    data: Arc<RwLock<HashMap<String, LearningContext>>>,
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryStore {
    /// Create a new in-memory store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new in-memory store with initial capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::with_capacity(capacity))),
        }
    }

    /// Get read access to the underlying data for testing.
    #[cfg(test)]
    pub(crate) async fn data(&self) -> tokio::sync::RwLockReadGuard<'_, HashMap<String, LearningContext>> {
        self.data.read().await
    }
}

#[async_trait]
impl KnowledgeStore<LearningContext> for InMemoryStore {
    async fn store_experience(&self, key: &str, experience: LearningContext) -> Result<()> {
        let mut data = self.data.write().await;
        data.insert(key.to_string(), experience);
        Ok(())
    }

    async fn retrieve_context(
        &self,
        query: &str,
        limit: usize,
        domain_filter: Option<AgentDomain>,
    ) -> Result<Vec<LearningContext>> {
        let data = self.data.read().await;
        let query_lower = query.to_lowercase();

        let mut results: Vec<(f32, LearningContext)> = data
            .values()
            .filter(|ctx| {
                // Apply domain filter if specified
                if let Some(domain) = domain_filter {
                    if ctx.domain != domain {
                        return false;
                    }
                }

                // Simple substring matching
                ctx.content.to_lowercase().contains(&query_lower)
            })
            .map(|ctx| {
                // Score based on importance and how well it matches
                let match_score = if ctx.content.to_lowercase() == query_lower {
                    1.0
                } else {
                    0.5
                };
                (ctx.importance * match_score, ctx.clone())
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take up to limit
        Ok(results.into_iter().take(limit).map(|(_, ctx)| ctx).collect())
    }

    async fn delete_experience(&self, key: &str) -> Result<()> {
        let mut data = self.data.write().await;
        data.remove(key)
            .ok_or_else(|| Error::Storage(StorageError::not_found(key)))?;
        Ok(())
    }

    async fn get_all_keys(&self) -> Result<Vec<String>> {
        let data = self.data.read().await;
        Ok(data.keys().cloned().collect())
    }

    async fn get_experience(&self, key: &str) -> Result<Option<LearningContext>> {
        let data = self.data.read().await;
        Ok(data.get(key).cloned())
    }

    async fn count(&self) -> Result<usize> {
        let data = self.data.read().await;
        Ok(data.len())
    }

    async fn clear(&self) -> Result<()> {
        let mut data = self.data.write().await;
        data.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_store_and_retrieve() {
        let store = InMemoryStore::new();

        let ctx1 = LearningContext::new("nginx service restart", AgentDomain::Infrastructure);
        let ctx2 = LearningContext::new("kubectl apply deployment", AgentDomain::Deployment);

        store.store_experience("key1", ctx1).await.unwrap();
        store.store_experience("key2", ctx2).await.unwrap();

        // Retrieve matching content
        let results = store.retrieve_context("nginx", 10, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("nginx"));

        // Retrieve with domain filter
        let results = store
            .retrieve_context("kubectl", 10, Some(AgentDomain::Deployment))
            .await
            .unwrap();
        assert_eq!(results.len(), 1);

        // Domain filter should exclude non-matching domains
        let results = store
            .retrieve_context("kubectl", 10, Some(AgentDomain::Infrastructure))
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_delete() {
        let store = InMemoryStore::new();
        let ctx = LearningContext::new("test", AgentDomain::General);

        store.store_experience("key1", ctx).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 1);

        store.delete_experience("key1").await.unwrap();
        assert_eq!(store.count().await.unwrap(), 0);

        // Deleting non-existent key should error
        let result = store.delete_experience("nonexistent").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_experience() {
        let store = InMemoryStore::new();
        let ctx = LearningContext::new("test content", AgentDomain::General);

        store.store_experience("key1", ctx.clone()).await.unwrap();

        let retrieved = store.get_experience("key1").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, ctx.content);

        let missing = store.get_experience("nonexistent").await.unwrap();
        assert!(missing.is_none());
    }

    #[tokio::test]
    async fn test_clear() {
        let store = InMemoryStore::new();

        store
            .store_experience("key1", LearningContext::new("test1", AgentDomain::General))
            .await
            .unwrap();
        store
            .store_experience("key2", LearningContext::new("test2", AgentDomain::General))
            .await
            .unwrap();

        assert_eq!(store.count().await.unwrap(), 2);

        store.clear().await.unwrap();
        assert_eq!(store.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_retrieval_ordering() {
        let store = InMemoryStore::new();

        // Store items with different importance
        let ctx1 = LearningContext::new("cpu monitoring", AgentDomain::Infrastructure)
            .with_importance(0.3);
        let ctx2 = LearningContext::new("cpu optimization", AgentDomain::Infrastructure)
            .with_importance(0.9);

        store.store_experience("key1", ctx1).await.unwrap();
        store.store_experience("key2", ctx2).await.unwrap();

        let results = store.retrieve_context("cpu", 10, None).await.unwrap();
        assert_eq!(results.len(), 2);
        // Higher importance should come first
        assert!(results[0].importance > results[1].importance);
    }

    #[tokio::test]
    async fn test_limit() {
        let store = InMemoryStore::new();

        for i in 0..10 {
            let ctx = LearningContext::new(format!("test content {i}"), AgentDomain::General);
            store
                .store_experience(&format!("key{i}"), ctx)
                .await
                .unwrap();
        }

        let results = store.retrieve_context("test", 5, None).await.unwrap();
        assert_eq!(results.len(), 5);
    }
}
