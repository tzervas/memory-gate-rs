//! Trait definitions for memory-gate-rs.
//!
//! This module defines the core trait interfaces that enable pluggable
//! adapters, storage backends, and memory-enabled agents.

use crate::{AgentDomain, LearningContext, Result};
use async_trait::async_trait;
use serde_json::Value;

/// Adapter for knowledge transformation before storage.
///
/// Memory adapters can transform, filter, or augment learning contexts
/// before they are stored. This enables custom processing pipelines.
///
/// # Example
///
/// ```rust,ignore
/// use memory_gate_rs::{MemoryAdapter, LearningContext, Result};
/// use async_trait::async_trait;
///
/// struct MyAdapter;
///
/// #[async_trait]
/// impl MemoryAdapter<LearningContext> for MyAdapter {
///     async fn adapt_knowledge(
///         &self,
///         mut context: LearningContext,
///         feedback: Option<f32>,
///     ) -> Result<LearningContext> {
///         if let Some(f) = feedback {
///             // Blend current importance with feedback
///             context.importance = (context.importance + f) / 2.0;
///         }
///         Ok(context)
///     }
/// }
/// ```
#[async_trait]
pub trait MemoryAdapter<T>: Send + Sync {
    /// Adapt knowledge based on context and optional feedback.
    ///
    /// # Arguments
    ///
    /// * `context` - The learning context to adapt
    /// * `feedback` - Optional feedback score in range `[0.0, 1.0]`
    ///
    /// # Errors
    ///
    /// Returns an error if adaptation fails.
    async fn adapt_knowledge(&self, context: T, feedback: Option<f32>) -> Result<T>;
}

/// Storage backend interface for knowledge persistence.
///
/// Knowledge stores handle the actual persistence of learning contexts,
/// including storage, retrieval, and deletion operations.
///
/// # Implementations
///
/// - [`InMemoryStore`](crate::storage::InMemoryStore) - HashMap-based storage for testing
/// - [`QdrantStore`](crate::storage::QdrantStore) - Qdrant vector database (feature: `qdrant`)
/// - [`SqliteVecStore`](crate::storage::SqliteVecStore) - SQLite with vectors (feature: `sqlite-vec`)
#[async_trait]
pub trait KnowledgeStore<T>: Send + Sync {
    /// Store an experience with the given key.
    ///
    /// # Arguments
    ///
    /// * `key` - Unique identifier for this experience
    /// * `experience` - The experience to store
    ///
    /// # Errors
    ///
    /// Returns an error if the store operation fails.
    async fn store_experience(&self, key: &str, experience: T) -> Result<()>;

    /// Retrieve relevant contexts based on a query.
    ///
    /// # Arguments
    ///
    /// * `query` - The search query string
    /// * `limit` - Maximum number of results to return
    /// * `domain_filter` - Optional domain to filter by
    ///
    /// # Errors
    ///
    /// Returns an error if the retrieval fails.
    async fn retrieve_context(
        &self,
        query: &str,
        limit: usize,
        domain_filter: Option<AgentDomain>,
    ) -> Result<Vec<T>>;

    /// Delete an experience by key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key of the experience to delete
    ///
    /// # Errors
    ///
    /// Returns an error if the delete operation fails.
    async fn delete_experience(&self, key: &str) -> Result<()>;

    /// Get all keys in the store.
    ///
    /// # Errors
    ///
    /// Returns an error if listing keys fails.
    async fn get_all_keys(&self) -> Result<Vec<String>>;

    /// Get an experience by key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key of the experience to retrieve
    ///
    /// # Errors
    ///
    /// Returns an error if the experience is not found or retrieval fails.
    async fn get_experience(&self, key: &str) -> Result<Option<T>>;

    /// Get the total count of items in the store.
    ///
    /// # Errors
    ///
    /// Returns an error if counting fails.
    async fn count(&self) -> Result<usize> {
        Ok(self.get_all_keys().await?.len())
    }

    /// Clear all items from the store.
    ///
    /// # Errors
    ///
    /// Returns an error if clearing fails.
    async fn clear(&self) -> Result<()>;
}

/// Base trait for memory-enabled agents.
///
/// Memory-enabled agents can process tasks while leveraging stored memories
/// for context-aware decision making.
#[async_trait]
pub trait MemoryEnabledAgent: Send + Sync {
    /// Process a task with optional context and memory storage.
    ///
    /// # Arguments
    ///
    /// * `task_input` - The task to process
    /// * `context` - Optional task-specific context
    /// * `store_memory` - Whether to store the interaction as a memory
    ///
    /// # Returns
    ///
    /// A tuple of (result, confidence) where confidence is in `[0.0, 1.0]`.
    ///
    /// # Errors
    ///
    /// Returns an error if task processing fails.
    async fn process_task(
        &self,
        task_input: &str,
        context: Option<Value>,
        store_memory: bool,
    ) -> Result<(String, f32)>;

    /// Get the domain this agent operates in.
    fn domain(&self) -> AgentDomain;

    /// Get the name of this agent.
    fn name(&self) -> &str;
}

/// Extension trait for knowledge stores that support batch operations.
#[async_trait]
pub trait BatchKnowledgeStore<T>: KnowledgeStore<T> {
    /// Store multiple experiences in a batch.
    ///
    /// # Errors
    ///
    /// Returns an error if any store operation fails.
    async fn store_batch(&self, items: Vec<(String, T)>) -> Result<()>;

    /// Delete multiple experiences by key.
    ///
    /// # Errors
    ///
    /// Returns an error if any delete operation fails.
    async fn delete_batch(&self, keys: &[String]) -> Result<()>;
}

/// Extension trait for stores that support metadata filtering.
#[async_trait]
pub trait FilterableStore<T>: KnowledgeStore<T> {
    /// Retrieve experiences matching a metadata filter.
    ///
    /// # Arguments
    ///
    /// * `filter` - JSON-style filter object
    /// * `limit` - Maximum results to return
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    async fn retrieve_by_filter(&self, filter: Value, limit: usize) -> Result<Vec<(String, T)>>;
}

/// Marker trait for stores that support vector similarity search.
pub trait VectorStore<T>: KnowledgeStore<T> {
    /// Get the embedding dimension used by this store.
    fn embedding_dimension(&self) -> usize;
}

/// Result of a task execution with memory context.
#[derive(Debug, Clone)]
pub struct TaskResult {
    /// The result output.
    pub output: String,

    /// Confidence score in `[0.0, 1.0]`.
    pub confidence: f32,

    /// Memories that were retrieved for context.
    pub retrieved_memories: Vec<LearningContext>,

    /// Whether the result was stored as a new memory.
    pub stored: bool,
}

impl TaskResult {
    /// Create a new task result.
    #[must_use]
    pub fn new(output: impl Into<String>, confidence: f32) -> Self {
        Self {
            output: output.into(),
            confidence: confidence.clamp(0.0, 1.0),
            retrieved_memories: Vec::new(),
            stored: false,
        }
    }

    /// Set the retrieved memories.
    #[must_use]
    pub fn with_memories(mut self, memories: Vec<LearningContext>) -> Self {
        self.retrieved_memories = memories;
        self
    }

    /// Mark as stored.
    #[must_use]
    pub const fn with_stored(mut self, stored: bool) -> Self {
        self.stored = stored;
        self
    }
}
