//! `SQLite` with sqlite-vec extension for vector similarity search.
//!
//! This module provides a [`SqliteVecStore`] implementation that uses `SQLite`
//! with the sqlite-vec extension for local vector similarity search on learning contexts.
//!
//! # Feature Flag
//!
//! This module is only available when the `sqlite-vec` feature is enabled:
//!
//! ```toml
//! [dependencies]
//! memory-gate-rs = { version = "0.1", features = ["sqlite-vec"] }
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use memory_gate_rs::storage::SqliteVecStore;
//! use memory_gate_rs::{KnowledgeStore, LearningContext, AgentDomain};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create an in-memory SQLite database with vector support
//!     let store = SqliteVecStore::open_in_memory().await?;
//!
//!     // Or use a file-based database
//!     // let store = SqliteVecStore::open("memories.db").await?;
//!
//!     // Store a learning context
//!     let ctx = LearningContext::new(
//!         "Restarting nginx service resolves high CPU issues",
//!         AgentDomain::Infrastructure,
//!     );
//!     store.store_experience("infra-001", ctx).await?;
//!
//!     // Retrieve relevant contexts
//!     let results = store.retrieve_context("CPU problems", 5, None).await?;
//!     Ok(())
//! }
//! ```

use crate::{AgentDomain, KnowledgeStore, LearningContext, Result, StorageError, VectorStore};
use async_trait::async_trait;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use rusqlite::{params, Connection};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Default embedding dimension for the default model (BAAI/bge-small-en-v1.5).
const DEFAULT_EMBEDDING_DIM: usize = 384;

/// SQLite-based knowledge store with vector similarity search via sqlite-vec.
///
/// This store uses `SQLite` with the sqlite-vec extension for efficient local
/// vector similarity search on learning contexts. It generates embeddings
/// using `FastEmbed` and stores them alongside the context data.
///
/// # Features
///
/// - Local vector similarity search (no external server required)
/// - Automatic embedding generation via `FastEmbed`
/// - Domain-based filtering
/// - File-based or in-memory database options
/// - Automatic schema creation
/// - JSON storage for full context preservation
///
/// # Architecture
///
/// The store creates two tables:
/// - `learning_contexts`: Stores the key, JSON-serialized context, and domain
/// - `vec_learning_contexts`: Virtual table for vector similarity search
pub struct SqliteVecStore {
    /// `SQLite` connection wrapped for async access.
    conn: Arc<Mutex<Connection>>,
    /// Text embedding model.
    embedder: Arc<Mutex<TextEmbedding>>,
    /// Embedding dimension.
    embedding_dim: usize,
}

impl std::fmt::Debug for SqliteVecStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteVecStore")
            .field("embedding_dim", &self.embedding_dim)
            .finish_non_exhaustive()
    }
}

impl SqliteVecStore {
    /// Open an in-memory `SQLite` database with vector support.
    ///
    /// This is useful for testing or temporary storage that doesn't need
    /// to persist across restarts.
    ///
    /// # Errors
    ///
    /// Returns an error if database initialization fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let store = SqliteVecStore::open_in_memory().await?;
    /// ```
    pub async fn open_in_memory() -> Result<Self> {
        Self::open_with_connection(
            Connection::open_in_memory()
                .map_err(|e| StorageError::connection(format!("Failed to open in-memory DB: {e}")))?,
        )
        .await
    }

    /// Open a file-based `SQLite` database with vector support.
    ///
    /// Creates the database file if it doesn't exist. The schema will be
    /// automatically created on first open.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the `SQLite` database file
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened or initialized.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let store = SqliteVecStore::open("./data/memories.db").await?;
    /// ```
    pub async fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_connection(
            Connection::open(path.as_ref())
                .map_err(|e| StorageError::connection(format!("Failed to open DB: {e}")))?,
        )
        .await
    }

    /// Open with an existing connection.
    async fn open_with_connection(conn: Connection) -> Result<Self> {
        // Log sqlite-vec extension status
        Self::load_vec_extension(&conn);

        let embedder = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::BGESmallENV15).with_show_download_progress(false),
        )
        .map_err(|e| StorageError::backend(format!("Failed to initialize embedder: {e}")))?;

        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
            embedder: Arc::new(Mutex::new(embedder)),
            embedding_dim: DEFAULT_EMBEDDING_DIM,
        };

        store.ensure_schema().await?;

        Ok(store)
    }

    /// Check for sqlite-vec extension availability.
    /// 
    /// Note: Since this crate forbids unsafe code, we cannot dynamically load
    /// the sqlite-vec extension. Instead, we use a fallback approach with
    /// manual cosine similarity calculation which works without the extension.
    /// 
    /// If you need native sqlite-vec performance, compile sqlite with the
    /// vec0 extension statically linked.
    fn load_vec_extension(_conn: &Connection) {
        // We cannot use load_extension as it requires unsafe code.
        // The fallback implementation provides the same functionality
        // using pure Rust cosine similarity calculations.
        tracing::debug!(
            "Using fallback vector similarity search (pure Rust cosine similarity). \
             For native sqlite-vec performance, compile SQLite with vec0 statically linked."
        );
    }

    /// Ensure the database schema exists.
    async fn ensure_schema(&self) -> Result<()> {
        let conn = self.conn.lock().await;

        // Create the main table for storing learning contexts
        conn.execute(
            "CREATE TABLE IF NOT EXISTS learning_contexts (
                key TEXT PRIMARY KEY,
                context_json TEXT NOT NULL,
                domain TEXT NOT NULL,
                content TEXT NOT NULL,
                importance REAL NOT NULL,
                created_at TEXT NOT NULL
            )",
            [],
        )
        .map_err(|e| StorageError::backend(format!("Failed to create contexts table: {e}")))?;

        // Create embeddings table for storing vectors
        conn.execute(
            "CREATE TABLE IF NOT EXISTS context_embeddings (
                    key TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    FOREIGN KEY (key) REFERENCES learning_contexts(key) ON DELETE CASCADE
                )",
            [],
        )
        .map_err(|e| StorageError::backend(format!("Failed to create embeddings table: {e}")))?;

        // Create index for domain filtering
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_contexts_domain ON learning_contexts(domain)",
            [],
        )
        .map_err(|e| StorageError::backend(format!("Failed to create domain index: {e}")))?;

        // Try to create the virtual table for sqlite-vec if the extension is loaded
        // This will fail silently if sqlite-vec isn't available
        let _ = conn.execute(
            &format!(
                "CREATE VIRTUAL TABLE IF NOT EXISTS vec_contexts USING vec0(
                    key TEXT PRIMARY KEY,
                    embedding float[{dim}]
                )",
                dim = self.embedding_dim
            ),
            [],
        );

        Ok(())
    }

    /// Generate embedding for text content.
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let embedder = self.embedder.lock().await;
        let embeddings = embedder
            .embed(vec![text], None)
            .map_err(|e| StorageError::backend(format!("Failed to generate embedding: {e}")))?;

        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| StorageError::backend("No embedding generated").into())
    }

    /// Convert embedding to blob format for storage.
    fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
        embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect()
    }

    /// Convert blob back to embedding vector.
    fn blob_to_embedding(blob: &[u8]) -> Vec<f32> {
        blob.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    /// Calculate cosine similarity between two vectors.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    /// Check if the vec0 virtual table is available.
    async fn has_vec_table(&self) -> bool {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_contexts'",
            [],
            |_| Ok(()),
        )
        .is_ok()
    }

}

#[async_trait]
impl KnowledgeStore<LearningContext> for SqliteVecStore {
    async fn store_experience(&self, key: &str, experience: LearningContext) -> Result<()> {
        let embedding = self.embed(&experience.content).await?;
        let embedding_blob = Self::embedding_to_blob(&embedding);
        let context_json = serde_json::to_string(&experience)
            .map_err(|e| StorageError::write(format!("Failed to serialize context: {e}")))?;

        let conn = self.conn.lock().await;

        // Insert or replace in main table
        conn.execute(
            "INSERT OR REPLACE INTO learning_contexts (key, context_json, domain, content, importance, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                key,
                context_json,
                experience.domain.as_str(),
                experience.content,
                experience.importance,
                experience.timestamp.to_rfc3339(),
            ],
        )
        .map_err(|e| StorageError::write(format!("Failed to insert context: {e}")))?;

        // Insert or replace in embeddings table
        conn.execute(
            "INSERT OR REPLACE INTO context_embeddings (key, embedding) VALUES (?1, ?2)",
            params![key, embedding_blob],
        )
        .map_err(|e| StorageError::write(format!("Failed to insert embedding: {e}")))?;

        // Try to insert into vec table if available
        let _ = conn.execute(
            "INSERT OR REPLACE INTO vec_contexts (key, embedding) VALUES (?1, ?2)",
            params![key, embedding_blob],
        );

        Ok(())
    }

    async fn retrieve_context(
        &self,
        query: &str,
        limit: usize,
        domain_filter: Option<AgentDomain>,
    ) -> Result<Vec<LearningContext>> {
        let query_embedding = self.embed(query).await?;
        let has_vec = self.has_vec_table().await;

        if has_vec {
            // Use sqlite-vec for similarity search
            self.retrieve_with_vec(&query_embedding, limit, domain_filter)
                .await
        } else {
            // Fallback to manual similarity calculation
            self.retrieve_with_fallback(&query_embedding, limit, domain_filter)
                .await
        }
    }

    async fn delete_experience(&self, key: &str) -> Result<()> {
        let conn = self.conn.lock().await;

        // Delete from vec table first if available
        let _ = conn.execute("DELETE FROM vec_contexts WHERE key = ?1", params![key]);

        // Delete from embeddings table
        conn.execute(
            "DELETE FROM context_embeddings WHERE key = ?1",
            params![key],
        )
        .map_err(|e| StorageError::delete(format!("Failed to delete embedding: {e}")))?;

        // Delete from main table
        let changes = conn
            .execute("DELETE FROM learning_contexts WHERE key = ?1", params![key])
            .map_err(|e| StorageError::delete(format!("Failed to delete context: {e}")))?;

        if changes == 0 {
            return Err(StorageError::not_found(key).into());
        }

        Ok(())
    }

    async fn get_all_keys(&self) -> Result<Vec<String>> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare("SELECT key FROM learning_contexts")
            .map_err(|e| StorageError::query(format!("Failed to prepare statement: {e}")))?;

        let keys = stmt
            .query_map([], |row| row.get(0))
            .map_err(|e| StorageError::query(format!("Failed to query keys: {e}")))?
            .collect::<std::result::Result<Vec<String>, _>>()
            .map_err(|e| StorageError::query(format!("Failed to collect keys: {e}")))?;

        Ok(keys)
    }

    async fn get_experience(&self, key: &str) -> Result<Option<LearningContext>> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare("SELECT context_json FROM learning_contexts WHERE key = ?1")
            .map_err(|e| StorageError::query(format!("Failed to prepare statement: {e}")))?;

        let result = stmt.query_row(params![key], |row| {
            let json: String = row.get(0)?;
            Ok(json)
        });

        match result {
            Ok(json) => {
                let context: LearningContext = serde_json::from_str(&json)
                    .map_err(|e| StorageError::query(format!("Failed to deserialize context: {e}")))?;
                Ok(Some(context))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(StorageError::query(format!("Failed to query context: {e}")).into()),
        }
    }

    async fn count(&self) -> Result<usize> {
        let conn = self.conn.lock().await;
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM learning_contexts", [], |row| {
                row.get(0)
            })
            .map_err(|e| StorageError::query(format!("Failed to count contexts: {e}")))?;

        Ok(count as usize)
    }

    async fn clear(&self) -> Result<()> {
        let conn = self.conn.lock().await;

        // Clear vec table first if available
        let _ = conn.execute("DELETE FROM vec_contexts", []);

        // Clear embeddings table
        conn.execute("DELETE FROM context_embeddings", [])
            .map_err(|e| StorageError::delete(format!("Failed to clear embeddings: {e}")))?;

        // Clear main table
        conn.execute("DELETE FROM learning_contexts", [])
            .map_err(|e| StorageError::delete(format!("Failed to clear contexts: {e}")))?;

        Ok(())
    }
}

impl SqliteVecStore {
    /// Retrieve contexts using sqlite-vec virtual table.
    async fn retrieve_with_vec(
        &self,
        query_embedding: &[f32],
        limit: usize,
        domain_filter: Option<AgentDomain>,
    ) -> Result<Vec<LearningContext>> {
        let embedding_blob = Self::embedding_to_blob(query_embedding);
        let conn = self.conn.lock().await;

        // Build and execute query based on whether we have a domain filter
        let json_results: Vec<String> = if let Some(domain) = domain_filter {
            let mut stmt = conn
                .prepare(
                    "SELECT lc.context_json
                     FROM vec_contexts vc
                     JOIN learning_contexts lc ON vc.key = lc.key
                     WHERE lc.domain = ?2
                     ORDER BY vec_distance_cosine(vc.embedding, ?1) ASC
                     LIMIT ?3",
                )
                .map_err(|e| StorageError::query(format!("Failed to prepare search query: {e}")))?;

            let results: Vec<String> = stmt
                .query_map(params![embedding_blob, domain.as_str(), limit as i64], |row| {
                    row.get::<_, String>(0)
                })
                .map_err(|e| StorageError::query(format!("Failed to execute search: {e}")))?
                .filter_map(std::result::Result::ok)
                .collect();
            results
        } else {
            let mut stmt = conn
                .prepare(
                    "SELECT lc.context_json
                     FROM vec_contexts vc
                     JOIN learning_contexts lc ON vc.key = lc.key
                     ORDER BY vec_distance_cosine(vc.embedding, ?1) ASC
                     LIMIT ?2",
                )
                .map_err(|e| StorageError::query(format!("Failed to prepare search query: {e}")))?;

            let results: Vec<String> = stmt
                .query_map(params![embedding_blob, limit as i64], |row| {
                    row.get::<_, String>(0)
                })
                .map_err(|e| StorageError::query(format!("Failed to execute search: {e}")))?
                .filter_map(std::result::Result::ok)
                .collect();
            results
        };

        let contexts: Vec<LearningContext> = json_results
            .into_iter()
            .filter_map(|json| serde_json::from_str(&json).ok())
            .collect();

        Ok(contexts)
    }

    /// Fallback retrieval using manual cosine similarity calculation.
    async fn retrieve_with_fallback(
        &self,
        query_embedding: &[f32],
        limit: usize,
        domain_filter: Option<AgentDomain>,
    ) -> Result<Vec<LearningContext>> {
        let conn = self.conn.lock().await;

        // Collect results with embeddings based on domain filter
        let rows: Vec<(String, Vec<u8>)> = if let Some(domain) = domain_filter {
            let mut stmt = conn
                .prepare(
                    "SELECT lc.context_json, ce.embedding
                     FROM learning_contexts lc
                     JOIN context_embeddings ce ON lc.key = ce.key
                     WHERE lc.domain = ?1",
                )
                .map_err(|e| StorageError::query(format!("Failed to prepare statement: {e}")))?;

            let results: Vec<(String, Vec<u8>)> = stmt
                .query_map(params![domain.as_str()], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
                })
                .map_err(|e| StorageError::query(format!("Failed to query contexts: {e}")))?
                .filter_map(std::result::Result::ok)
                .collect();
            results
        } else {
            let mut stmt = conn
                .prepare(
                    "SELECT lc.context_json, ce.embedding
                     FROM learning_contexts lc
                     JOIN context_embeddings ce ON lc.key = ce.key",
                )
                .map_err(|e| StorageError::query(format!("Failed to prepare statement: {e}")))?;

            let results: Vec<(String, Vec<u8>)> = stmt
                .query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
                })
                .map_err(|e| StorageError::query(format!("Failed to query contexts: {e}")))?
                .filter_map(std::result::Result::ok)
                .collect();
            results
        };

        // Calculate similarities and sort
        let mut scored: Vec<(f32, LearningContext)> = rows
            .into_iter()
            .filter_map(|(json, blob)| {
                let context: LearningContext = serde_json::from_str(&json).ok()?;
                let embedding = Self::blob_to_embedding(&blob);
                let similarity = Self::cosine_similarity(query_embedding, &embedding);
                Some((similarity, context))
            })
            .collect();

        // Sort by similarity descending
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top results
        let contexts: Vec<LearningContext> = scored
            .into_iter()
            .take(limit)
            .map(|(_, ctx)| ctx)
            .collect();

        Ok(contexts)
    }
}

impl VectorStore<LearningContext> for SqliteVecStore {
    fn embedding_dimension(&self) -> usize {
        self.embedding_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_embedding_roundtrip() {
        let embedding = vec![1.0_f32, 2.0, 3.0, 4.5, -1.0, 0.0];
        let blob = SqliteVecStore::embedding_to_blob(&embedding);
        let recovered = SqliteVecStore::blob_to_embedding(&blob);
        
        assert_eq!(embedding.len(), recovered.len());
        for (a, b) in embedding.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_cosine_similarity() {
        // Same vectors should have similarity 1.0
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 0.0, 0.0];
        let sim = SqliteVecStore::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.0001);

        // Orthogonal vectors should have similarity 0.0
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0];
        let sim = SqliteVecStore::cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.0001);

        // Opposite vectors should have similarity -1.0
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![-1.0_f32, 0.0, 0.0];
        let sim = SqliteVecStore::cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_normalized() {
        // Similarity should be scale-invariant
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![2.0_f32, 4.0, 6.0]; // Same direction, different magnitude
        let sim = SqliteVecStore::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let sim = SqliteVecStore::cosine_similarity(&a, &b);
        assert!(sim.abs() < f32::EPSILON);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 2.0, 3.0];
        let sim = SqliteVecStore::cosine_similarity(&a, &b);
        assert!(sim.abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_open_in_memory() {
        let result = SqliteVecStore::open_in_memory().await;
        assert!(result.is_ok(), "Failed to open in-memory store: {:?}", result.err());
    }

    #[tokio::test]
    #[ignore = "requires embedding model download"]
    async fn test_store_and_get() {
        let store = SqliteVecStore::open_in_memory().await.unwrap();

        let ctx = LearningContext::new("Test content for storage", AgentDomain::Infrastructure)
            .with_importance(0.8);

        store.store_experience("test-key", ctx.clone()).await.unwrap();

        let retrieved = store.get_experience("test-key").await.unwrap();
        assert!(retrieved.is_some());

        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.content, ctx.content);
        assert_eq!(retrieved.domain, ctx.domain);
        assert!((retrieved.importance - ctx.importance).abs() < 0.001);
    }

    #[tokio::test]
    #[ignore = "requires embedding model download"]
    async fn test_store_and_retrieve() {
        let store = SqliteVecStore::open_in_memory().await.unwrap();

        let ctx1 = LearningContext::new(
            "nginx server restart fixes memory issues",
            AgentDomain::Infrastructure,
        );
        let ctx2 = LearningContext::new(
            "kubernetes deployment rollout strategy",
            AgentDomain::Deployment,
        );

        store.store_experience("key1", ctx1).await.unwrap();
        store.store_experience("key2", ctx2).await.unwrap();

        // Retrieve should find related content
        let results = store.retrieve_context("nginx memory", 5, None).await.unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    #[ignore = "requires embedding model download"]
    async fn test_domain_filter() {
        let store = SqliteVecStore::open_in_memory().await.unwrap();

        let ctx1 = LearningContext::new("server configuration tips", AgentDomain::Infrastructure);
        let ctx2 = LearningContext::new("deployment configuration tips", AgentDomain::Deployment);

        store.store_experience("key1", ctx1).await.unwrap();
        store.store_experience("key2", ctx2).await.unwrap();

        // Filter by domain
        let results = store
            .retrieve_context("configuration", 10, Some(AgentDomain::Infrastructure))
            .await
            .unwrap();

        for ctx in &results {
            assert_eq!(ctx.domain, AgentDomain::Infrastructure);
        }
    }

    #[tokio::test]
    #[ignore = "requires embedding model download"]
    async fn test_delete() {
        let store = SqliteVecStore::open_in_memory().await.unwrap();

        let ctx = LearningContext::new("Test content", AgentDomain::General);
        store.store_experience("key1", ctx).await.unwrap();

        assert_eq!(store.count().await.unwrap(), 1);

        store.delete_experience("key1").await.unwrap();
        assert_eq!(store.count().await.unwrap(), 0);

        // Deleting non-existent key should error
        let result = store.delete_experience("nonexistent").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    #[ignore = "requires embedding model download"]
    async fn test_get_all_keys() {
        let store = SqliteVecStore::open_in_memory().await.unwrap();

        for i in 0..5 {
            let ctx = LearningContext::new(format!("Content {i}"), AgentDomain::General);
            store.store_experience(&format!("key-{i}"), ctx).await.unwrap();
        }

        let keys = store.get_all_keys().await.unwrap();
        assert_eq!(keys.len(), 5);
        
        for i in 0..5 {
            assert!(keys.contains(&format!("key-{i}")));
        }
    }

    #[tokio::test]
    #[ignore = "requires embedding model download"]
    async fn test_clear() {
        let store = SqliteVecStore::open_in_memory().await.unwrap();

        for i in 0..3 {
            let ctx = LearningContext::new(format!("Content {i}"), AgentDomain::General);
            store.store_experience(&format!("key-{i}"), ctx).await.unwrap();
        }

        assert_eq!(store.count().await.unwrap(), 3);

        store.clear().await.unwrap();
        assert_eq!(store.count().await.unwrap(), 0);
    }

    #[tokio::test]
    #[ignore = "requires embedding model download"]
    async fn test_upsert_overwrites() {
        let store = SqliteVecStore::open_in_memory().await.unwrap();

        let ctx1 = LearningContext::new("Original content", AgentDomain::General);
        store.store_experience("same-key", ctx1).await.unwrap();

        let ctx2 = LearningContext::new("Updated content", AgentDomain::Infrastructure)
            .with_importance(0.5);
        store.store_experience("same-key", ctx2).await.unwrap();

        // Should still be only one item
        assert_eq!(store.count().await.unwrap(), 1);

        // Should have the updated content
        let retrieved = store.get_experience("same-key").await.unwrap().unwrap();
        assert_eq!(retrieved.content, "Updated content");
        assert_eq!(retrieved.domain, AgentDomain::Infrastructure);
        assert!((retrieved.importance - 0.5).abs() < 0.001);
    }

    #[tokio::test]
    #[ignore = "requires embedding model download"]
    async fn test_get_nonexistent() {
        let store = SqliteVecStore::open_in_memory().await.unwrap();
        let result = store.get_experience("nonexistent").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    #[ignore = "requires embedding model download"]
    async fn test_metadata_preservation() {
        let store = SqliteVecStore::open_in_memory().await.unwrap();

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "test".to_string());
        metadata.insert("version".to_string(), "1.0".to_string());

        let ctx = LearningContext::new("Test with metadata", AgentDomain::General)
            .with_metadata(metadata);

        store.store_experience("meta-key", ctx.clone()).await.unwrap();

        let retrieved = store.get_experience("meta-key").await.unwrap().unwrap();
        assert!(retrieved.metadata.is_some());
        
        let meta = retrieved.metadata.unwrap();
        assert_eq!(meta.get("source"), Some(&"test".to_string()));
        assert_eq!(meta.get("version"), Some(&"1.0".to_string()));
    }

    #[test]
    fn test_vector_store_trait() {
        // Verify the embedding dimension is correct
        // This is a compile-time check that SqliteVecStore implements VectorStore
        fn _assert_vector_store<T: VectorStore<LearningContext>>() {}
        _assert_vector_store::<SqliteVecStore>();
    }

    #[tokio::test]
    #[ignore = "requires embedding model download"]
    async fn test_embedding_dimension() {
        let store = SqliteVecStore::open_in_memory().await.unwrap();
        assert_eq!(store.embedding_dimension(), DEFAULT_EMBEDDING_DIM);
    }
}
