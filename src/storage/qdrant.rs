//! Qdrant vector database storage backend.
//!
//! This module provides a [`QdrantStore`] implementation that uses the Qdrant
//! vector database for high-performance similarity search on learning contexts.
//!
//! # Feature Flag
//!
//! This module is only available when the `qdrant` feature is enabled:
//!
//! ```toml
//! [dependencies]
//! memory-gate-rs = { version = "0.1", features = ["qdrant"] }
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use memory_gate_rs::storage::QdrantStore;
//! use memory_gate_rs::{KnowledgeStore, LearningContext, AgentDomain};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Connect to Qdrant running locally
//!     let store = QdrantStore::new("http://localhost:6334", "memories").await?;
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

use crate::embedding::cache::{QueryEmbeddingCache, DEFAULT_QUERY_CACHE_CAPACITY};
use crate::embedding::{embed_batch, init_text_embedding, SupportedEmbeddingModel};
use crate::{
    AgentDomain, BatchKnowledgeStore, Error, KnowledgeStore, LearningContext, Result, StorageError,
    VectorStore,
};
use async_trait::async_trait;
use fastembed::TextEmbedding;
use qdrant_client::qdrant::{
    vectors_config, CollectionInfo, Condition, CreateCollectionBuilder, DeletePointsBuilder,
    Distance, Filter, GetPointsBuilder, PointId, PointStruct, PointsIdsList,
    ScalarQuantizationBuilder, ScrollPointsBuilder, SearchParamsBuilder, SearchPointsBuilder,
    UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::Qdrant;
use serde_json::Value;
use std::collections::HashMap;

/// Collection metadata key for catalog embedding model id (`mg/embed-catalog`).
const COLLECTION_META_EMBEDDING_MODEL: &str = "mg_embedding_model";
/// Collection metadata key for embedding dimension.
const COLLECTION_META_EMBEDDING_DIM: &str = "mg_embedding_dim";
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

/// Qdrant-backed knowledge store with vector similarity search.
///
/// This store uses Qdrant for efficient vector similarity search on learning
/// contexts. It generates embeddings using `FastEmbed` and stores them alongside
/// the context metadata.
///
/// # Features
///
/// - High-performance vector similarity search
/// - Automatic embedding generation via `FastEmbed`
/// - Domain-based filtering
/// - Automatic collection creation with optimal settings
/// - Payload storage for full context retrieval
///
/// # Architecture
///
/// Each learning context is stored as a Qdrant point with:
/// - ID: Hash of the key for consistent point identification
/// - Vector: Embedding of the context content
/// - Payload: Full serialized `LearningContext` plus the original key
pub struct QdrantStore {
    /// Qdrant client connection.
    client: Arc<Qdrant>,
    /// Collection name in Qdrant.
    collection_name: String,
    /// Text embedding model.
    embedder: Arc<RwLock<TextEmbedding>>,
    /// Catalog model bound to this collection.
    model: SupportedEmbeddingModel,
    /// Embedding dimension.
    embedding_dim: usize,
    /// LRU cache for retrieve-query embeddings (text + model id).
    query_embed_cache: Arc<Mutex<QueryEmbeddingCache>>,
}

impl std::fmt::Debug for QdrantStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QdrantStore")
            .field("collection_name", &self.collection_name)
            .field("model", &self.model.as_str())
            .field("embedding_dim", &self.embedding_dim)
            .finish_non_exhaustive()
    }
}

impl QdrantStore {
    /// Create a new Qdrant store with the default embedding model (`bge-small-en-v1.5`).
    ///
    /// This will create the collection if it doesn't exist, using optimal
    /// settings for similarity search.
    ///
    /// # Arguments
    ///
    /// * `url` - Qdrant server URL (e.g., `http://localhost:6334`)
    /// * `collection_name` - Name of the collection to use
    ///
    /// # Errors
    ///
    /// Returns an error if connection fails or collection creation fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let store = QdrantStore::new("http://localhost:6334", "my_memories").await?;
    /// ```
    pub async fn new(url: &str, collection_name: &str) -> Result<Self> {
        Self::with_model(url, collection_name, SupportedEmbeddingModel::DEFAULT).await
    }

    /// Create a new Qdrant store bound to a catalog embedding model.
    ///
    /// The collection vector size is set from [`SupportedEmbeddingModel::dimension`].
    /// Prefer this over [`Self::with_dimension`] so model identity is explicit.
    ///
    /// # Errors
    ///
    /// Returns an error if connection, embedder init, or collection creation fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use memory_gate_rs::{storage::QdrantStore, SupportedEmbeddingModel};
    /// let store = QdrantStore::with_model(
    ///     "http://localhost:6334",
    ///     "memories_minilm",
    ///     SupportedEmbeddingModel::AllMiniLmL6V2,
    /// ).await?;
    /// ```
    pub async fn with_model(
        url: &str,
        collection_name: &str,
        model: SupportedEmbeddingModel,
    ) -> Result<Self> {
        let client = Qdrant::from_url(url)
            .build()
            .map_err(|e| StorageError::connection(format!("Failed to connect to Qdrant: {e}")))?;

        let embedder =
            init_text_embedding(model).map_err(|e| StorageError::backend(e.to_string()))?;

        let store = Self {
            client: Arc::new(client),
            collection_name: collection_name.to_string(),
            embedder: Arc::new(RwLock::new(embedder)),
            model,
            embedding_dim: model.dimension(),
            query_embed_cache: Arc::new(Mutex::new(QueryEmbeddingCache::with_capacity(
                DEFAULT_QUERY_CACHE_CAPACITY,
            ))),
        };

        store.ensure_collection().await?;

        Ok(store)
    }

    /// Create a new Qdrant store with a custom embedding dimension.
    ///
    /// Prefer [`Self::with_model`] so the embedding model is explicit. This
    /// method keeps historical API compatibility: it always loads the default
    /// catalog model (`bge-small-en-v1.5`) but sets the collection size to
    /// `embedding_dim`. Mismatching dim and model output will fail at upsert.
    ///
    /// # Errors
    ///
    /// Returns an error if connection fails or collection creation fails.
    pub async fn with_dimension(
        url: &str,
        collection_name: &str,
        embedding_dim: usize,
    ) -> Result<Self> {
        let client = Qdrant::from_url(url)
            .build()
            .map_err(|e| StorageError::connection(format!("Failed to connect to Qdrant: {e}")))?;

        let model = SupportedEmbeddingModel::DEFAULT;
        let embedder =
            init_text_embedding(model).map_err(|e| StorageError::backend(e.to_string()))?;

        let store = Self {
            client: Arc::new(client),
            collection_name: collection_name.to_string(),
            embedder: Arc::new(RwLock::new(embedder)),
            model,
            embedding_dim,
            query_embed_cache: Arc::new(Mutex::new(QueryEmbeddingCache::with_capacity(
                DEFAULT_QUERY_CACHE_CAPACITY,
            ))),
        };

        store.ensure_collection().await?;

        Ok(store)
    }

    /// Catalog embedding model bound to this store.
    #[must_use]
    pub const fn embedding_model(&self) -> SupportedEmbeddingModel {
        self.model
    }

    /// Ensure the collection exists with proper configuration.
    async fn ensure_collection(&self) -> Result<()> {
        let exists = self
            .client
            .collection_exists(&self.collection_name)
            .await
            .map_err(|e| StorageError::connection(format!("Failed to check collection: {e}")))?;

        if !exists {
            let mut metadata = HashMap::new();
            metadata.insert(
                COLLECTION_META_EMBEDDING_MODEL.to_string(),
                Value::String(self.model.as_str().to_string()),
            );
            metadata.insert(
                COLLECTION_META_EMBEDDING_DIM.to_string(),
                Value::Number(serde_json::Number::from(self.embedding_dim)),
            );
            self.client
                .create_collection(
                    CreateCollectionBuilder::new(&self.collection_name)
                        .vectors_config(VectorParamsBuilder::new(
                            self.embedding_dim as u64,
                            Distance::Cosine,
                        ))
                        .quantization_config(ScalarQuantizationBuilder::default())
                        .metadata(metadata),
                )
                .await
                .map_err(|e| StorageError::backend(format!("Failed to create collection: {e}")))?;
            return Ok(());
        }

        let info = self
            .client
            .collection_info(&self.collection_name)
            .await
            .map_err(|e| StorageError::connection(format!("Failed to get collection info: {e}")))?;

        let info = info
            .result
            .ok_or_else(|| StorageError::backend("collection info missing from Qdrant response"))?;

        Self::validate_existing_collection(
            &self.collection_name,
            &info,
            self.model,
            self.embedding_dim,
        )?;

        Ok(())
    }

    /// Fail closed when an existing collection's vector size or stamped model disagrees with this store.
    fn validate_existing_collection(
        collection_name: &str,
        info: &CollectionInfo,
        model: SupportedEmbeddingModel,
        expected_dim: usize,
    ) -> Result<()> {
        let actual_dim = collection_vector_dimension(info).ok_or_else(|| {
            Error::invalid_config(format!(
                "collection '{collection_name}' has no vector size in config; cannot verify embedding dimension for model '{}'",
                model.as_str()
            ))
        })?;

        if actual_dim != expected_dim as u64 {
            return Err(Error::invalid_config(format!(
                "collection embedding dimension mismatch: expected {} for model '{}', found {}",
                expected_dim,
                model.as_str(),
                actual_dim
            )));
        }

        if let Some(stored_model) =
            collection_metadata_string(info, COLLECTION_META_EMBEDDING_MODEL)
        {
            if stored_model != model.as_str() {
                return Err(Error::invalid_config(format!(
                    "collection embedding model mismatch: expected '{}', found '{}' (dimension {})",
                    model.as_str(),
                    stored_model,
                    expected_dim
                )));
            }
        }

        if let Some(stored_dim) = collection_metadata_usize(info, COLLECTION_META_EMBEDDING_DIM) {
            if stored_dim != expected_dim {
                return Err(Error::invalid_config(format!(
                    "collection metadata embedding_dim mismatch: expected {}, found {} (model '{}')",
                    expected_dim,
                    stored_dim,
                    model.as_str()
                )));
            }
        }

        Ok(())
    }

    /// Generate embedding for document content (store path; not query-cached).
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut embedder = self.embedder.write().await;
        let embeddings = embed_batch(&mut embedder, &[text])
            .map_err(|e| StorageError::backend(e.to_string()))?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| StorageError::backend("No embedding generated").into())
    }

    /// Batch-embed multiple strings under one embedder lock.
    async fn embed_batch_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let mut embedder = self.embedder.write().await;
        Ok(embed_batch(&mut embedder, &refs).map_err(|e| StorageError::backend(e.to_string()))?)
    }

    /// Embed a retrieve query, using the bounded LRU cache when possible.
    async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        {
            let mut cache = self.query_embed_cache.lock().await;
            if let Some(hit) = cache.get(query, self.model) {
                return Ok(hit);
            }
        }
        let embedding = self.embed(query).await?;
        let mut cache = self.query_embed_cache.lock().await;
        cache.insert(query, self.model, embedding.clone());
        Ok(embedding)
    }

    /// Convert a key to a consistent point ID.
    fn key_to_point_id(key: &str) -> u64 {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(key.as_bytes());
        let result = hasher.finalize();
        // Take first 8 bytes as u64
        u64::from_le_bytes(result[..8].try_into().unwrap())
    }

    /// Serialize a learning context to a JSON payload.
    fn context_to_payload(key: &str, ctx: &LearningContext) -> Result<HashMap<String, Value>> {
        let mut payload = HashMap::new();
        payload.insert("key".to_string(), Value::String(key.to_string()));
        payload.insert("content".to_string(), Value::String(ctx.content.clone()));
        payload.insert(
            "domain".to_string(),
            Value::String(ctx.domain.as_str().to_string()),
        );
        payload.insert(
            "timestamp".to_string(),
            Value::String(ctx.timestamp.to_rfc3339()),
        );
        payload.insert(
            "importance".to_string(),
            Value::Number(serde_json::Number::from_f64(f64::from(ctx.importance)).unwrap()),
        );
        if let Some(ref meta) = ctx.metadata {
            payload.insert("metadata".to_string(), serde_json::to_value(meta)?);
        }
        Ok(payload)
    }

    /// Deserialize a learning context from a Qdrant payload.
    fn payload_to_context(payload: &HashMap<String, Value>) -> Result<(String, LearningContext)> {
        let key = payload
            .get("key")
            .and_then(|v| v.as_str())
            .ok_or_else(|| StorageError::backend("Missing key in payload"))?
            .to_string();

        let content = payload
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| StorageError::backend("Missing content in payload"))?
            .to_string();

        let domain_str = payload
            .get("domain")
            .and_then(|v| v.as_str())
            .ok_or_else(|| StorageError::backend("Missing domain in payload"))?;

        let domain: AgentDomain = domain_str
            .parse()
            .map_err(|e| StorageError::backend(format!("Invalid domain: {e}")))?;

        let timestamp_str = payload
            .get("timestamp")
            .and_then(|v| v.as_str())
            .ok_or_else(|| StorageError::backend("Missing timestamp in payload"))?;

        let timestamp = chrono::DateTime::parse_from_rfc3339(timestamp_str)
            .map_err(|e| StorageError::backend(format!("Invalid timestamp: {e}")))?
            .with_timezone(&chrono::Utc);

        let importance = payload
            .get("importance")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| StorageError::backend("Missing importance in payload"))?
            as f32;

        let metadata = payload
            .get("metadata")
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        let mut ctx = LearningContext::new(content, domain)
            .with_importance(importance)
            .with_timestamp(timestamp);

        if let Some(meta) = metadata {
            ctx = ctx.with_metadata(meta);
        }

        Ok((key, ctx))
    }

    /// Get the collection name.
    #[must_use]
    pub fn collection_name(&self) -> &str {
        &self.collection_name
    }
}

#[async_trait]
impl KnowledgeStore<LearningContext> for QdrantStore {
    async fn store_experience(&self, key: &str, experience: LearningContext) -> Result<()> {
        let embedding = self.embed(&experience.content).await?;
        let point_id = Self::key_to_point_id(key);
        let payload = Self::context_to_payload(key, &experience)?;

        let point = PointStruct::new(point_id, embedding, payload);

        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.collection_name, vec![point]).wait(true))
            .await
            .map_err(|e| StorageError::write(format!("Failed to upsert point: {e}")))?;

        Ok(())
    }

    async fn retrieve_context(
        &self,
        query: &str,
        limit: usize,
        domain_filter: Option<AgentDomain>,
    ) -> Result<Vec<LearningContext>> {
        let query_embedding = self.embed_query(query).await?;

        let mut search_builder =
            SearchPointsBuilder::new(&self.collection_name, query_embedding, limit as u64)
                .with_payload(true)
                .params(SearchParamsBuilder::default().exact(false));

        // Apply domain filter if specified
        if let Some(domain) = domain_filter {
            let filter = Filter::must([Condition::matches("domain", domain.as_str().to_string())]);
            search_builder = search_builder.filter(filter);
        }

        let results = self
            .client
            .search_points(search_builder)
            .await
            .map_err(|e| StorageError::query(format!("Search failed: {e}")))?;

        let mut contexts = Vec::with_capacity(results.result.len());
        for point in results.result {
            let payload: HashMap<String, Value> = point
                .payload
                .into_iter()
                .map(|(k, v)| (k, qdrant_value_to_json(v)))
                .collect();

            if let Ok((_, ctx)) = Self::payload_to_context(&payload) {
                contexts.push(ctx);
            }
        }

        Ok(contexts)
    }

    async fn delete_experience(&self, key: &str) -> Result<()> {
        let point_id = Self::key_to_point_id(key);

        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.collection_name)
                    .points(PointsIdsList {
                        ids: vec![PointId::from(point_id)],
                    })
                    .wait(true),
            )
            .await
            .map_err(|e| StorageError::delete(format!("Failed to delete point: {e}")))?;

        Ok(())
    }

    async fn get_all_keys(&self) -> Result<Vec<String>> {
        // Scroll through all points to get keys
        let mut keys = Vec::new();
        let mut offset: Option<PointId> = None;

        loop {
            let mut scroll_builder = ScrollPointsBuilder::new(&self.collection_name)
                .with_payload(true)
                .limit(100);

            if let Some(ref off) = offset {
                scroll_builder = scroll_builder.offset(off.clone());
            }

            let result = self
                .client
                .scroll(scroll_builder)
                .await
                .map_err(|e| StorageError::query(format!("Scroll failed: {e}")))?;

            for point in &result.result {
                if let Some(key_value) = point.payload.get("key") {
                    if let Some(key_str) = qdrant_value_to_json(key_value.clone()).as_str() {
                        keys.push(key_str.to_string());
                    }
                }
            }

            if result.result.len() < 100 {
                break;
            }

            offset = result.next_page_offset;
            if offset.is_none() {
                break;
            }
        }

        Ok(keys)
    }

    async fn get_experience(&self, key: &str) -> Result<Option<LearningContext>> {
        let point_id = Self::key_to_point_id(key);

        let result = self
            .client
            .get_points(
                GetPointsBuilder::new(&self.collection_name, vec![point_id.into()])
                    .with_payload(true),
            )
            .await
            .map_err(|e| StorageError::query(format!("Get point failed: {e}")))?;

        if let Some(point) = result.result.into_iter().next() {
            let payload: HashMap<String, Value> = point
                .payload
                .into_iter()
                .map(|(k, v)| (k, qdrant_value_to_json(v)))
                .collect();

            let (_, ctx) = Self::payload_to_context(&payload)?;
            return Ok(Some(ctx));
        }

        Ok(None)
    }

    async fn count(&self) -> Result<usize> {
        let info = self
            .client
            .collection_info(&self.collection_name)
            .await
            .map_err(|e| StorageError::query(format!("Failed to get collection info: {e}")))?;

        Ok(info
            .result
            .map_or(0, |r| r.points_count.unwrap_or(0) as usize))
    }

    async fn clear(&self) -> Result<()> {
        // Delete the collection and recreate it
        self.client
            .delete_collection(&self.collection_name)
            .await
            .map_err(|e| StorageError::delete(format!("Failed to delete collection: {e}")))?;

        self.ensure_collection().await?;

        Ok(())
    }
}

#[async_trait]
impl BatchKnowledgeStore<LearningContext> for QdrantStore {
    async fn store_batch(&self, items: Vec<(String, LearningContext)>) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        let texts: Vec<String> = items.iter().map(|(_, ctx)| ctx.content.clone()).collect();
        let embeddings = self.embed_batch_texts(&texts).await?;
        let mut points = Vec::with_capacity(items.len());
        for ((key, experience), embedding) in items.into_iter().zip(embeddings) {
            let point_id = Self::key_to_point_id(&key);
            let payload = Self::context_to_payload(&key, &experience)?;
            points.push(PointStruct::new(point_id, embedding, payload));
        }
        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.collection_name, points).wait(true))
            .await
            .map_err(|e| StorageError::write(format!("Failed to batch upsert: {e}")))?;
        Ok(())
    }

    async fn delete_batch(&self, keys: &[String]) -> Result<()> {
        if keys.is_empty() {
            return Ok(());
        }
        let ids: Vec<PointId> = keys
            .iter()
            .map(|k| PointId::from(Self::key_to_point_id(k)))
            .collect();
        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.collection_name)
                    .points(PointsIdsList { ids })
                    .wait(true),
            )
            .await
            .map_err(|e| StorageError::delete(format!("Failed to batch delete: {e}")))?;
        Ok(())
    }
}

impl VectorStore<LearningContext> for QdrantStore {
    fn embedding_dimension(&self) -> usize {
        self.embedding_dim
    }
}

/// Read the default (unnamed) vector dimension from collection info.
fn collection_vector_dimension(info: &CollectionInfo) -> Option<u64> {
    let vectors = info
        .config
        .as_ref()?
        .params
        .as_ref()?
        .vectors_config
        .as_ref()?;
    match &vectors.config {
        Some(vectors_config::Config::Params(p)) => Some(p.size),
        Some(vectors_config::Config::ParamsMap(m)) => m
            .map
            .get("")
            .or_else(|| m.map.values().next())
            .map(|p| p.size),
        None => None,
    }
}

fn collection_metadata_string(info: &CollectionInfo, key: &str) -> Option<String> {
    let value = info.config.as_ref()?.metadata.get(key)?;
    qdrant_metadata_value_as_str(value)
}

fn collection_metadata_usize(info: &CollectionInfo, key: &str) -> Option<usize> {
    let value = info.config.as_ref()?.metadata.get(key)?;
    match &value.kind {
        Some(qdrant_client::qdrant::value::Kind::IntegerValue(i)) => usize::try_from(*i).ok(),
        Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => s.parse().ok(),
        _ => None,
    }
}

fn qdrant_metadata_value_as_str(value: &qdrant_client::qdrant::Value) -> Option<String> {
    use qdrant_client::qdrant::value::Kind;
    match &value.kind {
        Some(Kind::StringValue(s)) => Some(s.clone()),
        _ => None,
    }
}

/// Convert a Qdrant Value to a `serde_json` Value.
fn qdrant_value_to_json(value: qdrant_client::qdrant::Value) -> Value {
    use qdrant_client::qdrant::value::Kind;

    match value.kind {
        Some(Kind::NullValue(_)) | None => Value::Null,
        Some(Kind::BoolValue(b)) => Value::Bool(b),
        Some(Kind::IntegerValue(i)) => Value::Number(i.into()),
        Some(Kind::DoubleValue(d)) => {
            serde_json::Number::from_f64(d).map_or(Value::Null, Value::Number)
        }
        Some(Kind::StringValue(s)) => Value::String(s),
        Some(Kind::ListValue(list)) => {
            Value::Array(list.values.into_iter().map(qdrant_value_to_json).collect())
        }
        Some(Kind::StructValue(s)) => Value::Object(
            s.fields
                .into_iter()
                .map(|(k, v)| (k, qdrant_value_to_json(v)))
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_key_to_point_id_consistency() {
        let key = "test-key-123";
        let id1 = QdrantStore::key_to_point_id(key);
        let id2 = QdrantStore::key_to_point_id(key);
        assert_eq!(id1, id2);

        // Different keys should produce different IDs
        let id3 = QdrantStore::key_to_point_id("different-key");
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_context_to_payload_and_back() {
        let key = "test-key";
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "test".to_string());

        let ctx = LearningContext::new("Test content for embedding", AgentDomain::Infrastructure)
            .with_importance(0.75)
            .with_metadata(metadata);

        let payload = QdrantStore::context_to_payload(key, &ctx).unwrap();

        // Verify payload contents
        assert_eq!(payload.get("key").unwrap().as_str().unwrap(), key);
        assert_eq!(
            payload.get("content").unwrap().as_str().unwrap(),
            "Test content for embedding"
        );
        assert_eq!(
            payload.get("domain").unwrap().as_str().unwrap(),
            "infrastructure"
        );
        assert!((payload.get("importance").unwrap().as_f64().unwrap() - 0.75).abs() < 0.001);

        // Round-trip test
        let (recovered_key, recovered_ctx) = QdrantStore::payload_to_context(&payload).unwrap();
        assert_eq!(recovered_key, key);
        assert_eq!(recovered_ctx.content, ctx.content);
        assert_eq!(recovered_ctx.domain, ctx.domain);
        assert!((recovered_ctx.importance - ctx.importance).abs() < 0.001);
        assert!(recovered_ctx.metadata.is_some());
        assert_eq!(
            recovered_ctx.metadata.as_ref().unwrap().get("source"),
            Some(&"test".to_string())
        );
    }

    #[test]
    fn test_payload_to_context_missing_fields() {
        let mut payload = HashMap::new();
        payload.insert("content".to_string(), Value::String("test".to_string()));
        // Missing key field
        let result = QdrantStore::payload_to_context(&payload);
        assert!(result.is_err());
    }

    #[test]
    fn test_qdrant_value_to_json() {
        use qdrant_client::qdrant::value::Kind;
        use qdrant_client::qdrant::Value as QdrantValue;

        // Test string conversion
        let string_val = QdrantValue {
            kind: Some(Kind::StringValue("hello".to_string())),
        };
        assert_eq!(
            qdrant_value_to_json(string_val),
            Value::String("hello".to_string())
        );

        // Test integer conversion
        let int_val = QdrantValue {
            kind: Some(Kind::IntegerValue(42)),
        };
        assert_eq!(qdrant_value_to_json(int_val), Value::Number(42.into()));

        // Test bool conversion
        let bool_val = QdrantValue {
            kind: Some(Kind::BoolValue(true)),
        };
        assert_eq!(qdrant_value_to_json(bool_val), Value::Bool(true));

        // Test null conversion
        let null_val = QdrantValue {
            kind: Some(Kind::NullValue(0)),
        };
        assert_eq!(qdrant_value_to_json(null_val), Value::Null);

        // Test None kind
        let none_val = QdrantValue { kind: None };
        assert_eq!(qdrant_value_to_json(none_val), Value::Null);
    }

    fn test_collection_info_with_dim(
        vector_size: u64,
        metadata: HashMap<String, qdrant_client::qdrant::Value>,
    ) -> CollectionInfo {
        use qdrant_client::qdrant::{
            CollectionConfig, CollectionInfo, CollectionParams, VectorParams, VectorsConfig,
        };

        CollectionInfo {
            status: 0,
            optimizer_status: None,
            segments_count: 1,
            config: Some(CollectionConfig {
                params: Some(CollectionParams {
                    shard_number: 1,
                    on_disk_payload: false,
                    vectors_config: Some(VectorsConfig {
                        config: Some(vectors_config::Config::Params(VectorParams {
                            size: vector_size,
                            distance: 0,
                            hnsw_config: None,
                            quantization_config: None,
                            on_disk: None,
                            datatype: None,
                            multivector_config: None,
                        })),
                    }),
                    replication_factor: None,
                    write_consistency_factor: None,
                    read_fan_out_factor: None,
                    sharding_method: None,
                    sparse_vectors_config: None,
                    read_fan_out_delay_ms: None,
                }),
                hnsw_config: None,
                optimizer_config: None,
                wal_config: None,
                quantization_config: None,
                strict_mode_config: None,
                metadata,
            }),
            payload_schema: HashMap::new(),
            points_count: Some(0),
            indexed_vectors_count: None,
            warnings: vec![],
            update_queue: None,
        }
    }

    #[test]
    fn test_collection_vector_dimension_from_params() {
        let info = test_collection_info_with_dim(384, HashMap::new());
        assert_eq!(collection_vector_dimension(&info), Some(384));
    }

    #[test]
    fn test_validate_existing_collection_dim_mismatch() {
        let info = test_collection_info_with_dim(768, HashMap::new());

        let err = QdrantStore::validate_existing_collection(
            "memories",
            &info,
            SupportedEmbeddingModel::BgeSmallEnV15,
            384,
        )
        .unwrap_err();
        assert!(err.to_string().contains("dimension mismatch"));
        assert!(err.to_string().contains("768"));
    }

    #[test]
    fn test_validate_existing_collection_model_mismatch_same_dim() {
        use qdrant_client::qdrant::{value::Kind, Value as QdrantValue};

        let mut metadata = HashMap::new();
        metadata.insert(
            COLLECTION_META_EMBEDDING_MODEL.to_string(),
            QdrantValue {
                kind: Some(Kind::StringValue("all-minilm-l6-v2".to_string())),
            },
        );

        let info = test_collection_info_with_dim(384, metadata);

        let err = QdrantStore::validate_existing_collection(
            "memories",
            &info,
            SupportedEmbeddingModel::BgeSmallEnV15,
            384,
        )
        .unwrap_err();
        assert!(err.to_string().contains("model mismatch"));
    }

    #[test]
    fn test_domain_serialization() {
        for domain in AgentDomain::all() {
            let ctx = LearningContext::new("test", *domain);
            let payload = QdrantStore::context_to_payload("key", &ctx).unwrap();
            let (_, recovered) = QdrantStore::payload_to_context(&payload).unwrap();
            assert_eq!(recovered.domain, *domain);
        }
    }

    #[test]
    fn test_timestamp_roundtrip() {
        use chrono::TimeZone;

        let timestamp = chrono::Utc
            .with_ymd_and_hms(2024, 6, 15, 12, 30, 45)
            .unwrap();
        let ctx = LearningContext::new("test", AgentDomain::General).with_timestamp(timestamp);

        let payload = QdrantStore::context_to_payload("key", &ctx).unwrap();
        let (_, recovered) = QdrantStore::payload_to_context(&payload).unwrap();

        // Timestamps should match to the second
        assert_eq!(
            recovered.timestamp.format("%Y-%m-%d %H:%M:%S").to_string(),
            timestamp.format("%Y-%m-%d %H:%M:%S").to_string()
        );
    }

    #[test]
    fn test_context_without_metadata() {
        let ctx = LearningContext::new("test content", AgentDomain::Deployment);
        assert!(ctx.metadata.is_none());

        let payload = QdrantStore::context_to_payload("key", &ctx).unwrap();
        assert!(!payload.contains_key("metadata"));

        let (_, recovered) = QdrantStore::payload_to_context(&payload).unwrap();
        assert!(recovered.metadata.is_none());
    }

    // Integration tests that require a running Qdrant instance
    // These are marked with #[ignore] and can be run with:
    // cargo test --features qdrant -- --ignored

    #[tokio::test]
    #[ignore = "requires running Qdrant instance"]
    async fn test_store_and_retrieve() {
        let store = QdrantStore::new("http://localhost:6334", "test_memories")
            .await
            .expect("Failed to create store");

        // Clear any existing data
        store.clear().await.expect("Failed to clear");

        let ctx = LearningContext::new(
            "Restarting nginx service resolves high CPU issues",
            AgentDomain::Infrastructure,
        )
        .with_importance(0.9);

        store
            .store_experience("test-key-1", ctx.clone())
            .await
            .expect("Failed to store");

        // Retrieve by similarity
        let results = store
            .retrieve_context("nginx CPU problems", 5, None)
            .await
            .expect("Failed to retrieve");

        assert!(!results.is_empty());
        assert!(results[0].content.contains("nginx"));
    }

    #[tokio::test]
    #[ignore = "requires running Qdrant instance"]
    async fn test_domain_filtering() {
        let store = QdrantStore::new("http://localhost:6334", "test_domain_filter")
            .await
            .expect("Failed to create store");

        store.clear().await.expect("Failed to clear");

        let ctx1 = LearningContext::new("nginx service management", AgentDomain::Infrastructure);
        let ctx2 = LearningContext::new("code review best practices", AgentDomain::CodeReview);

        store.store_experience("key1", ctx1).await.unwrap();
        store.store_experience("key2", ctx2).await.unwrap();

        // Search with domain filter
        let results = store
            .retrieve_context(
                "management practices",
                10,
                Some(AgentDomain::Infrastructure),
            )
            .await
            .unwrap();

        assert!(!results.is_empty());
        for ctx in &results {
            assert_eq!(ctx.domain, AgentDomain::Infrastructure);
        }
    }

    #[tokio::test]
    #[ignore = "requires running Qdrant instance"]
    async fn test_delete_and_get() {
        let store = QdrantStore::new("http://localhost:6334", "test_delete")
            .await
            .expect("Failed to create store");

        store.clear().await.expect("Failed to clear");

        let ctx = LearningContext::new("test delete operation", AgentDomain::General);
        store.store_experience("delete-key", ctx).await.unwrap();

        // Verify it exists
        let retrieved = store.get_experience("delete-key").await.unwrap();
        assert!(retrieved.is_some());

        // Delete it
        store.delete_experience("delete-key").await.unwrap();

        // Verify it's gone
        let retrieved = store.get_experience("delete-key").await.unwrap();
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    #[ignore = "requires running Qdrant instance"]
    async fn test_count_and_keys() {
        let store = QdrantStore::new("http://localhost:6334", "test_count")
            .await
            .expect("Failed to create store");

        store.clear().await.expect("Failed to clear");

        for i in 0..5 {
            let ctx = LearningContext::new(format!("test content {i}"), AgentDomain::General);
            store
                .store_experience(&format!("key-{i}"), ctx)
                .await
                .unwrap();
        }

        let count = store.count().await.unwrap();
        assert_eq!(count, 5);

        let keys = store.get_all_keys().await.unwrap();
        assert_eq!(keys.len(), 5);
        for i in 0..5 {
            assert!(keys.contains(&format!("key-{i}")));
        }
    }

    #[tokio::test]
    #[ignore = "requires running Qdrant instance"]
    async fn test_upsert_overwrites() {
        let store = QdrantStore::new("http://localhost:6334", "test_upsert")
            .await
            .expect("Failed to create store");

        store.clear().await.expect("Failed to clear");

        let ctx1 = LearningContext::new("original content", AgentDomain::General);
        store.store_experience("same-key", ctx1).await.unwrap();

        let ctx2 = LearningContext::new("updated content", AgentDomain::Infrastructure);
        store.store_experience("same-key", ctx2).await.unwrap();

        // Should still be only one item
        let count = store.count().await.unwrap();
        assert_eq!(count, 1);

        // Should have the updated content
        let retrieved = store.get_experience("same-key").await.unwrap().unwrap();
        assert_eq!(retrieved.content, "updated content");
        assert_eq!(retrieved.domain, AgentDomain::Infrastructure);
    }
}
