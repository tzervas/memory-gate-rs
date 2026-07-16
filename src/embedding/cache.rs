//! Bounded LRU cache for query embeddings (text + catalog model id).

use crate::SupportedEmbeddingModel;
use std::collections::HashMap;

/// Default maximum number of cached query vectors per store.
pub const DEFAULT_QUERY_CACHE_CAPACITY: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    text: String,
    model: &'static str,
}

/// LRU cache for embedding vectors keyed by query text and model id.
///
/// Cache hits return the same `Vec<f32>` values as a cold embed for the same inputs.
#[derive(Debug)]
pub struct QueryEmbeddingCache {
    capacity: usize,
    order: Vec<CacheKey>,
    map: HashMap<CacheKey, Vec<f32>>,
}

impl QueryEmbeddingCache {
    /// Create a cache with the given maximum entry count.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            order: Vec::with_capacity(capacity.min(64)),
            map: HashMap::new(),
        }
    }

    /// Lookup a cached embedding.
    #[must_use]
    pub fn get(&mut self, text: &str, model: SupportedEmbeddingModel) -> Option<Vec<f32>> {
        let key = CacheKey {
            text: text.to_string(),
            model: model.as_str(),
        };
        let value = self.map.get(&key)?.clone();
        self.touch(&key);
        Some(value)
    }

    /// Insert or refresh an embedding in the cache.
    pub fn insert(&mut self, text: &str, model: SupportedEmbeddingModel, embedding: Vec<f32>) {
        let key = CacheKey {
            text: text.to_string(),
            model: model.as_str(),
        };
        if self.map.contains_key(&key) {
            self.map.insert(key.clone(), embedding);
            self.touch(&key);
            return;
        }
        while self.map.len() >= self.capacity {
            if let Some(oldest) = self.order.first().cloned() {
                self.order.remove(0);
                self.map.remove(&oldest);
            } else {
                break;
            }
        }
        self.order.push(key.clone());
        self.map.insert(key, embedding);
    }

    fn touch(&mut self, key: &CacheKey) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            let k = self.order.remove(pos);
            self.order.push(k);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_vec(seed: u8) -> Vec<f32> {
        (0..4)
            .map(|i| f32::from(seed) + f32::from(u8::try_from(i).unwrap()) * 0.1)
            .collect()
    }

    #[test]
    fn hit_returns_identical_vector() {
        let mut cache = QueryEmbeddingCache::with_capacity(8);
        let v = sample_vec(1);
        cache.insert("hello", SupportedEmbeddingModel::BgeSmallEnV15, v.clone());
        let hit = cache
            .get("hello", SupportedEmbeddingModel::BgeSmallEnV15)
            .unwrap();
        assert_eq!(hit, v);
    }

    #[test]
    fn different_model_is_miss() {
        let mut cache = QueryEmbeddingCache::with_capacity(8);
        cache.insert(
            "hello",
            SupportedEmbeddingModel::BgeSmallEnV15,
            sample_vec(1),
        );
        assert!(cache
            .get("hello", SupportedEmbeddingModel::AllMiniLmL6V2)
            .is_none());
    }

    #[test]
    fn evicts_oldest_when_full() {
        let mut cache = QueryEmbeddingCache::with_capacity(2);
        cache.insert("a", SupportedEmbeddingModel::DEFAULT, sample_vec(1));
        cache.insert("b", SupportedEmbeddingModel::DEFAULT, sample_vec(2));
        cache.insert("c", SupportedEmbeddingModel::DEFAULT, sample_vec(3));
        assert!(cache.get("a", SupportedEmbeddingModel::DEFAULT).is_none());
        assert!(cache.get("b", SupportedEmbeddingModel::DEFAULT).is_some());
        assert!(cache.get("c", SupportedEmbeddingModel::DEFAULT).is_some());
    }
}
