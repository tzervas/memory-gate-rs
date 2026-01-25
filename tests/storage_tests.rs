//! Storage backend tests.

use memory_gate_rs::{
    storage::InMemoryStore,
    AgentDomain, KnowledgeStore, LearningContext,
};

#[tokio::test]
async fn test_in_memory_store_crud() {
    let store = InMemoryStore::new();

    // Create
    let ctx = LearningContext::new("Test content", AgentDomain::Infrastructure);
    store.store_experience("key1", ctx.clone()).await.unwrap();

    // Read
    let retrieved = store.get_experience("key1").await.unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().content, ctx.content);

    // Update (overwrite)
    let updated = LearningContext::new("Updated content", AgentDomain::Infrastructure);
    store.store_experience("key1", updated.clone()).await.unwrap();
    let retrieved = store.get_experience("key1").await.unwrap();
    assert_eq!(retrieved.unwrap().content, "Updated content");

    // Delete
    store.delete_experience("key1").await.unwrap();
    let retrieved = store.get_experience("key1").await.unwrap();
    assert!(retrieved.is_none());
}

#[tokio::test]
async fn test_in_memory_store_retrieval() {
    let store = InMemoryStore::new();

    // Add various memories
    let memories = vec![
        ("k1", "nginx server restart", AgentDomain::Infrastructure, 0.9),
        ("k2", "apache configuration", AgentDomain::Infrastructure, 0.7),
        ("k3", "kubectl deployment", AgentDomain::Deployment, 0.8),
        ("k4", "code review feedback", AgentDomain::CodeReview, 0.6),
    ];

    for (key, content, domain, importance) in memories {
        let ctx = LearningContext::new(content, domain).with_importance(importance);
        store.store_experience(key, ctx).await.unwrap();
    }

    // Search without filter
    let results = store.retrieve_context("server", 10, None).await.unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].content.contains("nginx"));

    // Search with domain filter
    let results = store
        .retrieve_context("config", 10, Some(AgentDomain::Infrastructure))
        .await
        .unwrap();
    assert_eq!(results.len(), 1);

    // Search with non-matching domain filter
    let results = store
        .retrieve_context("nginx", 10, Some(AgentDomain::Deployment))
        .await
        .unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_in_memory_store_ordering() {
    let store = InMemoryStore::new();

    // Add memories with different importance levels
    let memories = vec![
        ("k1", "low priority task", 0.2),
        ("k2", "medium priority task", 0.5),
        ("k3", "high priority task", 0.9),
    ];

    for (key, content, importance) in memories {
        let ctx = LearningContext::new(content, AgentDomain::General).with_importance(importance);
        store.store_experience(key, ctx).await.unwrap();
    }

    let results = store.retrieve_context("task", 10, None).await.unwrap();
    assert_eq!(results.len(), 3);

    // Should be ordered by importance (descending)
    assert!(results[0].importance >= results[1].importance);
    assert!(results[1].importance >= results[2].importance);
}

#[tokio::test]
async fn test_in_memory_store_limit() {
    let store = InMemoryStore::new();

    for i in 0..100 {
        let ctx = LearningContext::new(format!("Memory item {i}"), AgentDomain::General);
        store.store_experience(&format!("key{i}"), ctx).await.unwrap();
    }

    // Test various limits
    let results = store.retrieve_context("item", 5, None).await.unwrap();
    assert_eq!(results.len(), 5);

    let results = store.retrieve_context("item", 50, None).await.unwrap();
    assert_eq!(results.len(), 50);

    let results = store.retrieve_context("item", 200, None).await.unwrap();
    assert_eq!(results.len(), 100); // Can't return more than exists
}

#[tokio::test]
async fn test_in_memory_store_get_all_keys() {
    let store = InMemoryStore::new();

    let keys: Vec<_> = (0..5).map(|i| format!("key{i}")).collect();

    for key in &keys {
        let ctx = LearningContext::new("content", AgentDomain::General);
        store.store_experience(key, ctx).await.unwrap();
    }

    let all_keys = store.get_all_keys().await.unwrap();
    assert_eq!(all_keys.len(), 5);

    for key in keys {
        assert!(all_keys.contains(&key));
    }
}

#[tokio::test]
async fn test_in_memory_store_count_and_clear() {
    let store = InMemoryStore::new();

    assert_eq!(store.count().await.unwrap(), 0);

    for i in 0..10 {
        let ctx = LearningContext::new(format!("Memory {i}"), AgentDomain::General);
        store.store_experience(&format!("key{i}"), ctx).await.unwrap();
    }

    assert_eq!(store.count().await.unwrap(), 10);

    store.clear().await.unwrap();

    assert_eq!(store.count().await.unwrap(), 0);
}

#[tokio::test]
async fn test_in_memory_store_concurrent_access() {
    use std::sync::Arc;

    let store = Arc::new(InMemoryStore::new());

    // Spawn multiple writers
    let mut handles = vec![];

    for i in 0..10 {
        let store = Arc::clone(&store);
        let handle = tokio::spawn(async move {
            for j in 0..10 {
                let ctx = LearningContext::new(format!("Content {i}-{j}"), AgentDomain::General);
                store.store_experience(&format!("key-{i}-{j}"), ctx).await.unwrap();
            }
        });
        handles.push(handle);
    }

    // Wait for all writers
    for handle in handles {
        handle.await.unwrap();
    }

    // Should have all items
    assert_eq!(store.count().await.unwrap(), 100);
}
