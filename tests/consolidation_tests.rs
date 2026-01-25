//! Consolidation integration tests.

use chrono::{Duration, Utc};
use memory_gate_rs::{
    adapters::PassthroughAdapter,
    storage::InMemoryStore,
    AgentDomain, GatewayConfig, LearningContext, MemoryGateway,
};

fn create_gateway_with_config(config: GatewayConfig) -> MemoryGateway<PassthroughAdapter, InMemoryStore> {
    MemoryGateway::new(PassthroughAdapter, InMemoryStore::new(), config)
}

#[tokio::test]
async fn test_consolidation_removes_old_low_importance() {
    let config = GatewayConfig::default()
        .with_consolidation_enabled(false)
        .with_low_importance_threshold(0.3)
        .with_age_threshold_days(1);

    let gateway = create_gateway_with_config(config);

    // Add old, low-importance memory (should be removed)
    let old_low = LearningContext::new("Old low importance", AgentDomain::General)
        .with_importance(0.1)
        .with_timestamp(Utc::now() - Duration::days(5));
    gateway.learn_from_interaction(old_low, None).await.unwrap();

    // Add old, high-importance memory (should be kept)
    let old_high = LearningContext::new("Old high importance", AgentDomain::General)
        .with_importance(0.9)
        .with_timestamp(Utc::now() - Duration::days(5));
    gateway.learn_from_interaction(old_high, None).await.unwrap();

    // Add new, low-importance memory (should be kept - not old enough)
    let new_low = LearningContext::new("New low importance", AgentDomain::General)
        .with_importance(0.1);
    gateway.learn_from_interaction(new_low, None).await.unwrap();

    // Add new, high-importance memory (should be kept)
    let new_high = LearningContext::new("New high importance", AgentDomain::General)
        .with_importance(0.9);
    gateway.learn_from_interaction(new_high, None).await.unwrap();

    assert_eq!(gateway.count().await.unwrap(), 4);

    // Run consolidation
    let stats = gateway.run_consolidation_once().await.unwrap();

    // Only the old, low-importance memory should be deleted
    assert_eq!(stats.items_processed, 4);
    assert_eq!(stats.items_deleted, 1);
    assert_eq!(gateway.count().await.unwrap(), 3);

    // Verify the correct one was deleted
    let results = gateway.retrieve_context("Old low", None, None).await.unwrap();
    assert!(results.is_empty());

    let results = gateway.retrieve_context("Old high", None, None).await.unwrap();
    assert_eq!(results.len(), 1);
}

#[tokio::test]
async fn test_consolidation_respects_thresholds() {
    // Very strict thresholds - nothing should be deleted
    let strict_config = GatewayConfig::default()
        .with_consolidation_enabled(false)
        .with_low_importance_threshold(0.01) // Very low
        .with_age_threshold_days(365); // Must be very old

    let gateway = create_gateway_with_config(strict_config);

    // Add memories that would normally be consolidated
    let ctx = LearningContext::new("Medium importance", AgentDomain::General)
        .with_importance(0.2)
        .with_timestamp(Utc::now() - Duration::days(30));
    gateway.learn_from_interaction(ctx, None).await.unwrap();

    let stats = gateway.run_consolidation_once().await.unwrap();

    // Nothing should be deleted with strict thresholds
    assert_eq!(stats.items_deleted, 0);
    assert_eq!(gateway.count().await.unwrap(), 1);
}

#[tokio::test]
async fn test_consolidation_stats() {
    let config = GatewayConfig::default()
        .with_consolidation_enabled(false)
        .with_low_importance_threshold(0.5)
        .with_age_threshold_days(0); // Any age

    let gateway = create_gateway_with_config(config);

    // Add mix of memories
    for i in 0..10 {
        let importance = if i % 2 == 0 { 0.1 } else { 0.9 };
        let ctx = LearningContext::new(format!("Memory {i}"), AgentDomain::General)
            .with_importance(importance)
            .with_timestamp(Utc::now() - Duration::days(1)); // Old enough
        gateway.learn_from_interaction(ctx, None).await.unwrap();
    }

    let stats = gateway.run_consolidation_once().await.unwrap();

    assert_eq!(stats.items_processed, 10);
    assert_eq!(stats.items_deleted, 5); // Half have low importance
    assert!(stats.duration.as_millis() < 1000); // Should be fast
    assert!(stats.errors.is_empty());
}

#[tokio::test]
async fn test_consolidation_empty_store() {
    let config = GatewayConfig::default().with_consolidation_enabled(false);
    let gateway = create_gateway_with_config(config);

    let stats = gateway.run_consolidation_once().await.unwrap();

    assert_eq!(stats.items_processed, 0);
    assert_eq!(stats.items_deleted, 0);
}
