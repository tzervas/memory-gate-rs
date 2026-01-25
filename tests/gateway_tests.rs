//! Integration tests for the memory gateway.

use memory_gate_rs::{
    adapters::PassthroughAdapter,
    storage::InMemoryStore,
    AgentDomain, GatewayConfig, LearningContext, MemoryGateway,
};

fn create_gateway() -> MemoryGateway<PassthroughAdapter, InMemoryStore> {
    MemoryGateway::new(
        PassthroughAdapter,
        InMemoryStore::new(),
        GatewayConfig::default().with_consolidation_enabled(false),
    )
}

#[tokio::test]
async fn test_full_learning_cycle() {
    let gateway = create_gateway();

    // Store multiple learning contexts across domains
    let contexts = vec![
        LearningContext::new("nginx restart fixes high CPU", AgentDomain::Infrastructure),
        LearningContext::new("kubectl rollout restart deployment", AgentDomain::Deployment),
        LearningContext::new("Code review found SQL injection", AgentDomain::CodeReview),
        LearningContext::new("PagerDuty incident resolved via restart", AgentDomain::IncidentResponse),
    ];

    for ctx in contexts {
        gateway.learn_from_interaction(ctx, None).await.unwrap();
    }

    assert_eq!(gateway.count().await.unwrap(), 4);

    // Query across all domains
    let all_results = gateway.retrieve_context("restart", None, None).await.unwrap();
    assert!(all_results.len() >= 2); // nginx and kubectl both contain "restart"

    // Query with domain filter
    let infra_results = gateway
        .retrieve_context("restart", None, Some(AgentDomain::Infrastructure))
        .await
        .unwrap();
    assert_eq!(infra_results.len(), 1);
    assert!(infra_results[0].content.contains("nginx"));
}

#[tokio::test]
async fn test_feedback_affects_importance() {
    let gateway = create_gateway();

    // Learn with positive feedback
    let ctx1 = LearningContext::new("Good solution", AgentDomain::General).with_importance(0.5);
    gateway.learn_from_interaction(ctx1, Some(1.0)).await.unwrap();

    // Learn with negative feedback
    let ctx2 = LearningContext::new("Bad solution", AgentDomain::General).with_importance(0.5);
    gateway.learn_from_interaction(ctx2, Some(0.0)).await.unwrap();

    let results = gateway.retrieve_context("solution", None, None).await.unwrap();
    assert_eq!(results.len(), 2);

    // Good solution should have higher importance and come first
    let good = results.iter().find(|c| c.content.contains("Good")).unwrap();
    let bad = results.iter().find(|c| c.content.contains("Bad")).unwrap();

    assert!(good.importance > bad.importance);
}

#[tokio::test]
async fn test_retrieval_limit() {
    let gateway = create_gateway();

    // Store many items
    for i in 0..20 {
        let ctx = LearningContext::new(format!("Test memory {i}"), AgentDomain::General);
        gateway.learn_from_interaction(ctx, None).await.unwrap();
    }

    // Default limit
    let results = gateway.retrieve_context("memory", None, None).await.unwrap();
    assert_eq!(results.len(), 10); // Default limit is 10

    // Custom limit
    let results = gateway.retrieve_context("memory", Some(5), None).await.unwrap();
    assert_eq!(results.len(), 5);
}

#[tokio::test]
async fn test_deduplication() {
    let gateway = create_gateway();

    // Store the same content twice
    let ctx1 = LearningContext::new("Duplicate content", AgentDomain::General);
    let ctx2 = LearningContext::new("Duplicate content", AgentDomain::General);

    let key1 = gateway.learn_from_interaction(ctx1, None).await.unwrap();
    let key2 = gateway.learn_from_interaction(ctx2, None).await.unwrap();

    // Same content + domain should produce same key
    assert_eq!(key1, key2);

    // Should only have one item (overwritten)
    assert_eq!(gateway.count().await.unwrap(), 1);
}

#[tokio::test]
async fn test_clear() {
    let gateway = create_gateway();

    for i in 0..5 {
        let ctx = LearningContext::new(format!("Memory {i}"), AgentDomain::General);
        gateway.learn_from_interaction(ctx, None).await.unwrap();
    }

    assert_eq!(gateway.count().await.unwrap(), 5);

    gateway.clear().await.unwrap();

    assert_eq!(gateway.count().await.unwrap(), 0);
}
