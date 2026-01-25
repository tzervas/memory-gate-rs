//! Basic usage example for memory-gate-rs
//!
//! Run with: cargo run --example basic_usage

use memory_gate_rs::{
    adapters::PassthroughAdapter, storage::InMemoryStore, AgentDomain, GatewayConfig,
    LearningContext, MemoryGateway,
};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for observability
    tracing_subscriber::fmt::init();

    println!("=== Memory Gate Basic Usage Example ===\n");

    // 1. Create storage backend (in-memory for demo)
    let store = InMemoryStore::new();

    // 2. Create memory adapter
    let adapter = PassthroughAdapter::new();

    // 3. Configure the gateway
    let config = GatewayConfig::default()
        .with_consolidation_enabled(true)
        .with_consolidation_interval(Duration::from_secs(60))
        .with_low_importance_threshold(0.3)
        .with_retrieval_limit(100);

    // 4. Create the memory gateway
    let gateway = MemoryGateway::new(adapter, store, config);

    // 5. Learn from interactions
    println!("Learning from interactions...");

    let ctx1 = LearningContext::new(
        "High CPU usage resolved by restarting nginx service",
        AgentDomain::Infrastructure,
    );
    gateway.learn_from_interaction(ctx1, Some(0.9)).await?;

    let ctx2 = LearningContext::new(
        "Container logs show memory leak in worker process",
        AgentDomain::Infrastructure,
    );
    gateway.learn_from_interaction(ctx2, Some(0.7)).await?;

    let ctx3 = LearningContext::new(
        "Horizontal pod autoscaler triggered at 80% CPU threshold",
        AgentDomain::Infrastructure,
    );
    gateway.learn_from_interaction(ctx3, Some(0.8)).await?;

    // 6. Retrieve relevant context for a new query
    println!("\nRetrieving context for 'CPU'...");

    let contexts = gateway.retrieve_context("CPU", Some(5), None).await?;

    println!("\nFound {} relevant memories:", contexts.len());
    for (i, ctx) in contexts.iter().enumerate() {
        println!(
            "  {}. [importance: {:.2}] {}",
            i + 1,
            ctx.importance,
            ctx.content
        );
    }

    // 7. Domain-filtered retrieval
    println!("\nRetrieving Infrastructure-specific context...");

    let infra_contexts = gateway
        .retrieve_context("nginx", Some(10), Some(AgentDomain::Infrastructure))
        .await?;

    println!("Found {} infrastructure memories", infra_contexts.len());

    // 8. Manual consolidation
    println!("\nRunning memory consolidation...");
    let stats = gateway.run_consolidation_once().await?;
    println!(
        "Consolidation complete: {} processed, {} deleted",
        stats.items_processed, stats.items_deleted
    );

    println!("\n=== Example Complete ===");
    Ok(())
}
