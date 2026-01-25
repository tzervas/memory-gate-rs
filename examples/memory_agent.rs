//! Memory-enabled agent example
//!
//! Run with: cargo run --example memory_agent

use memory_gate_rs::{
    adapters::PassthroughAdapter,
    agents::BaseMemoryAgent,
    storage::InMemoryStore,
    AgentDomain, GatewayConfig, MemoryGateway, Result,
};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("=== Memory-Enabled Agent Example ===\n");

    // Create gateway components
    let store = InMemoryStore::new();
    let adapter = PassthroughAdapter::new();
    let config = GatewayConfig::default();
    let gateway = MemoryGateway::new(adapter, store, config);

    // Create a memory-enabled agent
    let agent = BaseMemoryAgent::new(
        "infra-agent-001",
        AgentDomain::Infrastructure,
        gateway,
        10, // retrieval limit
    );

    println!("Agent: {} (domain: {:?})", agent.name(), agent.domain());

    // Simulate processing tasks
    let tasks = vec![
        "Check nginx service status and restart if needed",
        "Investigate high memory usage on worker nodes",
        "Review Kubernetes pod autoscaling configuration",
    ];

    for task in tasks {
        println!("\n--- Processing Task ---");
        println!("Task: {}", task);

        // Process task with an executor function
        let output = agent
            .process_task(task, None, true, |enhanced| async move {
                // Simulate task execution using enhanced context
                let result = format!(
                    "Processed '{}' with {} memories from {}",
                    enhanced.task_input,
                    enhanced.retrieved_memories.len(),
                    enhanced.agent_name
                );
                let confidence = 0.85;
                Ok((result, confidence))
            })
            .await?;

        println!("Result: {}", output.result);
        println!("Confidence: {:.2}", output.confidence);
        println!("Memories used: {}", output.memories_used);
        if let Some(key) = &output.stored_key {
            println!("Stored as: {}", key);
        }
    }

    // Retrieve memories directly
    println!("\n--- Retrieving Agent Memories ---");
    let memories = agent.retrieve_memories("nginx").await?;
    println!("Found {} relevant memories for 'nginx'", memories.len());
    for mem in &memories {
        println!("  - {}", mem.content);
    }

    println!("\n=== Example Complete ===");
    Ok(())
}
