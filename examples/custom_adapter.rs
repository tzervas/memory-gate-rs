//! Custom adapter example showing how to create domain-specific memory processing
//!
//! Run with: cargo run --example custom_adapter

use async_trait::async_trait;
use memory_gate_rs::{
    storage::InMemoryStore, AgentDomain, GatewayConfig, LearningContext,
    MemoryGateway, MemoryAdapter, Result,
};
use std::collections::HashMap;

/// A custom adapter that boosts importance for security-related content
#[derive(Clone, Default)]
struct SecurityAwareAdapter {
    security_keywords: Vec<String>,
    security_boost: f32,
}

impl SecurityAwareAdapter {
    fn new() -> Self {
        Self {
            security_keywords: vec![
                "vulnerability".to_string(),
                "exploit".to_string(),
                "CVE".to_string(),
                "breach".to_string(),
                "unauthorized".to_string(),
                "injection".to_string(),
                "XSS".to_string(),
                "CSRF".to_string(),
            ],
            security_boost: 0.3, // Boost security items by 30%
        }
    }

    fn contains_security_keyword(&self, content: &str) -> bool {
        let content_lower = content.to_lowercase();
        self.security_keywords
            .iter()
            .any(|kw| content_lower.contains(&kw.to_lowercase()))
    }
}

#[async_trait]
impl MemoryAdapter<LearningContext> for SecurityAwareAdapter {
    async fn adapt_knowledge(
        &self,
        mut context: LearningContext,
        feedback: Option<f32>,
    ) -> Result<LearningContext> {
        // Apply feedback if provided
        if let Some(f) = feedback {
            context.importance = (context.importance + f.clamp(0.0, 1.0)) / 2.0;
        }

        // Boost security-related content
        if self.contains_security_keyword(&context.content) {
            context.importance = (context.importance + self.security_boost).min(1.0);
            
            // Add security flag to metadata
            let mut metadata = context.metadata.unwrap_or_default();
            metadata.insert("security_flagged".to_string(), "true".to_string());
            context.metadata = Some(metadata);
            
            println!(
                "  [SecurityAdapter] Boosted importance for security content: {:.2}",
                context.importance
            );
        }

        Ok(context)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("=== Custom Adapter Example ===\n");

    // Create gateway with security-aware adapter
    let adapter = SecurityAwareAdapter::new();
    let store = InMemoryStore::new();
    let config = GatewayConfig::default();
    let gateway = MemoryGateway::new(adapter, store, config);

    // Learn various items
    println!("Learning interactions...\n");

    let interactions = vec![
        ("Updated nginx configuration for better caching", 0.6),
        ("Applied CVE-2024-1234 security patch to OpenSSL", 0.7),
        ("Optimized database query performance by 40%", 0.8),
        ("Detected SQL injection attempt in user input", 0.6),
        ("Completed weekly backup successfully", 0.4),
    ];

    for (content, importance) in interactions {
        println!("Processing: {}", content);
        let ctx = LearningContext::new(content, AgentDomain::Infrastructure)
            .with_importance(importance);
        gateway.learn_from_interaction(ctx, None).await?;
    }

    // Retrieve all and see security items prioritized
    println!("\n--- Retrieving memories ---\n");

    let results = gateway.retrieve_context("", Some(10), None).await?;

    for (i, ctx) in results.iter().enumerate() {
        let security_flag = ctx.metadata.as_ref()
            .and_then(|m| m.get("security_flagged"))
            .map(|_| "🔒")
            .unwrap_or("  ");
        println!(
            "{} {}. [importance: {:.2}] {}",
            security_flag,
            i + 1,
            ctx.importance,
            ctx.content
        );
    }

    println!("\n=== Example Complete ===");
    Ok(())
}
