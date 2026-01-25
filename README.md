# memory-gate-rs

[![Crates.io](https://img.shields.io/crates/v/memory-gate-rs.svg)](https://crates.io/crates/memory-gate-rs)
[![Documentation](https://docs.rs/memory-gate-rs/badge.svg)](https://docs.rs/memory-gate-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Dynamic Memory Learning Layer for AI agents - persistent knowledge accumulation with vector retrieval and automatic consolidation.

## Overview

Memory-gate provides a production-ready memory layer for AI agents, solving the fundamental problem of stateless AI by enabling:

- **Persistent Knowledge Accumulation**: Agents retain and build upon operational knowledge across sessions
- **Context-Aware Retrieval**: Semantic vector search retrieves relevant past experiences
- **Automatic Consolidation**: Background cleanup of low-importance/stale memories
- **Domain Filtering**: Categorical filtering for multi-domain agent deployments

### Architecture: Complementary Learning Systems (CLS)

Inspired by neuroscience research on brain plasticity, memory-gate implements a dual-stream architecture:

- **Fast System** (hippocampal-like): Rapid pattern acquisition from new experiences
- **Slow System** (cortical-like): Stable long-term retention through consolidation

## Installation

```toml
[dependencies]
memory-gate-rs = "0.1"
```

### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `in-memory` | In-memory HashMap-based storage | ✓ |
| `qdrant` | Qdrant vector database backend | |
| `sqlite-vec` | SQLite with vector extension | |
| `metrics` | Prometheus metrics export | |
| `full` | All features enabled | |

## Quick Start

```rust
use memory_gate_rs::{
    MemoryGateway, GatewayConfig, LearningContext, AgentDomain,
    adapters::PassthroughAdapter,
    storage::InMemoryStore,
};

#[tokio::main]
async fn main() -> memory_gate_rs::Result<()> {
    // Create storage and adapter
    let store = InMemoryStore::new();
    let adapter = PassthroughAdapter;
    
    // Initialize gateway
    let gateway = MemoryGateway::new(adapter, store, GatewayConfig::default());
    
    // Learn from an interaction
    let context = LearningContext::new(
        "Resolved high CPU by restarting nginx service",
        AgentDomain::Infrastructure,
    );
    gateway.learn_from_interaction(context, Some(0.9)).await?;
    
    // Retrieve relevant memories
    let memories = gateway
        .retrieve_context("CPU usage issues", 5, Some(AgentDomain::Infrastructure))
        .await?;
    
    for memory in memories {
        println!("Recalled: {}", memory.content);
    }
    
    Ok(())
}
```

## Core Concepts

### LearningContext

The atomic unit of memory containing learned content:

```rust
pub struct LearningContext {
    pub content: String,           // The actual learned content
    pub domain: AgentDomain,       // Domain categorization
    pub timestamp: DateTime<Utc>,  // When the memory was created
    pub importance: f32,           // Importance score [0.0, 1.0]
    pub metadata: Option<HashMap<String, String>>,
}
```

### Agent Domains

Categorize memories by operational domain:

```rust
pub enum AgentDomain {
    Infrastructure,    // Server/network operations
    CodeReview,        // Code analysis and review
    Deployment,        // CI/CD and release management
    IncidentResponse,  // Incident handling
    General,           // Uncategorized
}
```

### Memory Gateway

Central orchestrator for learning and memory operations:

```rust
let gateway = MemoryGateway::new(adapter, store, config);

// Learn from interaction with optional feedback
gateway.learn_from_interaction(context, feedback).await?;

// Retrieve relevant memories
let memories = gateway.retrieve_context(query, limit, domain_filter).await?;

// Start background consolidation
gateway.start_consolidation().await?;
```

### Consolidation

Background process that maintains memory health:

- Prunes low-importance memories (`importance < 0.2`)
- Removes stale memories (older than configured threshold)
- Runs at configurable intervals (default: 1 hour)

## Storage Backends

### In-Memory (default)

Simple HashMap-based storage for testing and development:

```rust
use memory_gate_rs::storage::InMemoryStore;

let store = InMemoryStore::new();
```

### Qdrant (feature: `qdrant`)

Production vector database with semantic search:

```rust
use memory_gate_rs::storage::QdrantStore;

let store = QdrantStore::new("http://localhost:6334", "memories").await?;
```

### SQLite + Vector (feature: `sqlite-vec`)

Embedded vector storage with SQLite:

```rust
use memory_gate_rs::storage::SqliteVecStore;

let store = SqliteVecStore::new("./memory.db").await?;
```

## Metrics (feature: `metrics`)

Prometheus-compatible metrics:

| Metric | Type | Labels |
|--------|------|--------|
| `memory_gate_operations_total` | Counter | operation_type, status |
| `memory_gate_store_latency_seconds` | Histogram | store_type |
| `memory_gate_retrieval_latency_seconds` | Histogram | store_type |
| `memory_gate_items_count` | Gauge | store_type |
| `memory_gate_consolidation_runs_total` | Counter | status |

## Custom Adapters

Implement `MemoryAdapter` for custom knowledge transformation:

```rust
use memory_gate_rs::{MemoryAdapter, LearningContext, Result};
use async_trait::async_trait;

struct MyAdapter;

#[async_trait]
impl MemoryAdapter<LearningContext> for MyAdapter {
    async fn adapt_knowledge(
        &self,
        mut context: LearningContext,
        feedback: Option<f32>,
    ) -> Result<LearningContext> {
        if let Some(f) = feedback {
            context.importance = (context.importance + f) / 2.0;
        }
        Ok(context)
    }
}
```

## Custom Storage Backends

Implement `KnowledgeStore` for custom storage:

```rust
use memory_gate_rs::{KnowledgeStore, LearningContext, AgentDomain, Result};
use async_trait::async_trait;

struct MyStore { /* ... */ }

#[async_trait]
impl KnowledgeStore<LearningContext> for MyStore {
    async fn store_experience(&self, key: &str, experience: LearningContext) -> Result<()> {
        // Store implementation
        Ok(())
    }
    
    async fn retrieve_context(
        &self,
        query: &str,
        limit: usize,
        domain_filter: Option<AgentDomain>,
    ) -> Result<Vec<LearningContext>> {
        // Retrieval implementation
        Ok(vec![])
    }
    
    async fn delete_experience(&self, key: &str) -> Result<()> {
        // Delete implementation
        Ok(())
    }
    
    async fn get_all_keys(&self) -> Result<Vec<String>> {
        // List keys implementation
        Ok(vec![])
    }
}
```

## Integration with rust-ai Ecosystem

memory-gate-rs integrates with the broader rust-ai crate ecosystem:

- **rust-ai-core**: Shared device selection, error types, and traits
- **agent-mcp**: Multi-agent orchestration can use memory-gate for persistence
- **context-mcp**: Alternate MCP-based context storage
- **embeddenator-retrieval**: Future VSA-based similarity search integration

## License

MIT License - see [LICENSE-MIT](LICENSE-MIT)

## Contributing

Contributions welcome! Please read CONTRIBUTING.md first.
