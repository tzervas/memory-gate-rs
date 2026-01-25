//! Base memory-enabled agent implementation.

use crate::{
    AgentDomain, KnowledgeStore, LearningContext, MemoryAdapter, MemoryGateway,
    Result,
};
use serde_json::Value;
use std::sync::Arc;
use tracing::{debug, instrument};

/// Base agent that integrates with the memory gateway.
///
/// This provides a foundation for building memory-enabled agents. Extend this
/// by implementing the task execution logic.
///
/// # Type Parameters
///
/// * `A` - The memory adapter type
/// * `S` - The knowledge store type
///
/// # Example
///
/// ```rust,ignore
/// use memory_gate_rs::{
///     agents::BaseMemoryAgent,
///     MemoryGateway, GatewayConfig, AgentDomain, LearningContext,
///     adapters::PassthroughAdapter,
///     storage::InMemoryStore,
/// };
///
/// let store = InMemoryStore::new();
/// let adapter = PassthroughAdapter;
/// let gateway = MemoryGateway::new(adapter, store, GatewayConfig::default());
///
/// let agent = BaseMemoryAgent::new(
///     "InfraAgent",
///     AgentDomain::Infrastructure,
///     gateway,
///     10,
/// );
/// ```
pub struct BaseMemoryAgent<A, S>
where
    A: MemoryAdapter<LearningContext> + 'static,
    S: KnowledgeStore<LearningContext> + 'static,
{
    name: String,
    domain: AgentDomain,
    gateway: Arc<MemoryGateway<A, S>>,
    retrieval_limit: usize,
}

impl<A, S> BaseMemoryAgent<A, S>
where
    A: MemoryAdapter<LearningContext> + 'static,
    S: KnowledgeStore<LearningContext> + 'static,
{
    /// Create a new base memory agent.
    ///
    /// # Arguments
    ///
    /// * `name` - The agent's name
    /// * `domain` - The operational domain
    /// * `gateway` - The memory gateway to use
    /// * `retrieval_limit` - Maximum memories to retrieve per task
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        domain: AgentDomain,
        gateway: MemoryGateway<A, S>,
        retrieval_limit: usize,
    ) -> Self {
        Self {
            name: name.into(),
            domain,
            gateway: Arc::new(gateway),
            retrieval_limit,
        }
    }

    /// Create a new agent with a shared gateway.
    #[must_use]
    pub fn with_shared_gateway(
        name: impl Into<String>,
        domain: AgentDomain,
        gateway: Arc<MemoryGateway<A, S>>,
        retrieval_limit: usize,
    ) -> Self {
        Self {
            name: name.into(),
            domain,
            gateway,
            retrieval_limit,
        }
    }

    /// Get the agent's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the agent's domain.
    #[must_use]
    pub const fn domain(&self) -> AgentDomain {
        self.domain
    }

    /// Get a reference to the gateway.
    #[must_use]
    pub fn gateway(&self) -> &MemoryGateway<A, S> {
        &self.gateway
    }

    /// Retrieve relevant memories for a task.
    ///
    /// # Errors
    ///
    /// Returns an error if retrieval fails.
    #[instrument(skip(self))]
    pub async fn retrieve_memories(&self, task_input: &str) -> Result<Vec<LearningContext>> {
        debug!(
            agent = %self.name,
            domain = %self.domain,
            "Retrieving memories for task"
        );

        self.gateway
            .retrieve_context(task_input, Some(self.retrieval_limit), Some(self.domain))
            .await
    }

    /// Build an enhanced context combining task input with retrieved memories.
    ///
    /// # Errors
    ///
    /// Returns an error if memory retrieval fails.
    pub async fn build_enhanced_context(
        &self,
        task_input: &str,
        task_context: Option<Value>,
    ) -> Result<EnhancedContext> {
        let memories = self.retrieve_memories(task_input).await?;

        Ok(EnhancedContext {
            task_input: task_input.to_string(),
            task_context,
            retrieved_memories: memories,
            agent_name: self.name.clone(),
            agent_domain: self.domain,
        })
    }

    /// Store a learning from a completed task.
    ///
    /// # Arguments
    ///
    /// * `content` - The learned content
    /// * `feedback` - Optional feedback score
    ///
    /// # Errors
    ///
    /// Returns an error if storing fails.
    pub async fn store_learning(&self, content: &str, feedback: Option<f32>) -> Result<String> {
        let context = LearningContext::new(content, self.domain);
        self.gateway.learn_from_interaction(context, feedback).await
    }

    /// Process a task with memory integration.
    ///
    /// This is a template method that:
    /// 1. Retrieves relevant memories
    /// 2. Builds enhanced context
    /// 3. Calls the provided execution function
    /// 4. Optionally stores the interaction
    ///
    /// # Arguments
    ///
    /// * `task_input` - The task to process
    /// * `task_context` - Optional additional context
    /// * `store_memory` - Whether to store the interaction
    /// * `executor` - Function that executes the task with enhanced context
    ///
    /// # Errors
    ///
    /// Returns an error if any step fails.
    pub async fn process_task<F, Fut>(
        &self,
        task_input: &str,
        task_context: Option<Value>,
        store_memory: bool,
        executor: F,
    ) -> Result<TaskOutput>
    where
        F: FnOnce(EnhancedContext) -> Fut,
        Fut: std::future::Future<Output = Result<(String, f32)>>,
    {
        // Build enhanced context with memories
        let enhanced = self.build_enhanced_context(task_input, task_context).await?;
        let memory_count = enhanced.retrieved_memories.len();

        // Execute the task
        let (result, confidence) = executor(enhanced).await?;

        // Optionally store the interaction
        let stored_key = if store_memory {
            let learning_content = format!(
                "Task: {task_input}\nResult: {result}\nConfidence: {confidence:.2}"
            );
            Some(self.store_learning(&learning_content, Some(confidence)).await?)
        } else {
            None
        };

        Ok(TaskOutput {
            result,
            confidence,
            memories_used: memory_count,
            stored_key,
        })
    }
}

/// Enhanced context combining task input with retrieved memories.
#[derive(Debug, Clone)]
pub struct EnhancedContext {
    /// The original task input.
    pub task_input: String,

    /// Optional task-specific context.
    pub task_context: Option<Value>,

    /// Retrieved relevant memories.
    pub retrieved_memories: Vec<LearningContext>,

    /// Name of the agent processing this context.
    pub agent_name: String,

    /// Domain of the agent.
    pub agent_domain: AgentDomain,
}

impl EnhancedContext {
    /// Format memories as a context string for prompting.
    #[must_use]
    pub fn format_memories(&self) -> String {
        use std::fmt::Write;
        
        if self.retrieved_memories.is_empty() {
            return String::from("No relevant memories found.");
        }

        let mut output = String::from("Relevant memories:\n");
        for (i, mem) in self.retrieved_memories.iter().enumerate() {
            let _ = writeln!(
                output,
                "{}. [{}] (importance: {:.2}) {}",
                i + 1,
                mem.domain,
                mem.importance,
                mem.content
            );
        }
        output
    }

    /// Check if any memories were retrieved.
    #[must_use]
    pub const fn has_memories(&self) -> bool {
        !self.retrieved_memories.is_empty()
    }
}

/// Output from a processed task.
#[derive(Debug, Clone)]
pub struct TaskOutput {
    /// The task result.
    pub result: String,

    /// Confidence score in `[0.0, 1.0]`.
    pub confidence: f32,

    /// Number of memories used for context.
    pub memories_used: usize,

    /// Key if the interaction was stored.
    pub stored_key: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{adapters::PassthroughAdapter, storage::InMemoryStore, GatewayConfig};

    fn create_test_agent() -> BaseMemoryAgent<PassthroughAdapter, InMemoryStore> {
        let store = InMemoryStore::new();
        let adapter = PassthroughAdapter;
        let config = GatewayConfig::default().with_consolidation_enabled(false);
        let gateway = MemoryGateway::new(adapter, store, config);

        BaseMemoryAgent::new("TestAgent", AgentDomain::Infrastructure, gateway, 5)
    }

    #[tokio::test]
    async fn test_agent_properties() {
        let agent = create_test_agent();

        assert_eq!(agent.name(), "TestAgent");
        assert_eq!(agent.domain(), AgentDomain::Infrastructure);
    }

    #[tokio::test]
    async fn test_store_and_retrieve_learning() {
        let agent = create_test_agent();

        // Store a learning
        let key = agent
            .store_learning("nginx restart fixes high CPU", Some(0.9))
            .await
            .unwrap();
        assert!(!key.is_empty());

        // Retrieve memories - use a term that exists in the stored content
        let memories = agent.retrieve_memories("nginx").await.unwrap();
        assert_eq!(memories.len(), 1);
        assert!(memories[0].content.contains("nginx"));
    }

    #[tokio::test]
    async fn test_enhanced_context() {
        let agent = create_test_agent();

        // Store some memories
        agent
            .store_learning("Memory 1 about infrastructure", None)
            .await
            .unwrap();
        agent
            .store_learning("Memory 2 about infrastructure", None)
            .await
            .unwrap();

        // Build enhanced context - use a term that matches the content
        let enhanced = agent
            .build_enhanced_context("infrastructure", None)
            .await
            .unwrap();

        assert_eq!(enhanced.task_input, "infrastructure");
        assert_eq!(enhanced.agent_name, "TestAgent");
        assert!(enhanced.has_memories());
        assert_eq!(enhanced.retrieved_memories.len(), 2);
    }

    #[tokio::test]
    async fn test_format_memories() {
        let agent = create_test_agent();

        agent
            .store_learning("Important infrastructure fact", None)
            .await
            .unwrap();

        let enhanced = agent
            .build_enhanced_context("infrastructure", None)
            .await
            .unwrap();

        let formatted = enhanced.format_memories();
        assert!(formatted.contains("Relevant memories:"));
        assert!(formatted.contains("Important infrastructure fact"));
    }

    #[tokio::test]
    async fn test_process_task() {
        let agent = create_test_agent();

        // Store a memory first - use distinctive terms
        agent
            .store_learning("Previous nginx configuration fix", None)
            .await
            .unwrap();

        // Process a task - query must be a substring of stored content
        let output = agent
            .process_task(
                "nginx",  // Simple term that will match
                None,
                true,
                |ctx| async move {
                    let result = format!("Processed with {} memories", ctx.retrieved_memories.len());
                    Ok((result, 0.85))
                },
            )
            .await
            .unwrap();

        assert!(output.result.contains("1 memories"));
        assert!((output.confidence - 0.85).abs() < f32::EPSILON);
        assert_eq!(output.memories_used, 1);
        assert!(output.stored_key.is_some());
    }
}
