//! Passthrough adapter that applies basic importance adjustment.

use crate::{LearningContext, MemoryAdapter, Result};
use async_trait::async_trait;

/// A simple adapter that optionally adjusts importance based on feedback.
///
/// The `PassthroughAdapter` is the default adapter that performs minimal
/// transformation. If feedback is provided, it blends the current importance
/// with the feedback value.
///
/// # Example
///
/// ```
/// use memory_gate_rs::adapters::PassthroughAdapter;
/// use memory_gate_rs::{MemoryAdapter, LearningContext, AgentDomain};
///
/// # tokio_test::block_on(async {
/// let adapter = PassthroughAdapter;
/// let ctx = LearningContext::new("test", AgentDomain::General);
///
/// // Without feedback - passes through unchanged
/// let result = adapter.adapt_knowledge(ctx.clone(), None).await.unwrap();
/// assert!((result.importance - 1.0).abs() < f32::EPSILON);
///
/// // With feedback - blends importance
/// let result = adapter.adapt_knowledge(ctx, Some(0.5)).await.unwrap();
/// assert!((result.importance - 0.75).abs() < f32::EPSILON); // (1.0 + 0.5) / 2
/// # });
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct PassthroughAdapter;

impl PassthroughAdapter {
    /// Create a new passthrough adapter.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

#[async_trait]
impl MemoryAdapter<LearningContext> for PassthroughAdapter {
    async fn adapt_knowledge(
        &self,
        mut context: LearningContext,
        feedback: Option<f32>,
    ) -> Result<LearningContext> {
        if let Some(f) = feedback {
            // Validate feedback is in range
            let f = f.clamp(0.0, 1.0);
            // Blend current importance with feedback
            context.importance = f32::midpoint(context.importance, f);
        }
        Ok(context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AgentDomain;

    #[tokio::test]
    async fn test_passthrough_no_feedback() {
        let adapter = PassthroughAdapter::new();
        let ctx = LearningContext::new("test", AgentDomain::General).with_importance(0.8);

        let result = adapter.adapt_knowledge(ctx, None).await.unwrap();
        assert!((result.importance - 0.8).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_passthrough_with_feedback() {
        let adapter = PassthroughAdapter::new();
        let ctx = LearningContext::new("test", AgentDomain::General).with_importance(1.0);

        let result = adapter.adapt_knowledge(ctx, Some(0.5)).await.unwrap();
        assert!((result.importance - 0.75).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_passthrough_feedback_clamping() {
        let adapter = PassthroughAdapter::new();
        let ctx = LearningContext::new("test", AgentDomain::General).with_importance(0.5);

        // Feedback above 1.0 should be clamped
        let result = adapter.adapt_knowledge(ctx.clone(), Some(2.0)).await.unwrap();
        assert!((result.importance - 0.75).abs() < f32::EPSILON); // (0.5 + 1.0) / 2

        // Feedback below 0.0 should be clamped
        let result = adapter.adapt_knowledge(ctx, Some(-1.0)).await.unwrap();
        assert!((result.importance - 0.25).abs() < f32::EPSILON); // (0.5 + 0.0) / 2
    }
}
