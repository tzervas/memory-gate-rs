//! Core types for memory-gate-rs.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

/// Atomic unit of memory - contains learned content.
///
/// A `LearningContext` represents a single piece of knowledge that an agent
/// has learned from an interaction. It includes the content itself, domain
/// categorization, timestamp, importance score, and optional metadata.
///
/// # Example
///
/// ```
/// use memory_gate_rs::{LearningContext, AgentDomain};
///
/// let context = LearningContext::new(
///     "Resolved high CPU by restarting nginx service",
///     AgentDomain::Infrastructure,
/// );
/// assert!(context.importance > 0.0);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LearningContext {
    /// The actual learned content.
    pub content: String,

    /// Domain categorization for this memory.
    pub domain: AgentDomain,

    /// When the memory was created.
    pub timestamp: DateTime<Utc>,

    /// Importance score in range `[0.0, 1.0]`.
    /// Higher values indicate more important memories.
    pub importance: f32,

    /// Optional key-value metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

impl LearningContext {
    /// Create a new learning context with default importance.
    ///
    /// # Arguments
    ///
    /// * `content` - The learned content string
    /// * `domain` - The agent domain this applies to
    ///
    /// # Example
    ///
    /// ```
    /// use memory_gate_rs::{LearningContext, AgentDomain};
    ///
    /// let ctx = LearningContext::new("kubectl rollout restart deployment/api", AgentDomain::Deployment);
    /// assert_eq!(ctx.importance, 1.0);
    /// ```
    #[must_use]
    pub fn new(content: impl Into<String>, domain: AgentDomain) -> Self {
        Self {
            content: content.into(),
            domain,
            timestamp: Utc::now(),
            importance: 1.0,
            metadata: None,
        }
    }

    /// Create a new learning context with custom importance.
    #[must_use]
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    /// Add metadata to this learning context.
    #[must_use]
    pub fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Add a single metadata entry.
    #[must_use]
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata
            .get_or_insert_with(HashMap::new)
            .insert(key.into(), value.into());
        self
    }

    /// Set a custom timestamp.
    #[must_use]
    pub fn with_timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Calculate the age of this memory in days.
    #[must_use]
    pub fn age_days(&self) -> i64 {
        (Utc::now() - self.timestamp).num_days()
    }

    /// Check if this memory should be consolidated (pruned).
    ///
    /// A memory is eligible for consolidation if:
    /// - Its importance is below the threshold, AND
    /// - Its age exceeds the age threshold
    #[must_use]
    pub fn should_consolidate(&self, importance_threshold: f32, age_threshold_days: i64) -> bool {
        self.importance < importance_threshold && self.age_days() > age_threshold_days
    }
}

impl fmt::Display for LearningContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LearningContext({}, domain={}, importance={:.2})",
            truncate_content(&self.content, 50),
            self.domain,
            self.importance
        )
    }
}

/// Truncate content for display purposes.
fn truncate_content(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

/// Operational domains for agent categorization.
///
/// Domains allow filtering memories by the type of operation they relate to,
/// enabling domain-specific memory retrieval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AgentDomain {
    /// Server, network, and infrastructure operations.
    Infrastructure,

    /// Code analysis and review tasks.
    CodeReview,

    /// CI/CD and release management.
    Deployment,

    /// Incident handling and response.
    IncidentResponse,

    /// Uncategorized or general-purpose.
    #[default]
    General,
}

impl AgentDomain {
    /// Get all available domains.
    #[must_use]
    pub const fn all() -> &'static [Self] {
        &[
            Self::Infrastructure,
            Self::CodeReview,
            Self::Deployment,
            Self::IncidentResponse,
            Self::General,
        ]
    }

    /// Convert to string representation.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Infrastructure => "infrastructure",
            Self::CodeReview => "code_review",
            Self::Deployment => "deployment",
            Self::IncidentResponse => "incident_response",
            Self::General => "general",
        }
    }
}

impl fmt::Display for AgentDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for AgentDomain {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "infrastructure" => Ok(Self::Infrastructure),
            "code_review" | "codereview" => Ok(Self::CodeReview),
            "deployment" => Ok(Self::Deployment),
            "incident_response" | "incidentresponse" => Ok(Self::IncidentResponse),
            "general" => Ok(Self::General),
            _ => Err(format!("unknown domain: {s}")),
        }
    }
}

/// Configuration for the memory gateway.
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    /// Whether background consolidation is enabled.
    pub consolidation_enabled: bool,

    /// Interval between consolidation runs.
    pub consolidation_interval: Duration,

    /// Importance threshold for consolidation (memories below this may be pruned).
    pub low_importance_threshold: f32,

    /// Age threshold in days for consolidation.
    pub age_threshold_days: u32,

    /// Default limit for memory retrieval queries.
    pub retrieval_limit: usize,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            consolidation_enabled: true,
            consolidation_interval: Duration::from_secs(3600), // 1 hour
            low_importance_threshold: 0.2,
            age_threshold_days: 30,
            retrieval_limit: 10,
        }
    }
}

impl GatewayConfig {
    /// Create a new configuration with all defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether consolidation is enabled.
    #[must_use]
    pub const fn with_consolidation_enabled(mut self, enabled: bool) -> Self {
        self.consolidation_enabled = enabled;
        self
    }

    /// Set the consolidation interval.
    #[must_use]
    pub const fn with_consolidation_interval(mut self, interval: Duration) -> Self {
        self.consolidation_interval = interval;
        self
    }

    /// Set the low importance threshold for consolidation.
    #[must_use]
    pub fn with_low_importance_threshold(mut self, threshold: f32) -> Self {
        self.low_importance_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the age threshold in days for consolidation.
    #[must_use]
    pub const fn with_age_threshold_days(mut self, days: u32) -> Self {
        self.age_threshold_days = days;
        self
    }

    /// Set the default retrieval limit.
    #[must_use]
    pub const fn with_retrieval_limit(mut self, limit: usize) -> Self {
        self.retrieval_limit = limit;
        self
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn validate(&self) -> crate::Result<()> {
        if self.consolidation_interval.is_zero() && self.consolidation_enabled {
            return Err(crate::Error::invalid_config(
                "consolidation interval cannot be zero when enabled",
            ));
        }
        if self.retrieval_limit == 0 {
            return Err(crate::Error::invalid_config(
                "retrieval limit must be greater than zero",
            ));
        }
        Ok(())
    }
}

/// Statistics about consolidation operations.
#[derive(Debug, Clone, Default)]
pub struct ConsolidationStats {
    /// Number of items processed.
    pub items_processed: usize,

    /// Number of items deleted.
    pub items_deleted: usize,

    /// Duration of the consolidation run.
    pub duration: Duration,

    /// Any errors encountered (non-fatal).
    pub errors: Vec<String>,
}

impl ConsolidationStats {
    /// Create new empty stats.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            items_processed: 0,
            items_deleted: 0,
            duration: Duration::ZERO,
            errors: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learning_context_creation() {
        let ctx = LearningContext::new("test content", AgentDomain::Infrastructure);
        assert_eq!(ctx.content, "test content");
        assert_eq!(ctx.domain, AgentDomain::Infrastructure);
        assert!((ctx.importance - 1.0).abs() < f32::EPSILON);
        assert!(ctx.metadata.is_none());
    }

    #[test]
    fn test_learning_context_builder() {
        let ctx = LearningContext::new("test", AgentDomain::General)
            .with_importance(0.5)
            .with_meta("key", "value");

        assert!((ctx.importance - 0.5).abs() < f32::EPSILON);
        assert_eq!(ctx.metadata.as_ref().unwrap().get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_importance_clamping() {
        let ctx1 = LearningContext::new("test", AgentDomain::General).with_importance(2.0);
        assert!((ctx1.importance - 1.0).abs() < f32::EPSILON);

        let ctx2 = LearningContext::new("test", AgentDomain::General).with_importance(-0.5);
        assert!(ctx2.importance.abs() < f32::EPSILON);
    }

    #[test]
    fn test_agent_domain_parsing() {
        assert_eq!("infrastructure".parse::<AgentDomain>().unwrap(), AgentDomain::Infrastructure);
        assert_eq!("code_review".parse::<AgentDomain>().unwrap(), AgentDomain::CodeReview);
        assert_eq!("CodeReview".parse::<AgentDomain>().unwrap(), AgentDomain::CodeReview);
        assert!("unknown".parse::<AgentDomain>().is_err());
    }

    #[test]
    fn test_gateway_config_validation() {
        let config = GatewayConfig::default();
        assert!(config.validate().is_ok());

        let bad_config = GatewayConfig::new()
            .with_consolidation_enabled(true)
            .with_consolidation_interval(Duration::ZERO);
        assert!(bad_config.validate().is_err());
    }
}
