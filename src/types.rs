//! Core types for memory-gate-rs.
//!
//! This module contains the fundamental data structures that represent memories,
//! domain categorizations, and configuration. These types are designed to be
//! serializable for persistence and transport across different storage backends.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

/// Atomic unit of memory — contains learned content with contextual metadata.
///
/// A `LearningContext` represents a single piece of knowledge that an agent
/// has learned from an interaction. The design captures not just *what* was
/// learned, but *when*, *how important* it is, and *what domain* it belongs to.
///
/// # Why This Structure?
///
/// - **Content**: The actual knowledge — kept as free-form text to support any
///   type of learning without rigid schemas.
/// - **Domain**: Categorical filtering prevents cross-contamination in multi-agent
///   or multi-purpose deployments.
/// - **Timestamp**: Enables recency-based retrieval and time-based consolidation.
/// - **Importance**: Allows prioritizing high-value memories during retrieval and
///   protecting them during consolidation cleanup.
/// - **Metadata**: Extensible key-value storage for domain-specific attributes
///   without schema changes.
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
    pub const fn with_importance(mut self, importance: f32) -> Self {
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
    pub const fn with_timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
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
///
/// Per mint kickoff (M1): extended for workspace multi-layer memory integration:
/// repo scoping, layers (tero/context/gate), lang docs (rust/python). See README.
/// Tero-cited: readme--agent-domains .
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

    // --- M1 extensions for tero L1 + context-mcp + memory-gate integration (mint vision) ---
    /// Workspace-level orchestration / cross-repo coordination (e.g. wsfull orch).
    Workspace,

    /// Tero L1 structured cited index (deterministic corpus + lang docs dual-index).
    Tero,

    /// Context-mcp layer (session memory, temporal, future RAG over lang/workspace).
    Context,

    /// Memory-gate-rs itself (gating, consolidation, domain scoping facade).
    MemoryGate,

    /// Language reference docs: Rust (1.96 book/ref/std + tero-indexed).
    LangRust,

    /// Language reference docs: Python (3.13/3.14 stdlib/tutorial + dual index).
    LangPython,
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
            // M1 extensions (tero/context/gate/lang + workspace scoping)
            Self::Workspace,
            Self::Tero,
            Self::Context,
            Self::MemoryGate,
            Self::LangRust,
            Self::LangPython,
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
            // M1
            Self::Workspace => "workspace",
            Self::Tero => "tero",
            Self::Context => "context",
            Self::MemoryGate => "memory_gate",
            Self::LangRust => "lang_rust",
            Self::LangPython => "lang_python",
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
        let s = s.to_lowercase();
        match s.as_str() {
            "infrastructure" => Ok(Self::Infrastructure),
            "code_review" | "codereview" => Ok(Self::CodeReview),
            "deployment" => Ok(Self::Deployment),
            "incident_response" | "incidentresponse" => Ok(Self::IncidentResponse),
            "general" => Ok(Self::General),
            // M1 mint extensions + layer/repo/lang: prefix support for facade scoping
            "workspace" | "repo" => Ok(Self::Workspace),
            "tero" | "layer:tero" | "l1" => Ok(Self::Tero),
            "context" | "layer:context" | "context-mcp" | "rag" => Ok(Self::Context),
            "memory_gate" | "memory-gate" | "gate" | "layer:gate" => Ok(Self::MemoryGate),
            "lang_rust" | "lang:rust" | "rust" | "rust-1.96" => Ok(Self::LangRust),
            "lang_python" | "lang:python" | "python" | "python-3.13" | "python-3.14" => {
                Ok(Self::LangPython)
            }
            _ => {
                // Support "layer:xxx", "repo:xxx", "lang:xxx" prefixes for unified facade (M1)
                if let Some(rest) = s.strip_prefix("layer:") {
                    return Self::from_str(rest);
                }
                if let Some(_rest) = s.strip_prefix("repo:") {
                    // map generic repo to Workspace for scoping; specific use metadata or exact
                    return Ok(Self::Workspace);
                }
                if let Some(rest) = s.strip_prefix("lang:") {
                    let norm = rest.replace(['-', '.'], "_");
                    if norm.starts_with("rust") || norm == "rust" {
                        return Ok(Self::LangRust);
                    }
                    if norm.starts_with("python") || norm == "py" {
                        return Ok(Self::LangPython);
                    }
                    return Self::from_str(&format!("lang_{norm}"));
                }
                Err(format!("unknown domain: {s}"))
            }
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
    pub const fn with_low_importance_threshold(mut self, threshold: f32) -> Self {
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
        assert_eq!(
            ctx.metadata.as_ref().unwrap().get("key"),
            Some(&"value".to_string())
        );
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
        assert_eq!(
            "infrastructure".parse::<AgentDomain>().unwrap(),
            AgentDomain::Infrastructure
        );
        assert_eq!(
            "code_review".parse::<AgentDomain>().unwrap(),
            AgentDomain::CodeReview
        );
        assert_eq!(
            "CodeReview".parse::<AgentDomain>().unwrap(),
            AgentDomain::CodeReview
        );
        assert!("unknown".parse::<AgentDomain>().is_err());

        // M1 tests (tests-first for domain design)
        assert_eq!("tero".parse::<AgentDomain>().unwrap(), AgentDomain::Tero);
        assert_eq!(
            "layer:tero".parse::<AgentDomain>().unwrap(),
            AgentDomain::Tero
        );
        assert_eq!(
            "context".parse::<AgentDomain>().unwrap(),
            AgentDomain::Context
        );
        assert_eq!(
            "layer:context-mcp".parse::<AgentDomain>().unwrap(),
            AgentDomain::Context
        );
        assert_eq!(
            "gate".parse::<AgentDomain>().unwrap(),
            AgentDomain::MemoryGate
        );
        assert_eq!(
            "lang:rust".parse::<AgentDomain>().unwrap(),
            AgentDomain::LangRust
        );
        assert_eq!(
            "lang:python-3.14".parse::<AgentDomain>().unwrap(),
            AgentDomain::LangPython
        );
        assert_eq!(
            "workspace".parse::<AgentDomain>().unwrap(),
            AgentDomain::Workspace
        );
        assert_eq!(
            "repo:memory-gate-rs".parse::<AgentDomain>().unwrap(),
            AgentDomain::Workspace
        );
        assert_eq!(AgentDomain::LangRust.as_str(), "lang_rust");
        assert!(AgentDomain::all().contains(&AgentDomain::Tero));
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
