//! Prometheus metrics for memory-gate observability.
//!
//! This module provides Prometheus-compatible metrics for monitoring
//! memory gateway operations.

#[cfg(feature = "metrics")]
mod prometheus_metrics {
    use metrics::{counter, gauge, histogram};
    use std::time::Duration;

    /// Record a memory operation.
    pub fn record_operation(operation_type: &str, status: &str) {
        counter!(
            "memory_gate_operations_total",
            "operation_type" => operation_type.to_string(),
            "status" => status.to_string()
        )
        .increment(1);
    }

    /// Record store latency.
    pub fn record_store_latency(store_type: &str, duration: Duration) {
        histogram!(
            "memory_gate_store_latency_seconds",
            "store_type" => store_type.to_string()
        )
        .record(duration.as_secs_f64());
    }

    /// Record retrieval latency.
    pub fn record_retrieval_latency(store_type: &str, duration: Duration) {
        histogram!(
            "memory_gate_retrieval_latency_seconds",
            "store_type" => store_type.to_string()
        )
        .record(duration.as_secs_f64());
    }

    /// Set the items count gauge.
    pub fn set_items_count(store_type: &str, collection_name: &str, count: usize) {
        gauge!(
            "memory_gate_items_count",
            "store_type" => store_type.to_string(),
            "collection_name" => collection_name.to_string()
        )
        .set(count as f64);
    }

    /// Record a consolidation run.
    pub fn record_consolidation_run(status: &str) {
        counter!(
            "memory_gate_consolidation_runs_total",
            "status" => status.to_string()
        )
        .increment(1);
    }

    /// Record consolidation duration.
    pub fn record_consolidation_duration(duration: Duration) {
        histogram!("memory_gate_consolidation_duration_seconds").record(duration.as_secs_f64());
    }

    /// Record items processed during consolidation.
    pub fn record_consolidation_items(processed: usize, deleted: usize) {
        counter!("memory_gate_consolidation_items_processed_total").increment(processed as u64);
        counter!("memory_gate_consolidation_items_deleted_total").increment(deleted as u64);
    }

    /// Record an agent task.
    pub fn record_agent_task(agent_name: &str, domain: &str, status: &str) {
        counter!(
            "memory_gate_agent_tasks_processed_total",
            "agent_name" => agent_name.to_string(),
            "agent_domain" => domain.to_string(),
            "status" => status.to_string()
        )
        .increment(1);
    }

    /// Record agent task duration.
    pub fn record_agent_task_duration(agent_name: &str, domain: &str, duration: Duration) {
        histogram!(
            "memory_gate_agent_task_duration_seconds",
            "agent_name" => agent_name.to_string(),
            "agent_domain" => domain.to_string()
        )
        .record(duration.as_secs_f64());
    }

    /// Initialize the Prometheus exporter.
    ///
    /// # Errors
    ///
    /// Returns an error if the exporter fails to start.
    pub fn init_prometheus_exporter(
        addr: std::net::SocketAddr,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use metrics_exporter_prometheus::PrometheusBuilder;

        PrometheusBuilder::new()
            .with_http_listener(addr)
            .install()?;

        Ok(())
    }
}

#[cfg(feature = "metrics")]
pub use prometheus_metrics::*;

/// No-op metrics implementations when feature is disabled.
#[cfg(not(feature = "metrics"))]
mod noop_metrics {
    use std::time::Duration;

    /// No-op: Record a memory operation.
    #[inline]
    pub fn record_operation(_operation_type: &str, _status: &str) {}

    /// No-op: Record store latency.
    #[inline]
    pub fn record_store_latency(_store_type: &str, _duration: Duration) {}

    /// No-op: Record retrieval latency.
    #[inline]
    pub fn record_retrieval_latency(_store_type: &str, _duration: Duration) {}

    /// No-op: Set the items count gauge.
    #[inline]
    pub fn set_items_count(_store_type: &str, _collection_name: &str, _count: usize) {}

    /// No-op: Record a consolidation run.
    #[inline]
    pub fn record_consolidation_run(_status: &str) {}

    /// No-op: Record consolidation duration.
    #[inline]
    pub fn record_consolidation_duration(_duration: Duration) {}

    /// No-op: Record items processed during consolidation.
    #[inline]
    pub fn record_consolidation_items(_processed: usize, _deleted: usize) {}

    /// No-op: Record an agent task.
    #[inline]
    pub fn record_agent_task(_agent_name: &str, _domain: &str, _status: &str) {}

    /// No-op: Record agent task duration.
    #[inline]
    pub fn record_agent_task_duration(_agent_name: &str, _domain: &str, _duration: Duration) {}
}

#[cfg(not(feature = "metrics"))]
pub use noop_metrics::*;

/// Metric operation types.
pub mod ops {
    /// Store operation.
    pub const STORE: &str = "store";
    /// Retrieve operation.
    pub const RETRIEVE: &str = "retrieve";
    /// Delete operation.
    pub const DELETE: &str = "delete";
    /// Consolidation operation.
    pub const CONSOLIDATE: &str = "consolidate";
}

/// Metric status values.
pub mod status {
    /// Success status.
    pub const SUCCESS: &str = "success";
    /// Error status.
    pub const ERROR: &str = "error";
}

/// Store type labels.
pub mod store_types {
    /// In-memory store.
    pub const IN_MEMORY: &str = "in_memory";
    /// Qdrant store.
    pub const QDRANT: &str = "qdrant";
    /// `SQLite` vector store.
    pub const SQLITE_VEC: &str = "sqlite_vec";
}
