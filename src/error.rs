//! Error types for memory-gate-rs.

use std::fmt;

/// Result type alias for memory-gate operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Error type for memory-gate operations.
#[derive(Debug)]
pub enum Error {
    /// Storage operation failed.
    Storage(StorageError),
    /// Serialization/deserialization failed.
    Serialization(String),
    /// Invalid configuration.
    InvalidConfig(String),
    /// Consolidation operation failed.
    Consolidation(String),
    /// Embedding generation failed.
    Embedding(String),
    /// Gateway is not running.
    NotRunning,
    /// Operation was cancelled.
    Cancelled,
    /// Generic internal error.
    Internal(String),
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Storage(e) => Some(e),
            _ => None,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Storage(e) => write!(f, "storage error: {e}"),
            Self::Serialization(msg) => write!(f, "serialization error: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "invalid configuration: {msg}"),
            Self::Consolidation(msg) => write!(f, "consolidation error: {msg}"),
            Self::Embedding(msg) => write!(f, "embedding error: {msg}"),
            Self::NotRunning => write!(f, "gateway is not running"),
            Self::Cancelled => write!(f, "operation was cancelled"),
            Self::Internal(msg) => write!(f, "internal error: {msg}"),
        }
    }
}

impl From<StorageError> for Error {
    fn from(err: StorageError) -> Self {
        Self::Storage(err)
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization(err.to_string())
    }
}

/// Storage-specific errors.
#[derive(Debug)]
pub enum StorageError {
    /// Item not found.
    NotFound(String),
    /// Connection failed.
    Connection(String),
    /// Query failed.
    Query(String),
    /// Write operation failed.
    Write(String),
    /// Delete operation failed.
    Delete(String),
    /// Backend-specific error.
    Backend(String),
}

impl std::error::Error for StorageError {}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(key) => write!(f, "item not found: {key}"),
            Self::Connection(msg) => write!(f, "connection failed: {msg}"),
            Self::Query(msg) => write!(f, "query failed: {msg}"),
            Self::Write(msg) => write!(f, "write failed: {msg}"),
            Self::Delete(msg) => write!(f, "delete failed: {msg}"),
            Self::Backend(msg) => write!(f, "backend error: {msg}"),
        }
    }
}

impl StorageError {
    /// Create a not found error.
    #[must_use]
    pub fn not_found(key: impl Into<String>) -> Self {
        Self::NotFound(key.into())
    }

    /// Create a connection error.
    #[must_use]
    pub fn connection(msg: impl Into<String>) -> Self {
        Self::Connection(msg.into())
    }

    /// Create a query error.
    #[must_use]
    pub fn query(msg: impl Into<String>) -> Self {
        Self::Query(msg.into())
    }

    /// Create a write error.
    #[must_use]
    pub fn write(msg: impl Into<String>) -> Self {
        Self::Write(msg.into())
    }

    /// Create a delete error.
    #[must_use]
    pub fn delete(msg: impl Into<String>) -> Self {
        Self::Delete(msg.into())
    }

    /// Create a backend error.
    #[must_use]
    pub fn backend(msg: impl Into<String>) -> Self {
        Self::Backend(msg.into())
    }
}

impl Error {
    /// Create an invalid config error.
    #[must_use]
    pub fn invalid_config(msg: impl Into<String>) -> Self {
        Self::InvalidConfig(msg.into())
    }

    /// Create a consolidation error.
    #[must_use]
    pub fn consolidation(msg: impl Into<String>) -> Self {
        Self::Consolidation(msg.into())
    }

    /// Create an embedding error.
    #[must_use]
    pub fn embedding(msg: impl Into<String>) -> Self {
        Self::Embedding(msg.into())
    }

    /// Create an internal error.
    #[must_use]
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }
}
