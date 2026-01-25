//! Storage backends for knowledge persistence.

mod in_memory;

pub use in_memory::InMemoryStore;

#[cfg(feature = "qdrant")]
mod qdrant;
#[cfg(feature = "qdrant")]
pub use qdrant::QdrantStore;

#[cfg(feature = "sqlite-vec")]
mod sqlite_vec;
#[cfg(feature = "sqlite-vec")]
pub use sqlite_vec::SqliteVecStore;
