//! Vector Symbolic Architecture (VSA) / Hyperdimensional Computing (HDC) module.
//!
//! This module provides a complete implementation of VSA operations for
//! holographic memory storage and retrieval — enabling "RAG on crack" through
//! compositional, distributed representations.
//!
//! # Why VSA for AI Memory?
//!
//! Traditional vector embeddings store concepts as points in space. VSA instead
//! uses high-dimensional vectors (typically 10,000 dimensions) where:
//!
//! - **Random vectors are nearly orthogonal** — any two random concepts won't
//!   interfere, providing massive capacity.
//! - **Binding creates associations** — "Paris IS-CAPITAL-OF France" becomes a
//!   single vector that can later be queried.
//! - **Bundling superimposes multiple items** — a set of facts becomes one
//!   composite vector while preserving queryability.
//! - **Similarity via cosine distance** — related concepts cluster naturally.
//!
//! This enables analogical reasoning: "What is to Germany as Paris is to France?"
//! can be answered by vector algebra.
//!
//! # Core Operations
//!
//! - **Binding** (`bind`): Creates associations between concepts (reversible)
//! - **Bundling** (`bundle`): Superimposes multiple vectors into one
//! - **Permutation** (`permute`): Encodes position/sequence information
//! - **Similarity** (`cosine_similarity`): Measures relatedness
//!
//! # Example
//!
//! ```rust,ignore
//! use memory_gate_rs::vsa::{HolographicVector, VsaCodebook, HolographicStore};
//!
//! // Create a codebook for atomic symbols
//! let mut codebook = VsaCodebook::new(10000);
//!
//! // Get/create vectors for concepts
//! let country = codebook.get_or_create("country");
//! let capital = codebook.get_or_create("capital");
//! let france = codebook.get_or_create("france");
//! let paris = codebook.get_or_create("paris");
//!
//! // Bind country->france and capital->paris, then bundle
//! let france_record = country.bind(&france).bundle(&capital.bind(&paris));
//!
//! // Query: what is the capital of france?
//! let query = france_record.bind(&capital); // Unbind capital role
//! let result = codebook.find_nearest(&query, 5);
//! // result[0] should be "paris"
//! ```

mod codebook;
mod holographic_store;
mod ops;
mod vector;

pub use codebook::{CodebookEntry, VsaCodebook};
pub use holographic_store::{HolographicConfig, HolographicStore, MemoryTrace};
pub use ops::{BindingMode, BundlingMode, MapVsaOps, ResonatorNetwork, VsaOps};
pub use vector::{HolographicVector, Polarity};
