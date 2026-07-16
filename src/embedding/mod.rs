//! Shared embedding model catalog for vector backends.
//!
//! Stable string IDs are shared with the Python `memory-gate` package so both
//! implementations can run apples-to-apples retrieval comparisons.
//!
//! # Catalog (`mg/embed-catalog`)
//!
//! | ID | `FastEmbed` | `SentenceTransformers` | Dim |
//! |----|-----------|----------------------|-----|
//! | `all-minilm-l6-v2` | `AllMiniLML6V2` | `all-MiniLM-L6-v2` | 384 |
//! | `bge-small-en-v1.5` | `BGESmallENV15` | `BAAI/bge-small-en-v1.5` | 384 |
//! | `bge-base-en-v1.5` | `BGEBaseENV15` | `BAAI/bge-base-en-v1.5` | 768 |
//!
//! # Invariants
//!
//! - One model (and its dimension) is bound per collection / DB.
//! - Default for production vector backends remains **`bge-small-en-v1.5`**
//!   (backward compatible with prior hardcoding).
//! - Cross-port parity suites should pin **`all-minilm-l6-v2`**.

use crate::{Error, Result};
use std::fmt;
use std::str::FromStr;

#[cfg(any(feature = "qdrant", feature = "sqlite-vec"))]
pub mod cache;

/// First-class embedding models supported by memory-gate-rs vector backends.
///
/// String IDs match the Python `memory_gate.embedding_catalog` module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum SupportedEmbeddingModel {
    /// `sentence-transformers/all-MiniLM-L6-v2` (384-d). Preferred for cross-port parity.
    AllMiniLmL6V2,
    /// `BAAI/bge-small-en-v1.5` (384-d). Default for Qdrant / sqlite-vec.
    #[default]
    BgeSmallEnV15,
    /// `BAAI/bge-base-en-v1.5` (768-d). Larger accuracy tradeoff.
    BgeBaseEnV15,
}

impl SupportedEmbeddingModel {
    /// Default model for vector backends (matches historical BGE-small hardcoding).
    pub const DEFAULT: Self = Self::BgeSmallEnV15;

    /// Stable catalog ID used in config, metadata, and cross-port fixtures.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AllMiniLmL6V2 => "all-minilm-l6-v2",
            Self::BgeSmallEnV15 => "bge-small-en-v1.5",
            Self::BgeBaseEnV15 => "bge-base-en-v1.5",
        }
    }

    /// `HuggingFace` / `SentenceTransformers` model id for the Python port.
    #[must_use]
    pub const fn sentence_transformers_id(self) -> &'static str {
        match self {
            Self::AllMiniLmL6V2 => "all-MiniLM-L6-v2",
            Self::BgeSmallEnV15 => "BAAI/bge-small-en-v1.5",
            Self::BgeBaseEnV15 => "BAAI/bge-base-en-v1.5",
        }
    }

    /// Output embedding dimension for this model.
    #[must_use]
    pub const fn dimension(self) -> usize {
        match self {
            Self::AllMiniLmL6V2 | Self::BgeSmallEnV15 => 384,
            Self::BgeBaseEnV15 => 768,
        }
    }

    /// All first-class catalog entries (stable order).
    #[must_use]
    pub const fn all() -> [Self; 3] {
        [Self::AllMiniLmL6V2, Self::BgeSmallEnV15, Self::BgeBaseEnV15]
    }

    /// Parse a catalog ID or common alias (case-insensitive for the ID form).
    ///
    /// Accepts stable IDs, HF names, and FastEmbed-style names.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] when the id is not in the catalog.
    pub fn parse(id: &str) -> Result<Self> {
        id.parse()
    }

    /// Map to the `FastEmbed` enum variant used by vector backends.
    #[cfg(any(feature = "qdrant", feature = "sqlite-vec"))]
    #[must_use]
    pub const fn fastembed_model(self) -> fastembed::EmbeddingModel {
        match self {
            Self::AllMiniLmL6V2 => fastembed::EmbeddingModel::AllMiniLML6V2,
            Self::BgeSmallEnV15 => fastembed::EmbeddingModel::BGESmallENV15,
            Self::BgeBaseEnV15 => fastembed::EmbeddingModel::BGEBaseENV15,
        }
    }
}

impl fmt::Display for SupportedEmbeddingModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for SupportedEmbeddingModel {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        let normalized = s.trim().to_ascii_lowercase();
        // Strip common org prefixes for matching
        let key = normalized
            .strip_prefix("sentence-transformers/")
            .or_else(|| normalized.strip_prefix("baai/"))
            .unwrap_or(normalized.as_str());

        match key {
            "all-minilm-l6-v2" | "allminilml6v2" | "minilm" | "minilm-l6" => {
                Ok(Self::AllMiniLmL6V2)
            }
            "bge-small-en-v1.5" | "bgesmallenv15" | "bge-small" | "bge_small_en_v1.5" => {
                Ok(Self::BgeSmallEnV15)
            }
            "bge-base-en-v1.5" | "bgebaseenv15" | "bge-base" | "bge_base_en_v1.5" => {
                Ok(Self::BgeBaseEnV15)
            }
            _ => {
                let supported: Vec<&str> = Self::all().into_iter().map(Self::as_str).collect();
                Err(Error::invalid_config(format!(
                    "unknown embedding model '{s}'; supported: {}",
                    supported.join(", ")
                )))
            }
        }
    }
}

/// Initialize a `FastEmbed` `fastembed::TextEmbedding` for the given catalog model.
///
/// # Errors
///
/// Returns [`Error::Embedding`] if model download or ONNX init fails.
#[cfg(any(feature = "qdrant", feature = "sqlite-vec"))]
pub fn init_text_embedding(model: SupportedEmbeddingModel) -> Result<fastembed::TextEmbedding> {
    use fastembed::{InitOptions, TextEmbedding};

    TextEmbedding::try_new(
        InitOptions::new(model.fastembed_model()).with_show_download_progress(false),
    )
    .map_err(|e| Error::embedding(format!("Failed to initialize embedder ({model}): {e}")))
}

/// Embed multiple texts in one `FastEmbed` call (single lock acquisition on the caller side).
///
/// # Errors
///
/// Returns [`Error::Embedding`] when batch encoding fails or output length mismatches input.
#[cfg(any(feature = "qdrant", feature = "sqlite-vec"))]
pub fn embed_batch(
    embedder: &mut fastembed::TextEmbedding,
    texts: &[&str],
) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    let embeddings = embedder
        .embed(texts, None)
        .map_err(|e| Error::embedding(format!("Failed to batch-generate embeddings: {e}")))?;
    check_batch_embed_lengths(texts.len(), embeddings.len())?;
    Ok(embeddings)
}

/// Verify `FastEmbed` batch output length (unit-testable without model download).
#[cfg(any(feature = "qdrant", feature = "sqlite-vec"))]
fn check_batch_embed_lengths(expected: usize, actual: usize) -> Result<()> {
    if actual != expected {
        return Err(Error::embedding(format!(
            "batch embed length mismatch: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_stable_ids() {
        assert_eq!(
            SupportedEmbeddingModel::parse("all-minilm-l6-v2").unwrap(),
            SupportedEmbeddingModel::AllMiniLmL6V2
        );
        assert_eq!(
            SupportedEmbeddingModel::parse("bge-small-en-v1.5").unwrap(),
            SupportedEmbeddingModel::BgeSmallEnV15
        );
        assert_eq!(
            SupportedEmbeddingModel::parse("bge-base-en-v1.5").unwrap(),
            SupportedEmbeddingModel::BgeBaseEnV15
        );
    }

    #[test]
    fn parse_aliases_and_hf_names() {
        assert_eq!(
            SupportedEmbeddingModel::parse("all-MiniLM-L6-v2").unwrap(),
            SupportedEmbeddingModel::AllMiniLmL6V2
        );
        assert_eq!(
            SupportedEmbeddingModel::parse("BAAI/bge-small-en-v1.5").unwrap(),
            SupportedEmbeddingModel::BgeSmallEnV15
        );
        assert_eq!(
            SupportedEmbeddingModel::parse("sentence-transformers/all-MiniLM-L6-v2").unwrap(),
            SupportedEmbeddingModel::AllMiniLmL6V2
        );
    }

    #[test]
    fn parse_unknown_errors() {
        let err = SupportedEmbeddingModel::parse("not-a-model").unwrap_err();
        assert!(err.to_string().contains("unknown embedding model"));
    }

    #[test]
    fn dimensions() {
        assert_eq!(SupportedEmbeddingModel::AllMiniLmL6V2.dimension(), 384);
        assert_eq!(SupportedEmbeddingModel::BgeSmallEnV15.dimension(), 384);
        assert_eq!(SupportedEmbeddingModel::BgeBaseEnV15.dimension(), 768);
    }

    #[cfg(any(feature = "qdrant", feature = "sqlite-vec"))]
    #[test]
    fn check_batch_embed_lengths_mismatch() {
        let err = check_batch_embed_lengths(3, 2).unwrap_err();
        assert!(err.to_string().contains("batch embed length mismatch"));
        assert!(check_batch_embed_lengths(0, 0).is_ok());
    }

    #[test]
    fn default_is_bge_small() {
        assert_eq!(
            SupportedEmbeddingModel::DEFAULT,
            SupportedEmbeddingModel::BgeSmallEnV15
        );
        assert_eq!(
            SupportedEmbeddingModel::default(),
            SupportedEmbeddingModel::BgeSmallEnV15
        );
    }

    #[test]
    fn sentence_transformers_ids() {
        assert_eq!(
            SupportedEmbeddingModel::AllMiniLmL6V2.sentence_transformers_id(),
            "all-MiniLM-L6-v2"
        );
        assert_eq!(
            SupportedEmbeddingModel::BgeSmallEnV15.sentence_transformers_id(),
            "BAAI/bge-small-en-v1.5"
        );
        assert_eq!(
            SupportedEmbeddingModel::BgeBaseEnV15.sentence_transformers_id(),
            "BAAI/bge-base-en-v1.5"
        );
    }
}
