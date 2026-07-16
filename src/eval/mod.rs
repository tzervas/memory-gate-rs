//! Retrieval evaluation helpers (golden recall harness).
//!
//! Pure metrics with no embedding dependencies; used by `tests/golden_recall.rs`.
//! Not re-exported from the crate prelude (harness/library helper, not core gateway API).
//!
//! # Semantics
//!
//! - `recall_at_k`: set-based; duplicate IDs in either list count once.
//! - `k == 0`: top-k is empty → recall is `0.0` if any relevant IDs exist.
//! - Empty `relevant_ids`: returns `1.0` (vacuous success). Golden fixtures must not
//!   include empty relevance lists (enforced by the integration test).
//!
//! Run the full sqlite-vec golden path (downloads model on first run):
//! `cargo test --features sqlite-vec --test golden_recall -- --ignored`

use std::collections::HashSet;

/// Per-query recall@k: fraction of relevant IDs that appear in the top-k retrieved set.
///
/// `recall@k = |retrieved_top_k ∩ relevant| / |relevant|` (unique IDs).
///
/// Returns `1.0` when `relevant_ids` is empty (vacuous success).
#[must_use]
pub fn recall_at_k(
    retrieved_ids: &[impl AsRef<str>],
    relevant_ids: &[impl AsRef<str>],
    k: usize,
) -> f64 {
    if relevant_ids.is_empty() {
        return 1.0;
    }

    let top_k: HashSet<&str> = retrieved_ids.iter().take(k).map(AsRef::as_ref).collect();
    let relevant: HashSet<&str> = relevant_ids.iter().map(AsRef::as_ref).collect();
    let hits = relevant.intersection(&top_k).count();
    hits as f64 / relevant.len() as f64
}

/// Arithmetic mean of per-query recall@k scores.
///
/// Empty input returns `0.0` (no queries scored — distinct from all-zero recalls only if
/// callers also check length; golden harness requires non-empty query lists).
#[must_use]
pub fn mean_recall(per_query_recall: &[f64]) -> f64 {
    if per_query_recall.is_empty() {
        return 0.0;
    }
    per_query_recall.iter().sum::<f64>() / per_query_recall.len() as f64
}

#[cfg(test)]
mod tests {
    use super::{mean_recall, recall_at_k};

    #[test]
    fn recall_at_k_perfect_hit() {
        let retrieved = ["a", "b", "c"];
        let relevant = ["a", "b"];
        assert!((recall_at_k(&retrieved, &relevant, 5) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn recall_at_k_partial_within_k() {
        let retrieved = ["a", "x", "y"];
        let relevant = ["a", "b"];
        assert!((recall_at_k(&retrieved, &relevant, 3) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn recall_at_k_respects_k_cutoff() {
        let retrieved = ["x", "y", "a", "b"];
        let relevant = ["a", "b"];
        assert!((recall_at_k(&retrieved, &relevant, 2) - 0.0).abs() < f64::EPSILON);
        assert!((recall_at_k(&retrieved, &relevant, 4) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn recall_at_k_empty_relevant_is_vacuous_one() {
        assert!((recall_at_k(&["a"], &[] as &[&str], 1) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn recall_at_k_zero_is_empty_topk() {
        let retrieved = ["a", "b"];
        let relevant = ["a"];
        assert!((recall_at_k(&retrieved, &relevant, 0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn recall_at_k_dedupes_duplicate_ids() {
        let retrieved = ["a", "a", "b"];
        let relevant = ["a", "a", "b"];
        assert!((recall_at_k(&retrieved, &relevant, 3) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mean_recall_averages() {
        assert!((mean_recall(&[1.0, 0.5, 0.0]) - 0.5).abs() < f64::EPSILON);
        assert!((mean_recall(&[]) - 0.0).abs() < f64::EPSILON);
    }
}
