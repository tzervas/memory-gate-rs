//! VSA Codebook for atomic symbol management.
//!
//! The codebook maintains a mapping from string symbols to their
//! hyperdimensional vector representations, ensuring consistent
//! encoding across operations.

use crate::vsa::HolographicVector;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Entry in the codebook with metadata.
#[derive(Debug, Clone)]
pub struct CodebookEntry {
    /// The symbol name
    pub symbol: String,
    /// The hyperdimensional vector
    pub vector: HolographicVector,
    /// Number of times this symbol has been used
    pub usage_count: u64,
    /// When this entry was created (monotonic counter)
    pub created_at: u64,
}

/// Codebook for managing atomic symbol vectors.
///
/// Provides consistent symbol-to-vector mapping with automatic
/// generation of new random vectors for unseen symbols.
#[derive(Debug)]
pub struct VsaCodebook {
    /// Dimensionality of vectors
    dimensions: usize,
    /// Symbol to entry mapping
    entries: HashMap<String, CodebookEntry>,
    /// Seed for deterministic generation
    seed: u64,
    /// Counter for entry creation ordering
    counter: AtomicU64,
    /// Special vectors for roles/relations
    role_vectors: HashMap<String, HolographicVector>,
}

impl VsaCodebook {
    /// Create a new codebook with specified dimensionality.
    #[must_use]
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            entries: HashMap::new(),
            seed: 0xDEAD_BEEF_CAFE_BABE,
            counter: AtomicU64::new(0),
            role_vectors: HashMap::new(),
        }
    }

    /// Create with a custom seed for reproducibility.
    #[must_use]
    pub fn with_seed(dimensions: usize, seed: u64) -> Self {
        Self {
            dimensions,
            entries: HashMap::new(),
            seed,
            counter: AtomicU64::new(0),
            role_vectors: HashMap::new(),
        }
    }

    /// Get the dimensionality.
    #[must_use]
    pub const fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Get the number of symbols in the codebook.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if codebook is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Generate a deterministic seed for a symbol.
    fn symbol_seed(&self, symbol: &str) -> u64 {
        // Hash the symbol name combined with codebook seed
        let mut hash = self.seed;
        for byte in symbol.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(u64::from(byte));
        }
        hash
    }

    /// Get or create a vector for a symbol.
    #[must_use]
    pub fn get_or_create(&mut self, symbol: &str) -> HolographicVector {
        if let Some(entry) = self.entries.get_mut(symbol) {
            entry.usage_count += 1;
            return entry.vector.clone();
        }

        // Create new entry with deterministic vector
        let seed = self.symbol_seed(symbol);
        let vector = HolographicVector::random_bipolar_seeded(self.dimensions, seed);
        let created_at = self.counter.fetch_add(1, Ordering::SeqCst);

        let entry = CodebookEntry {
            symbol: symbol.to_string(),
            vector: vector.clone(),
            usage_count: 1,
            created_at,
        };

        self.entries.insert(symbol.to_string(), entry);
        vector
    }

    /// Get a vector without creating if it doesn't exist.
    #[must_use]
    pub fn get(&self, symbol: &str) -> Option<HolographicVector> {
        self.entries.get(symbol).map(|e| e.vector.clone())
    }

    /// Check if a symbol exists in the codebook.
    #[must_use]
    pub fn contains(&self, symbol: &str) -> bool {
        self.entries.contains_key(symbol)
    }

    /// Get entry with metadata.
    #[must_use]
    pub fn get_entry(&self, symbol: &str) -> Option<&CodebookEntry> {
        self.entries.get(symbol)
    }

    /// Get or create a role vector (for structural encoding).
    ///
    /// Role vectors are used for binding slots in frame-like structures.
    #[must_use]
    pub fn get_or_create_role(&mut self, role: &str) -> HolographicVector {
        if let Some(vec) = self.role_vectors.get(role) {
            return vec.clone();
        }

        // Create deterministic role vector with different seed space
        let seed = self.symbol_seed(&format!("__ROLE__{role}"));
        let vector = HolographicVector::random_bipolar_seeded(self.dimensions, seed);
        self.role_vectors.insert(role.to_string(), vector.clone());
        vector
    }

    /// Find the nearest symbol to a query vector.
    #[must_use]
    pub fn find_nearest(&self, query: &HolographicVector, top_k: usize) -> Vec<(String, f32)> {
        let mut results: Vec<(String, f32)> = self.entries.iter()
            .map(|(sym, entry)| (sym.clone(), query.cosine_similarity(&entry.vector)))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(top_k);
        results
    }

    /// Find nearest with threshold filtering.
    #[must_use]
    pub fn find_above_threshold(&self, query: &HolographicVector, threshold: f32) -> Vec<(String, f32)> {
        let mut results: Vec<(String, f32)> = self.entries.iter()
            .map(|(sym, entry)| (sym.clone(), query.cosine_similarity(&entry.vector)))
            .filter(|(_, sim)| *sim >= threshold)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
    }

    /// Decode a bundled vector to find all contributing symbols.
    ///
    /// This performs "soft" factorization of a bundle by checking
    /// similarity to all codebook entries.
    #[must_use]
    pub fn decode_bundle(&self, bundle: &HolographicVector, threshold: f32) -> Vec<(String, f32)> {
        self.find_above_threshold(bundle, threshold)
    }

    /// Create a record (frame) from role-filler pairs.
    ///
    /// A record binds each role to its filler and bundles all pairs:
    /// `role_1 ⊛ filler_1 + role_2 ⊛ filler_2 + ...`
    #[must_use]
    pub fn create_record(&mut self, pairs: &[(&str, &str)]) -> HolographicVector {
        let bound_pairs: Vec<HolographicVector> = pairs.iter()
            .map(|(role, filler)| {
                let role_vec = self.get_or_create_role(role);
                let filler_vec = self.get_or_create(filler);
                role_vec.bind(&filler_vec)
            })
            .collect();

        HolographicVector::bundle_all(&bound_pairs)
            .unwrap_or_else(|| HolographicVector::zeros(self.dimensions))
    }

    /// Query a record for the filler of a specific role.
    #[must_use]
    pub fn query_record(&mut self, record: &HolographicVector, role: &str) -> Vec<(String, f32)> {
        let role_vec = self.get_or_create_role(role);
        let query = record.unbind(&role_vec);
        self.find_nearest(&query, 5)
    }

    /// Get all symbols in the codebook.
    #[must_use]
    pub fn symbols(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    /// Get entries sorted by usage count (most used first).
    #[must_use]
    pub fn entries_by_usage(&self) -> Vec<&CodebookEntry> {
        let mut entries: Vec<&CodebookEntry> = self.entries.values().collect();
        entries.sort_by(|a, b| b.usage_count.cmp(&a.usage_count));
        entries
    }

    /// Merge another codebook into this one.
    pub fn merge(&mut self, other: &Self) {
        for (symbol, entry) in &other.entries {
            if let Some(existing) = self.entries.get_mut(symbol) {
                existing.usage_count += entry.usage_count;
            } else {
                self.entries.insert(symbol.clone(), entry.clone());
            }
        }
    }

    /// Export codebook to list of (symbol, vector) pairs.
    #[must_use]
    pub fn export(&self) -> Vec<(String, HolographicVector)> {
        self.entries.iter()
            .map(|(sym, entry)| (sym.clone(), entry.vector.clone()))
            .collect()
    }

    /// Create a cleanup vector that is orthogonal to all entries.
    ///
    /// Useful for "cleaning up" noisy bundled representations.
    #[must_use]
    pub fn create_cleanup_vector(&self) -> HolographicVector {
        // Random vector that should be orthogonal to all in high dimensions
        HolographicVector::random_bipolar_seeded(
            self.dimensions,
            self.counter.load(Ordering::SeqCst).wrapping_add(0xCAFE),
        )
    }
}

impl Clone for VsaCodebook {
    fn clone(&self) -> Self {
        Self {
            dimensions: self.dimensions,
            entries: self.entries.clone(),
            seed: self.seed,
            counter: AtomicU64::new(self.counter.load(Ordering::SeqCst)),
            role_vectors: self.role_vectors.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_or_create_consistent() {
        let mut cb = VsaCodebook::new(1000);
        
        let v1 = cb.get_or_create("test");
        let v2 = cb.get_or_create("test");
        
        assert!(v1.cosine_similarity(&v2) > 0.99, "Same symbol should return same vector");
    }

    #[test]
    fn test_different_symbols_orthogonal() {
        let mut cb = VsaCodebook::new(10000);
        
        let v1 = cb.get_or_create("apple");
        let v2 = cb.get_or_create("banana");
        
        let sim = v1.cosine_similarity(&v2);
        assert!(sim.abs() < 0.1, "Different symbols should be nearly orthogonal, got {sim}");
    }

    #[test]
    fn test_record_query() {
        let mut cb = VsaCodebook::with_seed(10000, 42);
        
        // Create a record: country=france, capital=paris
        let record = cb.create_record(&[
            ("country", "france"),
            ("capital", "paris"),
        ]);
        
        // Query: what is the capital?
        let results = cb.query_record(&record, "capital");
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "paris", "Should find paris as capital");
    }

    #[test]
    fn test_find_nearest() {
        let mut cb = VsaCodebook::new(10000);
        
        // Add some symbols
        let target = cb.get_or_create("target");
        cb.get_or_create("other1");
        cb.get_or_create("other2");
        
        // Add some noise to target
        let noisy = target.bundle(&HolographicVector::random_bipolar(10000));
        
        let results = cb.find_nearest(&noisy, 3);
        assert_eq!(results[0].0, "target", "Should find target as nearest");
    }
}
