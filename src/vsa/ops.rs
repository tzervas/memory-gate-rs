//! VSA operations configuration and utilities.

use crate::vsa::HolographicVector;

/// Binding mode for VSA operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BindingMode {
    /// Multiplicative binding (element-wise multiply)
    #[default]
    Multiplicative,
    /// Circular convolution binding
    Convolution,
    /// XOR binding (for binary vectors)
    Xor,
}

/// Bundling mode for VSA operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BundlingMode {
    /// Simple addition with majority vote (for bipolar)
    #[default]
    Additive,
    /// Weighted addition with decay
    WeightedDecay,
    /// Thresholded addition
    Thresholded,
}

/// VSA operations trait for extensible operation implementations.
pub trait VsaOps {
    /// Perform binding operation.
    fn bind(&self, a: &HolographicVector, b: &HolographicVector) -> HolographicVector;
    
    /// Perform unbinding operation.
    fn unbind(&self, bound: &HolographicVector, key: &HolographicVector) -> HolographicVector;
    
    /// Perform bundling operation.
    fn bundle(&self, vectors: &[&HolographicVector]) -> Option<HolographicVector>;
    
    /// Perform permutation.
    fn permute(&self, v: &HolographicVector, amount: i32) -> HolographicVector;
}

/// Default VSA operations implementation using MAP architecture.
#[derive(Debug, Clone, Default)]
pub struct MapVsaOps {
    /// Binding mode
    pub binding_mode: BindingMode,
    /// Bundling mode  
    pub bundling_mode: BundlingMode,
}

impl MapVsaOps {
    /// Create with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set binding mode.
    #[must_use]
    pub const fn with_binding_mode(mut self, mode: BindingMode) -> Self {
        self.binding_mode = mode;
        self
    }
    
    /// Set bundling mode.
    #[must_use]
    pub const fn with_bundling_mode(mut self, mode: BundlingMode) -> Self {
        self.bundling_mode = mode;
        self
    }
}

impl VsaOps for MapVsaOps {
    fn bind(&self, a: &HolographicVector, b: &HolographicVector) -> HolographicVector {
        match self.binding_mode {
            BindingMode::Multiplicative => a.bind(b),
            BindingMode::Convolution => {
                // Circular convolution via FFT would go here
                // For now, fall back to multiplicative
                a.bind(b)
            }
            BindingMode::Xor => {
                // XOR is equivalent to sign(a * b) for bipolar
                a.bind(b).to_bipolar()
            }
        }
    }
    
    fn unbind(&self, bound: &HolographicVector, key: &HolographicVector) -> HolographicVector {
        match self.binding_mode {
            BindingMode::Multiplicative | BindingMode::Xor => bound.unbind(key),
            BindingMode::Convolution => {
                // Correlation (inverse convolution)
                bound.unbind(key)
            }
        }
    }
    
    fn bundle(&self, vectors: &[&HolographicVector]) -> Option<HolographicVector> {
        if vectors.is_empty() {
            return None;
        }
        
        let refs: Vec<HolographicVector> = vectors.iter()
            .map(|&v| v.clone())
            .collect();
        
        match self.bundling_mode {
            BundlingMode::Additive => HolographicVector::bundle_all(&refs),
            BundlingMode::WeightedDecay => {
                // Weight earlier items less (recency bias)
                if refs.is_empty() {
                    return None;
                }
                let first = &refs[0];
                let n = refs.len();
                let others: Vec<(&HolographicVector, f32)> = refs[1..].iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let weight = (i + 2) as f32 / n as f32; // Increasing weight
                        (v, weight)
                    })
                    .collect();
                Some(first.bundle_weighted(&others))
            }
            BundlingMode::Thresholded => {
                // Standard bundle with hard threshold
                HolographicVector::bundle_all(&refs).map(|v| v.to_bipolar())
            }
        }
    }
    
    fn permute(&self, v: &HolographicVector, amount: i32) -> HolographicVector {
        v.permute(amount)
    }
}

/// Resonator network for factorization.
///
/// A resonator network iteratively refines factor estimates through
/// competitive dynamics, enabling recovery of bound components.
#[derive(Debug)]
pub struct ResonatorNetwork {
    /// Codebook vectors to search over
    codebook: Vec<(String, HolographicVector)>,
    /// Number of iterations
    max_iterations: usize,
    /// Convergence threshold
    convergence_threshold: f32,
}

impl ResonatorNetwork {
    /// Create a new resonator network.
    #[must_use]
    pub const fn new(codebook: Vec<(String, HolographicVector)>) -> Self {
        Self {
            codebook,
            max_iterations: 100,
            convergence_threshold: 0.001,
        }
    }
    
    /// Set maximum iterations.
    #[must_use]
    pub const fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }
    
    /// Set convergence threshold.
    #[must_use]
    pub const fn with_convergence_threshold(mut self, threshold: f32) -> Self {
        self.convergence_threshold = threshold;
        self
    }
    
    /// Factorize a bound vector to find its components.
    ///
    /// Given a vector that is the product of multiple bindings,
    /// attempts to recover the original factors.
    ///
    /// # Arguments
    ///
    /// * `bound` - The bound vector to factorize
    /// * `num_factors` - Number of factors to recover
    ///
    /// # Returns
    ///
    /// Vector of (name, vector, confidence) tuples for recovered factors.
    #[must_use]
    pub fn factorize(&self, bound: &HolographicVector, num_factors: usize) -> Vec<(String, HolographicVector, f32)> {
        if self.codebook.is_empty() || num_factors == 0 {
            return Vec::new();
        }
        
        // Initialize factor estimates randomly from codebook
        let mut factors: Vec<HolographicVector> = self.codebook.iter()
            .take(num_factors.min(self.codebook.len()))
            .map(|(_, v)| v.clone())
            .collect();
        
        // Pad with copies if needed
        while factors.len() < num_factors {
            factors.push(factors[0].clone());
        }
        
        let mut prev_sims = vec![0.0f32; num_factors];
        
        for _iter in 0..self.max_iterations {
            for i in 0..num_factors {
                // Compute product of all other factors
                let mut other_product = factors[(i + 1) % num_factors].clone();
                for (j, factor) in factors.iter().enumerate() {
                    if j != i && j != (i + 1) % num_factors {
                        other_product = other_product.bind(factor);
                    }
                }
                
                // Unbind to get estimate for factor i
                let estimate = bound.unbind(&other_product);
                
                // Find best match in codebook
                let (_best_name, best_vec, best_sim) = self.codebook.iter()
                    .map(|(name, vec)| (name.clone(), vec.clone(), estimate.cosine_similarity(vec)))
                    .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
                    .unwrap_or_else(|| (String::new(), estimate.clone(), 0.0));
                
                factors[i] = best_vec;
                
                // Check convergence
                if (best_sim - prev_sims[i]).abs() < self.convergence_threshold {
                    // This factor has converged
                }
                prev_sims[i] = best_sim;
            }
        }
        
        // Return final factors with confidence
        factors.into_iter()
            .filter_map(|factor| {
                self.codebook.iter()
                    .map(|(name, vec)| (name.clone(), vec.clone(), factor.cosine_similarity(vec)))
                    .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
            })
            .collect()
    }
    
    /// Find the single best matching item in codebook.
    #[must_use]
    pub fn find_best_match(&self, query: &HolographicVector) -> Option<(String, f32)> {
        self.codebook.iter()
            .map(|(name, vec)| (name.clone(), query.cosine_similarity(vec)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    }
    
    /// Find top-k matches in codebook.
    #[must_use]
    pub fn find_top_k(&self, query: &HolographicVector, k: usize) -> Vec<(String, f32)> {
        let mut matches: Vec<(String, f32)> = self.codebook.iter()
            .map(|(name, vec)| (name.clone(), query.cosine_similarity(vec)))
            .collect();
        
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        matches.truncate(k);
        matches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_vsa_ops_bind_unbind() {
        let ops = MapVsaOps::new();
        let dim = 1000;
        
        let a = HolographicVector::random_bipolar_seeded(dim, 42);
        let b = HolographicVector::random_bipolar_seeded(dim, 123);
        
        let bound = ops.bind(&a, &b);
        let recovered = ops.unbind(&bound, &b);
        
        assert!(a.cosine_similarity(&recovered) > 0.99);
    }

    #[test]
    fn test_resonator_simple() {
        let dim = 10000;
        
        // Create codebook
        let codebook: Vec<(String, HolographicVector)> = (0..10)
            .map(|i| (format!("item_{i}"), HolographicVector::random_bipolar_seeded(dim, i)))
            .collect();
        
        // Bind two items
        let item_3 = codebook[3].1.clone();
        let item_7 = codebook[7].1.clone();
        let bound = item_3.bind(&item_7);
        
        let network = ResonatorNetwork::new(codebook);
        
        // Query with one factor to find the other
        let query = bound.unbind(&item_3);
        let matches = network.find_top_k(&query, 3);
        
        assert_eq!(matches[0].0, "item_7", "Should find item_7 as best match");
    }
}
