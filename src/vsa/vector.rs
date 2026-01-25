//! Core hyperdimensional vector type and operations.
//!
//! Implements the fundamental data structure for VSA computations.

use std::ops::{Add, Mul, Neg};
use std::fmt;

/// Polarity mode for vector encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Polarity {
    /// Bipolar encoding: values are -1.0 or +1.0
    #[default]
    Bipolar,
    /// Binary encoding: values are 0.0 or 1.0
    Binary,
    /// Dense floating point: normalized continuous values
    Dense,
}

/// A high-dimensional vector for holographic memory operations.
///
/// This is the core data structure for Vector Symbolic Architecture (VSA).
/// Vectors are typically high-dimensional (1000-10000 dimensions) to ensure
/// near-orthogonality of random vectors with high probability.
#[derive(Clone)]
pub struct HolographicVector {
    /// The actual vector data
    pub(crate) data: Vec<f32>,
    /// Encoding polarity mode
    pub(crate) polarity: Polarity,
    /// Cached magnitude for efficient similarity
    pub(crate) magnitude: Option<f32>,
}

impl fmt::Debug for HolographicVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HolographicVector")
            .field("dimensions", &self.data.len())
            .field("polarity", &self.polarity)
            .field("magnitude", &self.magnitude())
            .finish()
    }
}

impl HolographicVector {
    /// Create a new zero vector with the specified dimensions.
    #[must_use]
    pub fn zeros(dimensions: usize) -> Self {
        Self {
            data: vec![0.0; dimensions],
            polarity: Polarity::Dense,
            magnitude: Some(0.0),
        }
    }

    /// Create a random bipolar vector (+1/-1 values).
    ///
    /// Random bipolar vectors are nearly orthogonal in high dimensions.
    #[must_use]
    pub fn random_bipolar(dimensions: usize) -> Self {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hasher};
        
        let state = RandomState::new();
        let mut data = Vec::with_capacity(dimensions);
        
        for i in 0..dimensions {
            let mut hasher = state.build_hasher();
            hasher.write_usize(i);
            let hash = hasher.finish();
            data.push(if hash & 1 == 0 { 1.0 } else { -1.0 });
        }
        
        Self {
            data,
            polarity: Polarity::Bipolar,
            magnitude: Some((dimensions as f32).sqrt()),
        }
    }

    /// Create a random bipolar vector with a seed for reproducibility.
    #[must_use]
    pub fn random_bipolar_seeded(dimensions: usize, seed: u64) -> Self {
        // Simple xorshift PRNG for reproducibility
        let mut state = seed;
        let mut data = Vec::with_capacity(dimensions);
        
        for _ in 0..dimensions {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            data.push(if state & 1 == 0 { 1.0 } else { -1.0 });
        }
        
        Self {
            data,
            polarity: Polarity::Bipolar,
            magnitude: Some((dimensions as f32).sqrt()),
        }
    }

    /// Create a vector from raw data.
    #[must_use]
    pub const fn from_data(data: Vec<f32>, polarity: Polarity) -> Self {
        Self {
            data,
            polarity,
            magnitude: None,
        }
    }

    /// Create a normalized dense vector from raw data.
    #[must_use]
    pub fn from_embedding(data: Vec<f32>) -> Self {
        let mut vec = Self {
            data,
            polarity: Polarity::Dense,
            magnitude: None,
        };
        vec.normalize();
        vec
    }

    /// Get the dimensionality of this vector.
    #[must_use]
    pub const fn dimensions(&self) -> usize {
        self.data.len()
    }

    /// Get the polarity mode.
    #[must_use]
    pub const fn polarity(&self) -> Polarity {
        self.polarity
    }

    /// Get the raw data slice.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Get mutable access to raw data.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        self.magnitude = None; // Invalidate cache
        &mut self.data
    }

    /// Compute and cache the magnitude (L2 norm).
    #[must_use]
    pub fn magnitude(&self) -> f32 {
        if let Some(mag) = self.magnitude {
            return mag;
        }
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalize the vector to unit length.
    pub fn normalize(&mut self) {
        let mag = self.magnitude();
        if mag > f32::EPSILON {
            for x in &mut self.data {
                *x /= mag;
            }
            self.magnitude = Some(1.0);
        }
    }

    /// Return a normalized copy of this vector.
    #[must_use]
    pub fn normalized(&self) -> Self {
        let mut result = self.clone();
        result.normalize();
        result
    }

    /// Compute cosine similarity with another vector.
    ///
    /// Returns a value in [-1.0, 1.0] where:
    /// - 1.0 means identical direction
    /// - 0.0 means orthogonal (unrelated)
    /// - -1.0 means opposite direction
    #[must_use]
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        assert_eq!(self.data.len(), other.data.len(), "Dimension mismatch");
        
        let dot: f32 = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum();
        
        let mag_self = self.magnitude();
        let mag_other = other.magnitude();
        
        if mag_self < f32::EPSILON || mag_other < f32::EPSILON {
            return 0.0;
        }
        
        (dot / (mag_self * mag_other)).clamp(-1.0, 1.0)
    }

    /// Compute dot product with another vector.
    #[must_use]
    pub fn dot(&self, other: &Self) -> f32 {
        assert_eq!(self.data.len(), other.data.len(), "Dimension mismatch");
        self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Compute Hamming distance (for bipolar/binary vectors).
    #[must_use]
    pub fn hamming_distance(&self, other: &Self) -> usize {
        assert_eq!(self.data.len(), other.data.len(), "Dimension mismatch");
        self.data.iter()
            .zip(other.data.iter())
            .filter(|(a, b)| (a.signum() - b.signum()).abs() > f32::EPSILON)
            .count()
    }

    // ==================== VSA OPERATIONS ====================

    /// Bind two vectors together (multiplicative binding).
    ///
    /// Binding creates an association between two concepts. The result
    /// is dissimilar to both inputs but can be "unbound" by binding
    /// with either input again (since bind is its own inverse for bipolar).
    ///
    /// For bipolar vectors: element-wise XOR (multiplication)
    /// For dense vectors: element-wise multiplication + normalization
    #[must_use]
    pub fn bind(&self, other: &Self) -> Self {
        assert_eq!(self.data.len(), other.data.len(), "Dimension mismatch");
        
        let data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        
        let mut result = Self {
            data,
            polarity: if self.polarity == Polarity::Bipolar && other.polarity == Polarity::Bipolar {
                Polarity::Bipolar
            } else {
                Polarity::Dense
            },
            magnitude: None,
        };
        
        // For dense vectors, normalize the result
        if result.polarity == Polarity::Dense {
            result.normalize();
        }
        
        result
    }

    /// Inverse bind (unbind) operation.
    ///
    /// For bipolar vectors, bind is self-inverse: unbind(A, B) = bind(A, B)
    /// For dense vectors, we use element-wise division + normalization.
    #[must_use]
    pub fn unbind(&self, other: &Self) -> Self {
        if self.polarity == Polarity::Bipolar && other.polarity == Polarity::Bipolar {
            // For bipolar, bind is self-inverse
            self.bind(other)
        } else {
            // For dense, use division with small epsilon to prevent div-by-zero
            assert_eq!(self.data.len(), other.data.len(), "Dimension mismatch");
            
            let data: Vec<f32> = self.data.iter()
                .zip(other.data.iter())
                .map(|(a, b)| {
                    if b.abs() < f32::EPSILON {
                        *a
                    } else {
                        a / b
                    }
                })
                .collect();
            
            let mut result = Self {
                data,
                polarity: Polarity::Dense,
                magnitude: None,
            };
            result.normalize();
            result
        }
    }

    /// Bundle multiple vectors together (additive superposition).
    ///
    /// Bundling combines multiple vectors into one that is similar to all
    /// inputs. This enables storing multiple items in a single vector.
    ///
    /// For bipolar: majority vote after summing
    /// For dense: weighted average
    #[must_use]
    pub fn bundle(&self, other: &Self) -> Self {
        self.bundle_weighted(&[(other, 1.0)])
    }

    /// Bundle with multiple weighted vectors.
    #[must_use]
    pub fn bundle_weighted(&self, others: &[(&Self, f32)]) -> Self {
        let mut sum = self.data.clone();
        
        for (vec, weight) in others {
            assert_eq!(sum.len(), vec.data.len(), "Dimension mismatch");
            for (s, v) in sum.iter_mut().zip(vec.data.iter()) {
                *s += v * weight;
            }
        }
        
        // Apply majority vote for bipolar, normalize for dense
        let polarity = if self.polarity == Polarity::Bipolar 
            && others.iter().all(|(v, _)| v.polarity == Polarity::Bipolar) {
            // Majority vote: sign of sum
            for s in &mut sum {
                *s = s.signum();
                if s.abs() < f32::EPSILON {
                    // Tie-break randomly (use index parity)
                    *s = 1.0;
                }
            }
            Polarity::Bipolar
        } else {
            Polarity::Dense
        };
        
        let mut result = Self {
            data: sum,
            polarity,
            magnitude: None,
        };
        
        if polarity == Polarity::Dense {
            result.normalize();
        }
        
        result
    }

    /// Bundle a collection of vectors.
    #[must_use]
    pub fn bundle_all(vectors: &[Self]) -> Option<Self> {
        if vectors.is_empty() {
            return None;
        }
        
        let first = &vectors[0];
        let others: Vec<(&Self, f32)> = vectors[1..].iter()
            .map(|v| (v, 1.0))
            .collect();
        
        Some(first.bundle_weighted(&others))
    }

    /// Permute the vector (cyclic shift).
    ///
    /// Permutation is used to encode position or sequence information.
    /// Different permutations create quasi-orthogonal variants.
    #[must_use]
    pub fn permute(&self, shifts: i32) -> Self {
        let n = self.data.len();
        let shifts = shifts.rem_euclid(n as i32) as usize;
        
        let mut data = vec![0.0; n];
        for (i, &val) in self.data.iter().enumerate() {
            data[(i + shifts) % n] = val;
        }
        
        Self {
            data,
            polarity: self.polarity,
            magnitude: self.magnitude,
        }
    }

    /// Inverse permute.
    #[must_use]
    pub fn inverse_permute(&self, shifts: i32) -> Self {
        self.permute(-shifts)
    }

    /// Apply threshold to convert dense vector to bipolar.
    #[must_use]
    pub fn to_bipolar(&self) -> Self {
        let data: Vec<f32> = self.data.iter()
            .map(|x| if *x >= 0.0 { 1.0 } else { -1.0 })
            .collect();
        
        Self {
            magnitude: Some((data.len() as f32).sqrt()),
            data,
            polarity: Polarity::Bipolar,
        }
    }

    /// Create a sequence encoding by binding with permuted position vectors.
    ///
    /// Encodes: `pos_0 ⊛ item_0 + pos_1 ⊛ item_1 + ...`
    #[must_use]
    pub fn encode_sequence(items: &[Self], position_base: &Self) -> Option<Self> {
        if items.is_empty() {
            return None;
        }
        
        let encoded: Vec<Self> = items.iter()
            .enumerate()
            .map(|(i, item)| position_base.permute(i as i32).bind(item))
            .collect();
        
        Self::bundle_all(&encoded)
    }

    /// Decode a specific position from a sequence encoding.
    #[must_use]
    pub fn decode_sequence_position(&self, position: usize, position_base: &Self) -> Self {
        let pos_vec = position_base.permute(position as i32);
        self.unbind(&pos_vec)
    }
}

// Implement operator overloading for convenience

impl Add for &HolographicVector {
    type Output = HolographicVector;
    
    fn add(self, rhs: Self) -> Self::Output {
        self.bundle(rhs)
    }
}

impl Mul for &HolographicVector {
    type Output = HolographicVector;
    
    fn mul(self, rhs: Self) -> Self::Output {
        self.bind(rhs)
    }
}

impl Neg for &HolographicVector {
    type Output = HolographicVector;
    
    fn neg(self) -> Self::Output {
        HolographicVector {
            data: self.data.iter().map(|x| -x).collect(),
            polarity: self.polarity,
            magnitude: self.magnitude,
        }
    }
}

impl PartialEq for HolographicVector {
    fn eq(&self, other: &Self) -> bool {
        self.data.len() == other.data.len() 
            && self.cosine_similarity(other) > 0.999
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_bipolar_orthogonality() {
        // Random bipolar vectors should be nearly orthogonal
        let dim = 10000;
        let a = HolographicVector::random_bipolar_seeded(dim, 42);
        let b = HolographicVector::random_bipolar_seeded(dim, 123);
        
        let sim = a.cosine_similarity(&b);
        assert!(sim.abs() < 0.1, "Random vectors should be nearly orthogonal, got {sim}");
    }

    #[test]
    fn test_bind_self_inverse() {
        let dim = 1000;
        let a = HolographicVector::random_bipolar_seeded(dim, 42);
        let b = HolographicVector::random_bipolar_seeded(dim, 123);
        
        // bind(bind(a, b), b) should recover a
        let bound = a.bind(&b);
        let recovered = bound.bind(&b);
        
        let sim = a.cosine_similarity(&recovered);
        assert!(sim > 0.99, "Bind should be self-inverse for bipolar, got {sim}");
    }

    #[test]
    fn test_bind_dissimilarity() {
        let dim = 1000;
        let a = HolographicVector::random_bipolar_seeded(dim, 42);
        let b = HolographicVector::random_bipolar_seeded(dim, 123);
        
        let bound = a.bind(&b);
        
        // Bound result should be dissimilar to both inputs
        assert!(a.cosine_similarity(&bound).abs() < 0.2);
        assert!(b.cosine_similarity(&bound).abs() < 0.2);
    }

    #[test]
    fn test_bundle_similarity() {
        let dim = 1000;
        let a = HolographicVector::random_bipolar_seeded(dim, 42);
        let b = HolographicVector::random_bipolar_seeded(dim, 123);
        
        let bundled = a.bundle(&b);
        
        // Bundled result should be somewhat similar to both inputs
        // For bipolar with majority vote, similarity is typically ~0.5 (half agree)
        assert!(a.cosine_similarity(&bundled) > 0.4, "Bundle should be similar to a");
        assert!(b.cosine_similarity(&bundled) > 0.4, "Bundle should be similar to b");
    }

    #[test]
    fn test_permute_inverse() {
        let dim = 1000;
        let a = HolographicVector::random_bipolar_seeded(dim, 42);
        
        let shifted = a.permute(5);
        let recovered = shifted.inverse_permute(5);
        
        let sim = a.cosine_similarity(&recovered);
        assert!(sim > 0.99, "Permute should be invertible, got {sim}");
    }

    #[test]
    fn test_permute_quasi_orthogonal() {
        let dim = 1000;
        let a = HolographicVector::random_bipolar_seeded(dim, 42);
        
        let shifted = a.permute(100);
        let sim = a.cosine_similarity(&shifted);
        
        // Different permutations should be quasi-orthogonal
        assert!(sim.abs() < 0.2, "Permuted vectors should be quasi-orthogonal, got {sim}");
    }

    #[test]
    fn test_sequence_encoding() {
        let dim = 10000;
        let pos_base = HolographicVector::random_bipolar_seeded(dim, 999);
        
        let items: Vec<HolographicVector> = (0..5)
            .map(|i| HolographicVector::random_bipolar_seeded(dim, i))
            .collect();
        
        let sequence = HolographicVector::encode_sequence(&items, &pos_base).unwrap();
        
        // Decode position 2
        let decoded = sequence.decode_sequence_position(2, &pos_base);
        
        // Should be most similar to items[2]
        let mut similarities: Vec<(usize, f32)> = items.iter()
            .enumerate()
            .map(|(i, item)| (i, decoded.cosine_similarity(item)))
            .collect();
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        assert_eq!(similarities[0].0, 2, "Decoded should be most similar to original item");
    }
}
