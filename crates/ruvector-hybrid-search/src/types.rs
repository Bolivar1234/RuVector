use thiserror::Error;

/// A sparse vector stored as sorted (term_id, weight) pairs.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseVec {
    /// Sorted by term_id ascending.
    pub terms: Vec<(u32, f32)>,
    /// Pre-computed L2 norm of weight values.
    pub l2_norm: f32,
}

impl SparseVec {
    pub fn new(mut terms: Vec<(u32, f32)>) -> Self {
        terms.sort_unstable_by_key(|(id, _)| *id);
        terms.dedup_by_key(|(id, _)| *id);
        let l2_norm = terms.iter().map(|(_, w)| w * w).sum::<f32>().sqrt();
        Self { terms, l2_norm }
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Dot product with another sparse vector. Both must be sorted by term_id.
    pub fn dot(&self, other: &SparseVec) -> f32 {
        let mut score = 0.0f32;
        let mut i = 0;
        let mut j = 0;
        while i < self.terms.len() && j < other.terms.len() {
            let (ai, aw) = self.terms[i];
            let (bi, bw) = other.terms[j];
            match ai.cmp(&bi) {
                std::cmp::Ordering::Equal => {
                    score += aw * bw;
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        score
    }
}

/// A single search result with component scores for diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub id: u64,
    /// Final fused score (higher = more relevant).
    pub score: f32,
    /// Dense cosine similarity component.
    pub dense_score: f32,
    /// Sparse BM25 component.
    pub sparse_score: f32,
}

impl SearchResult {
    pub fn new_dense(id: u64, dense_score: f32) -> Self {
        Self { id, score: dense_score, dense_score, sparse_score: 0.0 }
    }

    pub fn new_sparse(id: u64, sparse_score: f32) -> Self {
        Self { id, score: sparse_score, dense_score: 0.0, sparse_score }
    }

    pub fn new_fused(id: u64, score: f32, dense_score: f32, sparse_score: f32) -> Self {
        Self { id, score, dense_score, sparse_score }
    }
}

/// Trait that all hybrid search backends implement.
pub trait HybridSearcher: Send + Sync {
    /// Index a document with both dense and sparse representations.
    fn index(
        &mut self,
        id: u64,
        dense: &[f32],
        sparse: &SparseVec,
    ) -> Result<(), HybridError>;

    /// Search for the top-k most similar documents.
    fn search(
        &self,
        dense_query: &[f32],
        sparse_query: &SparseVec,
        k: usize,
    ) -> Vec<SearchResult>;

    /// Number of indexed documents.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Error)]
pub enum HybridError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("empty index")]
    EmptyIndex,

    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
}

/// Compute cosine similarity between two equal-length dense vectors.
#[inline(always)]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < f32::EPSILON {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_vec_dot_product() {
        let a = SparseVec::new(vec![(0, 1.0), (2, 2.0), (5, 3.0)]);
        let b = SparseVec::new(vec![(2, 1.0), (5, 1.0), (9, 4.0)]);
        // overlap at 2 (2*1=2) and 5 (3*1=3) → 5.0
        let got = a.dot(&b);
        assert!((got - 5.0).abs() < 1e-5, "expected 5.0 got {got}");
    }

    #[test]
    fn cosine_identical_vectors() {
        let v = vec![1.0f32, 2.0, 3.0, 4.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5, "identical vectors should have cosine 1.0, got {sim}");
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5, "orthogonal vectors should have cosine 0.0, got {sim}");
    }
}
