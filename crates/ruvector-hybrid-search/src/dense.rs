use crate::types::{cosine_similarity, HybridError, HybridSearcher, SearchResult, SparseVec};

/// Baseline: exact nearest-neighbour search using brute-force cosine similarity.
/// O(N*D) per query, 100% recall by definition. Used as ground truth.
pub struct DenseLinearSearcher {
    dim: usize,
    vectors: Vec<Vec<f32>>,
    ids: Vec<u64>,
}

impl DenseLinearSearcher {
    pub fn new(dim: usize) -> Self {
        Self { dim, vectors: Vec::new(), ids: Vec::new() }
    }

    /// Return ground-truth top-k ids for recall computation.
    pub fn ground_truth(&self, query: &[f32], k: usize) -> Vec<u64> {
        let results = self.search(query, &SparseVec::new(vec![]), k);
        results.into_iter().map(|r| r.id).collect()
    }
}

impl HybridSearcher for DenseLinearSearcher {
    fn index(&mut self, id: u64, dense: &[f32], _sparse: &SparseVec) -> Result<(), HybridError> {
        if dense.len() != self.dim {
            return Err(HybridError::DimensionMismatch {
                expected: self.dim,
                got: dense.len(),
            });
        }
        self.ids.push(id);
        self.vectors.push(dense.to_vec());
        Ok(())
    }

    fn search(&self, dense_query: &[f32], _sparse_query: &SparseVec, k: usize) -> Vec<SearchResult> {
        if self.vectors.is_empty() {
            return Vec::new();
        }
        let k = k.min(self.vectors.len());

        let mut scores: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cosine_similarity(dense_query, v)))
            .collect();

        // Partial sort: only move top-k to front.
        scores.select_nth_unstable_by(k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
        scores[..k].sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        scores[..k]
            .iter()
            .map(|(i, s)| SearchResult::new_dense(self.ids[*i], *s))
            .collect()
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_returns_k_results() {
        let mut idx = DenseLinearSearcher::new(4);
        // Vectors are unit-normalized direction vectors so cosine = dot product.
        // id=0: points toward (1,0,0,0), id=1 toward (0,1,0,0), id=2 toward (0,0,1,0),
        // id=3 toward (0,0,0,1) — all orthogonal.  Query points near id=0.
        let vecs: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![-1.0, 0.0, 0.0, 0.0],
        ];
        for (i, v) in vecs.iter().enumerate() {
            idx.index(i as u64, v, &SparseVec::new(vec![])).unwrap();
        }
        let q = vec![0.95f32, 0.1, 0.05, 0.01];
        let results = idx.search(&q, &SparseVec::new(vec![]), 3);
        assert_eq!(results.len(), 3);
        // id=0 ([1,0,0,0]) should be closest to the query.
        assert_eq!(results[0].id, 0, "id=0 should be nearest");
    }

    #[test]
    fn dense_nearest_is_identical() {
        let mut idx = DenseLinearSearcher::new(3);
        let vecs = vec![
            vec![1.0f32, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        for (i, v) in vecs.iter().enumerate() {
            idx.index(i as u64, v, &SparseVec::new(vec![])).unwrap();
        }
        let q = vec![0.0f32, 1.0, 0.0];
        let results = idx.search(&q, &SparseVec::new(vec![]), 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 1, "nearest to [0,1,0] should be id=1");
        assert!((results[0].score - 1.0).abs() < 1e-5);
    }
}
