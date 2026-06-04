use std::collections::HashMap;
use crate::types::{cosine_similarity, HybridError, HybridSearcher, SearchResult, SparseVec};
use crate::sparse::{Bm25Params, SparseInvertedIndex};

/// Reciprocal Rank Fusion constant. Standard value from Cormack et al. 2009 is 60.
/// Larger k de-emphasises rank differences at the top.
const RRF_K: f32 = 60.0;

/// Variant B: Hybrid RRF — run dense and sparse independently, fuse ranks.
///
/// score(d) = Σ_i  1 / (RRF_K + rank_i(d))
///
/// Each component contributes based on its rank of document d in its own top-N list.
pub struct HybridRrfSearcher {
    dim: usize,
    /// Dense vectors for cosine similarity.
    dense_vecs: Vec<Vec<f32>>,
    ids: Vec<u64>,
    /// Sparse inverted index.
    sparse: SparseInvertedIndex,
    /// Number of candidates to retrieve per component before fusion.
    candidate_multiplier: usize,
}

impl HybridRrfSearcher {
    /// candidate_multiplier: how many candidates to fetch from each component
    /// before fusing. Set to max(k*4, 20) in practice.
    pub fn new(dim: usize, bm25: Bm25Params, candidate_multiplier: usize) -> Self {
        Self {
            dim,
            dense_vecs: Vec::new(),
            ids: Vec::new(),
            sparse: SparseInvertedIndex::new(bm25),
            candidate_multiplier: candidate_multiplier.max(1),
        }
    }

    fn dense_topn(&self, query: &[f32], n: usize) -> Vec<(u64, f32)> {
        if self.dense_vecs.is_empty() {
            return Vec::new();
        }
        let n = n.min(self.dense_vecs.len());
        let mut scores: Vec<(usize, f32)> = self
            .dense_vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cosine_similarity(query, v)))
            .collect();
        scores.select_nth_unstable_by(n - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
        scores[..n].sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores[..n].iter().map(|(i, s)| (self.ids[*i], *s)).collect()
    }
}

impl HybridSearcher for HybridRrfSearcher {
    fn index(&mut self, id: u64, dense: &[f32], sparse: &SparseVec) -> Result<(), HybridError> {
        if dense.len() != self.dim {
            return Err(HybridError::DimensionMismatch {
                expected: self.dim,
                got: dense.len(),
            });
        }
        self.ids.push(id);
        self.dense_vecs.push(dense.to_vec());
        self.sparse.index(id, dense, sparse)?;
        Ok(())
    }

    fn search(&self, dense_query: &[f32], sparse_query: &SparseVec, k: usize) -> Vec<SearchResult> {
        if self.dense_vecs.is_empty() {
            return Vec::new();
        }
        let n_candidates = (k * self.candidate_multiplier).max(20).min(self.dense_vecs.len());

        let dense_results = self.dense_topn(dense_query, n_candidates);
        let sparse_results = self.sparse.search(dense_query, sparse_query, n_candidates);

        // RRF fusion: accumulate 1/(RRF_K+rank) for each doc from each component.
        let mut rrf_scores: HashMap<u64, (f32, f32, f32)> = HashMap::new(); // (rrf, dense_score, sparse_score)

        for (rank, (id, dscore)) in dense_results.iter().enumerate() {
            let entry = rrf_scores.entry(*id).or_insert((0.0, *dscore, 0.0));
            entry.0 += 1.0 / (RRF_K + rank as f32 + 1.0);
            entry.1 = *dscore;
        }
        for (rank, sr) in sparse_results.iter().enumerate() {
            let entry = rrf_scores.entry(sr.id).or_insert((0.0, 0.0, sr.sparse_score));
            entry.0 += 1.0 / (RRF_K + rank as f32 + 1.0);
            entry.2 = sr.sparse_score;
        }

        let k = k.min(rrf_scores.len());
        if k == 0 {
            return Vec::new();
        }

        let mut scored: Vec<SearchResult> = rrf_scores
            .into_iter()
            .map(|(id, (rrf, ds, ss))| SearchResult::new_fused(id, rrf, ds, ss))
            .collect();

        scored.select_nth_unstable_by(k - 1, |a, b| b.score.partial_cmp(&a.score).unwrap());
        scored[..k].sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        scored[..k].to_vec()
    }

    fn len(&self) -> usize {
        self.dense_vecs.len()
    }
}

/// Variant C: Hybrid Cascade — sparse retrieval narrows candidates, dense reranks.
///
/// Phase 1: sparse BM25 retrieves `sparse_candidates` docs.
/// Phase 2: dense cosine similarity reranks those candidates.
/// Falls back to dense-only when sparse returns < k results.
pub struct HybridCascadeSearcher {
    dim: usize,
    dense_vecs: Vec<Vec<f32>>,
    ids: Vec<u64>,
    sparse: SparseInvertedIndex,
    /// How many candidates to pull from sparse phase.
    sparse_candidates: usize,
}

impl HybridCascadeSearcher {
    pub fn new(dim: usize, bm25: Bm25Params, sparse_candidates: usize) -> Self {
        Self {
            dim,
            dense_vecs: Vec::new(),
            ids: Vec::new(),
            sparse: SparseInvertedIndex::new(bm25),
            sparse_candidates: sparse_candidates.max(1),
        }
    }
}

impl HybridSearcher for HybridCascadeSearcher {
    fn index(&mut self, id: u64, dense: &[f32], sparse: &SparseVec) -> Result<(), HybridError> {
        if dense.len() != self.dim {
            return Err(HybridError::DimensionMismatch {
                expected: self.dim,
                got: dense.len(),
            });
        }
        self.ids.push(id);
        self.dense_vecs.push(dense.to_vec());
        self.sparse.index(id, dense, sparse)?;
        Ok(())
    }

    fn search(&self, dense_query: &[f32], sparse_query: &SparseVec, k: usize) -> Vec<SearchResult> {
        if self.dense_vecs.is_empty() {
            return Vec::new();
        }

        let sparse_hits = self.sparse.search(dense_query, sparse_query, self.sparse_candidates);

        // Build lookup: id → dense vector index.
        let id_to_idx: HashMap<u64, usize> = self.ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

        let candidates: Vec<(u64, usize)> = if sparse_hits.len() >= k {
            sparse_hits.iter().filter_map(|r| id_to_idx.get(&r.id).map(|&i| (r.id, i))).collect()
        } else {
            // Fallback: use all documents for dense reranking.
            self.ids.iter().enumerate().map(|(i, &id)| (id, i)).collect()
        };

        let k = k.min(candidates.len());
        if k == 0 {
            return Vec::new();
        }

        let mut scored: Vec<(u64, f32, f32)> = candidates
            .iter()
            .map(|(id, idx)| {
                let ds = cosine_similarity(dense_query, &self.dense_vecs[*idx]);
                let ss = sparse_hits.iter().find(|r| r.id == *id).map(|r| r.sparse_score).unwrap_or(0.0);
                (*id, ds, ss)
            })
            .collect();

        scored.select_nth_unstable_by(k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
        scored[..k].sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        scored[..k]
            .iter()
            .map(|(id, ds, ss)| SearchResult::new_fused(*id, *ds, *ds, *ss))
            .collect()
    }

    fn len(&self) -> usize {
        self.dense_vecs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sparse(terms: &[(u32, f32)]) -> SparseVec {
        SparseVec::new(terms.to_vec())
    }

    fn make_dense(dim: usize, seed: f32) -> Vec<f32> {
        (0..dim).map(|i| (i as f32 * seed).sin()).collect()
    }

    #[test]
    fn rrf_returns_k_results() {
        let dim = 8;
        let mut idx = HybridRrfSearcher::new(dim, Bm25Params::default(), 4);
        for i in 0u64..20 {
            let dense = make_dense(dim, i as f32 * 0.3 + 0.1);
            let sparse = make_sparse(&[(i as u32 % 5, 1.0), ((i as u32 + 1) % 5, 0.5)]);
            idx.index(i, &dense, &sparse).unwrap();
        }
        let q_dense = make_dense(dim, 1.0);
        let q_sparse = make_sparse(&[(0, 1.0), (1, 1.0)]);
        let results = idx.search(&q_dense, &q_sparse, 5);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn cascade_returns_k_results() {
        let dim = 8;
        let mut idx = HybridCascadeSearcher::new(dim, Bm25Params::default(), 50);
        for i in 0u64..20 {
            let dense = make_dense(dim, i as f32 * 0.3 + 0.1);
            let sparse = make_sparse(&[(i as u32 % 5, 1.0), ((i as u32 + 2) % 5, 0.5)]);
            idx.index(i, &dense, &sparse).unwrap();
        }
        let q_dense = make_dense(dim, 1.0);
        let q_sparse = make_sparse(&[(0, 1.0), (2, 1.0)]);
        let results = idx.search(&q_dense, &q_sparse, 5);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn rrf_finds_pure_dense_nearest_when_sparse_is_empty() {
        let dim = 4;
        let mut idx = HybridRrfSearcher::new(dim, Bm25Params::default(), 4);
        let vecs = vec![
            vec![1.0f32, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        for (i, v) in vecs.iter().enumerate() {
            idx.index(i as u64, v, &SparseVec::new(vec![])).unwrap();
        }
        let q = vec![0.9f32, 0.1, 0.0, 0.0];
        let results = idx.search(&q, &SparseVec::new(vec![]), 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 0, "should find nearest dense neighbour");
    }
}
