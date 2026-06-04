use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use crate::types::SparseVec;

/// Synthetic dataset for benchmarking hybrid sparse-dense search.
///
/// Design goals:
/// 1. Dense and sparse features are CORRELATED but IMPERFECT — so each
///    component can retrieve relevant docs but neither achieves 100%.
/// 2. Dense retrieval excels on semantic similarity within a cluster.
/// 3. Sparse retrieval (BM25) excels on lexical overlap.
/// 4. The complement (docs found by dense but not sparse, or vice versa)
///    demonstrates the value of hybrid fusion.
///
/// Ground truth: brute-force dense cosine nearest-neighbour.
/// Cascade and RRF are measured by recall@10 against this ground truth.
pub struct SyntheticDataset {
    pub dense: Vec<Vec<f32>>,
    pub sparse: Vec<SparseVec>,
    pub ids: Vec<u64>,
    pub cluster_ids: Vec<usize>,
    pub dim: usize,
    pub vocab_size: u32,
    pub n_clusters: usize,
}

impl SyntheticDataset {
    /// Standard constructor used by the benchmark.
    ///
    /// Parameters chosen to create a meaningfully hard retrieval task:
    /// - 5K docs in 128D with noise_std=0.8 → ~80% intra-cluster dense similarity
    /// - vocab=2000, terms_per_cluster=40, avg_terms=12, primary_frac=0.65
    ///   → same-cluster BM25 recall of ~55-70% relative to dense NN ground truth
    pub fn generate(
        n_docs: usize,
        dim: usize,
        n_clusters: usize,
        vocab_size: u32,
        avg_terms: usize,
        seed: u64,
    ) -> Self {
        Self::generate_with_params(n_docs, dim, n_clusters, vocab_size, avg_terms, 0.65, 0.65, seed)
    }

    /// Full constructor with explicit difficulty controls.
    ///
    /// `dense_noise`: per-dim std of noise added to cluster centers before
    ///   L2 normalisation.  Larger → more inter-cluster overlap in dense space.
    /// `primary_frac`: fraction of sparse terms drawn from the cluster's
    ///   exclusive vocabulary zone.  Smaller → more cross-cluster contamination.
    pub fn generate_with_params(
        n_docs: usize,
        dim: usize,
        n_clusters: usize,
        vocab_size: u32,
        avg_terms: usize,
        dense_noise: f32,
        primary_frac: f32,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let center_dist = Normal::new(0.0f32, 2.0).unwrap();
        let noise_dist = Normal::new(0.0f32, dense_noise).unwrap();

        let terms_per_cluster = ((vocab_size as usize / n_clusters) as u32).max(5);

        let centers: Vec<Vec<f32>> = (0..n_clusters)
            .map(|_| (0..dim).map(|_| center_dist.sample(&mut rng)).collect())
            .collect();

        let mut dense = Vec::with_capacity(n_docs);
        let mut sparse = Vec::with_capacity(n_docs);
        let mut cluster_ids = Vec::with_capacity(n_docs);

        for _ in 0..n_docs {
            let cluster = rng.gen_range(0..n_clusters);
            cluster_ids.push(cluster);

            // Dense: cluster centre + noise, then L2-normalise.
            let v: Vec<f32> = centers[cluster].iter()
                .map(|&c| c + noise_dist.sample(&mut rng))
                .collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            dense.push(v.into_iter().map(|x| x / norm).collect());

            // Sparse: mixture of primary (cluster) and background terms.
            let n_total = ((avg_terms as f32 * (0.7 + rng.gen::<f32>() * 0.6)) as usize).max(3);
            let n_primary = ((n_total as f32 * primary_frac).round() as usize).max(1);
            let n_bg = n_total.saturating_sub(n_primary);

            let cluster_start = (cluster as u32) * terms_per_cluster;
            let mut terms_map: std::collections::HashMap<u32, f32> = Default::default();

            for _ in 0..n_primary {
                let offset = rng.gen_range(0..terms_per_cluster);
                let tid = (cluster_start + offset) % vocab_size;
                let w: f32 = rng.gen_range(0.8..3.0);
                *terms_map.entry(tid).or_insert(0.0) += w;
            }
            for _ in 0..n_bg {
                let base = (cluster_start + terms_per_cluster) % vocab_size;
                let avail = vocab_size.saturating_sub(terms_per_cluster);
                if avail == 0 { continue; }
                let offset = rng.gen_range(0..avail);
                let tid = (base + offset) % vocab_size;
                let w: f32 = rng.gen_range(0.05..0.5);
                *terms_map.entry(tid).or_insert(0.0) += w;
            }

            sparse.push(SparseVec::new(terms_map.into_iter().collect()));
        }

        let ids: Vec<u64> = (0..n_docs as u64).collect();
        Self { dense, sparse, ids, cluster_ids, dim, vocab_size, n_clusters }
    }

    /// Brute-force dense nearest-neighbour ground truth for queries.
    /// Returns top-k ids sorted by cosine similarity.
    pub fn ground_truth_dense(&self, query_dense: &[f32], k: usize) -> Vec<u64> {
        use crate::types::cosine_similarity;
        let k = k.min(self.dense.len());
        let mut scored: Vec<(usize, f32)> = self.dense.iter()
            .enumerate()
            .map(|(i, v)| (i, cosine_similarity(query_dense, v)))
            .collect();
        scored.select_nth_unstable_by(k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
        scored[..k].sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored[..k].iter().map(|(i, _)| self.ids[*i]).collect()
    }

    /// Generate queries paired with their source cluster IDs.
    /// Returns `(dense_query, sparse_query, source_cluster_id)`.
    pub fn generate_queries_with_clusters(
        &self,
        n_queries: usize,
        dense_noise_std: f32,
        sparse_overlap_frac: f32,
        seed: u64,
    ) -> Vec<(Vec<f32>, SparseVec, usize)> {
        let mut rng = StdRng::seed_from_u64(seed + 1000);
        let noise_dist = Normal::new(0.0f32, dense_noise_std).unwrap();

        (0..n_queries)
            .map(|_| {
                let doc_idx = rng.gen_range(0..self.dense.len());
                let source_cluster = self.cluster_ids[doc_idx];

                let qd: Vec<f32> = self.dense[doc_idx].iter()
                    .map(|&x| x + noise_dist.sample(&mut rng))
                    .collect();
                let norm = qd.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                let qd_norm: Vec<f32> = qd.into_iter().map(|x| x / norm).collect();

                let doc_terms = &self.sparse[doc_idx].terms;
                let n_take = ((doc_terms.len() as f32 * sparse_overlap_frac).ceil() as usize)
                    .max(2).min(doc_terms.len());

                let mut sorted = doc_terms.clone();
                sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                let qs_terms: Vec<(u32, f32)> = sorted[..n_take].iter()
                    .map(|(id, w)| (*id, w * (0.8 + rng.gen::<f32>() * 0.4)))
                    .collect();

                (qd_norm, SparseVec::new(qs_terms), source_cluster)
            })
            .collect()
    }

    /// Convenience wrapper without cluster IDs.
    pub fn generate_queries(
        &self,
        n_queries: usize,
        dense_noise_std: f32,
        sparse_overlap_frac: f32,
        seed: u64,
    ) -> Vec<(Vec<f32>, SparseVec)> {
        self.generate_queries_with_clusters(n_queries, dense_noise_std, sparse_overlap_frac, seed)
            .into_iter().map(|(d, s, _)| (d, s)).collect()
    }
}

/// Recall@k: fraction of ground-truth top-k ids found in the result list.
pub fn recall_at_k(ground_truth: &[u64], results: &[u64], k: usize) -> f32 {
    let k = k.min(ground_truth.len()).min(results.len());
    if k == 0 { return 0.0; }
    let gt_set: std::collections::HashSet<u64> = ground_truth[..k].iter().copied().collect();
    let found = results[..k].iter().filter(|id| gt_set.contains(id)).count();
    found as f32 / k as f32
}

/// Precision@k for set-based relevance (unordered ground-truth set).
pub fn precision_at_k_set(
    relevant: &std::collections::HashSet<u64>,
    results: &[u64],
    k: usize,
) -> f32 {
    let k = k.min(results.len());
    if k == 0 { return 0.0; }
    let found = results[..k].iter().filter(|id| relevant.contains(id)).count();
    found as f32 / k as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_correct_sizes() {
        let ds = SyntheticDataset::generate(100, 32, 5, 1000, 20, 42);
        assert_eq!(ds.dense.len(), 100);
        assert_eq!(ds.sparse.len(), 100);
        assert_eq!(ds.cluster_ids.len(), 100);
    }

    #[test]
    fn dense_vectors_normalized() {
        let ds = SyntheticDataset::generate(50, 16, 3, 300, 10, 7);
        for (i, v) in ds.dense.iter().enumerate() {
            let n: f32 = v.iter().map(|x| x*x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-4, "doc {i} norm={n}");
        }
    }

    #[test]
    fn sparse_terms_non_empty() {
        let ds = SyntheticDataset::generate(50, 8, 3, 150, 8, 99);
        for (i, sv) in ds.sparse.iter().enumerate() {
            assert!(!sv.is_empty(), "doc {i} has empty sparse vector");
        }
    }

    #[test]
    fn ground_truth_returns_k() {
        let ds = SyntheticDataset::generate(100, 8, 3, 100, 5, 11);
        let gt = ds.ground_truth_dense(&ds.dense[0], 5);
        assert_eq!(gt.len(), 5);
        assert_eq!(gt[0], 0, "identical vector should be nearest to itself");
    }

    #[test]
    fn recall_at_k_correct() {
        let gt = vec![1u64, 2, 3, 4];
        let res = vec![1u64, 5, 3, 6];
        let r = recall_at_k(&gt, &res, 4);
        assert!((r - 0.5).abs() < 1e-5, "expected 0.5 got {r}");
    }

    #[test]
    fn precision_at_k_set_correct() {
        use std::collections::HashSet;
        let rel: HashSet<u64> = [1, 3, 5].iter().copied().collect();
        let res = vec![1u64, 2, 3, 4];
        let p = precision_at_k_set(&rel, &res, 4);
        assert!((p - 0.5).abs() < 1e-5, "expected 0.5 got {p}");
    }
}
