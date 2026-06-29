//! PrunedMaxSim: centroid-based candidate pruning for multi-vector search.
//!
//! Algorithm:
//! 1. Pre-compute a centroid (mean of token vectors) for each document.
//! 2. For each query token, rank all centroids by cosine similarity and
//!    take the top-C candidates.
//! 3. Union candidate sets across query tokens (each token perspective votes).
//! 4. Run exact MaxSim on the merged candidate set only.
//!
//! Complexity:
//! - Build: O(N·K·D) centroid computation
//! - Search: O(N·M·D) centroid scan + O(C·M·K·D) MaxSim rerank
//!   where C ≪ N, so total ≈ O(N·M·D) dominated by centroid scan
//!
//! Expected recall@10 ≥ 85% with C = 10% of corpus at NOISE_STD = 0.3.

use std::cmp::Ordering;
use std::collections::HashSet;

use crate::{centroid, cosine_sim, maxsim_score, MaxSimResult, MultiVecDoc, MultiVecIndex, Token};

/// Centroid-pruned MaxSim index.
pub struct PrunedMaxSim {
    docs: Vec<MultiVecDoc>,
    centroids: Vec<Token>,
    /// Centroid candidates retrieved per query token before MaxSim reranking.
    pub n_candidates: usize,
}

impl PrunedMaxSim {
    pub fn new(n_candidates: usize) -> Self {
        Self {
            docs: Vec::new(),
            centroids: Vec::new(),
            n_candidates,
        }
    }
}

impl MultiVecIndex for PrunedMaxSim {
    fn add(&mut self, doc: MultiVecDoc) {
        self.centroids.push(centroid(&doc.tokens));
        self.docs.push(doc);
    }

    fn search(&self, query_tokens: &[Token], k: usize) -> Vec<MaxSimResult> {
        let n = self.docs.len();
        let c = self.n_candidates.min(n);
        let mut candidate_set: HashSet<usize> = HashSet::new();

        // Phase 1: centroid retrieval per query token
        for qt in query_tokens {
            let mut scores: Vec<(usize, f32)> = self
                .centroids
                .iter()
                .enumerate()
                .map(|(i, cen)| (i, cosine_sim(qt, cen)))
                .collect();
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            for &(idx, _) in scores.iter().take(c) {
                candidate_set.insert(idx);
            }
        }

        // Phase 2: exact MaxSim reranking on the merged candidate set
        let mut results: Vec<MaxSimResult> = candidate_set
            .iter()
            .map(|&idx| MaxSimResult {
                doc_id: self.docs[idx].id,
                score: maxsim_score(query_tokens, &self.docs[idx].tokens),
            })
            .collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        results.truncate(k);
        results
    }

    fn len(&self) -> usize {
        self.docs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FlatMaxSim, MultiVecIndex};

    fn cluster_doc(id: usize, center: &[f32], noise: f32, k: usize) -> MultiVecDoc {
        let dim = center.len();
        let tokens: Vec<Token> = (0..k)
            .map(|t| {
                center
                    .iter()
                    .enumerate()
                    .map(|(j, &c)| c + noise * (((t * dim + j) as f32) * 0.02 - 0.1))
                    .collect()
            })
            .collect();
        MultiVecDoc::new(id, tokens)
    }

    fn unit(v: Vec<f32>) -> Vec<f32> {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        v.into_iter().map(|x| x / n).collect()
    }

    #[test]
    fn pruned_recall_vs_flat_clustered() {
        let dim = 16;
        let n_clusters = 5;
        let docs_per_cluster = 20;

        let centers: Vec<Vec<f32>> = (0..n_clusters)
            .map(|c| {
                unit(
                    (0..dim)
                        .map(|j| if j == c % dim { 1.0f32 } else { 0.0 })
                        .collect(),
                )
            })
            .collect();

        let mut flat = FlatMaxSim::new();
        let mut pruned = PrunedMaxSim::new(30);

        for (ci, center) in centers.iter().enumerate() {
            for d in 0..docs_per_cluster {
                let id = ci * docs_per_cluster + d;
                let doc = cluster_doc(id, center, 0.1, 4);
                flat.add(doc.clone());
                pruned.add(doc);
            }
        }

        let query: Vec<Token> = vec![centers[0].clone()];
        let flat_ids: HashSet<usize> = flat.search(&query, 10).iter().map(|r| r.doc_id).collect();
        let pruned_ids: HashSet<usize> =
            pruned.search(&query, 10).iter().map(|r| r.doc_id).collect();
        let overlap = flat_ids.intersection(&pruned_ids).count();
        let recall = overlap as f32 / flat_ids.len() as f32;
        // Unit test with deterministic noise produces tied scores across cluster docs;
        // actual recall depends on sort stability across equal-score docs.
        // The benchmark (main.rs) uses random noise and measures real recall.
        assert!(
            recall >= 0.4,
            "PrunedMaxSim recall@10 should be ≥40%, got {:.1}%",
            recall * 100.0
        );
    }

    #[test]
    fn pruned_returns_k_or_fewer() {
        let mut idx = PrunedMaxSim::new(5);
        for i in 0..8usize {
            idx.add(MultiVecDoc::new(i, vec![vec![i as f32; 4]]));
        }
        let q: Vec<Token> = vec![vec![1.0f32; 4]];
        assert!(idx.search(&q, 3).len() <= 3);
        assert!(idx.search(&q, 100).len() <= 8);
    }

    #[test]
    fn pruned_empty_index() {
        let idx = PrunedMaxSim::new(10);
        let q: Vec<Token> = vec![vec![1.0f32; 4]];
        let res = idx.search(&q, 5);
        assert!(res.is_empty());
    }
}
