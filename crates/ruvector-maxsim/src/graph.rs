//! GraphMaxSim: greedy kNN graph on centroids + beam search + MaxSim reranking.
//!
//! Inspired by HNSW's layer-0 navigable graph, but simpler: a flat greedy kNN
//! graph on document centroid vectors. Beam search finds the approximate centroid
//! neighbourhood, then exact MaxSim reranks those candidates.
//!
//! Complexity:
//! - Build: O(N²·D) greedy kNN construction (offline; suitable for N ≤ 100K)
//! - Search: O(ef·M_graph·D) beam search + O(C·M_q·K_d·D) MaxSim rerank
//!
//! Typical parameters:
//! - m = 12  (neighbors per node)
//! - ef = 64 (beam width)
//! - n_candidates = 100 (passed to MaxSim reranker)

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::{centroid, cosine_sim, maxsim_score, MaxSimResult, MultiVecDoc, MultiVecIndex, Token};

// ── Ordered float wrapper for BinaryHeap ──────────────────────────────────────

#[derive(Clone, PartialEq)]
struct OrdF32(f32, usize);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(Ordering::Equal)
            .then(self.1.cmp(&other.1))
    }
}

// ── GraphMaxSim ───────────────────────────────────────────────────────────────

/// kNN centroid graph + beam search + MaxSim reranking.
pub struct GraphMaxSim {
    docs: Vec<MultiVecDoc>,
    centroids: Vec<Token>,
    /// graph[i] = neighbor indices sorted by descending centroid similarity
    graph: Vec<Vec<usize>>,
    /// Neighbors per node
    pub m: usize,
    /// Beam width during search (nodes explored per start point)
    pub ef: usize,
    /// Max candidates forwarded to MaxSim reranking
    pub n_candidates: usize,
    built: bool,
}

impl GraphMaxSim {
    pub fn new(m: usize, ef: usize, n_candidates: usize) -> Self {
        Self {
            docs: Vec::new(),
            centroids: Vec::new(),
            graph: Vec::new(),
            m,
            ef,
            n_candidates,
            built: false,
        }
    }

    /// Build the kNN graph on all added centroids. Call once after all add() calls.
    /// O(N²·D) — suitable for N ≤ 50K offline.
    pub fn build(&mut self) {
        let n = self.centroids.len();
        let m = self.m;
        self.graph = vec![Vec::new(); n];

        for i in 0..n {
            let mut nbrs: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, cosine_sim(&self.centroids[i], &self.centroids[j])))
                .collect();
            nbrs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            self.graph[i] = nbrs.iter().take(m).map(|(j, _)| *j).collect();
        }
        self.built = true;
    }

    /// Greedy beam search from multiple entry points. Returns up to `n_candidates`
    /// document indices ordered by centroid cosine similarity to the query centroid.
    fn beam_search(&self, query_centroid: &[f32]) -> Vec<usize> {
        let n = self.centroids.len();
        if n == 0 {
            return Vec::new();
        }

        let mut visited = vec![false; n];
        // Max-heap: explore highest-scoring frontier first
        let mut frontier: BinaryHeap<OrdF32> = BinaryHeap::new();
        let mut found: Vec<(f32, usize)> = Vec::new();

        // Seed from the first n_seeds consecutive docs.
        // Consecutive seeding ensures cluster coverage when docs are interleaved
        // by cluster label (e.g., doc i → cluster i % N_CLUSTERS).
        // Step-based seeding can miss clusters when step is a multiple of N_CLUSTERS.
        let n_seeds = 40usize.min(n);
        for e in 0..n_seeds {
            if !visited[e] {
                visited[e] = true;
                let s = cosine_sim(query_centroid, &self.centroids[e]);
                frontier.push(OrdF32(s, e));
            }
        }

        let budget = self.ef * 4;
        while let Some(OrdF32(score, curr)) = frontier.pop() {
            found.push((score, curr));
            if found.len() >= budget {
                break;
            }
            for &nbr in &self.graph[curr] {
                if !visited[nbr] {
                    visited[nbr] = true;
                    let ns = cosine_sim(query_centroid, &self.centroids[nbr]);
                    frontier.push(OrdF32(ns, nbr));
                }
            }
        }

        found.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        found
            .iter()
            .take(self.n_candidates)
            .map(|(_, idx)| *idx)
            .collect()
    }
}

impl MultiVecIndex for GraphMaxSim {
    fn add(&mut self, doc: MultiVecDoc) {
        self.centroids.push(centroid(&doc.tokens));
        self.docs.push(doc);
        self.built = false;
    }

    fn search(&self, query_tokens: &[Token], k: usize) -> Vec<MaxSimResult> {
        if !self.built || self.docs.is_empty() {
            return Vec::new();
        }
        let qc = centroid(query_tokens);
        let candidates = self.beam_search(&qc);

        let mut results: Vec<MaxSimResult> = candidates
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
    use std::collections::HashSet;

    fn unit(v: Vec<f32>) -> Vec<f32> {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        v.into_iter().map(|x| x / n).collect()
    }

    #[test]
    fn graph_builds_without_panic() {
        let mut idx = GraphMaxSim::new(4, 16, 20);
        for i in 0..30usize {
            let tokens: Vec<Token> = (0..3)
                .map(|k| vec![i as f32 * 0.1 + k as f32 * 0.01; 8])
                .collect();
            idx.add(MultiVecDoc::new(i, tokens));
        }
        idx.build();
        assert!(idx.built);
    }

    #[test]
    fn graph_search_returns_empty_before_build() {
        let mut idx = GraphMaxSim::new(4, 16, 20);
        for i in 0..10usize {
            idx.add(MultiVecDoc::new(i, vec![vec![i as f32; 4]]));
        }
        let q: Vec<Token> = vec![vec![1.0f32; 4]];
        // build() not called
        assert!(idx.search(&q, 5).is_empty());
    }

    #[test]
    fn graph_recall_acceptable() {
        let dim = 16;
        let n = 100usize;
        let k_tokens = 4;
        let n_clusters = 5;

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
        let mut graph = GraphMaxSim::new(8, 32, 30);

        for i in 0..n {
            let cluster = i % n_clusters;
            let tokens: Vec<Token> = (0..k_tokens)
                .map(|t| {
                    (0..dim)
                        .map(|j| centers[cluster][j] + (i * dim + t * dim + j) as f32 * 0.001)
                        .collect()
                })
                .collect();
            let doc = MultiVecDoc::new(i, tokens);
            flat.add(doc.clone());
            graph.add(doc);
        }
        graph.build();

        let query: Vec<Token> = vec![centers[0].clone()];
        let flat_ids: HashSet<usize> = flat.search(&query, 10).iter().map(|r| r.doc_id).collect();
        let graph_ids: HashSet<usize> = graph.search(&query, 10).iter().map(|r| r.doc_id).collect();

        let overlap = flat_ids.intersection(&graph_ids).count();
        let recall = overlap as f32 / flat_ids.len() as f32;
        assert!(
            recall >= 0.5,
            "GraphMaxSim recall@10 ≥50% required, got {:.1}%",
            recall * 100.0
        );
    }

    #[test]
    fn graph_empty_index() {
        let idx = GraphMaxSim::new(4, 16, 10);
        let q: Vec<Token> = vec![vec![1.0f32; 4]];
        assert!(idx.search(&q, 5).is_empty());
    }
}
