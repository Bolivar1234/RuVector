//! # ruvector-maxsim
//!
//! Multi-vector late interaction search using ColBERT-style MaxSim scoring.
//!
//! Each document stores K token embeddings (contextual token vectors from a language model).
//! Queries are also represented as M token embeddings. Relevance is:
//!
//! ```text
//! score(q, d) = Σ_{qi ∈ q}  max_{dj ∈ d}  cosine_sim(qi, dj)
//! ```
//!
//! This "late interaction" formula captures token-level semantic overlap without
//! collapsing the document to a single vector — outperforming single-vector retrieval
//! on BEIR benchmarks and enabling token-level RAG safety reasoning.
//!
//! ## Variants
//! - [`FlatMaxSim`]: brute-force O(N·M·K·D) baseline
//! - [`pruned::PrunedMaxSim`]: centroid-based candidate pruning + MaxSim reranking
//! - [`graph::GraphMaxSim`]: kNN centroid graph + beam search + MaxSim reranking

pub mod graph;
pub mod pruned;

use std::cmp::Ordering;

// ── Core types ────────────────────────────────────────────────────────────────

/// A dense float token embedding.
pub type Token = Vec<f32>;

/// A document represented as multiple token embeddings (e.g., one per contextual token).
#[derive(Clone, Debug)]
pub struct MultiVecDoc {
    pub id: usize,
    pub tokens: Vec<Token>,
}

impl MultiVecDoc {
    pub fn new(id: usize, tokens: Vec<Token>) -> Self {
        Self { id, tokens }
    }
}

/// Search result: document id + MaxSim relevance score.
#[derive(Clone, Debug)]
pub struct MaxSimResult {
    pub doc_id: usize,
    pub score: f32,
}

// ── Trait ─────────────────────────────────────────────────────────────────────

/// Uniform interface for all multi-vector index variants.
pub trait MultiVecIndex {
    fn add(&mut self, doc: MultiVecDoc);
    fn search(&self, query_tokens: &[Token], k: usize) -> Vec<MaxSimResult>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ── Math primitives ───────────────────────────────────────────────────────────

/// Cosine similarity clamped to [-1, 1].
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-9 || nb < 1e-9 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

/// ColBERT MaxSim: Σ_{qi ∈ query} max_{dj ∈ doc} cosine_sim(qi, dj).
pub fn maxsim_score(query_tokens: &[Token], doc_tokens: &[Token]) -> f32 {
    query_tokens
        .iter()
        .map(|qt| {
            doc_tokens
                .iter()
                .map(|dt| cosine_sim(qt, dt))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

/// Mean of a slice of token vectors (the document centroid).
pub fn centroid(tokens: &[Token]) -> Token {
    if tokens.is_empty() {
        return Vec::new();
    }
    let dim = tokens[0].len();
    let n = tokens.len() as f32;
    let mut c = vec![0.0f32; dim];
    for t in tokens {
        for (ci, ti) in c.iter_mut().zip(t.iter()) {
            *ci += ti / n;
        }
    }
    c
}

// ── FlatMaxSim ────────────────────────────────────────────────────────────────

/// Brute-force MaxSim: O(N·M·K·D) per query. Exact ground-truth baseline.
pub struct FlatMaxSim {
    docs: Vec<MultiVecDoc>,
}

impl FlatMaxSim {
    pub fn new() -> Self {
        Self { docs: Vec::new() }
    }
}

impl Default for FlatMaxSim {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiVecIndex for FlatMaxSim {
    fn add(&mut self, doc: MultiVecDoc) {
        self.docs.push(doc);
    }

    fn search(&self, query_tokens: &[Token], k: usize) -> Vec<MaxSimResult> {
        let mut scores: Vec<MaxSimResult> = self
            .docs
            .iter()
            .map(|doc| MaxSimResult {
                doc_id: doc.id,
                score: maxsim_score(query_tokens, &doc.tokens),
            })
            .collect();
        scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        scores.truncate(k);
        scores
    }

    fn len(&self) -> usize {
        self.docs.len()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_tok(v: Vec<f32>) -> Token {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        v.into_iter().map(|x| x / norm).collect()
    }

    #[test]
    fn cosine_sim_identical() {
        let v = unit_tok(vec![1.0, 0.0, 0.0]);
        assert!((cosine_sim(&v, &v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_sim_orthogonal() {
        let a = unit_tok(vec![1.0, 0.0, 0.0]);
        let b = unit_tok(vec![0.0, 1.0, 0.0]);
        assert!(cosine_sim(&a, &b).abs() < 1e-5);
    }

    #[test]
    fn cosine_sim_zero_vector() {
        let a = vec![0.0f32, 0.0, 0.0];
        let b = vec![1.0f32, 0.0, 0.0];
        assert_eq!(cosine_sim(&a, &b), 0.0);
    }

    #[test]
    fn maxsim_self_score_equals_query_len() {
        let tokens: Vec<Token> = vec![unit_tok(vec![1.0, 0.0, 0.0]), unit_tok(vec![0.0, 1.0, 0.0])];
        let score = maxsim_score(&tokens, &tokens);
        // Each query token matches itself perfectly: sum = M = 2.0
        assert!((score - 2.0).abs() < 1e-4, "expected ~2.0, got {}", score);
    }

    #[test]
    fn centroid_of_basis_vectors() {
        let tokens: Vec<Token> = vec![vec![2.0f32, 0.0], vec![0.0f32, 2.0]];
        let c = centroid(&tokens);
        assert!((c[0] - 1.0).abs() < 1e-5);
        assert!((c[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn flat_returns_k_results_sorted() {
        let mut idx = FlatMaxSim::new();
        for i in 0..20u32 {
            let t = vec![unit_tok(vec![i as f32, 0.0, 0.0, 0.0])];
            idx.add(MultiVecDoc::new(i as usize, t));
        }
        let q = vec![unit_tok(vec![1.0f32, 0.0, 0.0, 0.0])];
        let res = idx.search(&q, 5);
        assert_eq!(res.len(), 5);
        for w in res.windows(2) {
            assert!(w[0].score >= w[1].score, "results not sorted");
        }
    }

    #[test]
    fn flat_finds_exact_match_first() {
        let target: Vec<Token> = vec![
            unit_tok(vec![1.0, 0.0, 0.0, 0.0]),
            unit_tok(vec![0.0, 1.0, 0.0, 0.0]),
        ];
        let mut idx = FlatMaxSim::new();
        idx.add(MultiVecDoc::new(0, target.clone()));
        for i in 1usize..=9 {
            let t = vec![unit_tok(vec![i as f32 * 0.1, i as f32 * 0.1, 0.0, 0.0])];
            idx.add(MultiVecDoc::new(i, t));
        }
        let res = idx.search(&target, 1);
        assert_eq!(res[0].doc_id, 0);
    }

    #[test]
    fn flat_handles_k_larger_than_corpus() {
        let mut idx = FlatMaxSim::new();
        for i in 0..3usize {
            idx.add(MultiVecDoc::new(i, vec![vec![i as f32; 4]]));
        }
        let q = vec![vec![1.0f32; 4]];
        let res = idx.search(&q, 100);
        assert_eq!(res.len(), 3);
    }
}
