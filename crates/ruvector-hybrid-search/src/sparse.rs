use std::collections::HashMap;
use crate::types::{HybridError, HybridSearcher, SearchResult, SparseVec};

/// Posting list entry: (doc_id, term_weight).
#[derive(Debug, Clone)]
struct Posting {
    doc_id: u64,
    weight: f32,
}

/// BM25 parameters.
#[derive(Debug, Clone, Copy)]
pub struct Bm25Params {
    /// Term-frequency saturation (default 1.2; Robertson et al.).
    pub k1: f32,
    /// Length normalization (default 0.75).
    pub b: f32,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

/// Inverted index for sparse BM25 search.
///
/// BM25 score: Σ_t IDF(t) * tf_norm(t,d)
/// where IDF(t) = ln((N - df(t) + 0.5) / (df(t) + 0.5) + 1) (Robertson IDF)
/// and tf_norm = weight*(k1+1) / (weight + k1*(1-b+b*doclen/avgdl))
///
/// All IDF values are computed from maintained counts on each search call.
/// No unsafe interior mutability; df is derived directly from posting list lengths.
pub struct SparseInvertedIndex {
    params: Bm25Params,
    /// term_id → posting list (sorted by doc_id for potential skip-list extension)
    index: HashMap<u32, Vec<Posting>>,
    /// doc_id → doc length (sum of absolute term weights)
    doc_lengths: HashMap<u64, f32>,
    /// Total number of indexed documents
    n_docs: usize,
    /// Running sum of all doc lengths (for avgdl computation)
    total_length: f64,
}

impl SparseInvertedIndex {
    pub fn new(params: Bm25Params) -> Self {
        Self {
            params,
            index: HashMap::new(),
            doc_lengths: HashMap::new(),
            n_docs: 0,
            total_length: 0.0,
        }
    }

    fn avgdl(&self) -> f32 {
        if self.n_docs == 0 { 1.0 } else { (self.total_length / self.n_docs as f64) as f32 }
    }

    /// Robertson IDF: ln((N - df + 0.5) / (df + 0.5) + 1).  Always ≥ 0.
    fn idf(&self, df: usize) -> f32 {
        let n = self.n_docs as f32;
        let df = df as f32;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0)
    }

    /// BM25 TF normalization applied to a stored term weight.
    fn bm25_tf(&self, raw_weight: f32, doc_len: f32, avgdl: f32) -> f32 {
        let k1 = self.params.k1;
        let b = self.params.b;
        raw_weight * (k1 + 1.0) / (raw_weight + k1 * (1.0 - b + b * doc_len / avgdl))
    }
}

impl HybridSearcher for SparseInvertedIndex {
    fn index(&mut self, id: u64, _dense: &[f32], sparse: &SparseVec) -> Result<(), HybridError> {
        if sparse.is_empty() {
            // Still count the document so avgdl and n_docs stay consistent.
            self.doc_lengths.insert(id, 0.0);
            self.n_docs += 1;
            return Ok(());
        }
        let doc_len: f32 = sparse.terms.iter().map(|(_, w)| w.abs()).sum();
        self.doc_lengths.insert(id, doc_len);
        self.total_length += doc_len as f64;
        self.n_docs += 1;
        for (term_id, weight) in &sparse.terms {
            self.index
                .entry(*term_id)
                .or_default()
                .push(Posting { doc_id: id, weight: *weight });
        }
        Ok(())
    }

    fn search(&self, _dense_query: &[f32], sparse_query: &SparseVec, k: usize) -> Vec<SearchResult> {
        if self.n_docs == 0 || sparse_query.is_empty() {
            return Vec::new();
        }
        let avgdl = self.avgdl();
        let mut scores: HashMap<u64, f32> = HashMap::new();

        for (term_id, query_weight) in &sparse_query.terms {
            let postings = match self.index.get(term_id) {
                Some(p) if !p.is_empty() => p,
                _ => continue,
            };
            let idf = self.idf(postings.len());
            if idf <= 0.0 {
                continue;
            }
            for posting in postings {
                let doc_len = *self.doc_lengths.get(&posting.doc_id).unwrap_or(&1.0);
                let tf = self.bm25_tf(posting.weight, doc_len, avgdl);
                let contribution = idf * tf * query_weight;
                *scores.entry(posting.doc_id).or_insert(0.0) += contribution;
            }
        }

        let k = k.min(scores.len());
        if k == 0 {
            return Vec::new();
        }
        let mut scored: Vec<(u64, f32)> = scores.into_iter().collect();
        scored.select_nth_unstable_by(k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
        scored[..k].sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored[..k].iter().map(|(id, s)| SearchResult::new_sparse(*id, *s)).collect()
    }

    fn len(&self) -> usize {
        self.n_docs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sv(terms: &[(u32, f32)]) -> SparseVec {
        SparseVec::new(terms.to_vec())
    }

    #[test]
    fn sparse_index_basic_retrieval() {
        let mut idx = SparseInvertedIndex::new(Bm25Params::default());
        idx.index(0, &[], &sv(&[(1, 1.0), (2, 1.0), (3, 1.0)])).unwrap();
        idx.index(1, &[], &sv(&[(2, 1.0), (3, 1.0), (4, 1.0)])).unwrap();
        idx.index(2, &[], &sv(&[(4, 0.1), (5, 0.1), (6, 0.1)])).unwrap();

        let q = sv(&[(1, 1.0), (2, 1.0)]);
        let results = idx.search(&[], &q, 3);
        assert!(!results.is_empty());
        let top_ids: Vec<u64> = results.iter().map(|r| r.id).collect();
        assert!(top_ids.contains(&0), "doc 0 must appear (has both query terms)");
    }

    #[test]
    fn sparse_no_overlap_returns_empty() {
        let mut idx = SparseInvertedIndex::new(Bm25Params::default());
        idx.index(0, &[], &sv(&[(1, 1.0)])).unwrap();
        let results = idx.search(&[], &sv(&[(99, 1.0)]), 5);
        assert!(results.is_empty());
    }

    #[test]
    fn sparse_returns_at_most_k() {
        let mut idx = SparseInvertedIndex::new(Bm25Params::default());
        for i in 0u64..20 {
            idx.index(i, &[], &sv(&[(0, i as f32 + 0.1)])).unwrap();
        }
        let results = idx.search(&[], &sv(&[(0, 1.0)]), 5);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn sparse_idf_reduces_common_term_weight() {
        // A term appearing in many docs should have lower IDF than a rare term.
        let mut idx = SparseInvertedIndex::new(Bm25Params::default());
        // Term 0 appears in all 10 docs (common).
        // Term 1 appears only in doc 0 (rare).
        for i in 0u64..10 {
            let mut terms = vec![(0u32, 1.0f32)];
            if i == 0 { terms.push((1, 1.0)); }
            idx.index(i, &[], &sv(&terms)).unwrap();
        }
        let common_idf = idx.idf(10);
        let rare_idf = idx.idf(1);
        assert!(rare_idf > common_idf, "rare term IDF {rare_idf:.4} should be > common IDF {common_idf:.4}");
    }
}
