/// # ruvector-hybrid-search
///
/// Hybrid sparse-dense vector search combining BM25 inverted-index retrieval
/// with dense cosine-similarity ANN, fused via Reciprocal Rank Fusion (RRF)
/// and cascade reranking.
///
/// ## Variants
///
/// | Name | Strategy | Recall | Latency |
/// |------|----------|--------|---------|
/// | `DenseLinearSearcher` | Exact brute-force cosine | 100% (ground truth) | O(N·D) |
/// | `SparseInvertedIndex` | BM25 inverted index | Term-overlap dependent | O(vocab_hit) |
/// | `HybridRrfSearcher` | Dense + Sparse → RRF fusion | >dense or sparse alone | ~2x sparse |
/// | `HybridCascadeSearcher` | Sparse candidates → Dense rerank | High at large candidate set | O(C·D), C≪N |
///
/// ## Quick start
///
/// ```rust
/// use ruvector_hybrid_search::{HybridRrfSearcher, HybridSearcher, SparseVec, Bm25Params};
///
/// let mut idx = HybridRrfSearcher::new(128, Bm25Params::default(), 4);
/// let dense = vec![0.1f32; 128];
/// let sparse = SparseVec::new(vec![(0, 1.0), (5, 0.8)]);
/// idx.index(1, &dense, &sparse).unwrap();
///
/// let results = idx.search(&dense, &sparse, 1);
/// assert!(!results.is_empty());
/// ```

pub mod types;
pub mod dense;
pub mod sparse;
pub mod fusion;
pub mod dataset;

pub use types::{HybridError, HybridSearcher, SearchResult, SparseVec};
pub use dense::DenseLinearSearcher;
pub use sparse::{Bm25Params, SparseInvertedIndex};
pub use fusion::{HybridCascadeSearcher, HybridRrfSearcher};
pub use dataset::{precision_at_k_set, recall_at_k, SyntheticDataset};
// Re-export to allow benchmarks to call generate_queries_with_clusters.
