pub mod build;
pub mod error;
pub mod graph;
pub mod search;

pub use error::{Result, SymphonyError};
pub use search::{FlatExactIndex, GraphExactIndex, SearchResult, SymphonyIndex};

/// Distance metric.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Metric {
    /// Squared Euclidean distance (monotone for k-NN ranking).
    Euclidean,
    /// Cosine distance: 1 − cosine_similarity.
    Cosine,
}

/// Configuration for a SymphonyQG index.
#[derive(Debug, Clone)]
pub struct Config {
    /// Vector dimensionality. Must be a positive multiple of 8.
    pub dim: usize,
    /// Base neighbor count before BATCH_SIZE padding.
    pub m_base: usize,
    /// Number of candidates evaluated per vertex during construction.
    pub ef_construction: usize,
    pub metric: Metric,
    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dim: 128,
            m_base: 16,
            ef_construction: 200,
            metric: Metric::Euclidean,
            seed: 42,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.dim == 0 || self.dim % 8 != 0 {
            return Err(SymphonyError::InvalidConfig(format!(
                "dim must be > 0 and a multiple of 8; got {}",
                self.dim
            )));
        }
        if self.m_base == 0 {
            return Err(SymphonyError::InvalidConfig("m_base must be > 0".into()));
        }
        Ok(())
    }
}

/// Build all three index variants from the same dataset.
///
/// Shares one construction pass — the graph is cloned for the two
/// graph-based variants so both see identical edge topology.
pub fn build_all(
    vecs: &[Vec<f32>],
    config: &Config,
) -> (FlatExactIndex, GraphExactIndex, SymphonyIndex) {
    let flat = FlatExactIndex::build(vecs, config);
    let graph = build::build(vecs, config);
    let graph_clone = graph.clone();
    let graph_exact = GraphExactIndex::from_graph(graph, config.metric);
    let symphony = SymphonyIndex::from_graph(graph_clone, config.metric);
    (flat, graph_exact, symphony)
}
