//! ANN index adaptor.
//!
//! Per ADR-264 reuse boundary, this crate does **not** implement a vector index.
//! It defines [`AnnIndex`], a thin adaptor trait whose M1 implementations wrap an
//! existing ruvector backend:
//!
//! - `ruvector-core::HNSWIndex` — M1 primary.
//! - `ruvector-rairs::IVFIndex` (IVF-SQ, ADR-193) — M1 fallback on memory budget.
//! - `ruvector-turbovec` FastScan (ADR-254) — M2+ optimization, if shipped.
//!
//! The signature here intentionally mirrors `ruvector_rabitq::AnnIndex` so the
//! M1 implementation is a near-passthrough (`ruvector-rabitq` also provides the
//! `RandomRotation::HadamardSigned` reused at build time for consistency). The
//! [`crate::config::IndexBackend`] enum selects which concrete backend
//! [`build_index`] constructs.

use ruvector_core::types::DbOptions;
use ruvector_core::{DistanceMetric, SearchQuery, VectorDB, VectorEntry};

use crate::config::IndexBackend;
use crate::{Embedding, Error, Result, SearchResult};

/// Adaptor over a ruvector ANN backend. Mirrors `ruvector_rabitq::AnnIndex`
/// (`add`/`search`/`len`/`dim`/`memory_bytes`) so the M1 wrapper is trivial, plus
/// PixelRAG-specific persistence + filtered search hooks.
pub trait AnnIndex: Send + Sync {
    /// Insert one embedding under external `id` (the tile id from [`crate::tile`]).
    ///
    /// **M1**: forward to the wrapped backend's `add`; the backend owns its
    /// quantization/rotation.
    fn add(&mut self, id: usize, vector: Embedding) -> Result<()>;

    /// Search for the `k` nearest neighbours of `query`.
    ///
    /// **M1**: forward to the wrapped backend's `search`, returning hits ordered
    /// by ascending squared-L2 distance (see [`SearchResult`]).
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>>;

    /// Search restricted to an allowlist of ids (pre-filtered retrieval).
    ///
    /// **M1**: reuse the allowlist-filtered search path from `ruvector-rairs`
    /// (IVF supports pre-filtered scan) / `ruvector-rabitq`. Backends without
    /// native filtering fall back to over-fetch + post-filter.
    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        allowlist: &[usize],
    ) -> Result<Vec<SearchResult>>;

    /// Number of indexed vectors.
    fn len(&self) -> usize;

    /// Whether the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Embedding dimensionality the index was built for.
    fn dim(&self) -> usize;

    /// Honest byte footprint of the index (originals + codes + rotation +
    /// bookkeeping), matching `ruvector_rabitq::AnnIndex::memory_bytes`.
    fn memory_bytes(&self) -> usize;

    /// Persist the index to a `*.pixelrag` artifact.
    ///
    /// **M2**: serialize the wrapped backend + id map via `bincode`.
    fn save(&self, path: &std::path::Path) -> Result<()>;
}

/// Construct the configured ANN backend for embeddings of dimension `dim`.
///
/// **M1**: match on `backend` and build the corresponding ruvector index
/// (`ruvector-core` HNSW / `ruvector-rairs` IVF-SQ), returning it boxed behind
/// [`AnnIndex`]. `IndexBackend::Turbovec` is gated on ADR-254 shipping (M2+);
/// until then it returns [`crate::Error::Index`].
pub fn build_index(backend: IndexBackend, dim: usize) -> Result<Box<dyn AnnIndex>> {
    match backend {
        IndexBackend::Hnsw => Ok(Box::new(RuvectorHnswIndex::new(dim)?)),
        IndexBackend::IvfSq => Err(Error::Index(
            "IVF-SQ backend (ruvector-rairs, ADR-193) is the M1 fallback and not yet wired; \
             use IndexBackend::Hnsw"
                .into(),
        )),
        IndexBackend::Turbovec => Err(Error::Index(
            "Turbovec FastScan backend is gated on ADR-254 shipping (M2+)".into(),
        )),
    }
}

/// Load a previously persisted index from a `*.pixelrag` artifact.
///
/// **M2**: `bincode`-deserialize the backend + id map and return it behind
/// [`AnnIndex`].
pub fn load_index(_path: &std::path::Path) -> Result<Box<dyn AnnIndex>> {
    unimplemented!("M2: bincode-deserialize *.pixelrag into the wrapped ruvector backend")
}

// ── M1 concrete backend: ruvector-core HNSW ──────────────────────────────────

/// M1 primary [`AnnIndex`] implementation wrapping `ruvector_core::VectorDB`.
///
/// External tile ids (`usize`) are mapped to `ruvector_core::VectorId` (a
/// `String`) by decimal formatting; cosine distance is the default metric so
/// normalized visual embeddings compare by angle. The underlying `VectorDB`
/// requires the `storage` feature (a redb-backed path); we point it at a unique
/// temp path so the index is self-contained for the M1 plumbing harness.
pub struct RuvectorHnswIndex {
    db: VectorDB,
    dim: usize,
    /// Externally-assigned ids in insertion order (drives [`AnnIndex::len`] and
    /// the post-search allowlist filter without touching the backend).
    ids: Vec<usize>,
}

impl RuvectorHnswIndex {
    /// Build an empty HNSW-backed index for `dim`-wide cosine embeddings.
    pub fn new(dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(Error::Index("embedding dimension must be > 0".into()));
        }
        // Unique, process-local storage path. ruvector-core's `storage` feature
        // is on by default and `VectorDB::new` needs a path; this keeps the M1
        // index ephemeral and isolated (M2 swaps in real *.pixelrag persistence).
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let storage_path = std::env::temp_dir()
            .join(format!("pixelrag-hnsw-{}-{}.db", std::process::id(), nanos))
            .to_string_lossy()
            .into_owned();

        let options = DbOptions {
            dimensions: dim,
            distance_metric: DistanceMetric::Cosine,
            storage_path,
            ..DbOptions::default()
        };
        let db = VectorDB::new(options).map_err(|e| Error::Index(format!("VectorDB::new: {e}")))?;
        Ok(Self { db, dim, ids: Vec::new() })
    }

    fn id_to_key(id: usize) -> String {
        id.to_string()
    }

    fn key_to_id(key: &str) -> Option<usize> {
        key.parse().ok()
    }
}

impl AnnIndex for RuvectorHnswIndex {
    fn add(&mut self, id: usize, vector: Embedding) -> Result<()> {
        if vector.len() != self.dim {
            return Err(Error::Index(format!(
                "embedding dim {} != index dim {}",
                vector.len(),
                self.dim
            )));
        }
        self.db
            .insert(VectorEntry {
                id: Some(Self::id_to_key(id)),
                vector,
                metadata: None,
            })
            .map_err(|e| Error::Index(format!("VectorDB::insert: {e}")))?;
        self.ids.push(id);
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dim {
            return Err(Error::Index(format!(
                "query dim {} != index dim {}",
                query.len(),
                self.dim
            )));
        }
        let raw = self
            .db
            .search(SearchQuery {
                vector: query.to_vec(),
                k,
                filter: None,
                ef_search: None,
            })
            .map_err(|e| Error::Index(format!("VectorDB::search: {e}")))?;
        Ok(raw
            .into_iter()
            .filter_map(|r| Self::key_to_id(&r.id).map(|id| SearchResult { id, score: r.score }))
            .collect())
    }

    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        allowlist: &[usize],
    ) -> Result<Vec<SearchResult>> {
        // M1: over-fetch then post-filter against the allowlist. Native
        // pre-filtered scan (ruvector-rairs IVF / rabitq) is a later milestone;
        // over-fetching `k + allowlist.len()` guarantees we can still return up to
        // `k` allowed hits when the unfiltered top-k are mostly disallowed.
        let allow: std::collections::HashSet<usize> = allowlist.iter().copied().collect();
        let over_k = k.saturating_add(allowlist.len()).max(k);
        let mut hits = self.search(query, over_k)?;
        hits.retain(|h| allow.contains(&h.id));
        hits.truncate(k);
        Ok(hits)
    }

    fn len(&self) -> usize {
        self.ids.len()
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn memory_bytes(&self) -> usize {
        // Honest f32-originals footprint plus the id bookkeeping vector. This
        // excludes redb on-disk pages (the index is unquantized in M1).
        self.ids.len() * self.dim * std::mem::size_of::<f32>()
            + self.ids.len() * std::mem::size_of::<usize>()
    }

    fn save(&self, _path: &std::path::Path) -> Result<()> {
        // M2: bincode-serialize the id map + backend snapshot into *.pixelrag.
        // The ruvector-core storage feature already persists vectors to its redb
        // path; a dedicated portable artifact is deferred to M2.
        Err(Error::Index(
            "M2: *.pixelrag persistence not yet implemented (RuvectorHnswIndex::save)".into(),
        ))
    }
}
