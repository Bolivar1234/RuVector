//! Unified coherence engine.
//!
//! The `CoherenceEngine` ties together:
//! - Graph state (from [`graph`])
//! - MinCut computation (from [`mincut`] or [`bridge`])
//! - Coherence scoring (from [`scoring`] or [`bridge`])
//! - Cut pressure (from [`pressure`])
//! - Adaptive recomputation frequency (from [`adaptive`])
//!
//! This is the single entry point that the kernel calls on each epoch.
//!
//! ## Lifecycle
//!
//! ```text
//! engine.add_partition(id)          -- register a new partition
//! engine.record_communication(a, b) -- record inter-partition traffic
//! engine.tick(cpu_load)             -- advance epoch, recompute if adaptive says so
//! engine.score(id)                  -- read the latest coherence score
//! engine.pressure(id)               -- read the latest cut pressure
//! engine.recommend()                -- get split/merge recommendation
//! ```

use rvm_types::{CoherenceScore, CutPressure, PartitionId, RvmError};

use crate::adaptive::AdaptiveCoherenceEngine;
use crate::bridge::{BuiltinCoherence, BuiltinMinCut, CoherenceBackend, MinCutBackend};
use crate::graph::{CoherenceGraph, GraphError};
use crate::pressure::{self, MergeSignal, SPLIT_THRESHOLD_BP};

/// Maximum number of partitions tracked by the coherence engine.
const ENGINE_MAX_NODES: usize = 32;

/// Maximum number of directed edges tracked by the coherence engine.
const ENGINE_MAX_EDGES: usize = 128;

/// A recommendation produced by the coherence engine after an epoch tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoherenceDecision {
    /// No split or merge action is warranted.
    NoAction,
    /// A partition should be split due to high cut pressure.
    SplitRecommended {
        /// The partition that should be split.
        partition: PartitionId,
        /// The cut pressure that triggered the recommendation.
        pressure: CutPressure,
    },
    /// Two partitions should be merged due to high mutual coherence.
    MergeRecommended {
        /// First partition to merge.
        a: PartitionId,
        /// Second partition to merge.
        b: PartitionId,
        /// Mutual coherence score.
        mutual_coherence: CoherenceScore,
    },
}

/// Per-partition cached scoring data.
#[derive(Debug, Clone, Copy)]
struct PartitionEntry {
    /// Partition ID.
    id: PartitionId,
    /// Most recently computed coherence score.
    score: CoherenceScore,
    /// Most recently computed cut pressure.
    pressure: CutPressure,
    /// Whether this slot is active.
    active: bool,
}

impl PartitionEntry {
    const EMPTY: Self = Self {
        id: PartitionId::HYPERVISOR, // sentinel; never matched when !active
        score: CoherenceScore::MAX,
        pressure: CutPressure::ZERO,
        active: false,
    };
}

/// The unified coherence engine.
///
/// Generics `MCB` and `CB` allow injecting custom mincut and coherence
/// scoring backends for testing or for the ruvector bridge.
pub struct CoherenceEngine<MCB: MinCutBackend, CB: CoherenceBackend> {
    /// The communication topology graph.
    graph: CoherenceGraph<ENGINE_MAX_NODES, ENGINE_MAX_EDGES>,
    /// Adaptive recomputation controller.
    adaptive: AdaptiveCoherenceEngine,
    /// MinCut backend.
    mincut_backend: MCB,
    /// Coherence scoring backend.
    coherence_backend: CB,
    /// Per-partition cached scores and pressures.
    entries: [PartitionEntry; ENGINE_MAX_NODES],
    /// Epoch counter (incremented on each `tick`).
    epoch: u64,
    /// Decision computed on the last recompute (or after the last graph
    /// mutation). Returned on skip ticks instead of re-running the O(n^2)
    /// recommendation pass over stale data.
    cached_decision: CoherenceDecision,
    /// Whether `cached_decision` is still valid. Invalidated by graph
    /// mutations (`record_communication`, `add_partition`,
    /// `remove_partition`).
    cache_valid: bool,
}

// -----------------------------------------------------------------------
// Type alias for the default engine (built-in backends)
// -----------------------------------------------------------------------

/// Default coherence engine using built-in Stoer-Wagner and ratio scoring.
pub type DefaultCoherenceEngine =
    CoherenceEngine<BuiltinMinCut<ENGINE_MAX_NODES>, BuiltinCoherence>;

/// RuVector-backed coherence engine (available with `ruvector` feature).
#[cfg(feature = "ruvector")]
pub type RuVectorCoherenceEngine = CoherenceEngine<
    crate::bridge::RuVectorMinCut<ENGINE_MAX_NODES>,
    crate::bridge::SpectralCoherence,
>;

// -----------------------------------------------------------------------
// Implementation
// -----------------------------------------------------------------------

impl DefaultCoherenceEngine {
    /// Create a new default engine with built-in backends.
    ///
    /// `max_iterations` controls the Stoer-Wagner budget per mincut
    /// computation.
    #[must_use]
    pub fn with_defaults(max_iterations: u32) -> Self {
        Self::new(
            BuiltinMinCut::new(max_iterations),
            BuiltinCoherence,
        )
    }
}

#[cfg(feature = "ruvector")]
impl RuVectorCoherenceEngine {
    /// Create a new engine with RuVector backends.
    ///
    /// `max_iterations` is passed to the fallback Stoer-Wagner until the
    /// ruvector crates gain `no_std` support.
    #[must_use]
    pub fn with_ruvector(max_iterations: u32) -> Self {
        Self::new(
            crate::bridge::RuVectorMinCut::new(max_iterations),
            crate::bridge::SpectralCoherence,
        )
    }
}

impl<MCB: MinCutBackend, CB: CoherenceBackend> CoherenceEngine<MCB, CB> {
    /// Create a new engine with the given backends.
    #[must_use]
    pub fn new(mincut_backend: MCB, coherence_backend: CB) -> Self {
        Self {
            graph: CoherenceGraph::new(),
            adaptive: AdaptiveCoherenceEngine::new(),
            mincut_backend,
            coherence_backend,
            entries: [PartitionEntry::EMPTY; ENGINE_MAX_NODES],
            epoch: 0,
            cached_decision: CoherenceDecision::NoAction,
            cache_valid: false,
        }
    }

    /// Current epoch counter.
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Number of active partitions tracked by the engine.
    #[must_use]
    pub fn partition_count(&self) -> usize {
        self.graph.node_count() as usize
    }

    /// Register a new partition in the coherence graph.
    pub fn add_partition(&mut self, id: PartitionId) -> Result<(), RvmError> {
        self.cache_valid = false;
        self.graph
            .add_node(id)
            .map_err(|e| match e {
                GraphError::DuplicateNode => RvmError::InvalidPartitionState,
                GraphError::NodeCapacityExhausted => RvmError::ResourceLimitExceeded,
                _ => RvmError::InternalError,
            })?;

        // Find a free entry slot
        for entry in self.entries.iter_mut() {
            if !entry.active {
                entry.id = id;
                entry.score = CoherenceScore::MAX;
                entry.pressure = CutPressure::ZERO;
                entry.active = true;
                return Ok(());
            }
        }
        // Shouldn't happen because the graph already accepted the node,
        // but guard against it.
        Err(RvmError::ResourceLimitExceeded)
    }

    /// Remove a partition from the coherence graph.
    pub fn remove_partition(&mut self, id: PartitionId) -> Result<(), RvmError> {
        self.cache_valid = false;
        self.graph
            .remove_node(id)
            .map_err(|_| RvmError::PartitionNotFound)?;

        // Clear the entry
        for entry in self.entries.iter_mut() {
            if entry.active && entry.id == id {
                entry.active = false;
                break;
            }
        }
        Ok(())
    }

    /// Record a directed communication event between two partitions.
    ///
    /// If no edge exists yet, one is created. If an edge already exists,
    /// its weight is incremented by `weight`.
    ///
    /// Uses the graph's adjacency-matrix-backed `find_directed_edge` for
    /// O(1) existence check + O(out-degree) edge lookup instead of the
    /// previous O(E) scan over all active edges.
    pub fn record_communication(
        &mut self,
        from: PartitionId,
        to: PartitionId,
        weight: u64,
    ) -> Result<(), RvmError> {
        self.cache_valid = false;
        match self.graph.find_directed_edge(from, to) {
            Some(eidx) => {
                // Clamp to i64::MAX: a plain `as i64` cast would turn
                // weights >= 2^63 into negative deltas (weight decrease).
                let delta = weight.min(i64::MAX as u64) as i64;
                self.graph
                    .update_weight(eidx, delta)
                    .map_err(|_| RvmError::InternalError)?;
            }
            None => {
                self.graph.add_edge(from, to, weight).map_err(|e| match e {
                    GraphError::EdgeCapacityExhausted => RvmError::ResourceLimitExceeded,
                    GraphError::NodeNotFound => RvmError::PartitionNotFound,
                    _ => RvmError::InternalError,
                })?;
            }
        }
        Ok(())
    }

    /// Advance one epoch.
    ///
    /// Consults the adaptive engine to decide whether to recompute
    /// coherence scores and cut pressures. Returns the strongest
    /// split or merge recommendation found, or `NoAction`.
    /// Edge weight decay rate per epoch (basis points). 500 = 5% decay.
    const EDGE_DECAY_BP: u16 = 500;

    /// Advance one epoch.
    ///
    /// Decays edge weights by 5% per epoch to prevent stale communication
    /// patterns from dominating. Then consults the adaptive engine to
    /// decide whether to recompute scores and pressures. Returns the
    /// strongest split or merge recommendation, or `NoAction`.
    pub fn tick(&mut self, cpu_load_percent: u8) -> CoherenceDecision {
        self.epoch = self.epoch.wrapping_add(1);

        // Decay edge weights each epoch to prevent stale communication
        // patterns from dominating the graph.
        self.graph.decay_weights(Self::EDGE_DECAY_BP);

        let should_recompute = self.adaptive.tick(cpu_load_percent);
        if !should_recompute {
            // Skip tick: avoid the O(n^2) recommendation pass over stale
            // data. Reuse the cached decision unless the graph mutated
            // since it was computed.
            if !self.cache_valid {
                self.cached_decision = self.recommend();
                self.cache_valid = true;
            }
            return self.cached_decision;
        }

        // Recompute scores and pressures for all active partitions
        for entry in self.entries.iter_mut() {
            if !entry.active {
                continue;
            }
            entry.score = self.coherence_backend.compute_score(entry.id, &self.graph);

            let pr = pressure::compute_cut_pressure(entry.id, &self.graph);
            entry.pressure = pr.pressure;
        }

        self.adaptive.record_computation();
        let decision = self.recommend();
        self.cached_decision = decision;
        self.cache_valid = true;
        decision
    }

    /// Get the current coherence score for a partition.
    #[must_use]
    pub fn score(&self, id: PartitionId) -> CoherenceScore {
        for entry in &self.entries {
            if entry.active && entry.id == id {
                return entry.score;
            }
        }
        CoherenceScore::MAX // unknown partition treated as fully coherent
    }

    /// Get the current cut pressure for a partition.
    #[must_use]
    pub fn pressure(&self, id: PartitionId) -> CutPressure {
        for entry in &self.entries {
            if entry.active && entry.id == id {
                return entry.pressure;
            }
        }
        CutPressure::ZERO // unknown partition has no pressure
    }

    /// Get the strongest split or merge recommendation without advancing
    /// the epoch.
    #[must_use]
    pub fn recommend(&self) -> CoherenceDecision {
        // Find the partition with the highest split pressure
        let mut best_split: Option<(PartitionId, CutPressure)> = None;
        for entry in &self.entries {
            if !entry.active {
                continue;
            }
            if entry.pressure.as_fixed() > SPLIT_THRESHOLD_BP {
                match best_split {
                    None => best_split = Some((entry.id, entry.pressure)),
                    Some((_, prev)) if entry.pressure > prev => {
                        best_split = Some((entry.id, entry.pressure));
                    }
                    _ => {}
                }
            }
        }

        if let Some((partition, pressure)) = best_split {
            return CoherenceDecision::SplitRecommended {
                partition,
                pressure,
            };
        }

        // Check for merge candidates among all pairs
        let mut best_merge: Option<MergeSignal> = None;
        let active_entries: [Option<PartitionId>; ENGINE_MAX_NODES] = {
            let mut arr = [None; ENGINE_MAX_NODES];
            for (i, entry) in self.entries.iter().enumerate() {
                if entry.active {
                    arr[i] = Some(entry.id);
                }
            }
            arr
        };

        for i in 0..ENGINE_MAX_NODES {
            let a = match active_entries[i] {
                Some(id) => id,
                None => continue,
            };
            for j in (i + 1)..ENGINE_MAX_NODES {
                let b = match active_entries[j] {
                    Some(id) => id,
                    None => continue,
                };
                // Non-adjacent pairs can never merge: with zero connecting
                // weight, `evaluate_merge` always yields mutual_bp == 0,
                // which is below the merge threshold. Skip them via the
                // O(1) adjacency-matrix lookup.
                if self.graph.edge_weight_between(a, b) == 0 {
                    continue;
                }
                let signal = pressure::evaluate_merge(a, b, &self.graph);
                if signal.should_merge {
                    match best_merge {
                        None => best_merge = Some(signal),
                        Some(ref prev)
                            if signal.mutual_coherence > prev.mutual_coherence =>
                        {
                            best_merge = Some(signal);
                        }
                        _ => {}
                    }
                }
            }
        }

        if let Some(signal) = best_merge {
            return CoherenceDecision::MergeRecommended {
                a: signal.partition_a,
                b: signal.partition_b,
                mutual_coherence: signal.mutual_coherence,
            };
        }

        CoherenceDecision::NoAction
    }

    /// Access the underlying coherence graph (for inspection/testing).
    #[must_use]
    pub fn graph(&self) -> &CoherenceGraph<ENGINE_MAX_NODES, ENGINE_MAX_EDGES> {
        &self.graph
    }

    /// Access the adaptive engine (for inspection/testing).
    #[must_use]
    pub fn adaptive(&self) -> &AdaptiveCoherenceEngine {
        &self.adaptive
    }

    /// The name of the active mincut backend.
    #[must_use]
    pub fn mincut_backend_name(&self) -> &'static str {
        self.mincut_backend.backend_name()
    }

    /// The name of the active coherence scoring backend.
    #[must_use]
    pub fn coherence_backend_name(&self) -> &'static str {
        self.coherence_backend.backend_name()
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn pid(n: u32) -> PartitionId {
        PartitionId::new(n)
    }

    #[test]
    fn engine_creation_defaults() {
        let engine = DefaultCoherenceEngine::with_defaults(100);
        assert_eq!(engine.epoch(), 0);
        assert_eq!(engine.partition_count(), 0);
        assert_eq!(engine.mincut_backend_name(), "stoer-wagner-builtin");
        assert_eq!(engine.coherence_backend_name(), "ratio-builtin");
    }

    #[test]
    fn add_and_remove_partitions() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);

        engine.add_partition(pid(1)).unwrap();
        engine.add_partition(pid(2)).unwrap();
        assert_eq!(engine.partition_count(), 2);

        engine.remove_partition(pid(1)).unwrap();
        assert_eq!(engine.partition_count(), 1);
    }

    #[test]
    fn duplicate_partition_rejected() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();
        assert_eq!(
            engine.add_partition(pid(1)),
            Err(RvmError::InvalidPartitionState)
        );
    }

    #[test]
    fn remove_nonexistent_partition_fails() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        assert_eq!(
            engine.remove_partition(pid(99)),
            Err(RvmError::PartitionNotFound)
        );
    }

    #[test]
    fn record_communication_creates_edge() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();
        engine.add_partition(pid(2)).unwrap();

        engine.record_communication(pid(1), pid(2), 500).unwrap();
        assert_eq!(engine.graph().edge_count(), 1);

        // Second call increments weight rather than creating new edge
        engine.record_communication(pid(1), pid(2), 300).unwrap();
        assert_eq!(engine.graph().edge_count(), 1);
    }

    #[test]
    fn record_communication_to_unknown_partition_fails() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();
        assert_eq!(
            engine.record_communication(pid(1), pid(99), 100),
            Err(RvmError::PartitionNotFound)
        );
    }

    #[test]
    fn tick_advances_epoch() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();

        assert_eq!(engine.epoch(), 0);
        engine.tick(20);
        assert_eq!(engine.epoch(), 1);
        engine.tick(20);
        assert_eq!(engine.epoch(), 2);
    }

    #[test]
    fn score_after_tick() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();
        engine.add_partition(pid(2)).unwrap();
        engine.record_communication(pid(1), pid(2), 1000).unwrap();

        // Before tick, score is the initial MAX
        assert_eq!(engine.score(pid(1)), CoherenceScore::MAX);

        // After tick at low load, scores are recomputed
        engine.tick(10);

        // pid(1) has external-only edges, so score should be 0
        assert_eq!(engine.score(pid(1)).as_basis_points(), 0);
    }

    #[test]
    fn pressure_after_tick() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();
        engine.add_partition(pid(2)).unwrap();
        engine.record_communication(pid(1), pid(2), 1000).unwrap();

        engine.tick(10);

        // pid(1) has fully external traffic => max pressure
        assert_eq!(engine.pressure(pid(1)).as_fixed(), 10_000);
    }

    #[test]
    fn split_recommended_for_high_pressure() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();
        engine.add_partition(pid(2)).unwrap();
        engine.record_communication(pid(1), pid(2), 1000).unwrap();

        let decision = engine.tick(10);

        match decision {
            CoherenceDecision::SplitRecommended { partition, pressure } => {
                // Either pid(1) or pid(2) should be recommended for split
                assert!(partition == pid(1) || partition == pid(2));
                assert!(pressure.as_fixed() > SPLIT_THRESHOLD_BP);
            }
            _ => panic!("expected SplitRecommended"),
        }
    }

    #[test]
    fn no_action_for_isolated_partitions() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();
        engine.add_partition(pid(2)).unwrap();
        // No communication recorded

        let decision = engine.tick(10);
        assert_eq!(decision, CoherenceDecision::NoAction);
    }

    #[test]
    fn recommend_without_tick() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();
        // No edges, no pressure
        assert_eq!(engine.recommend(), CoherenceDecision::NoAction);
    }

    #[test]
    fn score_of_unknown_partition_returns_max() {
        let engine = DefaultCoherenceEngine::with_defaults(100);
        assert_eq!(engine.score(pid(99)), CoherenceScore::MAX);
    }

    #[test]
    fn pressure_of_unknown_partition_returns_zero() {
        let engine = DefaultCoherenceEngine::with_defaults(100);
        assert_eq!(engine.pressure(pid(99)), CutPressure::ZERO);
    }

    #[test]
    fn adaptive_skips_under_high_load() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();
        engine.add_partition(pid(2)).unwrap();
        engine.record_communication(pid(1), pid(2), 1000).unwrap();

        // First tick at high load -- always computes on first epoch
        let _ = engine.tick(90);
        assert_eq!(engine.epoch(), 1);
        // Score should have been computed
        assert_eq!(engine.score(pid(1)).as_basis_points(), 0);

        // Next 3 ticks at high load should skip recomputation
        // (interval = 4 at >80% load). Scores stay the same.
        let _ = engine.tick(90);
        let _ = engine.tick(90);
        let _ = engine.tick(90);
        assert_eq!(engine.epoch(), 4);
    }

    #[test]
    fn skip_tick_returns_cached_decision() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();
        engine.add_partition(pid(2)).unwrap();
        engine.record_communication(pid(1), pid(2), 1000).unwrap();

        // First tick at high load always computes (split expected).
        let first = engine.tick(90);
        assert!(matches!(
            first,
            CoherenceDecision::SplitRecommended { .. }
        ));

        // Subsequent skip ticks (interval = 4 at >80% load) must return
        // the same cached decision without recomputation.
        assert_eq!(engine.tick(90), first);
        assert_eq!(engine.tick(90), first);
    }

    #[test]
    fn graph_mutation_invalidates_decision_cache() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();
        engine.add_partition(pid(2)).unwrap();

        // First tick at high load computes: no edges => NoAction.
        assert_eq!(engine.tick(90), CoherenceDecision::NoAction);

        // Mutate the graph with heavy mutual traffic; the next skip tick
        // must not return the stale cached NoAction.
        engine.record_communication(pid(1), pid(2), 8000).unwrap();
        engine.record_communication(pid(2), pid(1), 8000).unwrap();

        let decision = engine.tick(90); // skip tick, cache invalidated
        assert!(matches!(
            decision,
            CoherenceDecision::MergeRecommended { .. }
        ));
    }

    #[test]
    fn record_communication_huge_weight_does_not_decrease_edge() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();
        engine.add_partition(pid(2)).unwrap();

        // Create the edge, then record a weight >= 2^63 on the update
        // path. A naive `as i64` cast would make the delta negative and
        // shrink the edge weight.
        engine.record_communication(pid(1), pid(2), 100).unwrap();
        engine
            .record_communication(pid(1), pid(2), u64::MAX)
            .unwrap();

        let w = engine.graph().edge_weight_between(pid(1), pid(2));
        // Clamped delta: 100 + i64::MAX.
        assert_eq!(w, 100u64 + i64::MAX as u64);
        assert!(w > 100);
    }

    #[cfg(feature = "ruvector")]
    #[test]
    fn ruvector_engine_creation() {
        let engine = RuVectorCoherenceEngine::with_ruvector(100);
        assert_eq!(engine.mincut_backend_name(), "ruvector-mincut-stub");
        assert_eq!(engine.coherence_backend_name(), "ruvector-spectral-stub");
    }

    #[cfg(feature = "ruvector")]
    #[test]
    fn ruvector_engine_lifecycle() {
        let mut engine = RuVectorCoherenceEngine::with_ruvector(100);
        engine.add_partition(pid(1)).unwrap();
        engine.add_partition(pid(2)).unwrap();
        engine.record_communication(pid(1), pid(2), 500).unwrap();

        let decision = engine.tick(10);
        // With only external traffic, should recommend split
        match decision {
            CoherenceDecision::SplitRecommended { .. } => {}
            _ => panic!("expected SplitRecommended from ruvector engine"),
        }
    }

    #[cfg(feature = "ruvector")]
    #[test]
    fn ruvector_matches_builtin_results() {
        // Since the ruvector stubs delegate to the builtin, results
        // should be identical.
        let mut default_engine = DefaultCoherenceEngine::with_defaults(100);
        let mut rv_engine = RuVectorCoherenceEngine::with_ruvector(100);

        for engine in [&mut default_engine as &mut dyn EngineOps, &mut rv_engine] {
            engine.add_p(pid(1)).unwrap();
            engine.add_p(pid(2)).unwrap();
            engine.record(pid(1), pid(2), 1000).unwrap();
            engine.do_tick(10);
        }

        assert_eq!(
            default_engine.score(pid(1)),
            rv_engine.score(pid(1))
        );
        assert_eq!(
            default_engine.pressure(pid(1)),
            rv_engine.pressure(pid(1))
        );
    }
}

// Helper trait for the ruvector_matches_builtin_results test
#[cfg(all(test, feature = "ruvector"))]
trait EngineOps {
    fn add_p(&mut self, id: PartitionId) -> Result<(), RvmError>;
    fn record(&mut self, from: PartitionId, to: PartitionId, w: u64) -> Result<(), RvmError>;
    fn do_tick(&mut self, load: u8) -> CoherenceDecision;
}

#[cfg(all(test, feature = "ruvector"))]
impl<MCB: MinCutBackend, CB: CoherenceBackend> EngineOps for CoherenceEngine<MCB, CB> {
    fn add_p(&mut self, id: PartitionId) -> Result<(), RvmError> {
        self.add_partition(id)
    }
    fn record(&mut self, from: PartitionId, to: PartitionId, w: u64) -> Result<(), RvmError> {
        self.record_communication(from, to, w)
    }
    fn do_tick(&mut self, load: u8) -> CoherenceDecision {
        self.tick(load)
    }
}
