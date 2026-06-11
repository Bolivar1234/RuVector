//! Per-cycle KV caches for the HRM-Text recurrence (ADR-199 Phase D item 3).
//!
//! Each (level, cycle, layer) triple attends over a *different* K/V tensor, so
//! the recurrence needs `(h_cycles * l_cycles + h_cycles) * half_layers` cache
//! slots — about 3x the KV memory of an equal-depth standard transformer for
//! representative configs (e.g. 2x2 cycles, 24 total layers: `6 * 12 / 24 = 3x`).
//! That 3x figure is the headline number this module makes measurable via
//! [`HrmKvCaches::memory_bytes`] and [`HrmKvCaches::projected_memory_bytes`].

use super::config::HrmTextConfig;
use crate::error::{Result, RuvLLMError};

/// KV cache for a single (level, cycle, layer) slot. Stores FP32 keys and
/// values laid out as `[cached_len, num_kv_heads * head_dim]`.
#[derive(Debug, Clone, Default)]
pub struct KvLayerCache {
    k: Vec<f32>,
    v: Vec<f32>,
    kv_dim: usize,
    cached_len: usize,
}

impl KvLayerCache {
    /// Create an empty cache with the given KV row width.
    pub fn new(kv_dim: usize) -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
            kv_dim,
            cached_len: 0,
        }
    }

    /// Append one or more positions of K/V (full prefill or single decode
    /// step). `k` and `v` must each be a whole number of `kv_dim` rows.
    pub fn append(&mut self, k: &[f32], v: &[f32]) -> Result<()> {
        if self.kv_dim == 0 {
            return Err(RuvLLMError::KvCache(
                "KvLayerCache kv_dim is zero".to_string(),
            ));
        }
        if k.len() != v.len() || k.len() % self.kv_dim != 0 {
            return Err(RuvLLMError::KvCache(format!(
                "KV append shape mismatch: k={}, v={}, kv_dim={}",
                k.len(),
                v.len(),
                self.kv_dim
            )));
        }
        self.k.extend_from_slice(k);
        self.v.extend_from_slice(v);
        self.cached_len += k.len() / self.kv_dim;
        Ok(())
    }

    /// Number of cached positions.
    pub fn len(&self) -> usize {
        self.cached_len
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cached_len == 0
    }

    /// Cached keys, `[cached_len, kv_dim]` row-major.
    pub fn keys(&self) -> &[f32] {
        &self.k
    }

    /// Cached values, `[cached_len, kv_dim]` row-major.
    pub fn values(&self) -> &[f32] {
        &self.v
    }

    /// KV row width (`num_kv_heads * head_dim`).
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Bytes currently held (FP32).
    pub fn memory_bytes(&self) -> usize {
        (self.k.len() + self.v.len()) * std::mem::size_of::<f32>()
    }

    /// Drop all cached positions (capacity is released too).
    pub fn clear(&mut self) {
        self.k = Vec::new();
        self.v = Vec::new();
        self.cached_len = 0;
    }

    /// Truncate to the first `len` cached positions. No-op if `len` is at or
    /// beyond the current length. Used for speculative-decoding rollback
    /// (ADR-199 R2): rejected draft positions are dropped from the cache.
    pub fn truncate(&mut self, len: usize) {
        if len < self.cached_len {
            self.k.truncate(len * self.kv_dim);
            self.v.truncate(len * self.kv_dim);
            self.cached_len = len;
        }
    }
}

/// All KV caches for one HRM-Text sequence, indexed by (level, cycle, layer).
///
/// Layout:
/// - `l_cycle(i, k)` -> the `half_layers` caches used by the L core during
///   H-cycle `i`, L-cycle `k` (`h_cycles * l_cycles` slots of `half_layers`).
/// - `h_cycle(i)` -> the `half_layers` caches used by the H core during
///   H-cycle `i` (`h_cycles` slots of `half_layers`).
///
/// Total slot count: `(h_cycles * l_cycles + h_cycles) * half_layers`.
#[derive(Debug, Clone)]
pub struct HrmKvCaches {
    l_slots: Vec<Vec<KvLayerCache>>,
    h_slots: Vec<Vec<KvLayerCache>>,
    h_cycles: usize,
    l_cycles: usize,
    half_layers: usize,
    kv_dim: usize,
}

impl HrmKvCaches {
    /// Allocate empty caches shaped for `config`.
    pub fn new(config: &HrmTextConfig) -> Self {
        let kv_dim = config.kv_dim();
        let make_slot = || -> Vec<KvLayerCache> {
            (0..config.half_layers).map(|_| KvLayerCache::new(kv_dim)).collect()
        };
        Self {
            l_slots: (0..config.h_cycles * config.l_cycles).map(|_| make_slot()).collect(),
            h_slots: (0..config.h_cycles).map(|_| make_slot()).collect(),
            h_cycles: config.h_cycles,
            l_cycles: config.l_cycles,
            half_layers: config.half_layers,
            kv_dim,
        }
    }

    /// Per-layer caches for L-cycle `(i, k)` (H-cycle `i`, L-cycle `k`).
    pub fn l_cycle(&mut self, i: usize, k: usize) -> Result<&mut [KvLayerCache]> {
        if i >= self.h_cycles || k >= self.l_cycles {
            return Err(RuvLLMError::KvCache(format!(
                "L cache slot ({}, {}) out of range ({} x {})",
                i, k, self.h_cycles, self.l_cycles
            )));
        }
        Ok(self.l_slots[i * self.l_cycles + k].as_mut_slice())
    }

    /// Per-layer caches for H-cycle `i`.
    pub fn h_cycle(&mut self, i: usize) -> Result<&mut [KvLayerCache]> {
        if i >= self.h_cycles {
            return Err(RuvLLMError::KvCache(format!(
                "H cache slot {} out of range ({})",
                i, self.h_cycles
            )));
        }
        Ok(self.h_slots[i].as_mut_slice())
    }

    /// Total number of (level, cycle, layer) slots:
    /// `(h_cycles * l_cycles + h_cycles) * half_layers`.
    pub fn slot_count(&self) -> usize {
        (self.h_cycles * self.l_cycles + self.h_cycles) * self.half_layers
    }

    /// Total bytes currently cached across every slot (FP32).
    pub fn memory_bytes(&self) -> usize {
        self.l_slots
            .iter()
            .chain(self.h_slots.iter())
            .flat_map(|slot| slot.iter())
            .map(KvLayerCache::memory_bytes)
            .sum()
    }

    /// Projected FP32 KV footprint for `config` at `seq_len` positions,
    /// computed analytically (no allocation):
    /// `slots * 2 (K+V) * seq_len * kv_dim * 4 bytes`.
    pub fn projected_memory_bytes(config: &HrmTextConfig, seq_len: usize) -> usize {
        let slots = (config.h_cycles * config.l_cycles + config.h_cycles) * config.half_layers;
        slots * 2 * seq_len * config.kv_dim() * std::mem::size_of::<f32>()
    }

    /// Truncate every slot to the first `len` positions (speculative-decoding
    /// rollback, ADR-199 R2). Slots already at or below `len` are untouched.
    pub fn truncate(&mut self, len: usize) {
        for slot in self.l_slots.iter_mut().chain(self.h_slots.iter_mut()) {
            for cache in slot.iter_mut() {
                cache.truncate(len);
            }
        }
    }

    /// Clear every slot.
    pub fn clear(&mut self) {
        for slot in self.l_slots.iter_mut().chain(self.h_slots.iter_mut()) {
            for cache in slot.iter_mut() {
                cache.clear();
            }
        }
    }

    /// Number of cached positions, validated to be uniform across all slots.
    pub fn cached_len(&self) -> Result<usize> {
        let mut len: Option<usize> = None;
        for cache in self
            .l_slots
            .iter()
            .chain(self.h_slots.iter())
            .flat_map(|slot| slot.iter())
        {
            match len {
                None => len = Some(cache.len()),
                Some(expected) if cache.len() != expected => {
                    return Err(RuvLLMError::KvCache(format!(
                        "non-uniform HRM KV cache lengths: {} vs {}",
                        expected,
                        cache.len()
                    )))
                }
                _ => {}
            }
        }
        Ok(len.unwrap_or(0))
    }

    /// Configured H cycle count.
    pub fn h_cycles(&self) -> usize {
        self.h_cycles
    }

    /// Configured L cycle count.
    pub fn l_cycles(&self) -> usize {
        self.l_cycles
    }

    /// Layers per core.
    pub fn half_layers(&self) -> usize {
        self.half_layers
    }

    /// KV row width.
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slot_count_formula() {
        let config = HrmTextConfig::tiny_test(); // 2x2 cycles, half_layers 1
        let caches = HrmKvCaches::new(&config);
        assert_eq!(caches.slot_count(), (2 * 2 + 2) * 1);

        let l = HrmTextConfig::hrm_text_l(); // 2x2 cycles, half_layers 12
        let caches = HrmKvCaches::new(&l);
        assert_eq!(caches.slot_count(), 72);
    }

    #[test]
    fn test_append_and_clear() {
        let config = HrmTextConfig::tiny_test();
        let mut caches = HrmKvCaches::new(&config);
        let kv_dim = config.kv_dim();
        let k = vec![0.5f32; 3 * kv_dim];
        let v = vec![0.25f32; 3 * kv_dim];
        for cache in caches.l_cycle(0, 0).unwrap() {
            cache.append(&k, &v).unwrap();
            assert_eq!(cache.len(), 3);
        }
        assert!(caches.memory_bytes() > 0);
        // Lengths are non-uniform now (only one slot filled).
        assert!(caches.cached_len().is_err());
        caches.clear();
        assert_eq!(caches.cached_len().unwrap(), 0);
        assert_eq!(caches.memory_bytes(), 0);
    }

    #[test]
    fn test_append_shape_mismatch_rejected() {
        let config = HrmTextConfig::tiny_test();
        let mut cache = KvLayerCache::new(config.kv_dim());
        let k = vec![0.0f32; config.kv_dim() + 1]; // not a whole row
        let v = vec![0.0f32; config.kv_dim() + 1];
        assert!(cache.append(&k, &v).is_err());
    }

    #[test]
    fn test_projected_memory_linear_in_cycles() {
        let mut config = HrmTextConfig::tiny_test();
        let base = HrmKvCaches::projected_memory_bytes(&config, 64);
        config.h_cycles = 4;
        config.l_cycles = 4;
        let big = HrmKvCaches::projected_memory_bytes(&config, 64);
        // slots scale as h * (l + 1): 2*(2+1)=6 -> 4*(4+1)=20
        assert_eq!(big * 6, base * 20);
    }

    #[test]
    fn test_truncate_rolls_back_positions() {
        let config = HrmTextConfig::tiny_test();
        let mut cache = KvLayerCache::new(config.kv_dim());
        let k = vec![0.5f32; 4 * config.kv_dim()];
        let v = vec![0.25f32; 4 * config.kv_dim()];
        cache.append(&k, &v).unwrap();
        assert_eq!(cache.len(), 4);
        cache.truncate(6); // beyond length: no-op
        assert_eq!(cache.len(), 4);
        cache.truncate(2);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.keys().len(), 2 * config.kv_dim());
        assert_eq!(cache.values().len(), 2 * config.kv_dim());
        // Appending after truncation continues from the new length.
        cache
            .append(&k[..config.kv_dim()], &v[..config.kv_dim()])
            .unwrap();
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_out_of_range_slots_error() {
        let config = HrmTextConfig::tiny_test();
        let mut caches = HrmKvCaches::new(&config);
        assert!(caches.l_cycle(2, 0).is_err());
        assert!(caches.l_cycle(0, 2).is_err());
        assert!(caches.h_cycle(2).is_err());
    }
}
