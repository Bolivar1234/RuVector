//! Attention mask modes for HRM-Text (ADR-199 Phase D).
//!
//! HRM-Text is PrefixLM-pretrained: prompt (prefix) tokens attend
//! bidirectionally while response tokens attend causally. This module models
//! that as a tiny mask object passed into the attention loops — a mask-function
//! change, not a new kernel (ADR-199 Phase D item 2).

use serde::{Deserialize, Serialize};

/// Attention masking mode.
///
/// The allowed-set rule is:
///
/// ```text
/// Causal:                allowed(q, k) = k <= q
/// PrefixLm{prefix_len}:  allowed(q, k) = k < prefix_len || k <= q
/// ```
///
/// `PrefixLm { prefix_len: 0 }` is therefore exactly equivalent to `Causal`,
/// and for query positions `q >= prefix_len` the PrefixLM allowed-set is
/// bit-for-bit identical to the causal one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AttentionMaskMode {
    /// Standard causal (autoregressive) masking.
    Causal,
    /// PrefixLM masking: the first `prefix_len` key positions are visible to
    /// every query; outside the prefix, masking is causal.
    PrefixLm {
        /// Number of leading positions forming the bidirectional prefix.
        prefix_len: usize,
    },
}

impl AttentionMaskMode {
    /// Whether query position `q_pos` may attend to key position `k_pos`.
    #[inline(always)]
    pub fn allowed(&self, q_pos: usize, k_pos: usize) -> bool {
        match self {
            Self::Causal => k_pos <= q_pos,
            Self::PrefixLm { prefix_len } => k_pos < *prefix_len || k_pos <= q_pos,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_zero_equals_causal() {
        let causal = AttentionMaskMode::Causal;
        let prefix0 = AttentionMaskMode::PrefixLm { prefix_len: 0 };
        for q in 0..16 {
            for k in 0..16 {
                assert_eq!(causal.allowed(q, k), prefix0.allowed(q, k));
            }
        }
    }

    #[test]
    fn test_prefix_rows_attend_full_prefix() {
        let mask = AttentionMaskMode::PrefixLm { prefix_len: 6 };
        // Every query inside the prefix sees the entire prefix (bidirectional).
        for q in 0..6 {
            for k in 0..6 {
                assert!(mask.allowed(q, k), "prefix q={} must see prefix k={}", q, k);
            }
            // ... but not beyond it (unless causal, i.e. k <= q, which can't
            // exceed the prefix here).
            for k in 6..16 {
                assert!(!mask.allowed(q, k));
            }
        }
    }

    #[test]
    fn test_causal_region_matches_causal() {
        let causal = AttentionMaskMode::Causal;
        let mask = AttentionMaskMode::PrefixLm { prefix_len: 6 };
        for q in 6..16 {
            for k in 0..16 {
                assert_eq!(causal.allowed(q, k), mask.allowed(q, k));
            }
        }
    }
}
