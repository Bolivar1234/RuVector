//! HRM-Text model configuration (ADR-199 Phase D).
//!
//! Mirrors the upstream `sapientinc/HRM-Text` hyperparameters: a dual-timescale
//! hierarchical recurrent transformer where `num_hidden_layers` TOTAL layers are
//! split evenly (`half_layers`) between the slow H-level and fast L-level cores,
//! iterated `h_cycles x l_cycles` times per forward pass.

use crate::error::{Result, RuvLLMError};
use serde::{Deserialize, Serialize};

/// Where layer normalization is applied within a decoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NormPosition {
    /// Pre-norm (Llama/Phi-3 style): `x + f(norm(x))`. Default.
    PreNorm,
    /// Post-norm (original Transformer style): `norm(x + f(x))`.
    PostNorm,
}

/// Configuration for the HRM-Text hierarchical recurrent architecture.
///
/// `num_hidden_layers` is the TOTAL layer count; `half_layers`
/// (`num_hidden_layers / 2`) layers go to each of the H and L cores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrmTextConfig {
    /// Hidden size (embedding dimension).
    pub hidden_size: usize,
    /// Intermediate size of the SwiGLU MLP. See [`HrmTextConfig::default_intermediate_size`].
    pub intermediate_size: usize,
    /// TOTAL number of decoder layers (split evenly between H and L cores).
    pub num_hidden_layers: usize,
    /// Number of attention (query) heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads (GQA when < `num_attention_heads`).
    pub num_kv_heads: usize,
    /// Vocabulary size (upstream HRM-Text uses a custom 65,536-token vocab).
    pub vocab_size: usize,
    /// Maximum sequence length supported.
    pub max_position_embeddings: usize,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// Pre-norm vs post-norm layer layout.
    pub norm_position: NormPosition,
    /// Number of slow H-level cycles.
    pub h_cycles: usize,
    /// Number of fast L-level cycles per H cycle.
    pub l_cycles: usize,
    /// Layers per core: must equal `num_hidden_layers / 2`.
    pub half_layers: usize,
    /// Whether prompt (prefix) tokens attend bidirectionally (PrefixLM
    /// objective; upstream requires `token_type_ids = ones`).
    pub prefix_bidirectional: bool,
    /// Beginning-of-sequence token id.
    pub bos_token_id: u32,
    /// End-of-sequence token id.
    pub eos_token_id: u32,
}

impl HrmTextConfig {
    /// Default SwiGLU intermediate size.
    ///
    /// Formula: take the standard 4x MLP expansion, scale by 2/3 to keep
    /// parameter count constant under SwiGLU's three projections
    /// (`4 * hidden * 2/3 = (8/3) * hidden`), then round UP to the next
    /// multiple of 256 for tile-friendly GEMMs:
    ///
    /// ```text
    /// intermediate = ceil( ceil(8 * hidden / 3) / 256 ) * 256
    /// ```
    ///
    /// Examples: hidden 1280 -> 3584, hidden 1536 -> 4096.
    pub fn default_intermediate_size(hidden_size: usize) -> usize {
        let raw = (8 * hidden_size + 2) / 3; // ceil(8h/3)
        ((raw + 255) / 256) * 256 // round up to multiple of 256
    }

    /// HRM-Text-L reference configuration (~1B class): 24 total layers
    /// (12 per core), hidden 1280, 10 heads, 65,536-token vocab.
    pub fn hrm_text_l() -> Self {
        Self {
            hidden_size: 1280,
            intermediate_size: Self::default_intermediate_size(1280), // 3584
            num_hidden_layers: 24,
            num_attention_heads: 10,
            num_kv_heads: 10,
            vocab_size: 65536,
            max_position_embeddings: 4096,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            norm_position: NormPosition::PreNorm,
            h_cycles: 2,
            l_cycles: 2,
            half_layers: 12,
            prefix_bidirectional: true,
            bos_token_id: 1,
            eos_token_id: 2,
        }
    }

    /// HRM-Text-XL configuration: 32 total layers (16 per core),
    /// hidden 1536, 12 heads, 65,536-token vocab.
    pub fn hrm_text_xl() -> Self {
        Self {
            hidden_size: 1536,
            intermediate_size: Self::default_intermediate_size(1536), // 4096
            num_hidden_layers: 32,
            num_attention_heads: 12,
            num_kv_heads: 12,
            vocab_size: 65536,
            max_position_embeddings: 4096,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            norm_position: NormPosition::PreNorm,
            h_cycles: 2,
            l_cycles: 2,
            half_layers: 16,
            prefix_bidirectional: true,
            bos_token_id: 1,
            eos_token_id: 2,
        }
    }

    /// Tiny configuration for tests: 2 total layers (1 per core), hidden 32,
    /// 2 heads (head_dim 16), vocab 128, 2x2 cycles. Intermediate size is set
    /// to 64 explicitly (the 256-multiple default would dominate test time).
    pub fn tiny_test() -> Self {
        Self {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_kv_heads: 2,
            vocab_size: 128,
            max_position_embeddings: 64,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            norm_position: NormPosition::PreNorm,
            h_cycles: 2,
            l_cycles: 2,
            half_layers: 1,
            prefix_bidirectional: true,
            bos_token_id: 1,
            eos_token_id: 2,
        }
    }

    /// Dimension per attention head.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Combined K (or V) row width: `num_kv_heads * head_dim`.
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim()
    }

    /// Validate internal consistency. Must be called before building a model.
    pub fn validate(&self) -> Result<()> {
        if self.num_hidden_layers == 0 || self.num_hidden_layers % 2 != 0 {
            return Err(RuvLLMError::Config(format!(
                "HRM-Text requires an even, non-zero num_hidden_layers to split \
                 between H and L cores; got {}",
                self.num_hidden_layers
            )));
        }
        if self.half_layers != self.num_hidden_layers / 2 {
            return Err(RuvLLMError::Config(format!(
                "half_layers ({}) must equal num_hidden_layers / 2 ({})",
                self.half_layers,
                self.num_hidden_layers / 2
            )));
        }
        if self.num_attention_heads == 0
            || self.hidden_size % self.num_attention_heads != 0
        {
            return Err(RuvLLMError::Config(format!(
                "hidden_size {} must be divisible by num_attention_heads {}",
                self.hidden_size, self.num_attention_heads
            )));
        }
        if self.num_kv_heads == 0 || self.num_attention_heads % self.num_kv_heads != 0 {
            return Err(RuvLLMError::Config(format!(
                "num_attention_heads {} must be divisible by num_kv_heads {}",
                self.num_attention_heads, self.num_kv_heads
            )));
        }
        if self.h_cycles == 0 || self.l_cycles == 0 {
            return Err(RuvLLMError::Config(
                "h_cycles and l_cycles must both be >= 1".to_string(),
            ));
        }
        if self.vocab_size == 0 || self.hidden_size == 0 || self.intermediate_size == 0 {
            return Err(RuvLLMError::Config(
                "vocab_size, hidden_size and intermediate_size must be non-zero".to_string(),
            ));
        }
        if self.max_position_embeddings == 0 {
            return Err(RuvLLMError::Config(
                "max_position_embeddings must be non-zero".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_intermediate_size_formula() {
        // ceil(8*1280/3) = 3414 -> rounded up to 3584
        assert_eq!(HrmTextConfig::default_intermediate_size(1280), 3584);
        // 8*1536/3 = 4096 exactly (already a multiple of 256)
        assert_eq!(HrmTextConfig::default_intermediate_size(1536), 4096);
        // Small sizes round up to at least 256
        assert_eq!(HrmTextConfig::default_intermediate_size(32), 256);
    }

    #[test]
    fn test_reference_configs_validate() {
        assert!(HrmTextConfig::hrm_text_l().validate().is_ok());
        assert!(HrmTextConfig::hrm_text_xl().validate().is_ok());
        assert!(HrmTextConfig::tiny_test().validate().is_ok());
    }

    #[test]
    fn test_odd_layers_rejected() {
        let mut config = HrmTextConfig::tiny_test();
        config.num_hidden_layers = 3;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_half_layers_mismatch_rejected() {
        let mut config = HrmTextConfig::tiny_test();
        config.half_layers = 2; // num_hidden_layers is 2, so half must be 1
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_head_dims() {
        let config = HrmTextConfig::tiny_test();
        assert_eq!(config.head_dim(), 16);
        assert_eq!(config.kv_dim(), 32);
        let l = HrmTextConfig::hrm_text_l();
        assert_eq!(l.head_dim(), 128);
    }
}
