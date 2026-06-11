//! Integration tests for the HRM-Text native backend (ADR-199 Phase D).
//!
//! Proves the Phase D verification properties on tiny seeded random-weight
//! models:
//! - config validation
//! - PrefixLM mask allowed-set properties
//! - the H/L recurrence actually runs (cycle counts change logits)
//! - per-cycle KV cache slot count and memory accounting (incl. the ~3x
//!   headline number for hrm_text_l at 4K context, FP32)
//! - decode parity: prefill + decode_step logits match the full-sequence
//!   forward within 1e-4 (the critical per-cycle cache correctness test)
//! - greedy generation determinism and eos handling
//! - the PrefixLM mask measurably changes logits vs causal

use ruvllm::backends::hrm_text::{
    AttentionMaskMode, HrmKvCaches, HrmTextConfig, HrmTextModel,
};

const SEED: u64 = 42;

fn tiny_model() -> HrmTextModel {
    HrmTextModel::new_random(HrmTextConfig::tiny_test(), SEED).expect("tiny model builds")
}

// ---------------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------------

#[test]
fn config_rejects_odd_layer_count() {
    let mut config = HrmTextConfig::tiny_test();
    config.num_hidden_layers = 3;
    assert!(config.validate().is_err());
    // And the model constructor surfaces it too.
    assert!(HrmTextModel::new_random(config, SEED).is_err());
}

#[test]
fn config_rejects_half_layers_mismatch() {
    let mut config = HrmTextConfig::tiny_test();
    config.num_hidden_layers = 4; // half_layers stays 1, must be 2
    assert!(config.validate().is_err());
}

// ---------------------------------------------------------------------------
// Mask properties (enumerated over a 16x16 grid)
// ---------------------------------------------------------------------------

#[test]
fn mask_prefix_zero_is_causal() {
    let causal = AttentionMaskMode::Causal;
    let prefix0 = AttentionMaskMode::PrefixLm { prefix_len: 0 };
    for q in 0..16 {
        for k in 0..16 {
            assert_eq!(causal.allowed(q, k), prefix0.allowed(q, k));
        }
    }
}

#[test]
fn mask_prefix_rows_attend_full_prefix_and_causal_region_is_bitwise_causal() {
    let prefix_len = 6;
    let mask = AttentionMaskMode::PrefixLm { prefix_len };
    let causal = AttentionMaskMode::Causal;
    for q in 0..16 {
        for k in 0..16 {
            let allowed = mask.allowed(q, k);
            if q < prefix_len {
                // Prefix rows see the full prefix (bidirectional) and
                // nothing beyond it.
                assert_eq!(allowed, k < prefix_len, "q={} k={}", q, k);
            } else {
                // Causal region matches Causal bit-for-bit.
                assert_eq!(allowed, causal.allowed(q, k), "q={} k={}", q, k);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Recurrence
// ---------------------------------------------------------------------------

#[test]
fn forward_returns_full_logit_grid() {
    let config = HrmTextConfig::tiny_test();
    let model = tiny_model();
    let mut caches = HrmKvCaches::new(&config);
    let ids = [3u32, 5, 7, 11, 13];
    let logits = model
        .forward(&ids, AttentionMaskMode::Causal, &mut caches)
        .unwrap();
    assert_eq!(logits.len(), ids.len() * config.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn cycle_counts_change_logits() {
    // Same weights, different recurrence depth => different logits,
    // proving the nested H/L loop actually runs.
    let ids = [3u32, 5, 7, 11];

    let mut model = tiny_model(); // 2x2 cycles
    let mut caches_22 = HrmKvCaches::new(&model.config);
    let logits_22 = model
        .forward(&ids, AttentionMaskMode::Causal, &mut caches_22)
        .unwrap();

    model.cycle_override(1, 1).unwrap();
    let mut caches_11 = HrmKvCaches::new(&model.config);
    let logits_11 = model
        .forward(&ids, AttentionMaskMode::Causal, &mut caches_11)
        .unwrap();

    model.cycle_override(3, 2).unwrap();
    let mut caches_32 = HrmKvCaches::new(&model.config);
    let logits_32 = model
        .forward(&ids, AttentionMaskMode::Causal, &mut caches_32)
        .unwrap();

    let max_diff = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    assert!(
        max_diff(&logits_22, &logits_11) > 1e-3,
        "1x1 cycles must differ from 2x2"
    );
    assert!(
        max_diff(&logits_22, &logits_32) > 1e-3,
        "3x2 cycles must differ from 2x2"
    );
}

// ---------------------------------------------------------------------------
// KV cache accounting
// ---------------------------------------------------------------------------

#[test]
fn kv_cache_slot_count_matches_formula() {
    for (config, name) in [
        (HrmTextConfig::tiny_test(), "tiny"),
        (HrmTextConfig::hrm_text_l(), "l"),
        (HrmTextConfig::hrm_text_xl(), "xl"),
    ] {
        let caches = HrmKvCaches::new(&config);
        let expected =
            (config.h_cycles * config.l_cycles + config.h_cycles) * config.half_layers;
        assert_eq!(caches.slot_count(), expected, "config {}", name);
    }
}

#[test]
fn kv_memory_positive_and_linear_in_cycles() {
    let ids = [3u32, 5, 7, 11];

    let config = HrmTextConfig::tiny_test(); // 2x2
    let model = HrmTextModel::new_random(config.clone(), SEED).unwrap();
    let mut caches = HrmKvCaches::new(&config);
    model
        .forward(&ids, AttentionMaskMode::Causal, &mut caches)
        .unwrap();
    let bytes_22 = caches.memory_bytes();
    assert!(bytes_22 > 0);

    let mut config_44 = HrmTextConfig::tiny_test();
    config_44.h_cycles = 4;
    config_44.l_cycles = 4;
    let model_44 = HrmTextModel::new_random(config_44.clone(), SEED).unwrap();
    let mut caches_44 = HrmKvCaches::new(&config_44);
    model_44
        .forward(&ids, AttentionMaskMode::Causal, &mut caches_44)
        .unwrap();
    let bytes_44 = caches_44.memory_bytes();

    // Slot count scales as h*(l+1): 2*(2+1)=6 -> 4*(4+1)=20.
    assert_eq!(bytes_44 * 6, bytes_22 * 20, "KV bytes must scale linearly with cycle slots");

    // Filled bytes match the analytic projection at this sequence length.
    assert_eq!(
        bytes_22,
        HrmKvCaches::projected_memory_bytes(&config, ids.len())
    );
}

#[test]
fn kv_memory_hrm_text_l_at_4k_is_3x_standard() {
    // The ADR-199 headline number: per-cycle caching costs
    // (h*l + h) * half_layers slots vs num_hidden_layers slots for an
    // equal-depth standard model => 6*12 / 24 = 3x for hrm_text_l (2x2).
    let config = HrmTextConfig::hrm_text_l();
    let seq = 4096;
    let hrm_bytes = HrmKvCaches::projected_memory_bytes(&config, seq);

    // Exact expectation: 72 slots * 2 (K+V) * 4096 * kv_dim(1280) * 4 bytes.
    assert_eq!(hrm_bytes, 72 * 2 * seq * 1280 * 4);
    assert_eq!(hrm_bytes, 3_019_898_880); // ~2.81 GiB FP32

    let standard_bytes =
        config.num_hidden_layers * 2 * seq * config.kv_dim() * std::mem::size_of::<f32>();
    assert_eq!(hrm_bytes, 3 * standard_bytes);

    println!(
        "hrm_text_l KV @ {} ctx FP32: {} bytes ({:.2} GiB), standard 24-layer: {} bytes, ratio {}x",
        seq,
        hrm_bytes,
        hrm_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        standard_bytes,
        hrm_bytes / standard_bytes
    );
}

// ---------------------------------------------------------------------------
// Decode parity — the critical per-cycle cache correctness test
// ---------------------------------------------------------------------------

#[test]
fn decode_parity_with_full_forward() {
    const TOL: f32 = 1e-4;
    let config = HrmTextConfig::tiny_test();
    let model = tiny_model();
    let vocab = config.vocab_size;

    // Random-ish 12-token sequence with a 6-token PrefixLM prefix.
    let ids: Vec<u32> = vec![3, 17, 42, 99, 5, 64, 23, 81, 7, 120, 56, 11];
    let prefix_len = 6;

    // Reference: full-sequence forward under the same PrefixLM mask.
    let mut ref_caches = HrmKvCaches::new(&config);
    let full_logits = model
        .forward(
            &ids,
            AttentionMaskMode::PrefixLm { prefix_len },
            &mut ref_caches,
        )
        .unwrap();

    // Incremental: prefill the prefix region's tokens... prefill takes the
    // whole prompt; emulate generation by prefilling positions 0..=p and
    // decoding the rest one token at a time.
    let prompt_len = prefix_len + 1; // prefill through position 6
    let mut caches = HrmKvCaches::new(&config);
    let mut step_logits = Vec::new(); // (position, logits) for pos >= prefix_len

    let prefill_logits = model
        .prefill(&ids[..prompt_len], prefix_len, &mut caches)
        .unwrap();
    step_logits.push((prompt_len - 1, prefill_logits));

    for pos in prompt_len..ids.len() {
        let logits = model.decode_step(ids[pos], pos, &mut caches).unwrap();
        step_logits.push((pos, logits));
    }

    // Also verify the earlier in-prefix-region rows directly from the full
    // forward against a second full forward (sanity: determinism).
    let mut caches2 = HrmKvCaches::new(&config);
    let full_logits2 = model
        .forward(
            &ids,
            AttentionMaskMode::PrefixLm { prefix_len },
            &mut caches2,
        )
        .unwrap();
    assert_eq!(full_logits, full_logits2, "forward must be deterministic");

    let mut worst = 0.0f32;
    for (pos, logits) in &step_logits {
        assert!(*pos >= prefix_len);
        let reference = &full_logits[pos * vocab..(pos + 1) * vocab];
        let max_diff = reference
            .iter()
            .zip(logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        worst = worst.max(max_diff);
        assert!(
            max_diff <= TOL,
            "decode parity failed at position {}: max diff {} > {}",
            pos,
            max_diff,
            TOL
        );
    }
    assert_eq!(step_logits.len(), ids.len() - prompt_len + 1);
    println!("decode parity: worst max-abs logit diff = {:e} (tolerance {:e})", worst, TOL);
}

// ---------------------------------------------------------------------------
// Greedy generation
// ---------------------------------------------------------------------------

#[test]
fn greedy_generation_is_deterministic_per_seed() {
    let prompt = [3u32, 5, 7, 11];
    let a = tiny_model().generate_greedy(&prompt, 2, 8).unwrap();
    let b = tiny_model().generate_greedy(&prompt, 2, 8).unwrap();
    assert_eq!(a, b, "same seed must generate identical tokens");
    assert!(a.len() <= 8);
    // No token may be the eos id (generation stops before emitting it).
    assert!(a.iter().all(|&t| t != HrmTextConfig::tiny_test().eos_token_id));
}

#[test]
fn greedy_generation_stops_at_eos() {
    let prompt = [3u32, 5, 7, 11];
    // Discover the first greedy token, then rebuild the same-weights model
    // with that token as eos: generation must stop immediately and return
    // nothing (eos excluded).
    let probe = tiny_model().generate_greedy(&prompt, 0, 1).unwrap();
    assert_eq!(probe.len(), 1);
    let first_token = probe[0];

    let mut config = HrmTextConfig::tiny_test();
    config.eos_token_id = first_token;
    let model = HrmTextModel::new_random(config, SEED).unwrap();
    let tokens = model.generate_greedy(&prompt, 0, 8).unwrap();
    assert!(tokens.is_empty(), "generation must stop at eos, got {:?}", tokens);
}

// ---------------------------------------------------------------------------
// PrefixLM effect
// ---------------------------------------------------------------------------

#[test]
fn prefixlm_mask_changes_logits() {
    let config = HrmTextConfig::tiny_test();
    let model = tiny_model();
    let vocab = config.vocab_size;
    let ids: Vec<u32> = vec![3, 17, 42, 99, 5, 64, 23, 81, 7, 120, 56, 11];
    let prefix_len = 6;

    let mut c1 = HrmKvCaches::new(&config);
    let causal = model
        .forward(&ids, AttentionMaskMode::Causal, &mut c1)
        .unwrap();
    let mut c2 = HrmKvCaches::new(&config);
    let prefix = model
        .forward(&ids, AttentionMaskMode::PrefixLm { prefix_len }, &mut c2)
        .unwrap();

    let row_diff = |pos: usize| {
        causal[pos * vocab..(pos + 1) * vocab]
            .iter()
            .zip(prefix[pos * vocab..(pos + 1) * vocab].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    };

    // Inside the prefix, position 0 now sees positions 1..6 bidirectionally.
    assert!(row_diff(0) > 1e-4, "prefix-region logits must differ");
    // Beyond the prefix, logits differ too: the prefix tokens' K/V (deeper
    // cycles) were computed under a different mask.
    assert!(
        row_diff(ids.len() - 1) > 1e-5,
        "prefix-conditioned region logits must differ"
    );
    // Position 0 attending only to itself is identical under both masks at
    // the FIRST L-cycle, so the difference must come from the mask — verify
    // it's not NaN garbage.
    assert!(causal.iter().chain(prefix.iter()).all(|v| v.is_finite()));
}
