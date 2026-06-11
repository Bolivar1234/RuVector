//! Phase-2 integration tests for the HRM-Text native backend (ADR-199):
//!
//! - safetensors weight I/O (Phase D item 4): bitwise round-trip, shape
//!   validation, missing/extra tensor reporting
//! - R2 self-speculative decoding: output ALWAYS identical to full-cycle
//!   greedy, acceptance accounting, determinism
//! - R3 latent state snapshot + warm start: shapes, fingerprint guard,
//!   warm-vs-cold effect, `embed_state`, determinism
//! - `decode_chunk` bitwise parity with sequential `decode_step`s (the
//!   property the R2 correctness invariant rests on)

use ruvllm::backends::hrm_text::{
    config_fingerprint, load_safetensors, save_safetensors, AttentionMaskMode, HrmCycles,
    HrmKvCaches, HrmStateSnapshot, HrmTextConfig, HrmTextModel, SelfSpeculativeDecoder,
};

const SEED: u64 = 42;
const PROMPT: [u32; 4] = [3, 5, 7, 11];

fn tiny_model() -> HrmTextModel {
    HrmTextModel::new_random(HrmTextConfig::tiny_test(), SEED).expect("tiny model builds")
}

fn assert_bitwise_eq(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{what}: bitwise mismatch at index {i}: {x} vs {y}"
        );
    }
}

// ---------------------------------------------------------------------------
// Weight I/O (Phase D item 4)
// ---------------------------------------------------------------------------

#[test]
fn safetensors_round_trip_is_bitwise_identical() {
    let config = HrmTextConfig::tiny_test();
    let model = tiny_model();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.safetensors");

    save_safetensors(&model, &path).unwrap();
    let loaded = load_safetensors(&config, &path).unwrap();

    // Weight buffers are bitwise identical...
    assert_bitwise_eq(&loaded.z_l_init, &model.z_l_init, "z_l_init");
    assert_bitwise_eq(
        &loaded.embed_tokens.weight,
        &model.embed_tokens.weight,
        "embed_tokens.weight",
    );
    assert_bitwise_eq(&loaded.lm_head.w, &model.lm_head.w, "lm_head.weight");

    // ...and so are forward logits on a fixed token sequence, under both
    // causal and PrefixLM masks.
    let ids: Vec<u32> = vec![3, 17, 42, 99, 5, 64, 23, 81];
    for mask in [
        AttentionMaskMode::Causal,
        AttentionMaskMode::PrefixLm { prefix_len: 4 },
    ] {
        let mut c1 = HrmKvCaches::new(&config);
        let mut c2 = HrmKvCaches::new(&config);
        let original = model.forward(&ids, mask, &mut c1).unwrap();
        let reloaded = loaded.forward(&ids, mask, &mut c2).unwrap();
        assert_bitwise_eq(&original, &reloaded, "round-trip forward logits");
    }
}

#[test]
fn safetensors_load_rejects_shape_mismatch() {
    let model = tiny_model();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.safetensors");
    save_safetensors(&model, &path).unwrap();

    // Same architecture but a different vocab: every vocab-shaped tensor
    // mismatches; the loader must name the offending tensor.
    let mut wrong = HrmTextConfig::tiny_test();
    wrong.vocab_size = 64;
    let err = load_safetensors(&wrong, &path).unwrap_err().to_string();
    assert!(err.contains("shape"), "got: {err}");
    assert!(
        err.contains("embed_tokens.weight") || err.contains("lm_head.weight"),
        "error must name the tensor, got: {err}"
    );
}

#[test]
fn safetensors_load_reports_missing_and_extra_tensors() {
    let model = tiny_model();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.safetensors");
    save_safetensors(&model, &path).unwrap();

    // A 4-layer config expects h_level/l_level.layers.1.* which the 2-layer
    // file does not contain -> reported missing by name.
    let mut bigger = HrmTextConfig::tiny_test();
    bigger.num_hidden_layers = 4;
    bigger.half_layers = 2;
    let err = load_safetensors(&bigger, &path).unwrap_err().to_string();
    assert!(err.contains("missing"), "got: {err}");
    assert!(err.contains("layers.1"), "must name a missing tensor, got: {err}");
}

// ---------------------------------------------------------------------------
// decode_chunk parity (foundation of the R2 invariant)
// ---------------------------------------------------------------------------

#[test]
fn decode_chunk_is_bitwise_identical_to_sequential_steps() {
    let config = HrmTextConfig::tiny_test();
    let model = tiny_model();
    let prompt = [3u32, 17, 42, 99];
    let tail = [5u32, 64, 23];

    let mut step_caches = HrmKvCaches::new(&config);
    model.prefill(&prompt, 2, &mut step_caches).unwrap();
    let mut step_logits = Vec::new();
    for (j, &t) in tail.iter().enumerate() {
        step_logits.extend(model.decode_step(t, prompt.len() + j, &mut step_caches).unwrap());
    }

    let mut chunk_caches = HrmKvCaches::new(&config);
    model.prefill(&prompt, 2, &mut chunk_caches).unwrap();
    let chunk_logits = model
        .decode_chunk(&tail, prompt.len(), &mut chunk_caches)
        .unwrap();

    assert_bitwise_eq(&chunk_logits, &step_logits, "decode_chunk vs decode_step");
    assert_eq!(
        step_caches.cached_len().unwrap(),
        chunk_caches.cached_len().unwrap()
    );
}

// ---------------------------------------------------------------------------
// R2 — self-speculative decoding
// ---------------------------------------------------------------------------

#[test]
fn spec_equal_cycles_full_acceptance_and_identical_output() {
    let config = HrmTextConfig::tiny_test(); // 2x2
    let reference = tiny_model().generate_greedy(&PROMPT, 2, 16).unwrap();

    let mut model = HrmTextModel::new_random(config, SEED).unwrap();
    let cycles = HrmCycles::new(2, 2);
    let mut dec = SelfSpeculativeDecoder::new(&mut model, cycles, cycles).unwrap();
    let (tokens, stats) = dec.generate_greedy(&PROMPT, 2, 16).unwrap();

    assert_eq!(tokens, reference, "draft==verify must reproduce greedy exactly");
    assert!(stats.drafted > 0);
    assert_eq!(stats.accepted, stats.drafted, "equal cycles: every draft accepted");
    assert_eq!(stats.acceptance_rate, 1.0);
}

#[test]
fn spec_draft_1x1_verify_2x2_matches_full_cycle_greedy() {
    // CORRECTNESS INVARIANT: output is identical to generate_greedy at
    // verify cycles regardless of the draft setting.
    let config = HrmTextConfig::tiny_test(); // 2x2 == verify setting
    let reference = tiny_model().generate_greedy(&PROMPT, 2, 24).unwrap();
    assert!(!reference.is_empty());

    let mut model = HrmTextModel::new_random(config, SEED).unwrap();
    let mut dec =
        SelfSpeculativeDecoder::new(&mut model, HrmCycles::new(1, 1), HrmCycles::new(2, 2))
            .unwrap();
    let (tokens, stats) = dec.generate_greedy(&PROMPT, 2, 24).unwrap();

    assert_eq!(
        tokens, reference,
        "1x1-draft output must equal 2x2 greedy: {tokens:?} vs {reference:?}"
    );
    assert!(stats.drafted > 0);
    assert!(
        stats.acceptance_rate > 0.0 && stats.acceptance_rate <= 1.0,
        "acceptance_rate must be in (0, 1], got {} ({}/{} drafts)",
        stats.acceptance_rate,
        stats.accepted,
        stats.drafted
    );
    assert!(stats.verify_forwards > 0 && stats.draft_forwards > 0);
    println!(
        "self-speculative 1x1 draft vs 2x2 verify: acceptance_rate = {:.3} \
         ({} accepted / {} drafted, {} draft fwds, {} verify fwds, {} tokens)",
        stats.acceptance_rate,
        stats.accepted,
        stats.drafted,
        stats.draft_forwards,
        stats.verify_forwards,
        tokens.len()
    );
}

#[test]
fn spec_invariant_holds_across_draft_settings_and_k() {
    // Sweep draft settings and draft lengths: the output never changes.
    let reference = tiny_model().generate_greedy(&PROMPT, 0, 20).unwrap();
    for (dh, dl) in [(1, 1), (1, 2), (3, 3), (2, 2)] {
        for k in [1usize, 2, 4, 8] {
            let mut model = tiny_model();
            let mut dec = SelfSpeculativeDecoder::new(
                &mut model,
                HrmCycles::new(dh, dl),
                HrmCycles::new(2, 2),
            )
            .unwrap()
            .with_draft_tokens(k)
            .unwrap();
            let (tokens, _stats) = dec.generate_greedy(&PROMPT, 0, 20).unwrap();
            assert_eq!(
                tokens, reference,
                "invariant violated for draft {dh}x{dl}, k={k}"
            );
        }
    }
}

#[test]
fn spec_is_deterministic_across_runs() {
    let run = || {
        let mut model = tiny_model();
        let mut dec = SelfSpeculativeDecoder::new(
            &mut model,
            HrmCycles::new(1, 1),
            HrmCycles::new(2, 2),
        )
        .unwrap();
        dec.generate_greedy(&PROMPT, 2, 16).unwrap()
    };
    let (tokens_a, stats_a) = run();
    let (tokens_b, stats_b) = run();
    assert_eq!(tokens_a, tokens_b);
    assert_eq!(stats_a, stats_b, "stats must be deterministic too");
}

#[test]
fn spec_stops_at_eos_like_greedy() {
    // Make the first greedy token the eos id: both paths emit nothing.
    let probe = tiny_model().generate_greedy(&PROMPT, 0, 1).unwrap();
    let mut config = HrmTextConfig::tiny_test();
    config.eos_token_id = probe[0];
    let mut model = HrmTextModel::new_random(config, SEED).unwrap();
    let reference = model.generate_greedy(&PROMPT, 0, 8).unwrap();
    assert!(reference.is_empty());
    let mut dec =
        SelfSpeculativeDecoder::new(&mut model, HrmCycles::new(1, 1), HrmCycles::new(2, 2))
            .unwrap();
    let (tokens, _stats) = dec.generate_greedy(&PROMPT, 0, 8).unwrap();
    assert!(tokens.is_empty(), "must stop at eos, got {tokens:?}");
}

// ---------------------------------------------------------------------------
// R3 — latent state snapshot + warm start
// ---------------------------------------------------------------------------

#[test]
fn capture_state_shape_and_fingerprint() {
    let config = HrmTextConfig::tiny_test();
    let model = tiny_model();
    let ids = [3u32, 17, 42, 99, 5];
    let (logits, snap) = model
        .capture_state(&ids, AttentionMaskMode::PrefixLm { prefix_len: 3 })
        .unwrap();
    assert_eq!(logits.len(), ids.len() * config.vocab_size);
    assert_eq!(snap.seq_len, ids.len());
    assert_eq!(snap.hidden_size, config.hidden_size);
    assert_eq!(snap.z_h.len(), ids.len() * config.hidden_size);
    assert_eq!(snap.config_fingerprint, config_fingerprint(&config));
    assert!(snap.z_h.iter().all(|v| v.is_finite()));
    snap.validate().unwrap();
}

#[test]
fn warm_start_rejects_fingerprint_mismatch() {
    let model = tiny_model();
    let (_logits, snap) = model
        .capture_state(&[3, 5, 7], AttentionMaskMode::Causal)
        .unwrap();

    // Architecture-different model (rope_theta participates in the
    // fingerprint): warm start must be rejected.
    let mut other_config = HrmTextConfig::tiny_test();
    other_config.rope_theta = 50000.0;
    let other = HrmTextModel::new_random(other_config.clone(), SEED).unwrap();
    let mut caches = HrmKvCaches::new(&other_config);
    let err = other
        .prefill_warm(&[3, 5, 7], 0, &snap, &mut caches)
        .unwrap_err()
        .to_string();
    assert!(err.contains("fingerprint"), "got: {err}");

    // Same architecture accepts it — even at a different cycle override
    // (cycles are runtime knobs, excluded from the fingerprint).
    let mut same = tiny_model();
    same.cycle_override(1, 1).unwrap();
    let mut caches = HrmKvCaches::new(&same.config);
    same.prefill_warm(&[3, 5, 7], 0, &snap, &mut caches).unwrap();
}

#[test]
fn warm_start_changes_logits_vs_cold_and_is_deterministic() {
    let config = HrmTextConfig::tiny_test();
    let model = tiny_model();
    let (_logits, snap) = model
        .capture_state(&[9, 13, 21, 33], AttentionMaskMode::Causal)
        .unwrap();
    let ids = [3u32, 5, 7, 11];

    let mut cold_caches = HrmKvCaches::new(&config);
    let cold = model.prefill(&ids, 2, &mut cold_caches).unwrap();
    let mut warm_caches = HrmKvCaches::new(&config);
    let warm = model.prefill_warm(&ids, 2, &snap, &mut warm_caches).unwrap();

    assert_eq!(cold.len(), warm.len());
    let max_diff = cold
        .iter()
        .zip(warm.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff > 1e-4, "warm start must alter logits (max diff {max_diff})");
    assert!(warm.iter().all(|v| v.is_finite()));
    assert_eq!(warm_caches.cached_len().unwrap(), ids.len());

    let mut warm_caches2 = HrmKvCaches::new(&config);
    let warm2 = model.prefill_warm(&ids, 2, &snap, &mut warm_caches2).unwrap();
    assert_bitwise_eq(&warm, &warm2, "prefill_warm determinism");
}

#[test]
fn embed_state_is_hidden_size_pooled_and_deterministic() {
    let config = HrmTextConfig::tiny_test();
    let model = tiny_model();
    let ids = [3u32, 17, 42, 99];
    let a = model.embed_state(&ids).unwrap();
    let b = model.embed_state(&ids).unwrap();
    assert_eq!(a.len(), config.hidden_size);
    assert!(a.iter().all(|v| v.is_finite()));
    assert_bitwise_eq(&a, &b, "embed_state determinism");

    // Matches manual mean-pooling of a captured snapshot under the same
    // (full-bidirectional) mask.
    let (_logits, snap) = model
        .capture_state(&ids, AttentionMaskMode::PrefixLm { prefix_len: ids.len() })
        .unwrap();
    assert_bitwise_eq(&a, &snap.mean_pooled(), "embed_state vs pooled snapshot");

    // Different inputs embed differently.
    let c = model.embed_state(&[8u32, 16, 24, 32]).unwrap();
    let diff = a
        .iter()
        .zip(c.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(diff > 1e-5, "different inputs must embed differently");
}

#[test]
fn snapshot_used_for_warm_start_must_validate() {
    let model = tiny_model();
    let mut caches = HrmKvCaches::new(&model.config);
    let bogus = HrmStateSnapshot {
        z_h: vec![0.0; 7], // not seq_len * hidden_size
        seq_len: 2,
        hidden_size: 4,
        config_fingerprint: config_fingerprint(&model.config),
    };
    assert!(model.prefill_warm(&[3, 5], 0, &bogus, &mut caches).is_err());
}
