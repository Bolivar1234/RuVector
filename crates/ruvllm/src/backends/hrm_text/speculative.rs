//! Self-speculative decoding via cycle asymmetry (ADR-199 R2).
//!
//! A recurrent model can draft with ITSELF at a low cycle count and verify at
//! the full cycle count — same weights, same tokenizer, no separate draft
//! model. The decoder borrows one [`HrmTextModel`] and toggles
//! `cycle_override` between a cheap draft setting and the full verify
//! setting, keeping SEPARATE [`HrmKvCaches`] for the two passes.
//!
//! ## Loop
//!
//! 1. Draft `draft_tokens` (default 4) greedy tokens at `draft_cycles`.
//! 2. Verify them with ONE batched [`HrmTextModel::decode_chunk`] forward at
//!    `verify_cycles` over the drafted positions.
//! 3. Accept the longest prefix where `draft == argmax(verify logits)`; on
//!    the first mismatch emit the verify argmax token and restart drafting.
//!    Rejected positions are rolled back from both caches
//!    ([`HrmKvCaches::truncate`]).
//!
//! ## Correctness invariant
//!
//! Every emitted token is an argmax of verify-cycle logits over the exact
//! committed context, and batched causal chunks are bitwise identical to
//! sequential single-token decodes — so the output sequence is ALWAYS
//! identical to [`HrmTextModel::generate_greedy`] run at `verify_cycles`,
//! regardless of the draft setting. Draft quality only affects speed
//! (acceptance rate), never output.
//!
//! Research gate (ADR-199): > 50 % acceptance, else the separate-draft-model
//! path wins and this track closes.

use serde::{Deserialize, Serialize};

use super::cache::HrmKvCaches;
use super::generate::{argmax, HrmTextModel};
use crate::error::{Result, RuvLLMError};

/// Default number of tokens drafted per round.
pub const DEFAULT_DRAFT_TOKENS: usize = 4;

/// A recurrence-depth setting `(h_cycles, l_cycles)` — the first-class,
/// routable compute knob of ADR-199 (R1), used here as the R2 draft/verify
/// asymmetry. Mirrors `cycle_override(h, l)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HrmCycles {
    /// Slow H-level cycles.
    pub h_cycles: usize,
    /// Fast L-level cycles per H cycle.
    pub l_cycles: usize,
}

impl HrmCycles {
    /// Construct a cycle setting.
    pub fn new(h_cycles: usize, l_cycles: usize) -> Self {
        Self { h_cycles, l_cycles }
    }
}

/// Statistics for one self-speculative generation run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct SpecStats {
    /// Draft tokens proposed.
    pub drafted: usize,
    /// Draft tokens accepted by verification.
    pub accepted: usize,
    /// `accepted / drafted` (0.0 when nothing was drafted).
    pub acceptance_rate: f32,
    /// Draft-pass model invocations (prefill + chunks + single steps).
    pub draft_forwards: usize,
    /// Verify-pass model invocations (prefill + batched chunks).
    pub verify_forwards: usize,
}

/// Greedy self-speculative decoder over a single borrowed [`HrmTextModel`].
#[derive(Debug)]
pub struct SelfSpeculativeDecoder<'m> {
    model: &'m mut HrmTextModel,
    /// Cheap drafting recurrence depth.
    pub draft_cycles: HrmCycles,
    /// Full verification recurrence depth (defines the output distribution).
    pub verify_cycles: HrmCycles,
    /// Tokens drafted per round (`>= 1`).
    pub draft_tokens: usize,
}

impl<'m> SelfSpeculativeDecoder<'m> {
    /// Create a decoder. Both cycle settings must be `>= 1x1`.
    pub fn new(
        model: &'m mut HrmTextModel,
        draft_cycles: HrmCycles,
        verify_cycles: HrmCycles,
    ) -> Result<Self> {
        for (name, c) in [("draft", draft_cycles), ("verify", verify_cycles)] {
            if c.h_cycles == 0 || c.l_cycles == 0 {
                return Err(RuvLLMError::Config(format!(
                    "{name}_cycles must be >= 1x1, got {}x{}",
                    c.h_cycles, c.l_cycles
                )));
            }
        }
        Ok(Self {
            model,
            draft_cycles,
            verify_cycles,
            draft_tokens: DEFAULT_DRAFT_TOKENS,
        })
    }

    /// Set the number of tokens drafted per round (builder style).
    pub fn with_draft_tokens(mut self, draft_tokens: usize) -> Result<Self> {
        if draft_tokens == 0 {
            return Err(RuvLLMError::Config(
                "draft_tokens must be >= 1".to_string(),
            ));
        }
        self.draft_tokens = draft_tokens;
        Ok(self)
    }

    /// Greedy self-speculative generation. Semantics (prompt, `prefix_len`
    /// PrefixLM positions, eos, `max_new_tokens`) match
    /// [`HrmTextModel::generate_greedy`] at `verify_cycles` exactly; returns
    /// the generated tokens and run statistics. The model's original cycle
    /// configuration is restored before returning (also on error).
    pub fn generate_greedy(
        &mut self,
        prompt_ids: &[u32],
        prefix_len: usize,
        max_new_tokens: usize,
    ) -> Result<(Vec<u32>, SpecStats)> {
        let orig = HrmCycles::new(self.model.config.h_cycles, self.model.config.l_cycles);
        let result = self.run(prompt_ids, prefix_len, max_new_tokens);
        self.model.config.h_cycles = orig.h_cycles;
        self.model.config.l_cycles = orig.l_cycles;
        result
    }

    fn set_cycles(&mut self, c: HrmCycles) -> Result<()> {
        self.model.cycle_override(c.h_cycles, c.l_cycles)
    }

    fn run(
        &mut self,
        prompt_ids: &[u32],
        prefix_len: usize,
        max_new_tokens: usize,
    ) -> Result<(Vec<u32>, SpecStats)> {
        let mut stats = SpecStats::default();
        if max_new_tokens == 0 {
            return Ok((Vec::new(), stats));
        }
        let eos = self.model.config.eos_token_id;
        let max_pos = self.model.config.max_position_embeddings;
        let vocab = self.model.config.vocab_size;
        let prompt_len = prompt_ids.len();

        // Verify prefill: produces the first verified token.
        self.set_cycles(self.verify_cycles)?;
        let mut vcaches = HrmKvCaches::new(&self.model.config);
        let first_logits = self.model.prefill(prompt_ids, prefix_len, &mut vcaches)?;
        stats.verify_forwards += 1;
        let first = argmax(&first_logits)?;
        if first == eos {
            return Ok((Vec::new(), stats));
        }
        // `seq` = committed tokens (prompt + emitted). Invariant at each
        // round top: vcaches hold seq[..len-1]; the last committed token is
        // pending (not yet fed to the verify pass).
        let mut seq: Vec<u32> = prompt_ids.to_vec();
        seq.push(first);
        if seq.len() - prompt_len >= max_new_tokens {
            return Ok((seq.split_off(prompt_len), stats));
        }

        // Draft prefill (same PrefixLM prompt handling, draft cycles).
        self.set_cycles(self.draft_cycles)?;
        let mut dcaches = HrmKvCaches::new(&self.model.config);
        self.model.prefill(prompt_ids, prefix_len, &mut dcaches)?;
        stats.draft_forwards += 1;

        loop {
            let cur = seq.len();
            let pend_pos = cur - 1; // position of the pending committed token
            if pend_pos >= max_pos {
                break; // mirrors generate_greedy's `pos >= max_pos` stop
            }
            let remaining = max_new_tokens - (cur - prompt_len);
            debug_assert!(remaining >= 1);
            let k = self.draft_tokens.min(remaining).min(max_pos - pend_pos).max(1);

            // --- Draft phase (draft cycles, draft caches) ---
            self.set_cycles(self.draft_cycles)?;
            let dlen = dcaches.cached_len()?;
            debug_assert!(dlen < cur);
            let sync_logits = self.model.decode_chunk(&seq[dlen..cur], dlen, &mut dcaches)?;
            stats.draft_forwards += 1;
            let mut last_row =
                sync_logits[sync_logits.len() - vocab..].to_vec();
            let mut drafts: Vec<u32> = Vec::with_capacity(k);
            loop {
                let d = argmax(&last_row)?;
                drafts.push(d);
                if drafts.len() == k || d == eos {
                    break; // drafting past eos is pointless (verify decides)
                }
                last_row = self
                    .model
                    .decode_step(d, cur + drafts.len() - 1, &mut dcaches)?;
                stats.draft_forwards += 1;
            }
            let k_used = drafts.len();
            stats.drafted += k_used;

            // --- Verify phase (verify cycles, verify caches) ---
            // One batched forward over [pending, d1..d_{k-1}]: row j checks
            // draft j+1... i.e. row j's argmax is the greedy token after
            // committed context + drafts[..j].
            self.set_cycles(self.verify_cycles)?;
            let mut chunk: Vec<u32> = Vec::with_capacity(k_used);
            chunk.push(seq[pend_pos]);
            chunk.extend_from_slice(&drafts[..k_used - 1]);
            let vlogits = self.model.decode_chunk(&chunk, pend_pos, &mut vcaches)?;
            stats.verify_forwards += 1;

            let mut stop = false;
            let mut mismatched = false;
            for (j, &d) in drafts.iter().enumerate() {
                let v = argmax(&vlogits[j * vocab..(j + 1) * vocab])?;
                if v == eos {
                    stop = true;
                    break;
                }
                if v == d {
                    seq.push(v);
                    stats.accepted += 1;
                } else {
                    seq.push(v); // the corrected (verified) token
                    mismatched = true;
                }
                if seq.len() - prompt_len >= max_new_tokens {
                    stop = true;
                }
                if stop || mismatched {
                    break;
                }
            }
            if stop {
                break;
            }
            // Roll back speculative cache rows beyond the committed sequence:
            // verify holds all but the pending last token; draft holds every
            // committed token it has consumed.
            vcaches.truncate(seq.len() - 1);
            dcaches.truncate(seq.len() - 1);
        }
        stats.acceptance_rate = if stats.drafted > 0 {
            stats.accepted as f32 / stats.drafted as f32
        } else {
            0.0
        };
        Ok((seq.split_off(prompt_len), stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::config::HrmTextConfig;

    #[test]
    fn test_rejects_zero_cycles_and_zero_draft_tokens() {
        let config = HrmTextConfig::tiny_test();
        let mut model = HrmTextModel::new_random(config, 42).unwrap();
        assert!(SelfSpeculativeDecoder::new(
            &mut model,
            HrmCycles::new(0, 1),
            HrmCycles::new(2, 2)
        )
        .is_err());
        assert!(SelfSpeculativeDecoder::new(
            &mut model,
            HrmCycles::new(1, 1),
            HrmCycles::new(2, 0)
        )
        .is_err());
        let dec =
            SelfSpeculativeDecoder::new(&mut model, HrmCycles::new(1, 1), HrmCycles::new(2, 2))
                .unwrap();
        assert!(dec.with_draft_tokens(0).is_err());
    }

    #[test]
    fn test_max_new_zero_and_cycle_restore() {
        let config = HrmTextConfig::tiny_test(); // 2x2
        let mut model = HrmTextModel::new_random(config, 42).unwrap();
        let mut dec =
            SelfSpeculativeDecoder::new(&mut model, HrmCycles::new(1, 1), HrmCycles::new(3, 3))
                .unwrap();
        let (tokens, stats) = dec.generate_greedy(&[3, 5, 7], 0, 0).unwrap();
        assert!(tokens.is_empty());
        assert_eq!(stats.drafted, 0);
        let (_tokens, _stats) = dec.generate_greedy(&[3, 5, 7], 0, 3).unwrap();
        // Original 2x2 cycle configuration is restored after the run.
        assert_eq!(model.config.h_cycles, 2);
        assert_eq!(model.config.l_cycles, 2);
    }
}
