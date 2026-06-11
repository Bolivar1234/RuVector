//! PrefixLM-aware prompting for HRM-Text (ADR-199 Layer 2 — the critical part).
//!
//! Model-card facts this module encodes:
//!
//! - The checkpoint is **pre-alignment**: PrefixLM-pretrained, not
//!   instruction-tuned, dialogue-tuned, or RLHF'd. Raw chat prompting will
//!   look broken; callers select an [`HrmMode`], never write raw prompts.
//! - Condition prefix tags are baked into pretraining: `direct`, `cot`,
//!   `noisy`, `synth` — composable (e.g. `synth,cot`). NLP tasks should use
//!   `direct` + 2–8 few-shot examples; reasoning/math should use `synth,cot`.
//! - PrefixLM caveat: inference requires `token_type_ids = ones` so prompt
//!   tokens attend bidirectionally. Through an OpenAI-compatible API this is
//!   the server's job — see [`crate::backend::PrefixMode`] and the ADR-199
//!   parity gate.

use crate::backend::GenerateRequest;
use serde::{Deserialize, Serialize};

/// Kernel task modes. Each maps to a condition-prefix template from ADR-199.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HrmMode {
    /// NLP tasks: `direct` + 2–8 few-shot examples.
    Direct,
    /// Reasoning/math: `synth,cot` composite prefix.
    SynthCot,
    /// Contradiction / evidence checking.
    Verify,
    /// Task decomposition.
    Plan,
    /// Strict JSON output.
    Extract,
}

/// Build the condition-prefix prompt for `mode`. Templates are verbatim from
/// ADR-199 — tests assert exact equality, do not edit casually.
pub fn build_hrm_prompt(mode: HrmMode, input: &str) -> String {
    match mode {
        HrmMode::Direct => format!("direct\nTask: Extract the answer.\nInput:\n{input}\nOutput:"),
        HrmMode::SynthCot => format!("synth,cot\nSolve carefully.\nProblem:\n{input}\nAnswer:"),
        HrmMode::Verify => format!(
            "direct\nCheck for contradictions, missing evidence, and invalid assumptions.\n{input}\nVerdict:"
        ),
        HrmMode::Plan => {
            format!("synth,cot\nDecompose this into executable steps.\n{input}\nPlan:")
        }
        HrmMode::Extract => format!("direct\nReturn strict JSON only.\n{input}\nJSON:"),
    }
}

/// Recommended generation parameters for a mode (temperature 0.2 for kernel
/// tasks, `max_tokens <= 512`, per ADR-199).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateParams {
    pub max_tokens: u32,
    pub temperature: f32,
    pub stop: Vec<String>,
}

/// Per-mode stop sequences.
pub fn stop_sequences(mode: HrmMode) -> Vec<String> {
    match mode {
        // Plans are multi-line numbered lists; only a hard break stops them.
        HrmMode::Plan => vec!["\n\n\n".to_string()],
        _ => vec!["\n\n".to_string()],
    }
}

/// Recommended parameters for a mode.
pub fn recommended_params(mode: HrmMode) -> GenerateParams {
    let max_tokens = match mode {
        HrmMode::Direct | HrmMode::Verify => 256,
        HrmMode::SynthCot | HrmMode::Plan | HrmMode::Extract => 512,
    };
    GenerateParams {
        max_tokens,
        temperature: 0.2,
        stop: stop_sequences(mode),
    }
}

/// One few-shot example: a task input and its ideal output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FewShot {
    pub input: String,
    pub output: String,
}

impl FewShot {
    pub fn new(input: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
        }
    }
}

/// Maximum few-shot bank size (model card recommends 2–8 shots).
pub const MAX_SHOTS: usize = 8;

/// Builds full prompts: condition tag first, then the few-shot bank, then the
/// real task — keeping the condition tokens at the very start of the
/// sequence, where pretraining placed them.
#[derive(Debug, Clone)]
pub struct PromptBuilder {
    mode: HrmMode,
    shots: Vec<FewShot>,
}

impl PromptBuilder {
    /// New builder with the built-in default bank for `mode` (Extract and
    /// Verify ship small banks; other modes start empty).
    pub fn new(mode: HrmMode) -> Self {
        Self {
            mode,
            shots: default_bank(mode),
        }
    }

    /// New builder with no few-shot bank.
    pub fn bare(mode: HrmMode) -> Self {
        Self {
            mode,
            shots: Vec::new(),
        }
    }

    /// Replace the few-shot bank (at most [`MAX_SHOTS`] examples).
    pub fn with_shots(mut self, shots: Vec<FewShot>) -> anyhow::Result<Self> {
        if shots.len() > MAX_SHOTS {
            anyhow::bail!(
                "few-shot bank too large: {} > {MAX_SHOTS} (model card recommends 2-8 shots)",
                shots.len()
            );
        }
        self.shots = shots;
        Ok(self)
    }

    /// Append one shot, enforcing the [`MAX_SHOTS`] cap.
    pub fn push_shot(&mut self, shot: FewShot) -> anyhow::Result<()> {
        if self.shots.len() >= MAX_SHOTS {
            anyhow::bail!("few-shot bank full ({MAX_SHOTS} shots)");
        }
        self.shots.push(shot);
        Ok(())
    }

    pub fn mode(&self) -> HrmMode {
        self.mode
    }

    pub fn shots(&self) -> &[FewShot] {
        &self.shots
    }

    /// Recommended generation parameters for this builder's mode.
    pub fn params(&self) -> GenerateParams {
        recommended_params(self.mode)
    }

    /// Build the full prompt: `tag\n[shot-blocks...]rest-of-template`.
    ///
    /// Each shot is rendered with the same mode template (minus the
    /// condition tag, which appears exactly once, first) followed by its
    /// output and a blank line.
    pub fn build(&self, input: &str) -> String {
        let template = build_hrm_prompt(self.mode, input);
        if self.shots.is_empty() {
            return template;
        }
        let (tag, rest) = template
            .split_once('\n')
            .expect("HRM templates always contain a condition-tag line");
        let mut out = String::with_capacity(template.len() + 256 * self.shots.len());
        out.push_str(tag);
        out.push('\n');
        for shot in &self.shots {
            let shot_template = build_hrm_prompt(self.mode, &shot.input);
            let (_, shot_rest) = shot_template
                .split_once('\n')
                .expect("HRM templates always contain a condition-tag line");
            out.push_str(shot_rest);
            out.push(' ');
            out.push_str(&shot.output);
            out.push_str("\n\n");
        }
        out.push_str(rest);
        out
    }

    /// Build a ready-to-send [`GenerateRequest`] with recommended params.
    pub fn to_request(&self, input: &str) -> GenerateRequest {
        let params = self.params();
        GenerateRequest {
            prompt: self.build(input),
            max_tokens: params.max_tokens,
            temperature: params.temperature,
            stop: params.stop,
            hrm_cycles: None,
        }
    }
}

/// Built-in default few-shot banks. Small but real; callers can replace them
/// via [`PromptBuilder::with_shots`].
fn default_bank(mode: HrmMode) -> Vec<FewShot> {
    match mode {
        HrmMode::Extract => vec![
            FewShot::new(
                "Order 1042 was placed by Dana Reyes for 3 widgets.",
                r#"{"order_id":1042,"customer":"Dana Reyes","quantity":3}"#,
            ),
            FewShot::new(
                "Sensor th-7 reported 21.5 degrees at hour 14.",
                r#"{"sensor":"th-7","temperature":21.5,"hour":14}"#,
            ),
        ],
        HrmMode::Verify => vec![
            FewShot::new(
                "Claim: The disk is full.\nRules: (none)\nEvidence: The disk is full.",
                "PASS",
            ),
            FewShot::new(
                "Claim: The build passed.\nRules: (none)\nEvidence: The build did not pass.",
                "FAIL\nContradictions:\n- claim conflicts with evidence: The build did not pass.",
            ),
        ],
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Templates must match ADR-199 verbatim.
    #[test]
    fn templates_exactly_match_adr199() {
        assert_eq!(
            build_hrm_prompt(HrmMode::Direct, "x"),
            "direct\nTask: Extract the answer.\nInput:\nx\nOutput:"
        );
        assert_eq!(
            build_hrm_prompt(HrmMode::SynthCot, "x"),
            "synth,cot\nSolve carefully.\nProblem:\nx\nAnswer:"
        );
        assert_eq!(
            build_hrm_prompt(HrmMode::Verify, "x"),
            "direct\nCheck for contradictions, missing evidence, and invalid assumptions.\nx\nVerdict:"
        );
        assert_eq!(
            build_hrm_prompt(HrmMode::Plan, "x"),
            "synth,cot\nDecompose this into executable steps.\nx\nPlan:"
        );
        assert_eq!(
            build_hrm_prompt(HrmMode::Extract, "x"),
            "direct\nReturn strict JSON only.\nx\nJSON:"
        );
    }

    #[test]
    fn builder_keeps_condition_tag_first_and_once() {
        let b = PromptBuilder::new(HrmMode::Extract);
        let p = b.build("fields: a=1");
        assert!(p.starts_with("direct\n"));
        assert_eq!(p.matches("direct\n").count(), 1);
        // Two default shots + the real task.
        assert_eq!(p.matches("Return strict JSON only.").count(), 3);
        assert!(p.ends_with("fields: a=1\nJSON:"));
    }

    #[test]
    fn bare_builder_is_pure_template() {
        let b = PromptBuilder::bare(HrmMode::Plan);
        assert_eq!(b.build("ship it"), build_hrm_prompt(HrmMode::Plan, "ship it"));
    }

    #[test]
    fn shot_cap_enforced() {
        let shots = (0..9)
            .map(|i| FewShot::new(format!("i{i}"), format!("o{i}")))
            .collect::<Vec<_>>();
        assert!(PromptBuilder::bare(HrmMode::Direct).with_shots(shots).is_err());
        let mut b = PromptBuilder::bare(HrmMode::Direct);
        for i in 0..MAX_SHOTS {
            b.push_shot(FewShot::new(format!("i{i}"), "o")).unwrap();
        }
        assert!(b.push_shot(FewShot::new("over", "o")).is_err());
    }

    #[test]
    fn params_follow_kernel_guidance() {
        for mode in [
            HrmMode::Direct,
            HrmMode::SynthCot,
            HrmMode::Verify,
            HrmMode::Plan,
            HrmMode::Extract,
        ] {
            let p = recommended_params(mode);
            assert!((p.temperature - 0.2).abs() < f32::EPSILON);
            assert!(p.max_tokens <= 512);
            assert!(!p.stop.is_empty());
        }
        let req = PromptBuilder::new(HrmMode::Verify).to_request("Claim: x\nEvidence: x");
        assert_eq!(req.max_tokens, 256);
    }
}
