//! Output parsers for kernel-mode generations (ADR-199 Layer 2).
//!
//! A pre-alignment model emits loosely formatted text; these parsers are
//! deliberately tolerant: `Verdict:` -> structured pass/fail and `JSON:` ->
//! serde-validated value with one retry on parse failure.

use crate::backend::LlmBackend;
use crate::prompt::PromptBuilder;
use crate::verifier::Verdict;

/// Parse a Verify-mode generation into a [`Verdict`].
///
/// Tolerant grammar:
/// - first line containing the word `PASS` or `FAIL` (case-insensitive,
///   word-boundary aware, optionally prefixed `Verdict:`) sets the result;
/// - a `Contradictions:`/`Missing facts:` heading opens a section; bullet
///   lines (`-`, `*`, `1.`, `1)`) below it are collected;
/// - inline items after the heading colon are split on `;`;
/// - `none`/`n/a`/`(none)` items are dropped.
///
/// Unparseable text yields a failing verdict (safe default for a gate).
pub fn parse_verdict(text: &str) -> Verdict {
    #[derive(PartialEq, Clone, Copy)]
    enum Section {
        None,
        Contradictions,
        Missing,
    }
    let mut pass: Option<bool> = None;
    let mut contradictions = Vec::new();
    let mut missing_facts = Vec::new();
    let mut section = Section::None;

    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let lower = line.to_lowercase();
        if pass.is_none() {
            if contains_word(&lower, "fail") {
                pass = Some(false);
            } else if contains_word(&lower, "pass") {
                pass = Some(true);
            }
        }
        if let Some(rest) = strip_heading(&lower, line, "contradiction") {
            section = Section::Contradictions;
            push_inline(&mut contradictions, rest);
            continue;
        }
        if let Some(rest) = strip_heading(&lower, line, "missing") {
            section = Section::Missing;
            push_inline(&mut missing_facts, rest);
            continue;
        }
        if let Some(item) = bullet_item(line) {
            match section {
                Section::Contradictions => push_item(&mut contradictions, item),
                Section::Missing => push_item(&mut missing_facts, item),
                Section::None => {}
            }
        }
    }

    Verdict {
        pass: pass.unwrap_or(false),
        contradictions,
        missing_facts,
    }
}

/// Word-boundary containment check on a lowercased haystack.
fn contains_word(haystack: &str, word: &str) -> bool {
    haystack.split(|c: char| !c.is_ascii_alphanumeric()).any(|w| w == word)
}

/// If `lower` starts with `key` (a section heading), return the original-case
/// text after the first `:` (may be empty).
fn strip_heading<'a>(lower: &str, original: &'a str, key: &str) -> Option<&'a str> {
    if !lower.starts_with(key) {
        return None;
    }
    original.split_once(':').map(|(_, rest)| rest)
}

fn push_inline(items: &mut Vec<String>, inline: &str) {
    for part in inline.split(';') {
        push_item(items, part);
    }
}

fn push_item(items: &mut Vec<String>, item: &str) {
    let cleaned = item.trim().trim_end_matches('.').trim();
    let lower = cleaned.to_lowercase();
    if cleaned.is_empty() || lower == "none" || lower == "(none)" || lower == "n/a" {
        return;
    }
    items.push(cleaned.to_string());
}

/// Strip a leading bullet marker (`-`, `*`, `•`, `1.`, `1)`) and return the
/// item text, or `None` if the line is not a bullet.
fn bullet_item(line: &str) -> Option<&str> {
    for marker in ["- ", "* ", "• "] {
        if let Some(rest) = line.strip_prefix(marker) {
            return Some(rest.trim());
        }
    }
    let digits = line.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits > 0 {
        let rest = &line[digits..];
        if let Some(item) = rest.strip_prefix('.').or_else(|| rest.strip_prefix(')')) {
            return Some(item.trim());
        }
    }
    None
}

/// Parse strict JSON out of a model generation: strips Markdown code fences
/// and any preamble/epilogue, then serde-validates the first balanced JSON
/// object or array.
pub fn parse_strict_json(text: &str) -> anyhow::Result<serde_json::Value> {
    let candidate = strip_fences(text);
    let trimmed = candidate.trim();
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
        return Ok(v);
    }
    if let Some(slice) = first_balanced_json(trimmed) {
        let v = serde_json::from_str::<serde_json::Value>(slice)?;
        return Ok(v);
    }
    anyhow::bail!("no valid JSON value found in model output ({} bytes)", text.len());
}

/// If the text contains a ```-fenced block, return the inside of the first
/// fence (dropping an optional language tag); otherwise return the text.
fn strip_fences(text: &str) -> &str {
    let Some(open) = text.find("```") else {
        return text;
    };
    let after = &text[open + 3..];
    // Skip an optional language tag up to end-of-line.
    let body_start = after.find('\n').map(|i| i + 1).unwrap_or(0);
    let body = &after[body_start..];
    match body.find("```") {
        Some(close) => &body[..close],
        None => body,
    }
}

/// Find the first balanced `{...}` or `[...]` slice, respecting strings.
fn first_balanced_json(text: &str) -> Option<&str> {
    let start = text.find(['{', '['])?;
    let bytes = text.as_bytes();
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(&text[start..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Extract-mode helper with one corrective retry (per ADR-199: "serde-
/// validated value with one retry on parse failure").
pub async fn extract_json_with_retry(
    backend: &dyn LlmBackend,
    builder: &PromptBuilder,
    input: &str,
) -> anyhow::Result<serde_json::Value> {
    let resp = backend.generate(builder.to_request(input)).await?;
    match parse_strict_json(&resp.text) {
        Ok(v) => Ok(v),
        Err(first_err) => {
            tracing::debug!(error = %first_err, "JSON parse failed, re-asking once");
            let retry_input = format!(
                "{input}\n\nThe previous output was not valid JSON ({first_err}). \
                 Return strict JSON only, with no prose before or after it."
            );
            let resp = backend.generate(builder.to_request(&retry_input)).await?;
            parse_strict_json(&resp.text)
        }
    }
}

/// Extract the last signed integer appearing in `text` (answer
/// normalization for math-style tasks). Returns `None` if there is none.
pub(crate) fn extract_final_int(text: &str) -> Option<i64> {
    let bytes = text.as_bytes();
    let mut last: Option<i64> = None;
    let mut i = 0;
    while i < bytes.len() {
        let negative = bytes[i] == b'-'
            && i + 1 < bytes.len()
            && bytes[i + 1].is_ascii_digit()
            // Treat "3-2" as two numbers, "-2" standalone as negative.
            && (i == 0 || !bytes[i - 1].is_ascii_digit());
        let start = if negative { i + 1 } else { i };
        if start < bytes.len() && bytes[start].is_ascii_digit() {
            let mut end = start;
            while end < bytes.len() && bytes[end].is_ascii_digit() {
                end += 1;
            }
            if let Ok(mut n) = text[start..end].parse::<i64>() {
                if negative {
                    n = -n;
                }
                last = Some(n);
            }
            i = end;
        } else {
            i += 1;
        }
    }
    last
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdict_plain_pass() {
        let v = parse_verdict("PASS");
        assert!(v.pass);
        assert!(v.contradictions.is_empty() && v.missing_facts.is_empty());
    }

    #[test]
    fn verdict_prefixed_and_lowercase() {
        assert!(parse_verdict("Verdict: pass").pass);
        assert!(!parse_verdict("verdict: FAIL.").pass);
    }

    #[test]
    fn verdict_word_boundaries() {
        // "passage" / "failure-free" must not trigger detection by themselves.
        let v = parse_verdict("the passage looks fine\nPASS");
        assert!(v.pass);
        assert!(!parse_verdict("a passage about compasses").pass); // no verdict => fail-safe
    }

    #[test]
    fn verdict_sections_collected() {
        let v = parse_verdict(
            "FAIL\nContradictions:\n- claim conflicts with evidence: X\n* second item\nMissing facts:\n1. deployment date\n2) owner name",
        );
        assert!(!v.pass);
        assert_eq!(v.contradictions.len(), 2);
        assert_eq!(v.missing_facts, vec!["deployment date", "owner name"]);
    }

    #[test]
    fn verdict_inline_items_and_none_dropped() {
        let v = parse_verdict("FAIL\nContradictions: a; b; none\nMissing facts: (none)");
        assert_eq!(v.contradictions, vec!["a", "b"]);
        assert!(v.missing_facts.is_empty());
    }

    #[test]
    fn verdict_empty_is_fail_safe() {
        assert!(!parse_verdict("").pass);
        assert!(!parse_verdict("   \n\t").pass);
    }

    #[test]
    fn json_plain_object() {
        let v = parse_strict_json(r#"{"a": 1, "b": "two"}"#).unwrap();
        assert_eq!(v["a"], 1);
    }

    #[test]
    fn json_with_preamble_and_epilogue() {
        let v = parse_strict_json("Sure! Here is the JSON: {\"a\":[1,2]} hope it helps").unwrap();
        assert_eq!(v["a"][1], 2);
    }

    #[test]
    fn json_fenced_block() {
        let v = parse_strict_json("```json\n{\"k\": true}\n```").unwrap();
        assert_eq!(v["k"], true);
    }

    #[test]
    fn json_braces_inside_strings() {
        let v = parse_strict_json(r#"out: {"s": "h{i}", "n": 1} tail }"#).unwrap();
        assert_eq!(v["s"], "h{i}");
    }

    #[test]
    fn json_array_top_level() {
        let v = parse_strict_json("[1, 2, 3]").unwrap();
        assert_eq!(v[2], 3);
    }

    #[test]
    fn json_adversarial_rejected() {
        assert!(parse_strict_json("no json here").is_err());
        assert!(parse_strict_json("{ truncated: ").is_err());
        assert!(parse_strict_json("").is_err());
    }

    #[test]
    fn final_int_extraction() {
        assert_eq!(extract_final_int("Answer: 42."), Some(42));
        assert_eq!(extract_final_int("3 - 2 = 1"), Some(1));
        assert_eq!(extract_final_int("temp is -5 degrees"), Some(-5));
        assert_eq!(extract_final_int("no digits"), None);
    }
}
