//! Safetensors weight I/O for the HRM-Text native backend
//! (ADR-199 Phase D item 4).
//!
//! Implements the safetensors v0.x container by hand — no new dependencies:
//!
//! ```text
//! [u64 LE header length N] [N bytes JSON header] [raw tensor data]
//! header = { "name": { "dtype": "F32", "shape": [..], "data_offsets": [b, e] }, ... }
//! ```
//!
//! Only `F32` tensors are supported; any other dtype is rejected with a clear
//! error. Data offsets are relative to the start of the data section.
//!
//! ## Canonical tensor names
//!
//! | Name | Shape |
//! |------|-------|
//! | `embed_tokens.weight` | `[vocab_size, hidden_size]` |
//! | `z_l_init` | `[hidden_size]` |
//! | `{h_level,l_level}.layers.{i}.attn.{q,k,v}_proj.weight` | `[heads*head_dim, hidden]` / `[kv_dim, hidden]` |
//! | `{h_level,l_level}.layers.{i}.attn.o_proj.weight` | `[hidden, heads*head_dim]` |
//! | `{h_level,l_level}.layers.{i}.mlp.{gate,up}_proj.weight` | `[intermediate, hidden]` |
//! | `{h_level,l_level}.layers.{i}.mlp.down_proj.weight` | `[hidden, intermediate]` |
//! | `{h_level,l_level}.layers.{i}.attn_norm.weight` | `[hidden_size]` |
//! | `{h_level,l_level}.layers.{i}.mlp_norm.weight` | `[hidden_size]` |
//! | `final_norm.weight` | `[hidden_size]` |
//! | `lm_head.weight` | `[vocab_size, hidden_size]` |
//!
//! An alias table maps plausible upstream HuggingFace-export names (e.g.
//! `model.embed_tokens.weight`, `model.H_level.layers.{i}.self_attn.q_proj.weight`)
//! onto the canonical set — see [`canonicalize_name`].
//!
//! TODO(ADR-199): verify the alias table against the actual upstream
//! `conversion/convert_to_hf.py` output once real HRM-Text-1B weights are
//! available; the upstream naming convention is currently an educated guess.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::Path;

use serde_json::{json, Value};

use super::config::HrmTextConfig;
use super::core::HrmDecoderLayer;
use super::generate::HrmTextModel;
use crate::error::{Result, RuvLLMError};

const DTYPE_F32: &str = "F32";
const F32_BYTES: usize = std::mem::size_of::<f32>();

// ---------------------------------------------------------------------------
// Name canonicalization (upstream alias table)
// ---------------------------------------------------------------------------

/// Map a tensor name from a safetensors file onto OUR canonical name.
///
/// Canonical names pass through unchanged. The alias rewrites cover the
/// plausible upstream HuggingFace-export convention:
///
/// - strip a leading `model.`
/// - `H_level.` / `L_level.` -> `h_level.` / `l_level.`
/// - `.self_attn.` -> `.attn.`
/// - `.input_layernorm.weight` -> `.attn_norm.weight`
/// - `.post_attention_layernorm.weight` -> `.mlp_norm.weight`
/// - top-level `norm.weight` -> `final_norm.weight`
/// - `z_L_init` / `z_l_init.weight` / `l_init` -> `z_l_init`
///
/// TODO(ADR-199): verify against upstream `convert_to_hf.py` output once
/// real weights are available.
fn canonicalize_name(raw: &str) -> String {
    let mut name = raw.strip_prefix("model.").unwrap_or(raw).to_string();
    name = name
        .replace("H_level.", "h_level.")
        .replace("L_level.", "l_level.")
        .replace(".self_attn.", ".attn.")
        .replace(".input_layernorm.weight", ".attn_norm.weight")
        .replace(".post_attention_layernorm.weight", ".mlp_norm.weight");
    match name.as_str() {
        "norm.weight" => "final_norm.weight".to_string(),
        "z_L_init" | "z_l_init.weight" | "l_init" => "z_l_init".to_string(),
        _ => name,
    }
}

// ---------------------------------------------------------------------------
// Canonical tensor enumeration (read-only and mutable views over the model)
// ---------------------------------------------------------------------------

/// Per-layer canonical tensors as `(suffix, shape, data)` triples.
fn layer_tensors(layer: &HrmDecoderLayer) -> Vec<(&'static str, Vec<usize>, &[f32])> {
    vec![
        ("attn.q_proj.weight", vec![layer.q_proj.out_dim, layer.q_proj.in_dim], layer.q_proj.w.as_slice()),
        ("attn.k_proj.weight", vec![layer.k_proj.out_dim, layer.k_proj.in_dim], layer.k_proj.w.as_slice()),
        ("attn.v_proj.weight", vec![layer.v_proj.out_dim, layer.v_proj.in_dim], layer.v_proj.w.as_slice()),
        ("attn.o_proj.weight", vec![layer.o_proj.out_dim, layer.o_proj.in_dim], layer.o_proj.w.as_slice()),
        ("mlp.gate_proj.weight", vec![layer.gate_proj.out_dim, layer.gate_proj.in_dim], layer.gate_proj.w.as_slice()),
        ("mlp.up_proj.weight", vec![layer.up_proj.out_dim, layer.up_proj.in_dim], layer.up_proj.w.as_slice()),
        ("mlp.down_proj.weight", vec![layer.down_proj.out_dim, layer.down_proj.in_dim], layer.down_proj.w.as_slice()),
        ("attn_norm.weight", vec![layer.attn_norm.gamma.len()], layer.attn_norm.gamma.as_slice()),
        ("mlp_norm.weight", vec![layer.mlp_norm.gamma.len()], layer.mlp_norm.gamma.as_slice()),
    ]
}

/// Every canonical tensor of `model` as `(name, shape, data)`, in a fixed
/// deterministic order (also the on-disk data order used by save).
fn model_tensors(model: &HrmTextModel) -> Vec<(String, Vec<usize>, &[f32])> {
    let c = &model.config;
    let mut out: Vec<(String, Vec<usize>, &[f32])> = vec![
        (
            "embed_tokens.weight".to_string(),
            vec![c.vocab_size, c.hidden_size],
            model.embed_tokens.weight.as_slice(),
        ),
        ("z_l_init".to_string(), vec![c.hidden_size], model.z_l_init.as_slice()),
    ];
    for (level, core) in [("h_level", &model.h_level), ("l_level", &model.l_level)] {
        for (i, layer) in core.layers.iter().enumerate() {
            for (suffix, shape, data) in layer_tensors(layer) {
                out.push((format!("{level}.layers.{i}.{suffix}"), shape, data));
            }
        }
    }
    out.push((
        "final_norm.weight".to_string(),
        vec![model.final_norm.gamma.len()],
        model.final_norm.gamma.as_slice(),
    ));
    out.push((
        "lm_head.weight".to_string(),
        vec![model.lm_head.out_dim, model.lm_head.in_dim],
        model.lm_head.w.as_slice(),
    ));
    out
}

/// Mutable counterpart of [`model_tensors`]: `(name, expected_shape, slot)`
/// in the same order, for assignment during load.
fn model_slots_mut(model: &mut HrmTextModel) -> Vec<(String, Vec<usize>, &mut Vec<f32>)> {
    let c = model.config.clone();
    let mut out: Vec<(String, Vec<usize>, &mut Vec<f32>)> = vec![
        (
            "embed_tokens.weight".to_string(),
            vec![c.vocab_size, c.hidden_size],
            &mut model.embed_tokens.weight,
        ),
        ("z_l_init".to_string(), vec![c.hidden_size], &mut model.z_l_init),
    ];
    for (level, core) in [("h_level", &mut model.h_level), ("l_level", &mut model.l_level)] {
        for (i, layer) in core.layers.iter_mut().enumerate() {
            let HrmDecoderLayer {
                q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj,
                attn_norm, mlp_norm, ..
            } = layer;
            let slots: Vec<(&'static str, Vec<usize>, &mut Vec<f32>)> = vec![
                ("attn.q_proj.weight", vec![q_proj.out_dim, q_proj.in_dim], &mut q_proj.w),
                ("attn.k_proj.weight", vec![k_proj.out_dim, k_proj.in_dim], &mut k_proj.w),
                ("attn.v_proj.weight", vec![v_proj.out_dim, v_proj.in_dim], &mut v_proj.w),
                ("attn.o_proj.weight", vec![o_proj.out_dim, o_proj.in_dim], &mut o_proj.w),
                ("mlp.gate_proj.weight", vec![gate_proj.out_dim, gate_proj.in_dim], &mut gate_proj.w),
                ("mlp.up_proj.weight", vec![up_proj.out_dim, up_proj.in_dim], &mut up_proj.w),
                ("mlp.down_proj.weight", vec![down_proj.out_dim, down_proj.in_dim], &mut down_proj.w),
                ("attn_norm.weight", vec![attn_norm.gamma.len()], &mut attn_norm.gamma),
                ("mlp_norm.weight", vec![mlp_norm.gamma.len()], &mut mlp_norm.gamma),
            ];
            for (suffix, shape, slot) in slots {
                out.push((format!("{level}.layers.{i}.{suffix}"), shape, slot));
            }
        }
    }
    let final_dim = model.final_norm.gamma.len();
    out.push(("final_norm.weight".to_string(), vec![final_dim], &mut model.final_norm.gamma));
    out.push((
        "lm_head.weight".to_string(),
        vec![model.lm_head.out_dim, model.lm_head.in_dim],
        &mut model.lm_head.w,
    ));
    out
}

// ---------------------------------------------------------------------------
// Save
// ---------------------------------------------------------------------------

/// Save `model` to `path` in safetensors format (F32, little-endian), using
/// the canonical tensor names documented at module level.
pub fn save_safetensors(model: &HrmTextModel, path: impl AsRef<Path>) -> Result<()> {
    let tensors = model_tensors(model);

    // Header: tensor name -> { dtype, shape, data_offsets }, offsets laid out
    // contiguously in enumeration order.
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for (name, shape, data) in &tensors {
        let numel: usize = shape.iter().product();
        if numel != data.len() {
            return Err(RuvLLMError::Serialization(format!(
                "safetensors save: tensor '{}' shape {:?} ({} elements) does not \
                 match buffer length {}",
                name, shape, numel, data.len()
            )));
        }
        let nbytes = numel * F32_BYTES;
        header.insert(
            name.clone(),
            json!({ "dtype": DTYPE_F32, "shape": shape, "data_offsets": [offset, offset + nbytes] }),
        );
        offset += nbytes;
    }
    let mut header_bytes = serde_json::to_vec(&Value::Object(header))
        .map_err(|e| RuvLLMError::Serialization(format!("safetensors header encode: {e}")))?;
    // Pad with spaces to an 8-byte multiple (matches the reference writer).
    while header_bytes.len() % 8 != 0 {
        header_bytes.push(b' ');
    }

    let file = std::fs::File::create(path.as_ref())?;
    let mut w = std::io::BufWriter::new(file);
    w.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
    w.write_all(&header_bytes)?;
    for (_name, _shape, data) in &tensors {
        for &v in *data {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    w.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

/// Load a model from a safetensors file at `path`, validating every tensor's
/// shape against `config` and reporting missing/extra tensors by name.
/// Upstream HF-export names are accepted via the alias table
/// ([`canonicalize_name`]); only `F32` dtype is supported.
pub fn load_safetensors(config: &HrmTextConfig, path: impl AsRef<Path>) -> Result<HrmTextModel> {
    config.validate()?;
    let bytes = std::fs::read(path.as_ref())?;
    if bytes.len() < 8 {
        return Err(RuvLLMError::Serialization(
            "safetensors: file shorter than the 8-byte header length".to_string(),
        ));
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
    let data_start = 8usize.checked_add(header_len).ok_or_else(|| {
        RuvLLMError::Serialization("safetensors: header length overflows".to_string())
    })?;
    if data_start > bytes.len() {
        return Err(RuvLLMError::Serialization(format!(
            "safetensors: header length {} exceeds file size {}",
            header_len,
            bytes.len()
        )));
    }
    let header: Value = serde_json::from_slice(&bytes[8..data_start])
        .map_err(|e| RuvLLMError::Serialization(format!("safetensors header parse: {e}")))?;
    let obj = header.as_object().ok_or_else(|| {
        RuvLLMError::Serialization("safetensors: header is not a JSON object".to_string())
    })?;
    let data = &bytes[data_start..];

    // Canonicalized name -> (shape, raw little-endian bytes).
    let mut raw: BTreeMap<String, (Vec<usize>, &[u8])> = BTreeMap::new();
    for (name, entry) in obj {
        if name == "__metadata__" {
            continue;
        }
        let (shape, span) = parse_tensor_entry(name, entry, data.len())?;
        let canonical = canonicalize_name(name);
        if raw.insert(canonical.clone(), (shape, &data[span.0..span.1])).is_some() {
            return Err(RuvLLMError::Serialization(format!(
                "safetensors: duplicate tensor '{}' (canonical '{}')",
                name, canonical
            )));
        }
    }

    // Allocate destination buffers at the right shapes, then overwrite.
    // TODO(perf): replace the random-init scaffold with a zeroed constructor
    // for large configs; load-time cost only.
    let mut model = HrmTextModel::new_random(config.clone(), 0)?;
    let mut missing: Vec<String> = Vec::new();
    let mut used: BTreeSet<String> = BTreeSet::new();
    for (name, expected_shape, slot) in model_slots_mut(&mut model) {
        match raw.get(&name) {
            None => missing.push(name),
            Some((shape, le_bytes)) => {
                if *shape != expected_shape {
                    return Err(RuvLLMError::Config(format!(
                        "safetensors: tensor '{}' has shape {:?} but config requires {:?}",
                        name, shape, expected_shape
                    )));
                }
                debug_assert_eq!(le_bytes.len(), slot.len() * F32_BYTES);
                for (dst, chunk) in slot.iter_mut().zip(le_bytes.chunks_exact(F32_BYTES)) {
                    *dst = f32::from_le_bytes(chunk.try_into().expect("4 bytes"));
                }
                used.insert(name);
            }
        }
    }
    let extra: Vec<&String> = raw.keys().filter(|k| !used.contains(*k)).collect();
    if !missing.is_empty() || !extra.is_empty() {
        return Err(RuvLLMError::Config(format!(
            "safetensors: tensor set mismatch — missing: {:?}, extra (unrecognized): {:?}",
            missing, extra
        )));
    }
    Ok(model)
}

/// Validate one header entry; returns `(shape, (begin, end))` data span.
fn parse_tensor_entry(name: &str, entry: &Value, data_len: usize) -> Result<(Vec<usize>, (usize, usize))> {
    let err = |msg: String| RuvLLMError::Serialization(format!("safetensors tensor '{name}': {msg}"));
    let obj = entry.as_object().ok_or_else(|| err("entry is not an object".into()))?;
    let dtype = obj.get("dtype").and_then(Value::as_str).ok_or_else(|| err("missing dtype".into()))?;
    if dtype != DTYPE_F32 {
        return Err(err(format!("dtype '{dtype}' unsupported — only F32 is implemented")));
    }
    let shape: Vec<usize> = obj
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| err("missing shape".into()))?
        .iter()
        .map(|v| v.as_u64().map(|x| x as usize).ok_or_else(|| err("non-integer shape entry".into())))
        .collect::<Result<_>>()?;
    let offsets = obj
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or_else(|| err("missing data_offsets".into()))?;
    if offsets.len() != 2 {
        return Err(err("data_offsets must have exactly 2 entries".into()));
    }
    let begin = offsets[0].as_u64().ok_or_else(|| err("bad begin offset".into()))? as usize;
    let end = offsets[1].as_u64().ok_or_else(|| err("bad end offset".into()))? as usize;
    if begin > end || end > data_len {
        return Err(err(format!("data_offsets [{begin}, {end}] out of bounds (data {data_len} bytes)")));
    }
    let numel: usize = shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d)).ok_or_else(|| err("shape overflows".into()))?;
    let expected = numel.checked_mul(F32_BYTES).ok_or_else(|| err("byte size overflows".into()))?;
    if end - begin != expected {
        return Err(err(format!(
            "shape {shape:?} needs {expected} bytes but data_offsets span {}",
            end - begin
        )));
    }
    Ok((shape, (begin, end)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonicalize_upstream_aliases() {
        for (alias, canonical) in [
            ("model.embed_tokens.weight", "embed_tokens.weight"),
            ("model.z_L_init", "z_l_init"),
            ("z_l_init.weight", "z_l_init"),
            (
                "model.H_level.layers.3.self_attn.q_proj.weight",
                "h_level.layers.3.attn.q_proj.weight",
            ),
            (
                "model.L_level.layers.0.input_layernorm.weight",
                "l_level.layers.0.attn_norm.weight",
            ),
            (
                "model.L_level.layers.0.post_attention_layernorm.weight",
                "l_level.layers.0.mlp_norm.weight",
            ),
            ("model.H_level.layers.1.mlp.gate_proj.weight", "h_level.layers.1.mlp.gate_proj.weight"),
            ("model.norm.weight", "final_norm.weight"),
            ("lm_head.weight", "lm_head.weight"),
            // Canonical names are fixed points.
            ("h_level.layers.0.attn.k_proj.weight", "h_level.layers.0.attn.k_proj.weight"),
        ] {
            assert_eq!(canonicalize_name(alias), canonical, "alias {alias}");
        }
    }

    #[test]
    fn test_tensor_enumerations_agree() {
        let config = HrmTextConfig::tiny_test();
        let mut model = HrmTextModel::new_random(config, 7).unwrap();
        let ro: Vec<(String, Vec<usize>)> = model_tensors(&model)
            .into_iter()
            .map(|(n, s, _)| (n, s))
            .collect();
        let rw: Vec<(String, Vec<usize>)> = model_slots_mut(&mut model)
            .into_iter()
            .map(|(n, s, _)| (n, s))
            .collect();
        assert_eq!(ro, rw);
        // tiny_test: 2 + 2 levels * 1 layer * 9 + 2 = 22 tensors.
        assert_eq!(ro.len(), 22);
    }

    #[test]
    fn test_loads_upstream_alias_names() {
        // Save canonically, rewrite every header key to the upstream-style
        // alias, and verify the loader still resolves the full tensor set.
        let config = HrmTextConfig::tiny_test();
        let model = HrmTextModel::new_random(config.clone(), 11).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("alias.safetensors");
        save_safetensors(&model, &path).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header: Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
        let mut renamed = serde_json::Map::new();
        for (name, entry) in header.as_object().unwrap() {
            let alias = match name.as_str() {
                "lm_head.weight" => name.clone(),
                "z_l_init" => "model.z_L_init".to_string(),
                "final_norm.weight" => "model.norm.weight".to_string(),
                other => format!("model.{}", other.replace("h_level.", "H_level.").replace("l_level.", "L_level.").replace(".attn.", ".self_attn.")),
            };
            renamed.insert(alias, entry.clone());
        }
        let mut new_header = serde_json::to_vec(&Value::Object(renamed)).unwrap();
        while new_header.len() % 8 != 0 {
            new_header.push(b' ');
        }
        let mut out = Vec::new();
        out.extend_from_slice(&(new_header.len() as u64).to_le_bytes());
        out.extend_from_slice(&new_header);
        out.extend_from_slice(&bytes[8 + header_len..]);
        let alias_path = dir.path().join("alias2.safetensors");
        std::fs::write(&alias_path, out).unwrap();

        let loaded = load_safetensors(&config, &alias_path).unwrap();
        assert_eq!(loaded.z_l_init, model.z_l_init);
        assert_eq!(loaded.lm_head.w, model.lm_head.w);
    }

    #[test]
    fn test_rejects_non_f32_dtype() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f16.safetensors");
        let header = json!({
            "embed_tokens.weight": { "dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8] }
        });
        let mut hb = serde_json::to_vec(&header).unwrap();
        while hb.len() % 8 != 0 {
            hb.push(b' ');
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(hb.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&hb);
        bytes.extend_from_slice(&[0u8; 8]);
        std::fs::write(&path, bytes).unwrap();
        let err = load_safetensors(&HrmTextConfig::tiny_test(), &path).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("F16") && msg.contains("only F32"), "got: {msg}");
    }
}
