//! The `WitnessReceipt` wire shape and its content address.
//!
//! Field-for-field the `registry-model.md` shape, so re-serializing a receipt
//! reproduces the bytes its id was computed over. That is the whole point: the
//! reader recomputes ids rather than trusting the ones in the file, and a
//! recomputation that used a different encoding would disagree with the
//! registry about intact receipts and agree with it about nothing.
//!
//! [`crate::witness_view`] is where the chain is verified; this module only
//! defines what a receipt is and how its address is derived.

use serde::{Deserialize, Serialize};

use rvf_forge_core::canonical::canonical_sha256_hex;

/// The `type` value every chained receipt carries.
pub const WITNESS_RECEIPT_TYPE: &str = "witness-receipt";
/// The prefix on every content address in the registry model.
pub const DIGEST_PREFIX: &str = "sha256:";
/// The schema version this viewer understands.
pub const WITNESS_SCHEMA_VERSION: u32 = 1;
/// Hex digits kept when shortening a content address for display.
const ID_SHORT_HEX: usize = 12;

/// One link in a subject's witness chain, as written by the registry or the
/// CLI.
///
/// `event` and `outcome` are strings rather than enums: a viewer that refused
/// to display a receipt whose event it did not recognise would hide exactly the
/// records an operator most needs to see, and an unknown value hashes the same
/// either way. `deny_unknown_fields` still holds, because a field this reader
/// does not model is a field it cannot hash, and silently dropping it would
/// make a tampered receipt recompute to its declared id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WitnessReceipt {
    pub schema_version: u32,
    #[serde(rename = "type")]
    pub object_type: String,
    pub receipt_id: String,
    pub subject: String,
    pub event: String,
    pub outcome: String,
    pub actor: Actor,
    pub evidence: Evidence,
    pub timestamp: String,
    pub prev_receipt: Option<String>,
    /// Not verified here — this viewer checks the chain, not the signers — and
    /// removed before the id is recomputed, so its shape is irrelevant.
    #[serde(default)]
    pub signatures: Vec<serde_json::Value>,
}

/// Who performed the witnessed operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Actor {
    pub kind: String,
    pub id: String,
}

/// What the actor observed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Evidence {
    pub details: String,
}

/// Recompute a receipt's content address.
///
/// `signatures` and `receiptId` are removed first: a receipt cannot commit to
/// its own id, and a countersignature added later must not change it. Everything
/// else is covered, so editing any other field moves the address.
pub fn recompute_id(receipt: &WitnessReceipt) -> Option<String> {
    let mut value = serde_json::to_value(receipt).ok()?;
    let map = value.as_object_mut()?;
    map.remove("signatures");
    map.remove("receiptId");
    canonical_sha256_hex(&value)
        .ok()
        .map(|hex| format!("{DIGEST_PREFIX}{hex}"))
}

/// A content address shortened for display, prefix kept so it stays readable
/// as a digest rather than as an opaque token.
pub fn short_id(id: &str) -> String {
    match id.strip_prefix(DIGEST_PREFIX) {
        Some(hex) if hex.len() > ID_SHORT_HEX => {
            format!("{DIGEST_PREFIX}{}…", &hex[..ID_SHORT_HEX])
        }
        _ if id.is_empty() => "—".to_string(),
        _ => id.to_string(),
    }
}
