//! Inspection of an `.rvf` package.
//!
//! # STUB — replaced by `rvf-forge-core` FFI
//!
//! Everything in this module is a placeholder. It reads path metadata and
//! nothing else: it does not open the archive, parse segments, resolve the
//! manifest, or check a signature. When `rvf-forge-core` lands, [`inspect`]
//! becomes a call across the `rvm_inspect` C ABI (ADR-289 §4) and the status
//! enums below gain their `Verified` / `Failed` variants.
//!
//! Two invariants must survive that replacement:
//!
//! - `inspect` and `verify` **never execute RVF content** (ADR-289 §3).
//!   Inspection of an untrusted package must be safe.
//! - Until a signature has actually been checked, the reported status is
//!   [`VerificationStatus::Unverified`]. The stub does not pretend to verify.

use serde::Serialize;
use std::path::Path;

use crate::capability::CapabilityCard;

/// Result of a signature and hash check.
///
/// Single-variant on purpose: nothing in this crate can currently produce any
/// other answer, and a `Verified` variant that no code path sets would be an
/// invitation to default to it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum VerificationStatus {
    /// No verification has been performed. Execution must not proceed.
    Unverified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SignatureStatus {
    /// No signature has been checked; the publisher is unknown.
    Unverified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InspectionSummary {
    pub path: String,
    pub file_name: String,
    /// The path ends in `.rvf`. A filename check, not a format check.
    pub extension_ok: bool,
    /// `false` when the path does not exist or could not be stat'd.
    pub exists: bool,
    pub size_bytes: u64,
    /// Content hash of the base artifact. `None` until `rvf-forge-core` lands.
    pub identity: Option<String>,
    pub publisher: Option<String>,
    pub verification: VerificationStatus,
    pub signature: SignatureStatus,
    /// Runtime profiles the package declares. Empty until the manifest is read.
    pub declared_runtime_profiles: Vec<String>,
    pub capability_manifest_present: bool,
    /// Always `false`. Inspection never executes package content.
    pub executed: bool,
    /// Always `true` while this is a stub, so the UI can label it.
    pub stub: bool,
    pub notes: Vec<String>,
}

/// Inspect a package without executing it.
///
/// Deterministic for a given path and filesystem state: the output is a pure
/// function of the path string and the file's size.
pub fn inspect(path: &Path) -> InspectionSummary {
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let extension_ok = path
        .extension()
        .map(|e| e.eq_ignore_ascii_case("rvf"))
        .unwrap_or(false);

    let (exists, size_bytes) = match std::fs::metadata(path) {
        Ok(meta) if meta.is_file() => (true, meta.len()),
        Ok(_) => (false, 0),
        Err(_) => (false, 0),
    };

    let mut notes = vec![
        "Inspection is a stub: no segment, manifest, or signature was read.".to_string(),
        "Package content was not executed.".to_string(),
    ];
    if !exists {
        notes.push("No readable file at this path.".to_string());
    }
    if !extension_ok {
        notes.push("Filename does not end in .rvf.".to_string());
    }
    notes.push(
        "Identity, publisher, and signature become available when rvf-forge-core is wired in."
            .to_string(),
    );

    let manifest_present = sidecar_manifest_path(path).is_some();
    if manifest_present {
        notes.push(
            "A development capability-manifest sidecar was found next to this file.".to_string(),
        );
    }

    InspectionSummary {
        path: path.to_string_lossy().into_owned(),
        file_name,
        extension_ok,
        exists,
        size_bytes,
        identity: None,
        publisher: None,
        verification: VerificationStatus::Unverified,
        signature: SignatureStatus::Unverified,
        declared_runtime_profiles: Vec::new(),
        capability_manifest_present: manifest_present,
        executed: false,
        stub: true,
        notes,
    }
}

/// Derive the P6 capability card for a package.
///
/// The real implementation reads the signed `CapabilityManifest` segment out of
/// the RVF. Until then this looks for a `<file>.manifest.json` sidecar so the
/// install screen has something real to render, and falls back to the
/// everything-denied card when there is none.
pub fn capability_card(path: &Path) -> CapabilityCard {
    let Some(sidecar) = sidecar_manifest_path(path) else {
        return CapabilityCard::manifest_unavailable(
            "This package's capability manifest has not been read (inspection is a stub).",
        );
    };
    match std::fs::read_to_string(&sidecar) {
        Ok(json) => match CapabilityCard::from_manifest_json(&json) {
            Ok(mut card) => {
                card.notes.push(format!(
                    "Derived from development sidecar {}, not from a verified RVF segment.",
                    sidecar.display()
                ));
                card
            }
            Err(e) => CapabilityCard::manifest_unavailable(&format!(
                "The capability manifest was rejected: {e}"
            )),
        },
        Err(e) => CapabilityCard::manifest_unavailable(&format!(
            "The capability manifest could not be read: {e}"
        )),
    }
}

/// `<path>.manifest.json`, when it exists.
fn sidecar_manifest_path(path: &Path) -> Option<std::path::PathBuf> {
    let mut name = path.as_os_str().to_os_string();
    name.push(".manifest.json");
    let candidate = std::path::PathBuf::from(name);
    candidate.is_file().then_some(candidate)
}
