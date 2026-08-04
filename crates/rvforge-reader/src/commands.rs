//! Tauri command surface.
//!
//! Thin wrappers only: every decision lives in [`crate::inspect`],
//! [`crate::capability`], and [`crate::runtime`] so it can be tested without a
//! webview. None of these commands executes package content.

use std::path::PathBuf;

use crate::capability::CapabilityCard;
use crate::inspect::{self, InspectionSummary};
use crate::runtime::{self, HostProfile, RuntimeChoice};

/// Read identity and verification status for a package. Never executes it.
#[tauri::command]
pub fn inspect_rvf(path: String) -> InspectionSummary {
    inspect::inspect(&PathBuf::from(path))
}

/// The P6 install-time capability contract for a package.
#[tauri::command]
pub fn capability_card(path: String) -> CapabilityCard {
    inspect::capability_card(&PathBuf::from(path))
}

/// The FR004 runtime selection for this host.
///
/// The choice is a property of the host and the vendored compatibility matrix;
/// `path` only labels which package the answer was reported for.
#[tauri::command]
pub fn runtime_selection(path: String) -> Result<RuntimeChoice, String> {
    let mut choice = runtime::select_for_host(&HostProfile::detect()).map_err(|e| e.to_string())?;
    choice.subject = Some(path);
    Ok(choice)
}
