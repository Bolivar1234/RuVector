//! Tauri command surface.
//!
//! Thin wrappers only: every decision lives in [`crate::inspect`],
//! [`crate::capability`], and [`crate::runtime`] so it can be tested without a
//! webview. None of these commands executes package content.

use std::path::PathBuf;
use std::sync::Mutex;

use crate::capability::CapabilityCard;
use crate::dock::{
    AgentDockEntry, IsolationClass, NetworkIndicator, PermissionSummary, ResourceUsage,
    SystemOwnedStatus, TrustBadge, WitnessStatus,
};
use crate::dock_events::Notification;
use crate::dock_roster::{AgentPill, DockRoster, ExpandedAgentView, RosterView};
use crate::dock_state::DockState;
use crate::inspect::{self, InspectionSummary, VerificationOutcome};
use crate::runtime::{self, HostProfile, RuntimeChoice};

/// Read identity, segments, and declared capabilities. Checks nothing, executes
/// nothing: the returned verification status is always `unverified`.
#[tauri::command]
pub fn inspect_rvf(path: String) -> InspectionSummary {
    inspect::inspect(&PathBuf::from(path))
}

/// Run the root-manifest, per-segment-hash, and unsigned-executable checks, and
/// append the witness receipt. Still executes nothing.
#[tauri::command]
pub fn verify_rvf(path: String) -> VerificationOutcome {
    inspect::verify(&PathBuf::from(path))
}

/// The P6 install-time capability contract for a package. Verifies first, and
/// returns the everything-denied card if verification does not pass.
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

/// The dock's roster, managed by the Tauri app.
///
/// Chrome state lives on the Rust side of the process boundary, which is what
/// makes ADR-295 §7 structural here: the webview can read it and can invoke
/// user controls, but the agent-facing command is
/// [`dock_agent_report`] and it carries only task text and progress.
#[derive(Default)]
pub struct DockSurface(pub Mutex<DockRoster>);

impl DockSurface {
    fn with<R>(&self, f: impl FnOnce(&mut DockRoster) -> R) -> Result<R, String> {
        let mut roster = self
            .0
            .lock()
            .map_err(|_| "dock roster lock is poisoned".to_string())?;
        Ok(f(&mut roster))
    }
}

/// The collapsed bar: active agent, two secondary slots, overflow count,
/// aggregate usage, approval count.
#[tauri::command]
pub fn dock_view(dock: tauri::State<'_, DockSurface>) -> Result<RosterView, String> {
    dock.with(|r| r.view())
}

/// Interaction one of the acceptance test.
#[tauri::command]
pub fn dock_expand(
    dock: tauri::State<'_, DockSurface>,
    agent_id: String,
) -> Result<ExpandedAgentView, String> {
    dock.with(|r| r.expand(&agent_id))?
        .map_err(|e| e.to_string())
}

/// One action, from any non-terminal state.
#[tauri::command]
pub fn dock_pause(
    dock: tauri::State<'_, DockSurface>,
    agent_id: String,
) -> Result<DockState, String> {
    dock.with(|r| r.pause(&agent_id))?
        .map_err(|e| e.to_string())
}

/// Interaction two of the acceptance test. No confirmation chain.
#[tauri::command]
pub fn dock_terminate(
    dock: tauri::State<'_, DockSurface>,
    agent_id: String,
) -> Result<DockState, String> {
    dock.with(|r| r.terminate(&agent_id))?
        .map_err(|e| e.to_string())
}

/// Behind the overflow count.
#[tauri::command]
pub fn dock_swarm_view(dock: tauri::State<'_, DockSurface>) -> Result<Vec<AgentPill>, String> {
    dock.with(|r| r.swarm_view())
}

/// Threshold events only (ADR-295 §9).
#[tauri::command]
pub fn dock_notifications(
    dock: tauri::State<'_, DockSurface>,
) -> Result<Vec<Notification>, String> {
    dock.with(|r| r.event_log().notifications().to_vec())
}

/// **The agent channel.** The whole agent-supplied surface is these two
/// arguments, and both are sanitized and clamped before they are stored. There
/// is no command through which an agent sets its own state, trust badge,
/// network indicator, permissions, witness status, cost, or resource usage.
#[tauri::command]
pub fn dock_agent_report(
    dock: tauri::State<'_, DockSurface>,
    agent_id: String,
    task: String,
    progress: i64,
) -> Result<(), String> {
    dock.with(|r| r.apply_agent_update(&agent_id, &task, progress))?
        .map_err(|e| e.to_string())
}

/// Register a placeholder agent so the dock can be exercised before `rvm-ffi`
/// is wired in.
///
/// Every security-bearing field reports its unknown value — unverified
/// publisher, no confinement engaged, no witness chain — because the scaffold
/// has measured none of them. It does not fabricate a trusted-looking agent.
#[tauri::command]
pub fn dock_add_scaffold_agent(
    dock: tauri::State<'_, DockSurface>,
    agent_id: String,
    name: String,
    icon: String,
) -> Result<RosterView, String> {
    let system = SystemOwnedStatus::measured(
        DockState::Idle,
        TrustBadge::Unverified,
        NetworkIndicator::Disabled,
        PermissionSummary {
            isolation: IsolationClass::None,
            local_model: false,
            network_allowed: false,
            selected_folder_access: false,
            encrypted_memory: false,
            grants: Vec::new(),
            pending_approvals: Vec::new(),
        },
        WitnessStatus::Unavailable,
        ResourceUsage::ZERO,
        0,
        None,
    );
    dock.with(|r| {
        r.insert(AgentDockEntry::new(agent_id, name, icon, system));
        r.view()
    })
}
