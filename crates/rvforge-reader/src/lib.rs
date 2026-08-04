//! RVForge Reader — desktop reader for signed `.rvf` agent packages.
//!
//! This crate is the Tauri v2 host described in ADR-289. It is split so that
//! every decision the user sees is made in a plain library module that can be
//! tested without a webview:
//!
//! - [`inspect`] — read an `.rvf` without executing it (ADR-289 §3).
//! - [`capability`] — derive the install-time capability contract (requirements P6).
//! - [`runtime`] — apply the FR004 runtime selection order (ADR-289 §2).
//! - [`state`] — encrypted state-capsule layout (ADR-288).
//! - [`dock`], [`dock_state`], [`dock_roster`], [`dock_events`] — the Agent
//!   Dock: what a running agent shows and how it is stopped (ADR-295).
//!
//! # Security invariants
//!
//! These hold for every module here and must survive the replacement of the
//! stubs by `rvf-forge-core`:
//!
//! 1. **RVF content is never executed.** Inspection reads metadata only. There
//!    is no code path in this crate that loads, links, or interprets a segment.
//! 2. **Verification precedes any load.** Until `rvf-forge-core` lands, every
//!    inspection reports [`inspect::VerificationStatus::Unverified`] rather
//!    than asserting a signature it did not check.
//! 3. **Capability rendering is default-deny.** A class that was not requested
//!    is rendered in the "cannot" list. A missing manifest yields a card where
//!    everything is denied, not an empty card.
//! 4. **No vague permission prose.** Broad scopes and phrases such as "access
//!    your computer" are rejected at derivation time (requirements P6).
//! 5. **No network calls.** Nothing in this crate opens a socket; the packaged
//!    application's CSP forbids remote content.
//! 6. **Runtime order is not configurable.** The selection order comes from the
//!    vendored compatibility matrix. No environment variable, CLI flag, or
//!    setting reorders it; only signed policy may, and signed policy is not yet
//!    implemented (ADR-289 §2).
//! 7. **Agent input cannot reach dock chrome.** An agent supplies task text and
//!    progress and nothing else, through
//!    [`dock::AgentProvidedStatus::from_agent`]. Trust badge, network
//!    indicator, permission summary, witness status, resource usage, cost, and
//!    the runtime state live in [`dock::SystemOwnedStatus`], which has no
//!    constructor or setter that takes an agent-derived value (ADR-295 §7).

pub mod capability;
pub mod dock;
pub mod dock_events;
pub mod dock_roster;
pub mod dock_state;
pub mod dock_text;
pub mod inspect;
pub mod runtime;
pub mod state;

#[cfg(feature = "desktop")]
pub mod commands;

/// Launch the desktop shell.
#[cfg(feature = "desktop")]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        // Dock chrome state lives here, on the Rust side of the process
        // boundary, not in the webview that renders agent content.
        .manage(commands::DockSurface::default())
        .invoke_handler(tauri::generate_handler![
            commands::inspect_rvf,
            commands::capability_card,
            commands::runtime_selection,
            commands::dock_view,
            commands::dock_expand,
            commands::dock_pause,
            commands::dock_terminate,
            commands::dock_swarm_view,
            commands::dock_notifications,
            commands::dock_agent_report,
            commands::dock_add_scaffold_agent,
        ])
        .run(tauri::generate_context!())
        .expect("error while running RVForge Reader");
}
