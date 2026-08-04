//! Encrypted state capsule layout (ADR-288).
//!
//! # STUB — encryption is not implemented
//!
//! The directory layout and the lineage rules below are real; the sealing is
//! not. [`seal`] and [`unseal`] return [`StateError::EncryptionNotImplemented`]
//! rather than writing plaintext under a name that implies otherwise.
//!
//! The ADR-288 contract this module holds the shape of:
//!
//! 1. **The base RVF is immutable.** It is opened read-only, never written
//!    after signing, and any in-place modification is tampering.
//! 2. **State is a separate chain of encrypted delta segments** — a
//!    `CompressedCheckpoint` plus ordered `WitnessDelta` records — stored apart
//!    from both the base artifact and the installer payload, and deletable
//!    without touching either.
//! 3. **Every delta and checkpoint carries the base RVF identity it belongs
//!    to.** On open, a mismatch is a lineage rejection: state is refused,
//!    execution does not begin with partial state, and the rejection is
//!    witnessed.
//!
//! ```text
//! <install_root>/base.rvf                  immutable, signed, read-only
//! <state_root>/<base-identity>/
//!     checkpoint-000.ckpt                  CompressedCheckpoint
//!     delta-001.wdelta                     WitnessDelta (encrypted)
//!     delta-002.wdelta
//! ```

use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateError {
    /// The state root is inside the install root. State must be deletable
    /// without touching the immutable base artifact (ADR-288 §2).
    StateInsideInstallRoot {
        install_root: String,
        state_root: String,
    },
    /// State belongs to a different base RVF identity (ADR-288 §4).
    LineageRejected { expected: String, found: String },
    /// Deliberate: no plaintext fallback.
    EncryptionNotImplemented,
    /// A base identity must be a `sha256:<hex>` content address.
    InvalidBaseIdentity(String),
}

impl std::fmt::Display for StateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StateError::StateInsideInstallRoot {
                install_root,
                state_root,
            } => write!(
                f,
                "state root {state_root} is inside install root {install_root}; \
                 state must be deletable independently of the base artifact"
            ),
            StateError::LineageRejected { expected, found } => write!(
                f,
                "lineage rejection: state records base identity {found}, loaded base is {expected}"
            ),
            StateError::EncryptionNotImplemented => write!(
                f,
                "state encryption is not implemented; refusing to write unencrypted state"
            ),
            StateError::InvalidBaseIdentity(id) => {
                write!(f, "'{id}' is not a sha256:<hex> base identity")
            }
        }
    }
}

impl std::error::Error for StateError {}

/// Paths for one agent's state, bound to one base RVF identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateCapsule {
    install_root: PathBuf,
    state_root: PathBuf,
    base_identity: String,
}

impl StateCapsule {
    pub fn new(
        install_root: impl AsRef<Path>,
        state_root: impl AsRef<Path>,
        base_identity: impl Into<String>,
    ) -> Result<Self, StateError> {
        let install_root = install_root.as_ref().to_path_buf();
        let state_root = state_root.as_ref().to_path_buf();
        let base_identity = base_identity.into();

        if !is_base_identity(&base_identity) {
            return Err(StateError::InvalidBaseIdentity(base_identity));
        }
        if state_root.starts_with(&install_root) {
            return Err(StateError::StateInsideInstallRoot {
                install_root: install_root.display().to_string(),
                state_root: state_root.display().to_string(),
            });
        }
        Ok(Self {
            install_root,
            state_root,
            base_identity,
        })
    }

    pub fn install_root(&self) -> &Path {
        &self.install_root
    }

    pub fn state_root(&self) -> &Path {
        &self.state_root
    }

    pub fn base_identity(&self) -> &str {
        &self.base_identity
    }

    /// `<state_root>/<base-identity>/`. The identity is part of the path so
    /// two lineages cannot share a directory.
    pub fn capsule_dir(&self) -> PathBuf {
        self.state_root.join(self.base_identity.replace(':', "-"))
    }

    pub fn checkpoint_path(&self, seq: u32) -> PathBuf {
        self.capsule_dir().join(format!("checkpoint-{seq:03}.ckpt"))
    }

    pub fn delta_path(&self, seq: u32) -> PathBuf {
        self.capsule_dir().join(format!("delta-{seq:03}.wdelta"))
    }

    /// The base artifact is read-only for the lifetime of its identity.
    pub fn base_artifact_path(&self) -> PathBuf {
        self.install_root.join("base.rvf")
    }

    /// Reject state that was recorded against a different base identity.
    ///
    /// ADR-288 §4 also allows migrating state whose recorded identity is an
    /// accepted *ancestor* of the loaded base. Ancestry needs the release
    /// lineage chain from the registry, so this check is exact-match only for
    /// now and rejects rather than guessing.
    pub fn check_lineage(&self, recorded_base_identity: &str) -> Result<(), StateError> {
        if recorded_base_identity == self.base_identity {
            Ok(())
        } else {
            Err(StateError::LineageRejected {
                expected: self.base_identity.clone(),
                found: recorded_base_identity.to_string(),
            })
        }
    }
}

fn is_base_identity(id: &str) -> bool {
    match id.strip_prefix("sha256:") {
        Some(hex) => hex.len() == 64 && hex.chars().all(|c| c.is_ascii_hexdigit()),
        None => false,
    }
}

/// Encrypt a delta or checkpoint payload.
///
/// TODO(rvf-forge-core): implement AEAD sealing bound to the base identity,
/// with the customer-held key path required by ADR-288. Returning an error is
/// deliberate — a passthrough implementation would write plaintext state under
/// a name that claims encryption.
pub fn seal(_base_identity: &str, _plaintext: &[u8]) -> Result<Vec<u8>, StateError> {
    Err(StateError::EncryptionNotImplemented)
}

/// Decrypt a delta or checkpoint payload.
///
/// TODO(rvf-forge-core): see [`seal`].
pub fn unseal(_base_identity: &str, _sealed: &[u8]) -> Result<Vec<u8>, StateError> {
    Err(StateError::EncryptionNotImplemented)
}

/// Per-user state root, kept out of the install directory.
///
/// Falls back to a relative path when the environment has no home directory
/// rather than writing somewhere surprising.
pub fn default_state_root() -> PathBuf {
    let base = if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support"))
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
    };
    base.unwrap_or_else(|| PathBuf::from("."))
        .join("io.ruv.rvforge.reader")
        .join("state")
}
