//! ADR-288 state capsule layout. Encryption is stubbed; the layout is not.

use rvforge_reader::state::{self, StateCapsule, StateError};

const BASE: &str = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const OTHER: &str = "sha256:fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";

fn capsule() -> StateCapsule {
    StateCapsule::new("/opt/rvforge/agent", "/var/lib/rvforge/state", BASE).unwrap()
}

#[test]
fn state_lives_outside_the_install_directory() {
    let err = StateCapsule::new("/opt/rvforge/agent", "/opt/rvforge/agent/state", BASE)
        .expect_err("state inside the install root must be refused");
    assert!(matches!(err, StateError::StateInsideInstallRoot { .. }));
}

#[test]
fn the_capsule_directory_is_keyed_by_base_identity() {
    let c = capsule();
    let dir = c.capsule_dir();
    assert!(dir.starts_with("/var/lib/rvforge/state"));
    assert!(dir.to_string_lossy().contains(&BASE.replace(':', "-")));
}

#[test]
fn checkpoint_and_delta_paths_follow_the_adr_288_layout() {
    let c = capsule();
    assert!(c
        .checkpoint_path(0)
        .to_string_lossy()
        .ends_with("checkpoint-000.ckpt"));
    assert!(c
        .delta_path(1)
        .to_string_lossy()
        .ends_with("delta-001.wdelta"));
    assert!(c
        .delta_path(12)
        .to_string_lossy()
        .ends_with("delta-012.wdelta"));
}

#[test]
fn the_base_artifact_stays_in_the_install_root() {
    let c = capsule();
    assert!(c.base_artifact_path().starts_with(c.install_root()));
    assert!(!c.base_artifact_path().starts_with(c.state_root()));
}

#[test]
fn state_from_another_lineage_is_rejected() {
    let c = capsule();
    assert!(c.check_lineage(BASE).is_ok());
    assert!(matches!(
        c.check_lineage(OTHER),
        Err(StateError::LineageRejected { .. })
    ));
}

#[test]
fn a_base_identity_must_be_a_content_address() {
    for bad in ["", "not-a-hash", "sha256:short", "md5:0123"] {
        assert!(matches!(
            StateCapsule::new("/opt/a", "/var/b", bad),
            Err(StateError::InvalidBaseIdentity(_))
        ));
    }
}

#[test]
fn sealing_refuses_rather_than_writing_plaintext() {
    assert_eq!(
        state::seal(BASE, b"secret"),
        Err(StateError::EncryptionNotImplemented)
    );
    assert_eq!(
        state::unseal(BASE, b"secret"),
        Err(StateError::EncryptionNotImplemented)
    );
}

#[test]
fn the_default_state_root_is_not_the_current_directory() {
    let root = state::default_state_root();
    assert!(root.to_string_lossy().contains("io.ruv.rvforge.reader"));
    assert!(root.ends_with("state"));
}
