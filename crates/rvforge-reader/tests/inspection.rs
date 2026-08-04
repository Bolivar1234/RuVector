//! Inspection stub: deterministic, non-executing, never claims verification.

use rvforge_reader::inspect::{self, SignatureStatus, VerificationStatus};
use std::path::PathBuf;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("rvforge-reader-tests");
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir.join(name)
}

#[test]
fn inspection_is_deterministic_for_the_same_path() {
    let path = scratch("determinism.rvf");
    std::fs::write(&path, b"not a real rvf").unwrap();

    let first = inspect::inspect(&path);
    let second = inspect::inspect(&path);
    assert_eq!(first, second);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn inspection_never_reports_verified() {
    let path = scratch("unverified.rvf");
    std::fs::write(&path, b"bytes").unwrap();

    let summary = inspect::inspect(&path);
    assert_eq!(summary.verification, VerificationStatus::Unverified);
    assert_eq!(summary.signature, SignatureStatus::Unverified);
    assert!(
        summary.identity.is_none(),
        "the stub must not invent an identity"
    );
    assert!(summary.publisher.is_none());
    assert!(summary.stub);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn inspection_never_executes() {
    let path = scratch("no-exec.rvf");
    std::fs::write(&path, b"bytes").unwrap();
    assert!(!inspect::inspect(&path).executed);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_missing_file_is_reported_not_guessed() {
    let summary = inspect::inspect(&scratch("does-not-exist.rvf"));
    assert!(!summary.exists);
    assert_eq!(summary.size_bytes, 0);
    assert!(summary.extension_ok);
    assert_eq!(summary.verification, VerificationStatus::Unverified);
}

#[test]
fn a_non_rvf_extension_is_flagged() {
    let summary = inspect::inspect(&scratch("agent.zip"));
    assert!(!summary.extension_ok);
}

#[test]
fn size_is_read_from_metadata() {
    let path = scratch("sized.rvf");
    std::fs::write(&path, vec![0u8; 1234]).unwrap();
    let summary = inspect::inspect(&path);
    assert!(summary.exists);
    assert_eq!(summary.size_bytes, 1234);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_package_with_no_manifest_gets_the_everything_denied_card() {
    let path = scratch("cardless.rvf");
    std::fs::write(&path, b"bytes").unwrap();

    let card = inspect::capability_card(&path);
    assert!(card.requests.is_empty());
    assert_eq!(
        card.status,
        rvforge_reader::capability::CardStatus::ManifestUnavailable
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_rejected_sidecar_manifest_does_not_produce_a_permissive_card() {
    let path = scratch("bad-sidecar.rvf");
    std::fs::write(&path, b"bytes").unwrap();
    std::fs::write(
        scratch("bad-sidecar.rvf.manifest.json"),
        r#"{"schemaVersion":1,"manifestId":"sha256:c","defaultPolicy":"deny",
            "requests":[{"class":"filesystem","scope":"all-files"}],"denials":[]}"#,
    )
    .unwrap();

    let card = inspect::capability_card(&path);
    assert!(
        card.requests.is_empty(),
        "a rejected manifest grants nothing"
    );
    assert!(
        card.notes.iter().any(|n| n.contains("rejected")),
        "the rejection must be visible: {:?}",
        card.notes
    );

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(scratch("bad-sidecar.rvf.manifest.json"));
}
