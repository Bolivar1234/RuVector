//! Bounded metadata history for ADR-280 (§1 "META_SEG is the source of truth",
//! §5 replay ceilings, and the "Alternatives Considered" rejection of writing a
//! complete metadata snapshot on every mutation).
//!
//! Only the newest committed `META_SEG` generation is authoritative. Everything
//! older is superseded history, so neither the bytes a commit costs nor the work
//! an open performs may scale with the number of metadata commits, compaction
//! must reclaim the superseded generations, and corruption confined to a
//! superseded generation must not brick the artifact.

use rvf_runtime::options::{
    DistanceMetric, MetadataEntry, MetadataValue, RvfOptions, VectorMetadata,
};
use rvf_runtime::RvfStore;
use rvf_types::{SegmentType, SEGMENT_HEADER_SIZE, SEGMENT_MAGIC};
use std::time::Instant;
use tempfile::TempDir;

fn make_options(dim: u16) -> RvfOptions {
    RvfOptions {
        dimension: dim,
        metric: DistanceMetric::L2,
        ..Default::default()
    }
}

fn record(vector_id: u64, value: &str) -> VectorMetadata {
    VectorMetadata {
        vector_id,
        fields: vec![MetadataEntry {
            field_id: 1,
            value: MetadataValue::String(value.into()),
        }],
        delete_record: false,
    }
}

/// Walk the append-only file and return `(payload_start, payload_len)` for every
/// structurally complete `META_SEG`, in file order.
fn meta_segments(bytes: &[u8]) -> Vec<(usize, usize)> {
    let mut found = Vec::new();
    let mut offset = 0usize;
    while offset + SEGMENT_HEADER_SIZE <= bytes.len() {
        let magic = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        if magic != SEGMENT_MAGIC {
            break;
        }
        let seg_type = bytes[offset + 0x05];
        let payload_len =
            u64::from_le_bytes(bytes[offset + 0x10..offset + 0x18].try_into().unwrap()) as usize;
        let end = match offset.checked_add(SEGMENT_HEADER_SIZE + payload_len) {
            Some(end) if end <= bytes.len() => end,
            _ => break,
        };
        if seg_type == SegmentType::Meta as u8 {
            found.push((offset + SEGMENT_HEADER_SIZE, payload_len));
        }
        offset = end;
    }
    found
}

/// Several hundred single-vector metadata commits must not cost several hundred
/// full snapshots: the committed metadata bytes stay proportional to the live
/// snapshot, the artifact reopens with bounded work, and `compact()` reclaims
/// the superseded generations.
#[test]
fn metadata_history_stays_bounded_and_compact_reclaims_it() {
    const COMMITS: u64 = 600;

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("metadata_growth.rvf");
    let dim: u16 = 4;

    let mut store = RvfStore::create(&path, make_options(dim)).unwrap();
    for i in 0..COMMITS {
        let vector = [i as f32, 1.0, 2.0, 3.0];
        store
            .ingest_batch_with_metadata(&[&vector], &[i], &[record(i, &format!("record-{i}"))])
            .unwrap();
    }
    let size_before_compact = std::fs::metadata(&path).unwrap().len();
    store.close().unwrap();

    // Writing a full snapshot per commit is quadratic: the live snapshot after
    // `COMMITS` commits is roughly 26 KiB, so replaying it every time costs
    // several MiB. Delta generations with periodic materialization keep the
    // committed metadata within a small multiple of one live snapshot.
    let bytes = std::fs::read(&path).unwrap();
    let metadata_bytes: usize = meta_segments(&bytes).iter().map(|&(_, len)| len).sum();
    assert!(
        metadata_bytes < 1024 * 1024,
        "metadata history is unbounded: {COMMITS} commits wrote {metadata_bytes} META_SEG bytes"
    );

    // Reopening must replay the newest generation and its delta chain, not
    // every generation ever committed.
    let started = Instant::now();
    let reopened = RvfStore::open_readonly(&path)
        .unwrap_or_else(|e| panic!("{COMMITS} metadata commits must stay openable: {e:?}"));
    let open_time = started.elapsed();
    assert!(
        open_time.as_secs() < 10,
        "open replayed superseded metadata history: took {open_time:?}"
    );

    for i in 0..COMMITS {
        assert_eq!(
            reopened.get_metadata(i).unwrap(),
            vec![MetadataEntry {
                field_id: 1,
                value: MetadataValue::String(format!("record-{i}")),
            }],
            "record {i} must survive reopen"
        );
    }
    drop(reopened);

    // Compaction is the repair path: it drops every superseded generation and
    // rewrites one full snapshot.
    let mut store = RvfStore::open(&path).unwrap();
    store.compact().unwrap();
    store.close().unwrap();

    let size_after_compact = std::fs::metadata(&path).unwrap().len();
    assert!(
        size_after_compact < size_before_compact / 2,
        "compact() must reclaim superseded history: {size_before_compact} -> {size_after_compact}"
    );
    let compacted = std::fs::read(&path).unwrap();
    assert_eq!(
        meta_segments(&compacted).len(),
        1,
        "compaction must leave exactly one authoritative metadata generation"
    );

    let reopened = RvfStore::open_readonly(&path).unwrap();
    for i in 0..COMMITS {
        assert_eq!(
            reopened.get_metadata(i).unwrap(),
            vec![MetadataEntry {
                field_id: 1,
                value: MetadataValue::String(format!("record-{i}")),
            }],
            "record {i} must survive compaction"
        );
    }
}

/// Build a store with `commits` single-record metadata commits and return its
/// raw bytes alongside the file path.
fn commit_metadata_history(path: &std::path::Path, commits: u64) {
    let mut store = RvfStore::create(path, make_options(2)).unwrap();
    for i in 0..commits {
        store
            .ingest_batch_with_metadata(
                &[&[i as f32, 1.0]],
                &[i],
                &[record(i, &format!("value-{i}"))],
            )
            .unwrap();
    }
    store.close().unwrap();
}

fn expected_record(i: u64) -> Vec<MetadataEntry> {
    vec![MetadataEntry {
        field_id: 1,
        value: MetadataValue::String(format!("value-{i}")),
    }]
}

/// A generation older than the newest full snapshot contributes nothing to the
/// committed state, so corrupting one must not make the artifact unopenable
/// (ADR-280 §1: only the newest generation is authoritative).
#[test]
fn corrupt_superseded_generation_does_not_brick_the_artifact() {
    // Long enough that the runtime has materialized at least one later full
    // snapshot, which is what makes the earliest generations superseded.
    const COMMITS: u64 = 96;

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("superseded_corruption.rvf");
    commit_metadata_history(&path, COMMITS);

    let mut bytes = std::fs::read(&path).unwrap();
    let segments = meta_segments(&bytes);
    assert_eq!(segments.len() as u64, COMMITS);

    // Flip a byte in the oldest generation's payload.
    let (start, len) = segments[0];
    assert!(len > 0);
    bytes[start + len / 2] ^= 0xFF;
    std::fs::write(&path, &bytes).unwrap();

    let reopened = RvfStore::open_readonly(&path).unwrap_or_else(|e| {
        panic!("corruption in a superseded metadata generation must be tolerated: {e:?}")
    });
    for i in 0..COMMITS {
        assert_eq!(
            reopened.get_metadata(i).unwrap(),
            expected_record(i),
            "the authoritative snapshot must still resolve record {i}"
        );
    }
}

/// A torn newest generation -- a crash between appending the payload and
/// publishing the manifest -- must fall back to the previous complete snapshot
/// rather than failing the open (ADR-280 §4: never a mixed or torn snapshot).
#[test]
fn torn_newest_generation_falls_back_to_the_previous_snapshot() {
    const COMMITS: u64 = 3;

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("torn_newest.rvf");
    commit_metadata_history(&path, COMMITS);

    let mut bytes = std::fs::read(&path).unwrap();
    let segments = meta_segments(&bytes);
    assert_eq!(segments.len() as u64, COMMITS);

    let (start, len) = segments[segments.len() - 1];
    assert!(len > 0);
    bytes[start + len / 2] ^= 0xFF;
    std::fs::write(&path, &bytes).unwrap();

    let reopened = RvfStore::open_readonly(&path)
        .unwrap_or_else(|e| panic!("a torn newest generation must fall back: {e:?}"));
    for i in 0..COMMITS - 1 {
        assert_eq!(
            reopened.get_metadata(i).unwrap(),
            expected_record(i),
            "the previous complete snapshot must resolve record {i}"
        );
    }
    assert!(
        reopened.get_metadata(COMMITS - 1).is_none(),
        "the torn generation must not be partially applied"
    );
}

/// One delta generation may both delete a field from one record and set that
/// same field on another. The deletion names the field without constraining its
/// type, so it must not collide with the real type declared by the other record.
#[test]
fn delta_mixing_field_deletion_and_reuse_round_trips() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("delta_field_reuse.rvf");

    let mut store = RvfStore::create(&path, make_options(2)).unwrap();
    store
        .ingest_batch_with_metadata(
            &[&[1.0, 0.0], &[0.0, 1.0]],
            &[1, 2],
            &[
                VectorMetadata {
                    vector_id: 1,
                    fields: vec![MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::U64(10),
                    }],
                    delete_record: false,
                },
                VectorMetadata {
                    vector_id: 2,
                    fields: vec![MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::U64(20),
                    }],
                    delete_record: false,
                },
            ],
        )
        .unwrap();

    // Vector 1 (the lower identifier, so encoded first) drops field 1 while
    // vector 3 introduces it with a concrete type in the same generation.
    store
        .ingest_batch_with_metadata(
            &[&[0.5, 0.5]],
            &[3],
            &[
                VectorMetadata {
                    vector_id: 1,
                    fields: vec![MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::DeleteField,
                    }],
                    delete_record: false,
                },
                VectorMetadata {
                    vector_id: 3,
                    fields: vec![MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::U64(30),
                    }],
                    delete_record: false,
                },
            ],
        )
        .unwrap();
    store.close().unwrap();

    let reopened = RvfStore::open_readonly(&path).unwrap();
    assert_eq!(
        reopened.get_metadata(1).unwrap(),
        vec![],
        "the deleted field must not survive reopen"
    );
    assert_eq!(
        reopened.get_metadata(2).unwrap(),
        vec![MetadataEntry {
            field_id: 1,
            value: MetadataValue::U64(20),
        }],
        "an untouched record must be inherited from the base generation"
    );
    assert_eq!(
        reopened.get_metadata(3).unwrap(),
        vec![MetadataEntry {
            field_id: 1,
            value: MetadataValue::U64(30),
        }]
    );
}

/// A `derive()` child starts with no vectors, so it must not persist metadata
/// records that reference identifiers only the parent holds (ADR-280 §4: records
/// are validated against the resulting committed vector snapshot).
#[test]
fn derived_child_metadata_write_stays_openable() {
    let dir = TempDir::new().unwrap();
    let parent_path = dir.path().join("derive_parent.rvf");
    let child_path = dir.path().join("derive_child.rvf");
    let dim: u16 = 2;

    let mut parent = RvfStore::create(&parent_path, make_options(dim)).unwrap();
    parent
        .ingest_batch_with_metadata(
            &[&[1.0, 0.0], &[0.0, 1.0]],
            &[1, 2],
            &[record(1, "parent-one"), record(2, "parent-two")],
        )
        .unwrap();

    let mut child = parent
        .derive(&child_path, rvf_types::DerivationType::Clone, None)
        .unwrap();

    // Any metadata-bearing write on the child commits a new META generation.
    child
        .ingest_batch_with_metadata(&[&[0.5, 0.5]], &[7], &[record(7, "child-seven")])
        .unwrap();
    child.close().unwrap();
    parent.close().unwrap();

    let reopened = RvfStore::open_readonly(&child_path)
        .unwrap_or_else(|e| panic!("derived child with metadata must reopen: {e:?}"));
    assert_eq!(
        reopened.get_metadata(7).unwrap(),
        vec![MetadataEntry {
            field_id: 1,
            value: MetadataValue::String("child-seven".into()),
        }]
    );
    assert!(
        reopened.get_metadata(1).is_none(),
        "a derived child holds no parent vectors, so it carries no parent record"
    );
}
