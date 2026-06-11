//! Version-2 witness log: keyed-BLAKE3 chain MACs with 128-bit links
//! (ADR-134 v2).
//!
//! Fixes the two structural weaknesses of the v1 chain:
//!
//! 1. **Cost on the syscall path.** v1 computed two hashes per append
//!    (record hash + chain hash, SHA-256 when `crypto-sha256` is on).
//!    v2 computes exactly **one** keyed BLAKE3 compression: the MAC
//!    input is `content[0..44] || prev_mac[16]` = 60 bytes, which fits
//!    a single 64-byte BLAKE3 block.
//! 2. **32-bit folded links.** v1 folded its 64-bit chain values to
//!    32 bits to fit the 64-byte record; a forged link collides after
//!    ~2^16 attempts. v2 stores full 128-bit MACs (`prev_mac`,
//!    `chain_mac`), and because the MAC is keyed, forging any link
//!    requires the chain key, not just a hash collision.
//!
//! Per-record signatures are intentionally **absent** in v2: tamper
//! evidence for exported segments comes from Merkle sealing (see
//! [`crate::seal`]), which amortizes one signature over a whole
//! segment instead of paying HMAC per record.

use crate::seal::{SealedSegment, SegmentAccumulator, SegmentSealSigner, DEFAULT_SEGMENT_SIZE};
use rvm_types::{WitnessRecord, WitnessRecordV2};
use spin::Mutex;

/// Domain-separation context string for chain key derivation.
pub const CHAIN_KEY_CONTEXT: &str = "rvm-witness 2026 v2 chain key";

/// Derive a 32-byte chain key from arbitrary key material using
/// BLAKE3's `derive_key` mode with the [`CHAIN_KEY_CONTEXT`] domain.
#[must_use]
pub fn derive_chain_key(material: &[u8]) -> [u8; 32] {
    blake3::derive_key(CHAIN_KEY_CONTEXT, material)
}

/// The compile-time default chain key.
///
/// **Security warning:** this key is public. Production deployments
/// MUST supply a TEE- or boot-derived key via [`WitnessLogV2::with_key`].
#[must_use]
pub fn default_chain_key() -> [u8; 32] {
    derive_chain_key(b"rvm-witness-default-chain-key-v2")
}

/// Compute a v2 chain MAC: `trunc128(BLAKE3_keyed(key, content || prev_mac))`.
///
/// `content` must be the first [`WitnessRecordV2::CONTENT_LEN`] bytes of
/// the record's canonical serialization (which includes `sequence`).
/// The 60-byte input fits one BLAKE3 block: exactly one keyed
/// compression per call.
#[must_use]
pub fn compute_chain_mac_v2(
    key: &[u8; 32],
    content: &[u8],
    prev_mac: &[u8; 16],
) -> [u8; 16] {
    debug_assert_eq!(content.len(), WitnessRecordV2::CONTENT_LEN);
    let mut buf = [0u8; WitnessRecordV2::MAC_INPUT_LEN];
    buf[..WitnessRecordV2::CONTENT_LEN].copy_from_slice(content);
    buf[WitnessRecordV2::CONTENT_LEN..].copy_from_slice(prev_mac);
    let hash = blake3::keyed_hash(key, &buf);
    let mut mac = [0u8; 16];
    mac.copy_from_slice(&hash.as_bytes()[..16]);
    mac
}

/// Append-only ring buffer of v2 witness records.
///
/// `N` is the ring capacity; `SEG` is the Merkle segment size (leaves
/// accumulated between seals, default [`DEFAULT_SEGMENT_SIZE`]).
pub struct WitnessLogV2<const N: usize, const SEG: usize = DEFAULT_SEGMENT_SIZE> {
    inner: Mutex<Inner<N, SEG>>,
}

struct Inner<const N: usize, const SEG: usize> {
    records: [WitnessRecordV2; N],
    write_pos: usize,
    /// Current 128-bit chain head (the last record's `chain_mac`, or
    /// the genesis value if empty).
    head_mac: [u8; 16],
    /// Genesis value the chain started from (zero, or a v1 anchor).
    genesis: [u8; 16],
    sequence: u64,
    total_emitted: u64,
    total_overwritten: u64,
    key: [u8; 32],
    segment: SegmentAccumulator<SEG>,
    /// Records appended while the current segment was already full
    /// (their leaves were NOT accumulated; seal more often to avoid).
    segment_dropped: u64,
}

impl<const N: usize, const SEG: usize> WitnessLogV2<N, SEG> {
    const _ASSERT_N_NONZERO: () = assert!(N > 0, "witness log capacity must be > 0");
    const _ASSERT_SEG_NONZERO: () = assert!(SEG > 0, "segment size must be > 0");

    /// Create an empty v2 log using the default chain key.
    ///
    /// **Security warning:** the default key is public; use
    /// [`Self::with_key`] with a TEE/boot-derived key in production.
    #[must_use]
    pub fn new() -> Self {
        Self::with_key(default_chain_key())
    }

    /// Create an empty v2 log with the given 32-byte chain key.
    #[must_use]
    pub fn with_key(key: [u8; 32]) -> Self {
        Self::with_key_and_genesis(key, [0u8; 16])
    }

    /// Create an empty v2 log whose chain starts from `genesis` instead
    /// of zero.
    ///
    /// Used to anchor a migrated v1 log: pass
    /// [`crate::versioned::v1_head_to_genesis`] of the verified v1 chain
    /// head so the first v2 record's `prev_mac` cryptographically binds
    /// the v1 history.
    #[must_use]
    pub fn with_key_and_genesis(key: [u8; 32], genesis: [u8; 16]) -> Self {
        let () = Self::_ASSERT_N_NONZERO;
        let () = Self::_ASSERT_SEG_NONZERO;
        Self {
            inner: Mutex::new(Inner {
                records: [WitnessRecordV2::zeroed(); N],
                write_pos: 0,
                head_mac: genesis,
                genesis,
                sequence: 0,
                total_emitted: 0,
                total_overwritten: 0,
                key,
                segment: SegmentAccumulator::new(0),
                segment_dropped: 0,
            }),
        }
    }

    /// Append a v2 record built from content fields.
    ///
    /// Fills `version`, `sequence`, `prev_mac`, and `chain_mac`, then
    /// stores the record. Returns the assigned sequence number.
    ///
    /// Cost: one keyed BLAKE3 compression (60-byte input) plus
    /// bookkeeping. No per-record signature is computed; use
    /// [`Self::seal_segment`] for exportable tamper evidence.
    pub fn append(&self, mut record: WitnessRecordV2) -> u64 {
        let mut inner = self.inner.lock();

        record.version = WitnessRecordV2::VERSION;
        record.sequence = inner.sequence;
        record.prev_mac = inner.head_mac;
        let bytes = record.to_bytes();
        record.chain_mac = compute_chain_mac_v2(
            &inner.key,
            &bytes[..WitnessRecordV2::CONTENT_LEN],
            &record.prev_mac,
        );

        let seq = record.sequence;
        if inner.total_emitted >= N as u64 {
            inner.total_overwritten += 1;
        }
        let pos = inner.write_pos;
        inner.records[pos] = record;
        inner.write_pos = (pos + 1) % N;
        inner.head_mac = record.chain_mac;
        inner.sequence = seq.wrapping_add(1);
        inner.total_emitted += 1;

        // Accumulate the leaf for Merkle sealing (bookkeeping only).
        if !inner.segment.push(record.chain_mac) {
            inner.segment_dropped += 1;
        }

        seq
    }

    /// Append using the content fields of a v1 [`WitnessRecord`].
    ///
    /// Convenience for callers that still build v1 structs (emitters,
    /// gates); the v1 chain fields are ignored and replaced by v2 MACs.
    pub fn append_v1_content(&self, record: &WitnessRecord) -> u64 {
        self.append(WitnessRecordV2::from_v1_content(record))
    }

    /// Seal the current Merkle segment with `signer` and start a new one.
    ///
    /// Returns the [`SealedSegment`] (root + signature + metadata) and a
    /// copy of the [`SegmentAccumulator`] so the caller can export
    /// inclusion proofs for any record in the sealed segment. Returns
    /// `None` if no records were accumulated since the last seal.
    ///
    /// This is the **only** place signature cost is paid: one signature
    /// per up-to-`SEG` records, off the per-record append path.
    pub fn seal_segment<G: SegmentSealSigner>(
        &self,
        signer: &G,
    ) -> Option<(SealedSegment, SegmentAccumulator<SEG>)> {
        let mut inner = self.inner.lock();
        if inner.segment.is_empty() {
            return None;
        }
        let acc = inner.segment;
        let sealed = acc.seal(signer)?;
        inner.segment = SegmentAccumulator::new(inner.sequence);
        Some((sealed, acc))
    }

    /// Current 128-bit chain head (the last record's `chain_mac`).
    ///
    /// Export this to external anchoring (CT log, QMDB, remote
    /// attestation) so truncation of the tail is detectable.
    pub fn chain_head(&self) -> [u8; 16] {
        self.inner.lock().head_mac
    }

    /// The genesis value this chain started from.
    pub fn genesis(&self) -> [u8; 16] {
        self.inner.lock().genesis
    }

    /// The 32-byte chain MAC key (needed by verifiers; treat as secret).
    pub fn chain_key(&self) -> [u8; 32] {
        self.inner.lock().key
    }

    /// Number of leaves in the current (unsealed) segment.
    pub fn segment_len(&self) -> usize {
        self.inner.lock().segment.len()
    }

    /// True when the current segment is full and should be sealed.
    pub fn segment_is_full(&self) -> bool {
        self.inner.lock().segment.is_full()
    }

    /// Number of records appended while the segment was full (their
    /// leaves were not accumulated; they remain chain-protected only).
    pub fn segment_dropped(&self) -> u64 {
        self.inner.lock().segment_dropped
    }

    /// Total number of records ever emitted.
    pub fn total_emitted(&self) -> u64 {
        self.inner.lock().total_emitted
    }

    /// Number of records silently overwritten by ring wrap-around.
    pub fn total_overwritten(&self) -> u64 {
        self.inner.lock().total_overwritten
    }

    /// True when the used slot count has reached `watermark`.
    pub fn needs_drain(&self, watermark: usize) -> bool {
        self.len() >= watermark
    }

    /// Number of records currently in the buffer.
    #[allow(clippy::cast_possible_truncation)]
    pub fn len(&self) -> usize {
        let total = self.inner.lock().total_emitted;
        if total >= N as u64 { N } else { total as usize }
    }

    /// True if no records have been emitted.
    pub fn is_empty(&self) -> bool {
        self.inner.lock().total_emitted == 0
    }

    /// Copy of the record at the given ring index.
    pub fn get(&self, ring_index: usize) -> Option<WitnessRecordV2> {
        if ring_index >= N {
            return None;
        }
        let inner = self.inner.lock();
        if inner.total_emitted == 0 {
            return None;
        }
        Some(inner.records[ring_index])
    }

    /// Copies the most recent records into `buf`. Returns count copied.
    pub fn snapshot(&self, buf: &mut [WitnessRecordV2]) -> usize {
        let inner = self.inner.lock();
        #[allow(clippy::cast_possible_truncation)]
        let available = if inner.total_emitted >= N as u64 {
            N
        } else {
            inner.total_emitted as usize
        };
        let to_copy = buf.len().min(available);
        if to_copy == 0 {
            return 0;
        }
        let start = if inner.total_emitted >= N as u64 {
            inner.write_pos
        } else {
            0
        };
        for (i, slot) in buf.iter_mut().enumerate().take(to_copy) {
            let idx = (start + (available - to_copy) + i) % N;
            *slot = inner.records[idx];
        }
        to_copy
    }
}

impl<const N: usize, const SEG: usize> Default for WitnessLogV2<N, SEG> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvm_types::ActionKind;

    fn make_record(kind: ActionKind, actor: u32, target: u64, ts: u64) -> WitnessRecordV2 {
        let mut r = WitnessRecordV2::zeroed();
        r.action_kind = kind as u8;
        r.actor_partition_id = actor;
        r.target_object_id = target;
        r.timestamp_ns = ts;
        r
    }

    #[test]
    fn append_assigns_sequence_and_macs() {
        let log = WitnessLogV2::<16>::new();
        let s0 = log.append(make_record(ActionKind::PartitionCreate, 1, 100, 1000));
        let s1 = log.append(make_record(ActionKind::CapabilityGrant, 1, 200, 2000));
        assert_eq!(s0, 0);
        assert_eq!(s1, 1);

        let r0 = log.get(0).unwrap();
        let r1 = log.get(1).unwrap();
        assert_eq!(r0.version, 2);
        assert_eq!(r0.prev_mac, [0u8; 16]); // genesis
        assert_ne!(r0.chain_mac, [0u8; 16]);
        assert_eq!(r1.prev_mac, r0.chain_mac); // full-width link
        assert_eq!(log.chain_head(), r1.chain_mac);
    }

    #[test]
    fn chain_mac_is_keyed() {
        let mut r = WitnessRecordV2::zeroed();
        r.sequence = 7;
        let bytes = r.to_bytes();
        let content = &bytes[..WitnessRecordV2::CONTENT_LEN];
        let m1 = compute_chain_mac_v2(&derive_chain_key(b"key-a"), content, &[0; 16]);
        let m2 = compute_chain_mac_v2(&derive_chain_key(b"key-b"), content, &[0; 16]);
        assert_ne!(m1, m2, "different keys must produce different MACs");
    }

    #[test]
    fn chain_mac_binds_prev() {
        let r = WitnessRecordV2::zeroed();
        let bytes = r.to_bytes();
        let content = &bytes[..WitnessRecordV2::CONTENT_LEN];
        let key = default_chain_key();
        let m1 = compute_chain_mac_v2(&key, content, &[0x11; 16]);
        let m2 = compute_chain_mac_v2(&key, content, &[0x22; 16]);
        assert_ne!(m1, m2);
    }

    #[test]
    fn ring_wrap_and_counters() {
        let log = WitnessLogV2::<4>::new();
        for i in 0..10u64 {
            log.append(make_record(ActionKind::SchedulerEpoch, 1, i, i * 100));
        }
        assert_eq!(log.total_emitted(), 10);
        assert_eq!(log.len(), 4);
        assert_eq!(log.total_overwritten(), 6);
    }

    #[test]
    fn genesis_anchoring() {
        let key = default_chain_key();
        let genesis = [0xA5u8; 16];
        let log = WitnessLogV2::<8>::with_key_and_genesis(key, genesis);
        assert_eq!(log.genesis(), genesis);
        assert_eq!(log.chain_head(), genesis); // empty log: head == genesis
        log.append(make_record(ActionKind::BootAttestation, 0, 0, 1));
        assert_eq!(log.get(0).unwrap().prev_mac, genesis);
    }

    #[test]
    fn snapshot_returns_most_recent() {
        let log = WitnessLogV2::<16>::new();
        for i in 0..5u64 {
            log.append(make_record(ActionKind::SchedulerEpoch, 1, i, i * 100));
        }
        let mut buf = [WitnessRecordV2::zeroed(); 3];
        let copied = log.snapshot(&mut buf);
        assert_eq!(copied, 3);
        assert_eq!(buf[0].sequence, 2);
        assert_eq!(buf[2].sequence, 4);
    }

    #[test]
    fn append_v1_content_carries_fields() {
        let log = WitnessLogV2::<8>::new();
        let mut v1 = WitnessRecord::zeroed();
        v1.action_kind = ActionKind::RegionMap as u8;
        v1.proof_tier = 2;
        v1.actor_partition_id = 9;
        v1.target_object_id = 77;
        v1.capability_hash = 0xBEEF;
        v1.payload = [3; 8];
        v1.timestamp_ns = 123;
        // v1 chain fields must be ignored:
        v1.sequence = 999;
        v1.prev_hash = 0xDEAD;
        v1.record_hash = 0xFEED;

        log.append_v1_content(&v1);
        let r = log.get(0).unwrap();
        assert_eq!(r.sequence, 0); // log-assigned, not 999
        assert_eq!(r.action_kind, ActionKind::RegionMap as u8);
        assert_eq!(r.actor_partition_id, 9);
        assert_eq!(r.target_object_id, 77);
        assert_eq!(r.capability_hash, 0xBEEF);
        assert_eq!(r.payload, [3; 8]);
    }

    #[test]
    fn segment_tracking() {
        let log = WitnessLogV2::<64, 4>::new();
        assert_eq!(log.segment_len(), 0);
        for i in 0..4u64 {
            log.append(make_record(ActionKind::SchedulerEpoch, 1, i, i));
        }
        assert!(log.segment_is_full());
        assert_eq!(log.segment_dropped(), 0);
        // Appending past a full segment drops leaves (counted).
        log.append(make_record(ActionKind::SchedulerEpoch, 1, 4, 4));
        assert_eq!(log.segment_dropped(), 1);
    }
}
