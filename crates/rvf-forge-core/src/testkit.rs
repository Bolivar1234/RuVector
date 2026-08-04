//! Synthetic RVF construction, for tests and fixtures.
//!
//! Forge never writes RVF files in production — it consumes them — so nothing
//! here is part of the packaging path. It exists so that a test can build a
//! container byte by byte and then tamper with it, which is the only way to
//! test that verification actually refuses a tampered artifact.
//!
//! The segment layout written here is the one [`crate::container`] reads:
//!
//! ```text
//! [ 64-byte header ][ payload ][ signature footer, if SIGNED ][ zero pad to 64 ]
//! ```

use ed25519_dalek::SigningKey;
use rvf_types::{SegmentFlags, SegmentType, SEGMENT_ALIGNMENT, SEGMENT_HEADER_SIZE};

/// An Ed25519 keypair derived deterministically from a seed, so that a test
/// that signs a fixture produces the same bytes on every run.
pub struct TestKeypair {
    /// The signing key.
    pub signing: SigningKey,
    /// The 32-byte public key, as [`crate::VerifyOptions::trusted_keys`] takes it.
    pub public: [u8; 32],
}

impl TestKeypair {
    /// Derive a keypair from `seed`. The same seed always yields the same key.
    pub fn deterministic(seed: u8) -> Self {
        let secret = rvf_types::sha256(&[seed; 32]);
        let signing = SigningKey::from_bytes(&secret);
        let public = signing.verifying_key().to_bytes();
        Self { signing, public }
    }
}

/// Serialize a 64-byte segment header.
fn header_bytes(
    seg_type: u8,
    payload: &[u8],
    flags: SegmentFlags,
    segment_id: u64,
    alignment_pad: u32,
) -> [u8; SEGMENT_HEADER_SIZE] {
    const CHECKSUM_ALGO: u8 = 1; // XXH3-128, the format default.
    let content_hash = rvf_wire::hash::compute_content_hash(CHECKSUM_ALGO, payload);

    let mut b = [0u8; SEGMENT_HEADER_SIZE];
    b[0x00..0x04].copy_from_slice(&rvf_types::SEGMENT_MAGIC.to_le_bytes());
    b[0x04] = rvf_types::SEGMENT_VERSION;
    b[0x05] = seg_type;
    b[0x06..0x08].copy_from_slice(&flags.bits().to_le_bytes());
    b[0x08..0x10].copy_from_slice(&segment_id.to_le_bytes());
    b[0x10..0x18].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    // timestamp_ns stays zero: fixtures must be byte-reproducible.
    b[0x20] = CHECKSUM_ALGO;
    b[0x28..0x38].copy_from_slice(&content_hash);
    b[0x3C..0x40].copy_from_slice(&alignment_pad.to_le_bytes());
    b
}

fn pad_to_alignment(buf: &mut Vec<u8>) {
    let padded = buf.len().div_ceil(SEGMENT_ALIGNMENT) * SEGMENT_ALIGNMENT;
    buf.resize(padded, 0);
}

/// Build an unsigned segment.
pub fn unsigned_segment(seg_type: SegmentType, payload: &[u8], segment_id: u64) -> Vec<u8> {
    let unpadded = SEGMENT_HEADER_SIZE + payload.len();
    let pad = unpadded.div_ceil(SEGMENT_ALIGNMENT) * SEGMENT_ALIGNMENT - unpadded;

    let mut out = Vec::new();
    out.extend_from_slice(&header_bytes(
        seg_type as u8,
        payload,
        SegmentFlags::empty(),
        segment_id,
        pad as u32,
    ));
    out.extend_from_slice(payload);
    pad_to_alignment(&mut out);
    out
}

/// Build a segment signed with `keypair`, carrying an Ed25519 signature footer.
pub fn signed_segment(
    seg_type: SegmentType,
    payload: &[u8],
    segment_id: u64,
    keypair: &TestKeypair,
) -> Vec<u8> {
    let flags = SegmentFlags::empty().with(SegmentFlags::SIGNED);
    let footer_len = rvf_types::SignatureFooter::compute_footer_length(64) as usize;
    let unpadded = SEGMENT_HEADER_SIZE + payload.len() + footer_len;
    let pad = unpadded.div_ceil(SEGMENT_ALIGNMENT) * SEGMENT_ALIGNMENT - unpadded;

    let head = header_bytes(seg_type as u8, payload, flags, segment_id, pad as u32);
    let header = rvf_wire::read_segment_header(&head).expect("testkit built a valid header");
    let footer = rvf_crypto::sign_segment(&header, payload, &keypair.signing);

    let mut out = Vec::new();
    out.extend_from_slice(&head);
    out.extend_from_slice(payload);
    out.extend_from_slice(&rvf_crypto::encode_signature_footer(&footer));
    pad_to_alignment(&mut out);
    out
}

/// Build a minimal but complete container: a `META` segment carrying the given
/// capability declaration line, then a `MANIFEST` segment as the root.
pub fn minimal_container(capabilities: &str) -> Vec<u8> {
    let meta = format!("{}={}", crate::CAPABILITY_DECLARATION_KEY, capabilities);
    let mut out = unsigned_segment(SegmentType::Meta, meta.as_bytes(), 1);
    out.extend(unsigned_segment(SegmentType::Manifest, b"root", 2));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsigned_segment_is_aligned_and_parses() {
        let data = unsigned_segment(SegmentType::Meta, b"hello", 3);
        assert_eq!(data.len() % SEGMENT_ALIGNMENT, 0);
        let got = crate::inspect_bytes(&data).unwrap();
        assert_eq!(got.segments.len(), 1);
        assert_eq!(got.segments[0].segment_id, 3);
        assert!(!got.segments[0].signed);
    }

    #[test]
    fn signed_segment_is_aligned_and_carries_a_footer() {
        let kp = TestKeypair::deterministic(1);
        let data = signed_segment(SegmentType::Wasm, b"\0asm\x01\0\0\0", 4, &kp);
        assert_eq!(data.len() % SEGMENT_ALIGNMENT, 0);
        let got = crate::inspect_bytes(&data).unwrap();
        assert!(got.segments[0].signed);
        assert_eq!(
            got.segments[0].signature_algo.as_ref().unwrap().name,
            "ed25519"
        );
        assert_eq!(
            got.segments[0]
                .signature_algo
                .as_ref()
                .unwrap()
                .signature_length,
            64
        );
    }

    #[test]
    fn fixtures_are_byte_reproducible() {
        let kp = TestKeypair::deterministic(2);
        assert_eq!(
            signed_segment(SegmentType::Wasm, b"code", 1, &kp),
            signed_segment(SegmentType::Wasm, b"code", 1, &kp)
        );
        assert_eq!(minimal_container("network"), minimal_container("network"));
    }

    #[test]
    fn the_same_seed_yields_the_same_key() {
        assert_eq!(
            TestKeypair::deterministic(9).public,
            TestKeypair::deterministic(9).public
        );
        assert_ne!(
            TestKeypair::deterministic(9).public,
            TestKeypair::deterministic(10).public
        );
    }
}
