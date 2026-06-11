//! Merkle segment sealing for the v2 witness log (ADR-134 v2).
//!
//! Per-record appends only *accumulate* the 16-byte chain MAC as a
//! Merkle leaf (a memcpy). All expensive crypto -- leaf hashing, tree
//! construction, and the seal signature -- is paid once per segment in
//! [`SegmentAccumulator::seal`], CT/QMDB-style. A sealed root can be
//! anchored externally, and any record in the segment can be proven
//! included via a logarithmic [`MerkleProof`].
//!
//! Domain separation: leaves are hashed as `BLAKE3(0x00 || seq || mac)`
//! and internal nodes as `BLAKE3(0x01 || left || right)`, preventing
//! leaf/node confusion attacks. Odd nodes are promoted unchanged.

/// Default number of leaves per segment.
///
/// 256 leaves = 4 KiB of buffered MACs and an 8 KiB scratch level at
/// seal time (kernel-stack friendly), with one seal signature amortized
/// over 256 appends. Larger deployments can instantiate
/// `WitnessLogV2<N, 1024>` for cheaper amortization.
pub const DEFAULT_SEGMENT_SIZE: usize = 256;

/// Maximum supported Merkle depth (2^32 leaves; far above any segment).
pub const MAX_MERKLE_DEPTH: usize = 32;

const LEAF_DOMAIN: u8 = 0x00;
const NODE_DOMAIN: u8 = 0x01;
const SEAL_DOMAIN: u8 = 0x02;

/// Signs and verifies sealed segment roots.
///
/// Implemented by [`Blake3SealSigner`] (symmetric, in-crate) and, via
/// the adapter in `rvm-proof`, by every proof-crate `WitnessSigner`
/// (HMAC-SHA256, dual-HMAC, Ed25519, TEE-backed).
pub trait SegmentSealSigner {
    /// Produce a 64-byte signature over a 32-byte seal digest.
    fn sign_root(&self, digest: &[u8; 32]) -> [u8; 64];

    /// Verify a 64-byte signature over a 32-byte seal digest.
    fn verify_root(&self, digest: &[u8; 32], signature: &[u8; 64]) -> bool;
}

/// Keyed-BLAKE3 segment seal signer (symmetric MAC).
///
/// Signature layout: `sig[0..32] = BLAKE3_keyed(key, digest)`,
/// `sig[32..64] = 0`. Not publicly verifiable; single trust domain only.
#[derive(Clone)]
pub struct Blake3SealSigner {
    key: [u8; 32],
}

impl Blake3SealSigner {
    /// Create a seal signer from a 32-byte key.
    #[must_use]
    pub const fn new(key: [u8; 32]) -> Self {
        Self { key }
    }

    fn mac(&self, digest: &[u8; 32]) -> [u8; 32] {
        *blake3::keyed_hash(&self.key, digest).as_bytes()
    }
}

impl SegmentSealSigner for Blake3SealSigner {
    fn sign_root(&self, digest: &[u8; 32]) -> [u8; 64] {
        let mac = self.mac(digest);
        let mut sig = [0u8; 64];
        sig[..32].copy_from_slice(&mac);
        sig
    }

    fn verify_root(&self, digest: &[u8; 32], signature: &[u8; 64]) -> bool {
        let expected = self.sign_root(digest);
        // Branchless comparison (constant time w.r.t. content).
        let mut diff = 0u8;
        for i in 0..64 {
            diff |= expected[i] ^ signature[i];
        }
        diff == 0
    }
}

/// Hash a leaf: `BLAKE3(0x00 || sequence_le || chain_mac)`.
fn leaf_hash(sequence: u64, mac: &[u8; 16]) -> [u8; 32] {
    let mut buf = [0u8; 25];
    buf[0] = LEAF_DOMAIN;
    buf[1..9].copy_from_slice(&sequence.to_le_bytes());
    buf[9..25].copy_from_slice(mac);
    *blake3::hash(&buf).as_bytes()
}

/// Hash an internal node: `BLAKE3(0x01 || left || right)`.
fn node_hash(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut buf = [0u8; 65];
    buf[0] = NODE_DOMAIN;
    buf[1..33].copy_from_slice(left);
    buf[33..65].copy_from_slice(right);
    *blake3::hash(&buf).as_bytes()
}

/// Domain-separated digest that the seal signature covers:
/// `BLAKE3(0x02 || root || first_sequence_le || count_le)`.
///
/// Binding the sequence range prevents replaying a valid (root,
/// signature) pair for a different position in the log.
#[must_use]
pub fn seal_digest(root: &[u8; 32], first_sequence: u64, count: u32) -> [u8; 32] {
    let mut buf = [0u8; 45];
    buf[0] = SEAL_DOMAIN;
    buf[1..33].copy_from_slice(root);
    buf[33..41].copy_from_slice(&first_sequence.to_le_bytes());
    buf[41..45].copy_from_slice(&count.to_le_bytes());
    *blake3::hash(&buf).as_bytes()
}

/// A sealed Merkle segment: exportable, externally anchorable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SealedSegment {
    /// Merkle root over the segment's record chain MACs.
    pub root: [u8; 32],
    /// Sequence number of the first record in the segment.
    pub first_sequence: u64,
    /// Number of records (leaves) in the segment.
    pub count: u32,
    /// Signature over [`seal_digest`]`(root, first_sequence, count)`.
    pub signature: [u8; 64],
}

/// Verify a sealed segment's signature.
#[must_use]
pub fn verify_seal<G: SegmentSealSigner>(segment: &SealedSegment, signer: &G) -> bool {
    let digest = seal_digest(&segment.root, segment.first_sequence, segment.count);
    signer.verify_root(&digest, &segment.signature)
}

/// Merkle inclusion proof for a single record in a sealed segment.
///
/// `siblings[0..depth]` are the authentication path bottom-up;
/// `present` marks levels where the node had a sibling (promotion
/// levels carry the hash up unchanged).
#[derive(Debug, Clone, Copy)]
pub struct MerkleProof {
    /// Sibling hashes, bottom-up. Only `[0..depth]` are meaningful.
    pub siblings: [[u8; 32]; MAX_MERKLE_DEPTH],
    /// Whether a sibling exists at each level (false = promotion).
    pub present: [bool; MAX_MERKLE_DEPTH],
    /// Number of tree levels above the leaves.
    pub depth: u8,
    /// Leaf index within the segment (0-based).
    pub index: u32,
    /// Sequence number of the proven record.
    pub sequence: u64,
}

/// Verify a Merkle inclusion proof against a sealed root.
///
/// `chain_mac` is the 16-byte chain MAC of the record claimed included.
#[must_use]
pub fn verify_inclusion(root: &[u8; 32], chain_mac: &[u8; 16], proof: &MerkleProof) -> bool {
    if usize::from(proof.depth) > MAX_MERKLE_DEPTH {
        return false;
    }
    let mut hash = leaf_hash(proof.sequence, chain_mac);
    let mut index = proof.index;
    for level in 0..usize::from(proof.depth) {
        if proof.present[level] {
            hash = if index & 1 == 0 {
                node_hash(&hash, &proof.siblings[level])
            } else {
                node_hash(&proof.siblings[level], &hash)
            };
        }
        // Promotion: hash carries up unchanged.
        index >>= 1;
    }
    let mut diff = 0u8;
    for i in 0..32 {
        diff |= hash[i] ^ root[i];
    }
    diff == 0
}

/// Fixed-capacity Merkle leaf accumulator for one segment.
///
/// Appends are a 16-byte memcpy; the tree is only computed at seal /
/// proof time. `Copy` so [`crate::WitnessLogV2::seal_segment`] can hand
/// the caller a snapshot for proof generation.
#[derive(Clone, Copy)]
pub struct SegmentAccumulator<const S: usize> {
    leaves: [[u8; 16]; S],
    first_sequence: u64,
    len: usize,
}

impl<const S: usize> SegmentAccumulator<S> {
    /// Create an empty accumulator whose first leaf will correspond to
    /// the record with sequence number `first_sequence`.
    #[must_use]
    pub fn new(first_sequence: u64) -> Self {
        Self {
            leaves: [[0u8; 16]; S],
            first_sequence,
            len: 0,
        }
    }

    /// Append a record's chain MAC as the next leaf.
    ///
    /// Returns `false` (leaf dropped) if the segment is already full.
    pub fn push(&mut self, chain_mac: [u8; 16]) -> bool {
        if self.len >= S {
            return false;
        }
        self.leaves[self.len] = chain_mac;
        self.len += 1;
        true
    }

    /// Number of accumulated leaves.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if no leaves have been accumulated.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// True when the accumulator holds `S` leaves.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.len >= S
    }

    /// Sequence number of the first record in this segment.
    #[must_use]
    pub fn first_sequence(&self) -> u64 {
        self.first_sequence
    }

    /// Compute the Merkle root over the accumulated leaves.
    ///
    /// Returns `None` if the segment is empty.
    #[must_use]
    pub fn compute_root(&self) -> Option<[u8; 32]> {
        if self.len == 0 {
            return None;
        }
        let mut level = [[0u8; 32]; S];
        for (i, (slot, leaf)) in level
            .iter_mut()
            .zip(self.leaves.iter())
            .enumerate()
            .take(self.len)
        {
            *slot = leaf_hash(self.first_sequence + i as u64, leaf);
        }
        let mut width = self.len;
        while width > 1 {
            let mut next = 0;
            let mut i = 0;
            while i < width {
                if i + 1 < width {
                    level[next] = node_hash(&level[i], &level[i + 1]);
                } else {
                    level[next] = level[i]; // promotion
                }
                next += 1;
                i += 2;
            }
            width = next;
        }
        Some(level[0])
    }

    /// Build an inclusion proof for the leaf at `offset` (0-based
    /// within the segment). Returns `None` if out of range.
    #[must_use]
    pub fn inclusion_proof(&self, offset: usize) -> Option<MerkleProof> {
        if offset >= self.len {
            return None;
        }
        let mut proof = MerkleProof {
            siblings: [[0u8; 32]; MAX_MERKLE_DEPTH],
            present: [false; MAX_MERKLE_DEPTH],
            depth: 0,
            #[allow(clippy::cast_possible_truncation)]
            index: offset as u32,
            sequence: self.first_sequence + offset as u64,
        };
        let mut level = [[0u8; 32]; S];
        for (i, (slot, leaf)) in level
            .iter_mut()
            .zip(self.leaves.iter())
            .enumerate()
            .take(self.len)
        {
            *slot = leaf_hash(self.first_sequence + i as u64, leaf);
        }
        let mut width = self.len;
        let mut idx = offset;
        let mut depth = 0usize;
        while width > 1 {
            let sibling = idx ^ 1;
            if sibling < width {
                proof.siblings[depth] = level[sibling];
                proof.present[depth] = true;
            }
            let mut next = 0;
            let mut i = 0;
            while i < width {
                if i + 1 < width {
                    level[next] = node_hash(&level[i], &level[i + 1]);
                } else {
                    level[next] = level[i];
                }
                next += 1;
                i += 2;
            }
            width = next;
            idx >>= 1;
            depth += 1;
            if depth > MAX_MERKLE_DEPTH {
                return None;
            }
        }
        #[allow(clippy::cast_possible_truncation)]
        {
            proof.depth = depth as u8;
        }
        Some(proof)
    }

    /// Build an inclusion proof addressed by record sequence number.
    #[must_use]
    pub fn proof_for_sequence(&self, sequence: u64) -> Option<MerkleProof> {
        let offset = sequence.checked_sub(self.first_sequence)?;
        if offset >= self.len as u64 {
            return None;
        }
        #[allow(clippy::cast_possible_truncation)]
        self.inclusion_proof(offset as usize)
    }

    /// The raw chain MAC stored for the leaf at `offset`.
    #[must_use]
    pub fn leaf(&self, offset: usize) -> Option<[u8; 16]> {
        if offset >= self.len {
            return None;
        }
        Some(self.leaves[offset])
    }

    /// Seal this segment: compute the root and sign
    /// [`seal_digest`]`(root, first_sequence, len)`.
    ///
    /// Returns `None` if the segment is empty.
    #[must_use]
    pub fn seal<G: SegmentSealSigner>(&self, signer: &G) -> Option<SealedSegment> {
        let root = self.compute_root()?;
        #[allow(clippy::cast_possible_truncation)]
        let count = self.len as u32;
        let digest = seal_digest(&root, self.first_sequence, count);
        Some(SealedSegment {
            root,
            first_sequence: self.first_sequence,
            count,
            signature: signer.sign_root(&digest),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filled_acc<const S: usize>(n: usize, first_seq: u64) -> SegmentAccumulator<S> {
        let mut acc = SegmentAccumulator::<S>::new(first_seq);
        for i in 0..n {
            let mut mac = [0u8; 16];
            mac[0] = i as u8;
            mac[1] = 0xC3;
            assert!(acc.push(mac));
        }
        acc
    }

    fn test_signer() -> Blake3SealSigner {
        Blake3SealSigner::new([0x42u8; 32])
    }

    #[test]
    fn root_deterministic_and_content_sensitive() {
        let a = filled_acc::<8>(5, 0);
        let b = filled_acc::<8>(5, 0);
        assert_eq!(a.compute_root(), b.compute_root());

        let mut c = filled_acc::<8>(4, 0);
        let mut mac = [0u8; 16];
        mac[0] = 0xFF;
        c.push(mac);
        assert_ne!(a.compute_root(), c.compute_root());
    }

    #[test]
    fn root_binds_sequence_numbers() {
        let a = filled_acc::<8>(5, 0);
        let b = filled_acc::<8>(5, 100);
        assert_ne!(a.compute_root(), b.compute_root());
    }

    #[test]
    fn empty_segment_has_no_root_or_seal() {
        let acc = SegmentAccumulator::<8>::new(0);
        assert!(acc.compute_root().is_none());
        assert!(acc.seal(&test_signer()).is_none());
    }

    #[test]
    fn seal_and_verify_round_trip() {
        let acc = filled_acc::<8>(7, 10);
        let signer = test_signer();
        let sealed = acc.seal(&signer).unwrap();
        assert_eq!(sealed.first_sequence, 10);
        assert_eq!(sealed.count, 7);
        assert!(verify_seal(&sealed, &signer));
    }

    #[test]
    fn seal_tamper_detected() {
        let acc = filled_acc::<8>(7, 10);
        let signer = test_signer();
        let sealed = acc.seal(&signer).unwrap();

        let mut bad = sealed;
        bad.root[0] ^= 1;
        assert!(!verify_seal(&bad, &signer));

        let mut bad = sealed;
        bad.first_sequence += 1; // range replay
        assert!(!verify_seal(&bad, &signer));

        let mut bad = sealed;
        bad.count -= 1; // truncation claim
        assert!(!verify_seal(&bad, &signer));

        let mut bad = sealed;
        bad.signature[5] ^= 0x80;
        assert!(!verify_seal(&bad, &signer));

        // Wrong key fails.
        assert!(!verify_seal(&sealed, &Blake3SealSigner::new([0x43u8; 32])));
    }

    #[test]
    fn inclusion_proof_verifies_for_every_leaf() {
        // Cover power-of-two and odd (promotion) widths.
        for n in [1usize, 2, 3, 5, 7, 8] {
            let acc = filled_acc::<8>(n, 20);
            let root = acc.compute_root().unwrap();
            for i in 0..n {
                let proof = acc.inclusion_proof(i).unwrap();
                let mac = acc.leaf(i).unwrap();
                assert!(
                    verify_inclusion(&root, &mac, &proof),
                    "leaf {i} of {n} failed"
                );
            }
        }
    }

    #[test]
    fn inclusion_proof_tamper_detected() {
        let acc = filled_acc::<8>(6, 0);
        let root = acc.compute_root().unwrap();
        let proof = acc.inclusion_proof(2).unwrap();
        let mac = acc.leaf(2).unwrap();

        // Wrong MAC fails.
        let mut bad_mac = mac;
        bad_mac[3] ^= 0xFF;
        assert!(!verify_inclusion(&root, &bad_mac, &proof));

        // Wrong sequence fails (leaf hash binds sequence).
        let mut bad = proof;
        bad.sequence += 1;
        assert!(!verify_inclusion(&root, &mac, &bad));

        // Wrong index (position swap) fails.
        let mut bad = proof;
        bad.index ^= 1;
        assert!(!verify_inclusion(&root, &mac, &bad));

        // Corrupted sibling fails.
        let mut bad = proof;
        bad.siblings[0][0] ^= 1;
        assert!(!verify_inclusion(&root, &mac, &bad));

        // Wrong root fails.
        let mut bad_root = root;
        bad_root[0] ^= 1;
        assert!(!verify_inclusion(&bad_root, &mac, &proof));
    }

    #[test]
    fn proof_for_sequence_addressing() {
        let acc = filled_acc::<8>(5, 100);
        let root = acc.compute_root().unwrap();
        let proof = acc.proof_for_sequence(103).unwrap();
        assert_eq!(proof.index, 3);
        assert!(verify_inclusion(&root, &acc.leaf(3).unwrap(), &proof));
        assert!(acc.proof_for_sequence(99).is_none());
        assert!(acc.proof_for_sequence(105).is_none());
    }

    #[test]
    fn push_past_capacity_drops() {
        let mut acc = filled_acc::<4>(4, 0);
        assert!(acc.is_full());
        assert!(!acc.push([9u8; 16]));
        assert_eq!(acc.len(), 4);
    }
}
