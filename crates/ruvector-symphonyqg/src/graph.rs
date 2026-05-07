/// Number of neighbors evaluated per XNOR-popcount pass.
/// 32 → for dim=128 (16 bytes/code) one batch reads 512 bytes (8 cache lines).
/// Aligning the graph degree to a multiple of BATCH_SIZE eliminates wasted
/// SIMD lanes — the core architectural innovation of SymphonyQG (SIGMOD 2025).
pub const BATCH_SIZE: usize = 32;

/// Round `m` up to the nearest multiple of BATCH_SIZE.
#[inline]
pub fn batch_pad(m: usize) -> usize {
    ((m + BATCH_SIZE - 1) / BATCH_SIZE) * BATCH_SIZE
}

/// Squared Euclidean distance (no sqrt — monotone for ranking).
#[inline]
pub fn dist_l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Cosine distance: 1 − cos(θ).
#[inline]
pub fn dist_cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < f32::EPSILON || nb < f32::EPSILON {
        return 1.0;
    }
    (1.0 - dot / (na * nb)).max(0.0)
}

/// Encode a vector into a packed 1-bit code after random-sign rotation.
///
/// Rotation: y[i] = signs[i] * v[perm[i]]
/// Code bit i = 1 iff y[i] > 0
///
/// This is the randomised sign-flip + permutation approximation to the full
/// random orthogonal rotation used in RaBitQ. For the PoC it produces
/// well-conditioned codes without requiring QR decomposition.
pub fn encode(v: &[f32], signs: &[f32], perm: &[usize]) -> Vec<u8> {
    let dim = v.len();
    let code_bytes = dim / 8;
    let mut code = vec![0u8; code_bytes];
    for i in 0..dim {
        if signs[i] * v[perm[i]] > 0.0 {
            code[i >> 3] |= 1u8 << (i & 7);
        }
    }
    code
}

/// Batch XNOR-popcount estimated cosine distance.
///
/// For each of `n` neighbors whose packed codes are laid out contiguously in
/// `neighbor_codes` (stride = `code_bytes`), returns the estimated distance
/// to `query_code`:
///
///   dist_est ≈ 2 · |{bit positions where q ≠ d}| / (code_bytes * 8)
///
/// The rustc auto-vectoriser (and LLVM's loop vectoriser) will emit
/// VPXOR+VPOPCNT on AVX-512BITALG targets without explicit intrinsics.
pub fn batch_hamming_dist(
    query_code: &[u8],
    neighbor_codes: &[u8],
    n: usize,
    code_bytes: usize,
) -> Vec<f32> {
    let dim_f = (code_bytes * 8) as f32;
    (0..n)
        .map(|i| {
            let c = &neighbor_codes[i * code_bytes..(i + 1) * code_bytes];
            let differing: u32 = query_code
                .iter()
                .zip(c)
                .map(|(&q, &d)| (q ^ d).count_ones())
                .sum();
            2.0 * differing as f32 / dim_f
        })
        .collect()
}

/// CSR-style graph with inline 1-bit codes.
///
/// Memory layout (all flat `Vec` for cache locality):
///
/// ```text
///   vectors   : [n × dim]              f32  — full precision for re-ranking
///   neighbors : [n × m]                u32  — adjacency (m = BATCH_SIZE multiple)
///   nb_codes  : [n × m × code_bytes]   u8   — 1-bit codes inline with edges
///   self_codes: [n × code_bytes]        u8   — vertex's own 1-bit code (for ep seed)
///   signs     : [dim]                  f32
///   perm      : [dim]                  usize
/// ```
#[derive(Clone)]
pub struct SymphonyGraph {
    pub n: usize,
    pub dim: usize,
    pub code_bytes: usize,
    pub m: usize,

    pub vectors: Vec<f32>,
    pub neighbors: Vec<u32>,
    /// Inline 1-bit codes: nb_codes[v*m*code_bytes .. (v+1)*m*code_bytes]
    pub nb_codes: Vec<u8>,
    /// Self codes: self_codes[v*code_bytes .. (v+1)*code_bytes]
    pub self_codes: Vec<u8>,

    pub signs: Vec<f32>,
    pub perm: Vec<usize>,
    pub entry: usize,
}

impl SymphonyGraph {
    /// Neighbor IDs for vertex `v`.
    #[inline]
    pub fn neighbors_of(&self, v: usize) -> &[u32] {
        &self.neighbors[v * self.m..(v + 1) * self.m]
    }

    /// Inline 1-bit codes for vertex `v`'s neighbors.
    #[inline]
    pub fn nb_codes_of(&self, v: usize) -> &[u8] {
        let base = v * self.m * self.code_bytes;
        &self.nb_codes[base..base + self.m * self.code_bytes]
    }

    /// Full-precision vector for vertex `v`.
    #[inline]
    pub fn vector_of(&self, v: usize) -> &[f32] {
        &self.vectors[v * self.dim..(v + 1) * self.dim]
    }

    /// 1-bit self-code for vertex `v`.
    #[inline]
    pub fn self_code_of(&self, v: usize) -> &[u8] {
        &self.self_codes[v * self.code_bytes..(v + 1) * self.code_bytes]
    }

    /// Encode a query with this graph's rotation parameters.
    pub fn encode_query(&self, query: &[f32]) -> Vec<u8> {
        encode(query, &self.signs, &self.perm)
    }

    /// Total heap-allocated bytes.
    pub fn memory_bytes(&self) -> usize {
        self.vectors.len() * 4
            + self.neighbors.len() * 4
            + self.nb_codes.len()
            + self.self_codes.len()
            + self.signs.len() * 4
            + self.perm.len() * 8
    }
}
