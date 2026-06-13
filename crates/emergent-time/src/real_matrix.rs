//! Dense real matrices with a symmetric eigensolver.
//!
//! The symmetric Jacobi eigendecomposition is the numerical workhorse of the
//! whole crate: every "time-from-the-inside" construction ultimately reduces to
//! the spectrum of some Hermitian operator (a Hamiltonian, a density matrix, or
//! a modular Hamiltonian). We keep real symmetric Hamiltonians/density matrices,
//! which is enough to drive complex unitary evolution downstream.

/// Row-major dense `n x n` real matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct RealMatrix {
    pub n: usize,
    pub data: Vec<f64>,
}

impl RealMatrix {
    /// Zero matrix of dimension `n`.
    pub fn zeros(n: usize) -> Self {
        RealMatrix {
            n,
            data: vec![0.0; n * n],
        }
    }

    /// Identity matrix of dimension `n`.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    /// Diagonal matrix from the supplied entries.
    pub fn diag(d: &[f64]) -> Self {
        let mut m = Self::zeros(d.len());
        for (i, &v) in d.iter().enumerate() {
            m.set(i, i, v);
        }
        m
    }

    /// Build from a closure `f(row, col)`.
    pub fn from_fn(n: usize, f: impl Fn(usize, usize) -> f64) -> Self {
        let mut m = Self::zeros(n);
        for r in 0..n {
            for c in 0..n {
                m.set(r, c, f(r, c));
            }
        }
        m
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.n + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.n + c] = v;
    }

    /// Matrix product `self * other`.
    pub fn matmul(&self, other: &RealMatrix) -> RealMatrix {
        assert_eq!(self.n, other.n);
        let n = self.n;
        let mut out = RealMatrix::zeros(n);
        for r in 0..n {
            for k in 0..n {
                let a = self.get(r, k);
                if a == 0.0 {
                    continue;
                }
                for c in 0..n {
                    out.data[r * n + c] += a * other.get(k, c);
                }
            }
        }
        out
    }

    /// Matrix-vector product `self * v`.
    pub fn matvec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(self.n, v.len());
        let n = self.n;
        let mut out = vec![0.0; n];
        for r in 0..n {
            let mut acc = 0.0;
            for c in 0..n {
                acc += self.get(r, c) * v[c];
            }
            out[r] = acc;
        }
        out
    }

    /// Column `c` as a vector.
    pub fn column(&self, c: usize) -> Vec<f64> {
        (0..self.n).map(|r| self.get(r, c)).collect()
    }

    /// Maximum absolute off-diagonal entry — a symmetry/convergence probe.
    pub fn max_offdiag(&self) -> f64 {
        let mut m = 0.0f64;
        for r in 0..self.n {
            for c in 0..self.n {
                if r != c {
                    m = m.max(self.get(r, c).abs());
                }
            }
        }
        m
    }

    /// Eigendecomposition of a **symmetric** matrix via cyclic two-sided Jacobi
    /// rotations.
    ///
    /// Returns `(eigenvalues, eigenvectors)` where the eigenvectors are the
    /// columns of the returned orthogonal matrix `V`, so `self == V * diag(λ) * Vᵀ`.
    /// Robust and accurate for the small matrices used throughout this crate.
    pub fn symmetric_eigen(&self) -> (Vec<f64>, RealMatrix) {
        let n = self.n;
        let mut a = self.clone();
        let mut v = RealMatrix::identity(n);
        if n == 0 {
            return (Vec::new(), v);
        }
        if n == 1 {
            return (vec![a.get(0, 0)], v);
        }

        for _sweep in 0..100 {
            // Sum of squared off-diagonal elements — the Jacobi convergence measure.
            let mut off = 0.0;
            for p in 0..n {
                for q in (p + 1)..n {
                    off += a.get(p, q).powi(2);
                }
            }
            if off < 1e-28 {
                break;
            }

            for p in 0..n {
                for q in (p + 1)..n {
                    let apq = a.get(p, q);
                    if apq.abs() < 1e-300 {
                        continue;
                    }
                    let app = a.get(p, p);
                    let aqq = a.get(q, q);

                    // Rotation angle that zeros the (p, q) entry.
                    let theta = (aqq - app) / (2.0 * apq);
                    let t = if theta == 0.0 {
                        1.0
                    } else {
                        theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                    };
                    let c = 1.0 / (t * t + 1.0).sqrt();
                    let s = t * c;

                    // Left rotation: update columns p, q of A.
                    for k in 0..n {
                        let akp = a.get(k, p);
                        let akq = a.get(k, q);
                        a.set(k, p, c * akp - s * akq);
                        a.set(k, q, s * akp + c * akq);
                    }
                    // Right rotation: update rows p, q of A.
                    for k in 0..n {
                        let apk = a.get(p, k);
                        let aqk = a.get(q, k);
                        a.set(p, k, c * apk - s * aqk);
                        a.set(q, k, s * apk + c * aqk);
                    }
                    // Accumulate the rotation into the eigenvector matrix.
                    for k in 0..n {
                        let vkp = v.get(k, p);
                        let vkq = v.get(k, q);
                        v.set(k, p, c * vkp - s * vkq);
                        v.set(k, q, s * vkp + c * vkq);
                    }
                }
            }
        }

        let eigvals: Vec<f64> = (0..n).map(|i| a.get(i, i)).collect();
        (eigvals, v)
    }

    /// Reconstruct a symmetric matrix from a spectrum and an eigenvector matrix:
    /// `V * diag(λ) * Vᵀ`.
    pub fn from_spectrum(eigvals: &[f64], vecs: &RealMatrix) -> RealMatrix {
        let n = vecs.n;
        RealMatrix::from_fn(n, |r, c| {
            let mut acc = 0.0;
            for k in 0..n {
                acc += vecs.get(r, k) * eigvals[k] * vecs.get(c, k);
            }
            acc
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigen_of_diagonal() {
        let m = RealMatrix::diag(&[3.0, -1.0, 2.0]);
        let (vals, _v) = m.symmetric_eigen();
        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - -1.0).abs() < 1e-10);
        assert!((sorted[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn eigen_reconstructs() {
        // Symmetric 2x2.
        let m = RealMatrix::from_fn(2, |r, c| if r == c { 2.0 } else { 0.5 });
        let (vals, v) = m.symmetric_eigen();
        let recon = RealMatrix::from_spectrum(&vals, &v);
        for i in 0..4 {
            assert!((recon.data[i] - m.data[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn eigenvectors_orthonormal() {
        let m = RealMatrix::from_fn(3, |r, c| ((r + 1) * (c + 1)) as f64 % 5.0 + if r == c { 1.0 } else { 0.0 });
        // symmetrize
        let m = RealMatrix::from_fn(3, |r, c| 0.5 * (m.get(r, c) + m.get(c, r)));
        let (_vals, v) = m.symmetric_eigen();
        let vt = RealMatrix::from_fn(3, |r, c| v.get(c, r));
        let id = vt.matmul(&v);
        for r in 0..3 {
            for c in 0..3 {
                let expect = if r == c { 1.0 } else { 0.0 };
                assert!((id.get(r, c) - expect).abs() < 1e-9);
            }
        }
    }
}
