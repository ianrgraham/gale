//! Nodal LGL reference element on the 2D quadrilateral `[-1, 1]²`.
//!
//! Built as the tensor product of two [`Reference1d`] lines. Two properties carry
//! straight over from 1D and are the reason quad/hex elements are the GPU fast path
//! (see `docs/dg-gpu-fluid-simulation.md` §5, `docs/mesh-and-adaptivity-strategy.md` §1):
//!
//! - the **mass matrix stays diagonal** (`w_i · w_j` by LGL collocation), and
//! - the derivative operators apply **by sum factorization** — the 1D
//!   differentiation matrix swept along one direction, `O(p^{d+1})` work, *not* a
//!   dense `(p+1)²×(p+1)²` matrix (`O(p^{2d})`).
//!
//! ## Node ordering
//! Node `(i, j)` (with `i` the `r`-index, `j` the `s`-index) lives at flat index
//! `idx = i + j·(p+1)` — i.e. the `r`-direction is contiguous ("fastest"). A field
//! is stored as a length-`(p+1)²` vector in this layout.

use super::reference::Reference1d;

/// A nodal LGL reference element on `[-1, 1]²` of order `p` (`(p+1)²` nodes).
#[derive(Clone, Debug)]
pub struct Reference2dQuad {
    /// Polynomial order `p` (per direction).
    pub order: usize,
    /// The underlying 1D LGL line element (nodes, weights, differentiation matrix).
    pub line: Reference1d,
    /// Node coordinates `[r, s]`, length `(p+1)²`, in `idx = i + j·(p+1)` order.
    pub nodes: Vec<[f64; 2]>,
    /// Diagonal DG-SEM mass (`w_i · w_j`), length `(p+1)²`, same ordering.
    pub mass: Vec<f64>,
}

impl Reference2dQuad {
    /// Construct the order-`p` quad reference element.
    pub fn new(order: usize) -> Self {
        let line = Reference1d::new(order);
        let n = order + 1;
        let mut nodes = Vec::with_capacity(n * n);
        let mut mass = Vec::with_capacity(n * n);
        for j in 0..n {
            for i in 0..n {
                nodes.push([line.nodes[i], line.nodes[j]]);
                mass.push(line.weights[i] * line.weights[j]);
            }
        }
        Self { order, line, nodes, mass }
    }

    /// Number of nodes per direction (`p + 1`).
    pub fn n_1d(&self) -> usize {
        self.order + 1
    }

    /// Total number of nodes (`(p + 1)²`).
    pub fn n_nodes(&self) -> usize {
        let n = self.order + 1;
        n * n
    }

    /// Flat node index for tensor indices `(i, j)`: `i + j·(p+1)`.
    #[inline]
    pub fn node_index(&self, i: usize, j: usize) -> usize {
        i + j * (self.order + 1)
    }

    /// Diagonal of the DG-SEM mass matrix (`= w_i · w_j`), in node order.
    pub fn mass_diagonal(&self) -> &[f64] {
        &self.mass
    }

    /// `∂f/∂r` at every node, via sum factorization (1D `D` along the `r`-direction).
    /// Exact for any polynomial of degree ≤ `p` in `r`.
    pub fn diff_r(&self, f: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; f.len()];
        self.diff_r_into(f, &mut out);
        out
    }

    /// [`diff_r`] writing into a caller-provided `out` (no allocation; for hot matrix-free
    /// loops that reuse per-thread scratch). Bit-for-bit identical to [`diff_r`].
    #[inline]
    pub fn diff_r_into(&self, f: &[f64], out: &mut [f64]) {
        let n = self.n_1d();
        assert_eq!(f.len(), n * n, "diff_r: expected {} values", n * n);
        let d = &self.line.diff; // n×n, row-major
        for j in 0..n {
            for i in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += d[i * n + k] * f[k + j * n];
                }
                out[i + j * n] = acc;
            }
        }
    }

    /// Transpose of [`diff_r`]: `out_i = Σ_k Dᵀ_ik f_k` swept along `r`. Used for
    /// the adjoint (stiffness / SIPG lift) terms.
    pub fn diff_r_t(&self, f: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; f.len()];
        self.diff_r_t_into(f, &mut out);
        out
    }

    /// [`diff_r_t`] writing into a caller-provided `out` (no allocation).
    #[inline]
    pub fn diff_r_t_into(&self, f: &[f64], out: &mut [f64]) {
        let n = self.n_1d();
        assert_eq!(f.len(), n * n, "diff_r_t: expected {} values", n * n);
        let d = &self.line.diff;
        for j in 0..n {
            for i in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += d[k * n + i] * f[k + j * n];
                }
                out[i + j * n] = acc;
            }
        }
    }

    /// Transpose of [`diff_s`], swept along `s`.
    pub fn diff_s_t(&self, f: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; f.len()];
        self.diff_s_t_into(f, &mut out);
        out
    }

    /// [`diff_s_t`] writing into a caller-provided `out` (no allocation).
    #[inline]
    pub fn diff_s_t_into(&self, f: &[f64], out: &mut [f64]) {
        let n = self.n_1d();
        assert_eq!(f.len(), n * n, "diff_s_t: expected {} values", n * n);
        let d = &self.line.diff;
        for j in 0..n {
            for i in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += d[k * n + j] * f[i + k * n];
                }
                out[i + j * n] = acc;
            }
        }
    }

    /// `∂f/∂s` at every node, via sum factorization (1D `D` along the `s`-direction).
    /// Exact for any polynomial of degree ≤ `p` in `s`.
    pub fn diff_s(&self, f: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; f.len()];
        self.diff_s_into(f, &mut out);
        out
    }

    /// [`diff_s`] writing into a caller-provided `out` (no allocation).
    #[inline]
    pub fn diff_s_into(&self, f: &[f64], out: &mut [f64]) {
        let n = self.n_1d();
        assert_eq!(f.len(), n * n, "diff_s: expected {} values", n * n);
        let d = &self.line.diff; // n×n, row-major
        for j in 0..n {
            for i in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += d[j * n + k] * f[i + k * n];
                }
                out[i + j * n] = acc;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX_P: usize = 6;

    fn eval(nodes: &[[f64; 2]], a: i32, b: i32) -> Vec<f64> {
        nodes.iter().map(|c| c[0].powi(a) * c[1].powi(b)).collect()
    }

    #[test]
    fn node_count_and_corners() {
        for p in 1..=MAX_P {
            let e = Reference2dQuad::new(p);
            let n = p + 1;
            assert_eq!(e.n_nodes(), n * n);
            assert_eq!(e.nodes.len(), n * n);
            // The four corners are (±1, ±1).
            assert_eq!(e.nodes[e.node_index(0, 0)], [-1.0, -1.0]);
            assert_eq!(e.nodes[e.node_index(p, 0)], [1.0, -1.0]);
            assert_eq!(e.nodes[e.node_index(0, p)], [-1.0, 1.0]);
            assert_eq!(e.nodes[e.node_index(p, p)], [1.0, 1.0]);
        }
    }

    #[test]
    fn mass_sums_to_area_four() {
        for p in 1..=MAX_P {
            let e = Reference2dQuad::new(p);
            let s: f64 = e.mass.iter().sum();
            assert!((s - 4.0).abs() < 1e-12, "p={p} area={s}");
        }
    }

    #[test]
    fn tensor_quadrature_exact_to_degree_2p_minus_1() {
        // ∫∫ r^a s^b over [-1,1]² = I(a)·I(b), exact for a,b ≤ 2p-1.
        let i1 = |k: usize| if k % 2 == 1 { 0.0 } else { 2.0 / (k as f64 + 1.0) };
        for p in 1..=MAX_P {
            let e = Reference2dQuad::new(p);
            for a in 0..=(2 * p - 1) {
                for b in 0..=(2 * p - 1) {
                    let q: f64 = (0..e.n_nodes())
                        .map(|k| e.mass[k] * e.nodes[k][0].powi(a as i32) * e.nodes[k][1].powi(b as i32))
                        .sum();
                    assert!((q - i1(a) * i1(b)).abs() < 1e-9, "p={p} ∫r^{a}s^{b}={q}");
                }
            }
        }
    }

    #[test]
    fn diff_r_exact_on_polynomials() {
        for p in 1..=MAX_P {
            let e = Reference2dQuad::new(p);
            for a in 0..=p {
                for b in 0..=p {
                    let f = eval(&e.nodes, a as i32, b as i32);
                    let dr = e.diff_r(&f);
                    for k in 0..e.n_nodes() {
                        let (r, s) = (e.nodes[k][0], e.nodes[k][1]);
                        let exact = if a == 0 { 0.0 } else { a as f64 * r.powi(a as i32 - 1) * s.powi(b as i32) };
                        assert!((dr[k] - exact).abs() < 1e-8, "p={p} ∂r r^{a}s^{b}");
                    }
                }
            }
        }
    }

    #[test]
    fn diff_s_exact_on_polynomials() {
        for p in 1..=MAX_P {
            let e = Reference2dQuad::new(p);
            for a in 0..=p {
                for b in 0..=p {
                    let f = eval(&e.nodes, a as i32, b as i32);
                    let ds = e.diff_s(&f);
                    for k in 0..e.n_nodes() {
                        let (r, s) = (e.nodes[k][0], e.nodes[k][1]);
                        let exact = if b == 0 { 0.0 } else { b as f64 * r.powi(a as i32) * s.powi(b as i32 - 1) };
                        assert!((ds[k] - exact).abs() < 1e-8, "p={p} ∂s r^{a}s^{b}");
                    }
                }
            }
        }
    }

    #[test]
    fn mixed_partials_commute() {
        // ∂s(∂r f) == ∂r(∂s f) for a polynomial of degree ≤ p in each direction.
        for p in 2..=MAX_P {
            let e = Reference2dQuad::new(p);
            let pp = p as i32;
            let f: Vec<f64> = e
                .nodes
                .iter()
                .map(|c| c[0].powi(1) * c[1].powi(2) + 3.0 * c[0].powi(pp) * c[1] - c[1].powi(pp))
                .collect();
            let rs = e.diff_s(&e.diff_r(&f));
            let sr = e.diff_r(&e.diff_s(&f));
            for k in 0..e.n_nodes() {
                assert!((rs[k] - sr[k]).abs() < 1e-7, "p={p} mixed-partial mismatch at {k}");
            }
        }
    }
}
