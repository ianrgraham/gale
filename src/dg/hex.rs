//! Nodal LGL reference element on the 3D hexahedron `[-1, 1]³`.
//!
//! The tensor product of three [`Reference1d`] lines — the 3D analogue of
//! [`Reference2dQuad`](super::quad::Reference2dQuad). The same two properties that
//! make quads/hexes the GPU fast path carry over from 1D/2D:
//!
//! - the **mass matrix stays diagonal** (`w_i · w_j · w_k` by LGL collocation), and
//! - the derivative operators apply **by sum factorization** — the 1D
//!   differentiation matrix swept along one axis, `O(p⁴)` work per element, not a
//!   dense `(p+1)³×(p+1)³` matrix.
//!
//! ## Node ordering
//! Node `(i, j, k)` (with `i` the `r`-index, `j` the `s`-index, `k` the `t`-index)
//! lives at flat index `idx = i + j·n + k·n²` where `n = p+1` — i.e. `r` is
//! contiguous ("fastest"), then `s`, then `t`. A field is a length-`n³` vector.

use super::reference::Reference1d;

/// A nodal LGL reference element on `[-1, 1]³` of order `p` (`(p+1)³` nodes).
#[derive(Clone, Debug)]
pub struct Reference3dHex {
    /// Polynomial order `p` (per direction).
    pub order: usize,
    /// The underlying 1D LGL line element (nodes, weights, differentiation matrix).
    pub line: Reference1d,
    /// Node coordinates `[r, s, t]`, length `(p+1)³`, in `idx = i + j·n + k·n²` order.
    pub nodes: Vec<[f64; 3]>,
    /// Diagonal DG-SEM mass (`w_i · w_j · w_k`), length `(p+1)³`, same ordering.
    pub mass: Vec<f64>,
}

impl Reference3dHex {
    /// Construct the order-`p` hex reference element.
    pub fn new(order: usize) -> Self {
        let line = Reference1d::new(order);
        let n = order + 1;
        let mut nodes = Vec::with_capacity(n * n * n);
        let mut mass = Vec::with_capacity(n * n * n);
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    nodes.push([line.nodes[i], line.nodes[j], line.nodes[k]]);
                    mass.push(line.weights[i] * line.weights[j] * line.weights[k]);
                }
            }
        }
        Self { order, line, nodes, mass }
    }

    /// Number of nodes per direction (`p + 1`).
    pub fn n_1d(&self) -> usize {
        self.order + 1
    }

    /// Total number of nodes (`(p + 1)³`).
    pub fn n_nodes(&self) -> usize {
        let n = self.order + 1;
        n * n * n
    }

    /// Flat node index for tensor indices `(i, j, k)`: `i + j·n + k·n²`.
    #[inline]
    pub fn node_index(&self, i: usize, j: usize, k: usize) -> usize {
        let n = self.order + 1;
        i + j * n + k * n * n
    }

    /// Diagonal of the DG-SEM mass matrix (`= w_i · w_j · w_k`), in node order.
    pub fn mass_diagonal(&self) -> &[f64] {
        &self.mass
    }

    /// `∂f/∂r` at every node (1D `D` swept along the contiguous `r`-axis). Exact for
    /// any polynomial of degree ≤ `p` in `r`.
    pub fn diff_r(&self, f: &[f64]) -> Vec<f64> {
        let n = self.n_1d();
        assert_eq!(f.len(), n * n * n, "diff_r: expected {} values", n * n * n);
        let d = &self.line.diff;
        let mut out = vec![0.0; n * n * n];
        for k in 0..n {
            for j in 0..n {
                let base = j * n + k * n * n;
                for i in 0..n {
                    let mut acc = 0.0;
                    for a in 0..n {
                        acc += d[i * n + a] * f[a + base];
                    }
                    out[i + base] = acc;
                }
            }
        }
        out
    }

    /// `∂f/∂s` at every node (1D `D` swept along the `s`-axis, stride `n`).
    pub fn diff_s(&self, f: &[f64]) -> Vec<f64> {
        let n = self.n_1d();
        assert_eq!(f.len(), n * n * n, "diff_s: expected {} values", n * n * n);
        let d = &self.line.diff;
        let mut out = vec![0.0; n * n * n];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let mut acc = 0.0;
                    for a in 0..n {
                        acc += d[j * n + a] * f[i + a * n + k * n * n];
                    }
                    out[i + j * n + k * n * n] = acc;
                }
            }
        }
        out
    }

    /// `∂f/∂t` at every node (1D `D` swept along the `t`-axis, stride `n²`).
    pub fn diff_t(&self, f: &[f64]) -> Vec<f64> {
        let n = self.n_1d();
        assert_eq!(f.len(), n * n * n, "diff_t: expected {} values", n * n * n);
        let d = &self.line.diff;
        let mut out = vec![0.0; n * n * n];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let mut acc = 0.0;
                    for a in 0..n {
                        acc += d[k * n + a] * f[i + j * n + a * n * n];
                    }
                    out[i + j * n + k * n * n] = acc;
                }
            }
        }
        out
    }

    /// Transpose of [`diff_r`]: `Dᵀ` swept along `r`. Used for adjoint/stiffness terms.
    pub fn diff_r_t(&self, f: &[f64]) -> Vec<f64> {
        let n = self.n_1d();
        assert_eq!(f.len(), n * n * n, "diff_r_t: expected {} values", n * n * n);
        let d = &self.line.diff;
        let mut out = vec![0.0; n * n * n];
        for k in 0..n {
            for j in 0..n {
                let base = j * n + k * n * n;
                for i in 0..n {
                    let mut acc = 0.0;
                    for a in 0..n {
                        acc += d[a * n + i] * f[a + base];
                    }
                    out[i + base] = acc;
                }
            }
        }
        out
    }

    /// Transpose of [`diff_s`], swept along `s`.
    pub fn diff_s_t(&self, f: &[f64]) -> Vec<f64> {
        let n = self.n_1d();
        assert_eq!(f.len(), n * n * n, "diff_s_t: expected {} values", n * n * n);
        let d = &self.line.diff;
        let mut out = vec![0.0; n * n * n];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let mut acc = 0.0;
                    for a in 0..n {
                        acc += d[a * n + j] * f[i + a * n + k * n * n];
                    }
                    out[i + j * n + k * n * n] = acc;
                }
            }
        }
        out
    }

    /// Transpose of [`diff_t`], swept along `t`.
    pub fn diff_t_t(&self, f: &[f64]) -> Vec<f64> {
        let n = self.n_1d();
        assert_eq!(f.len(), n * n * n, "diff_t_t: expected {} values", n * n * n);
        let d = &self.line.diff;
        let mut out = vec![0.0; n * n * n];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let mut acc = 0.0;
                    for a in 0..n {
                        acc += d[a * n + k] * f[i + j * n + a * n * n];
                    }
                    out[i + j * n + k * n * n] = acc;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX_P: usize = 4;

    fn eval(nodes: &[[f64; 3]], a: i32, b: i32, c: i32) -> Vec<f64> {
        nodes.iter().map(|p| p[0].powi(a) * p[1].powi(b) * p[2].powi(c)).collect()
    }

    #[test]
    fn node_count_and_corners() {
        for p in 1..=MAX_P {
            let e = Reference3dHex::new(p);
            let n = p + 1;
            assert_eq!(e.n_nodes(), n * n * n);
            assert_eq!(e.nodes.len(), n * n * n);
            // The eight corners are (±1, ±1, ±1).
            assert_eq!(e.nodes[e.node_index(0, 0, 0)], [-1.0, -1.0, -1.0]);
            assert_eq!(e.nodes[e.node_index(p, 0, 0)], [1.0, -1.0, -1.0]);
            assert_eq!(e.nodes[e.node_index(0, p, 0)], [-1.0, 1.0, -1.0]);
            assert_eq!(e.nodes[e.node_index(0, 0, p)], [-1.0, -1.0, 1.0]);
            assert_eq!(e.nodes[e.node_index(p, p, p)], [1.0, 1.0, 1.0]);
        }
    }

    #[test]
    fn mass_sums_to_volume_eight() {
        for p in 1..=MAX_P {
            let e = Reference3dHex::new(p);
            let s: f64 = e.mass.iter().sum();
            assert!((s - 8.0).abs() < 1e-12, "p={p} volume={s}");
        }
    }

    #[test]
    fn tensor_quadrature_exact_to_degree_2p_minus_1() {
        // ∫ r^a s^b t^c over [-1,1]³ = I(a)·I(b)·I(c), exact for each ≤ 2p-1.
        let i1 = |k: usize| if k % 2 == 1 { 0.0 } else { 2.0 / (k as f64 + 1.0) };
        for p in 1..=MAX_P {
            let e = Reference3dHex::new(p);
            let hi = 2 * p - 1;
            for a in 0..=hi {
                for b in 0..=hi {
                    for c in 0..=hi {
                        let q: f64 = (0..e.n_nodes())
                            .map(|k| {
                                e.mass[k]
                                    * e.nodes[k][0].powi(a as i32)
                                    * e.nodes[k][1].powi(b as i32)
                                    * e.nodes[k][2].powi(c as i32)
                            })
                            .sum();
                        let want = i1(a) * i1(b) * i1(c);
                        assert!((q - want).abs() < 1e-9, "p={p} ∫r^{a}s^{b}t^{c}={q}");
                    }
                }
            }
        }
    }

    #[test]
    fn diff_exact_on_polynomials() {
        for p in 1..=MAX_P {
            let e = Reference3dHex::new(p);
            for a in 0..=p {
                for b in 0..=p {
                    for c in 0..=p {
                        let f = eval(&e.nodes, a as i32, b as i32, c as i32);
                        let dr = e.diff_r(&f);
                        let ds = e.diff_s(&f);
                        let dt = e.diff_t(&f);
                        for k in 0..e.n_nodes() {
                            let (r, s, t) = (e.nodes[k][0], e.nodes[k][1], e.nodes[k][2]);
                            let er = if a == 0 { 0.0 } else { a as f64 * r.powi(a as i32 - 1) * s.powi(b as i32) * t.powi(c as i32) };
                            let es = if b == 0 { 0.0 } else { b as f64 * r.powi(a as i32) * s.powi(b as i32 - 1) * t.powi(c as i32) };
                            let et = if c == 0 { 0.0 } else { c as f64 * r.powi(a as i32) * s.powi(b as i32) * t.powi(c as i32 - 1) };
                            assert!((dr[k] - er).abs() < 1e-7, "p={p} ∂r r^{a}s^{b}t^{c}");
                            assert!((ds[k] - es).abs() < 1e-7, "p={p} ∂s r^{a}s^{b}t^{c}");
                            assert!((dt[k] - et).abs() < 1e-7, "p={p} ∂t r^{a}s^{b}t^{c}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn mixed_partials_commute() {
        for p in 2..=MAX_P {
            let e = Reference3dHex::new(p);
            let pp = p as i32;
            let f: Vec<f64> = e
                .nodes
                .iter()
                .map(|c| c[0].powi(1) * c[1].powi(2) * c[2] + 3.0 * c[0].powi(pp) * c[2] - c[1].powi(pp))
                .collect();
            let rs = e.diff_s(&e.diff_r(&f));
            let sr = e.diff_r(&e.diff_s(&f));
            let rt = e.diff_t(&e.diff_r(&f));
            let tr = e.diff_r(&e.diff_t(&f));
            for k in 0..e.n_nodes() {
                assert!((rs[k] - sr[k]).abs() < 1e-7, "rs≠sr at {k}");
                assert!((rt[k] - tr[k]).abs() < 1e-7, "rt≠tr at {k}");
            }
        }
    }
}
