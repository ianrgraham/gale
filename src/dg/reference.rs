//! Nodal Legendre–Gauss–Lobatto (LGL) reference element in 1D.
//!
//! The building block of nodal DG-SEM (see `docs/dg-gpu-fluid-simulation.md` §3.2):
//! collocating interpolation *and* quadrature at the LGL points makes the mass
//! matrix **diagonal** and is what enables sum factorization on tensor-product
//! (quad/hex) elements — the GPU fast path. Everything here lives on the reference
//! interval `r ∈ [-1, 1]`; physical elements apply an affine/curved map on top.
//!
//! Operators are formed in closed form (no matrix inversion):
//! - LGL nodes/weights via Newton iteration on the Legendre polynomial,
//! - the differentiation matrix via the classical LGL closed form,
//! - the mass matrix is `diag(weights)` by LGL collocation.

/// A 1D nodal LGL reference element of polynomial order `p` (so `p + 1` nodes).
#[derive(Clone, Debug)]
pub struct Reference1d {
    /// Polynomial order `p`.
    pub order: usize,
    /// LGL nodes on `[-1, 1]`, strictly ascending, length `p + 1`.
    /// Endpoints are exactly `-1` and `+1`.
    pub nodes: Vec<f64>,
    /// LGL quadrature weights, length `p + 1`; strictly positive and summing to 2.
    pub weights: Vec<f64>,
    /// Dense differentiation matrix `D` (row-major, `(p+1) × (p+1)`):
    /// `(D · f)_i = f'(r_i)` exactly for any polynomial `f` of degree ≤ `p`.
    pub diff: Vec<f64>,
}

impl Reference1d {
    /// Construct the order-`p` reference element (`p ≥ 1` for a meaningful element).
    pub fn new(order: usize) -> Self {
        let (nodes, weights) = lgl_nodes_weights(order);
        let diff = differentiation_matrix(order, &nodes);
        Self { order, nodes, weights, diff }
    }

    /// Number of nodes (`p + 1`).
    pub fn n(&self) -> usize {
        self.order + 1
    }

    /// The diagonal DG-SEM mass matrix, returned as its diagonal (= the LGL weights).
    pub fn mass_diagonal(&self) -> &[f64] {
        &self.weights
    }

    /// Apply the differentiation matrix: `out_i = Σ_j D_ij f_j = f'(r_i)`.
    pub fn differentiate(&self, f: &[f64]) -> Vec<f64> {
        let np = self.n();
        assert_eq!(f.len(), np, "differentiate: expected {np} values");
        (0..np)
            .map(|i| (0..np).map(|j| self.diff[i * np + j] * f[j]).sum())
            .collect()
    }
}

/// Legendre polynomials `P_0(x) .. P_n(x)` via the three-term recurrence.
pub(crate) fn legendre_all(n: usize, x: f64) -> Vec<f64> {
    let mut p = vec![0.0; n + 1];
    p[0] = 1.0;
    if n >= 1 {
        p[1] = x;
    }
    for k in 2..=n {
        let kf = k as f64;
        p[k] = ((2.0 * kf - 1.0) * x * p[k - 1] - (kf - 1.0) * p[k - 2]) / kf;
    }
    p
}

/// LGL nodes and weights for polynomial order `n`.
///
/// Newton iteration from the Chebyshev–Gauss–Lobatto initial guess finds the
/// interior roots of `(1 - x²) P_n'(x)`; the endpoints `±1` are fixed points of
/// the iteration. Weights are `w_i = 2 / (n (n+1) P_n(x_i)²)`.
fn lgl_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    let np = n + 1;
    if n == 0 {
        // Degenerate (not used for real elements); define a single centered node.
        return (vec![0.0], vec![2.0]);
    }

    // Initial guess: Chebyshev–Gauss–Lobatto points x_i = cos(π i / n).
    let mut x: Vec<f64> = (0..np)
        .map(|i| (std::f64::consts::PI * i as f64 / n as f64).cos())
        .collect();

    // Newton iteration. Each node's equation x P_n - P_{n-1} = 0 is independent,
    // so the in-place sweep is exact Newton per node.
    let tol = 1e-15;
    for _ in 0..100 {
        let mut max_delta = 0.0_f64;
        for xi in x.iter_mut() {
            let leg = legendre_all(n, *xi);
            let dx = (*xi * leg[n] - leg[n - 1]) / (np as f64 * leg[n]);
            *xi -= dx;
            max_delta = max_delta.max(dx.abs());
        }
        if max_delta < tol {
            break;
        }
    }

    // Weights from the converged nodes.
    let mut w = vec![0.0; np];
    for i in 0..np {
        let pn = legendre_all(n, x[i])[n];
        w[i] = 2.0 / (n as f64 * np as f64 * pn * pn);
    }

    // The cos() guess yields nodes descending (+1 .. -1); sort ascending.
    let mut idx: Vec<usize> = (0..np).collect();
    idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());
    let mut nodes: Vec<f64> = idx.iter().map(|&i| x[i]).collect();
    let weights: Vec<f64> = idx.iter().map(|&i| w[i]).collect();

    // Snap endpoints to exact values (the iteration leaves them at ±1 already).
    nodes[0] = -1.0;
    nodes[np - 1] = 1.0;

    (nodes, weights)
}

/// Classical LGL differentiation matrix (row-major `(n+1)×(n+1)`):
/// `D_ij = P_n(x_i) / (P_n(x_j) (x_i - x_j))` for `i ≠ j`,
/// `D_00 = -n(n+1)/4`, `D_nn = +n(n+1)/4`, interior diagonal 0.
fn differentiation_matrix(n: usize, nodes: &[f64]) -> Vec<f64> {
    let np = n + 1;
    let pn: Vec<f64> = nodes.iter().map(|&x| legendre_all(n, x)[n]).collect();
    let mut d = vec![0.0; np * np];
    for i in 0..np {
        for j in 0..np {
            if i != j {
                d[i * np + j] = pn[i] / (pn[j] * (nodes[i] - nodes[j]));
            }
        }
    }
    if n >= 1 {
        let c = n as f64 * (n + 1) as f64 / 4.0;
        d[0] = -c;
        d[np * np - 1] = c;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX_P: usize = 8;

    #[test]
    fn nodes_endpoints_symmetry_monotone() {
        for p in 1..=MAX_P {
            let e = Reference1d::new(p);
            let np = e.n();
            assert!((e.nodes[0] + 1.0).abs() < 1e-13, "p={p} left endpoint");
            assert!((e.nodes[np - 1] - 1.0).abs() < 1e-13, "p={p} right endpoint");
            for i in 0..np {
                // LGL nodes are symmetric about 0.
                assert!(
                    (e.nodes[i] + e.nodes[np - 1 - i]).abs() < 1e-12,
                    "p={p} symmetry at {i}"
                );
            }
            for i in 1..np {
                assert!(e.nodes[i] > e.nodes[i - 1], "p={p} monotone at {i}");
            }
        }
    }

    #[test]
    fn weights_positive_and_sum_to_two() {
        for p in 1..=MAX_P {
            let e = Reference1d::new(p);
            let s: f64 = e.weights.iter().sum();
            assert!((s - 2.0).abs() < 1e-12, "p={p} sum(w)={s}");
            assert!(e.weights.iter().all(|&w| w > 0.0), "p={p} positive weights");
        }
    }

    #[test]
    fn diff_matrix_is_exact_on_polynomials() {
        // D differentiates any polynomial of degree ≤ p exactly.
        for p in 1..=MAX_P {
            let e = Reference1d::new(p);
            for k in 0..=p {
                let f: Vec<f64> = e.nodes.iter().map(|&x| x.powi(k as i32)).collect();
                let df = e.differentiate(&f);
                for i in 0..e.n() {
                    let exact = if k == 0 {
                        0.0
                    } else {
                        k as f64 * e.nodes[i].powi(k as i32 - 1)
                    };
                    assert!(
                        (df[i] - exact).abs() < 1e-9,
                        "p={p} k={k} i={i}: got {} want {}",
                        df[i],
                        exact
                    );
                }
            }
        }
    }

    #[test]
    fn diff_matrix_rows_sum_to_zero() {
        // D · 1 = 0 (derivative of a constant).
        for p in 1..=MAX_P {
            let e = Reference1d::new(p);
            for v in e.differentiate(&vec![1.0; e.n()]) {
                assert!(v.abs() < 1e-9, "p={p} nonzero const-derivative {v}");
            }
        }
    }

    #[test]
    fn lgl_quadrature_exact_to_degree_2p_minus_1() {
        // (p+1)-point LGL integrates polynomials up to degree 2p-1 exactly.
        for p in 1..=MAX_P {
            let e = Reference1d::new(p);
            for k in 0..=(2 * p - 1) {
                let q: f64 = (0..e.n())
                    .map(|i| e.weights[i] * e.nodes[i].powi(k as i32))
                    .sum();
                let exact = if k % 2 == 1 { 0.0 } else { 2.0 / (k as f64 + 1.0) };
                assert!((q - exact).abs() < 1e-9, "p={p} ∫x^{k}: got {q} want {exact}");
            }
        }
    }
}
