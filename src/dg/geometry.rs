//! Physical-element geometry for quads: the isoparametric map, Jacobian, metric
//! terms, physical gradients, and the physical (curved) mass diagonal.
//!
//! A physical element is the image of the reference square `[-1,1]²` under a
//! **bilinear map from 4 corner vertices** (straight-sided quad; a parallelogram or
//! Cartesian cell is the affine special case). The coordinate fields `x(r,s)`,
//! `y(r,s)` are polynomials of degree ≤ 1 in each direction, hence exactly
//! represented in the order-`p` nodal basis (`p ≥ 1`), so the geometric factors are
//! obtained by differentiating the *nodal coordinate values* with the reference
//! operators — no separate analytic Jacobian needed. This generalizes unchanged to
//! curved (higher-order) elements later.
//!
//! Design: geometry is **decoupled from the operators** — one shared
//! [`Reference2dQuad`] supplies `diff_r`/`diff_s`; each [`QuadGeometry`] stores only
//! per-node metrics. That is the one-reference / many-elements layout real DG codes
//! use (and what the GPU port wants).
//!
//! Corner order (CCW): `corners[0..4]` are the physical positions of the reference
//! corners `(-1,-1)`, `(1,-1)`, `(1,1)`, `(-1,1)`.

use super::quad::Reference2dQuad;

/// Per-node geometric factors of one physical quad element.
#[derive(Clone, Debug)]
pub struct QuadGeometry {
    /// Polynomial order (matches the reference element it was built from).
    pub order: usize,
    /// Physical coordinates at the nodes, in the reference node ordering.
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    /// Metric terms ∂r/∂x, ∂r/∂y, ∂s/∂x, ∂s/∂y at each node.
    pub rx: Vec<f64>,
    pub ry: Vec<f64>,
    pub sx: Vec<f64>,
    pub sy: Vec<f64>,
    /// Jacobian determinant `detJ = x_r y_s − x_s y_r` at each node (> 0 for CCW).
    pub jac: Vec<f64>,
    /// Physical mass diagonal `detJ · w_ref` at each node.
    pub jw: Vec<f64>,
}

impl QuadGeometry {
    /// Build the geometry of a straight-sided quad from its 4 CCW corner vertices,
    /// using the shared reference element `refq` for differentiation.
    pub fn from_corners(refq: &Reference2dQuad, corners: [[f64; 2]; 4]) -> Self {
        let nn = refq.n_nodes();
        let mut x = vec![0.0; nn];
        let mut y = vec![0.0; nn];
        for (k, c) in refq.nodes.iter().enumerate() {
            let (r, s) = (c[0], c[1]);
            // Bilinear shape functions at the reference corners.
            let n = [
                0.25 * (1.0 - r) * (1.0 - s),
                0.25 * (1.0 + r) * (1.0 - s),
                0.25 * (1.0 + r) * (1.0 + s),
                0.25 * (1.0 - r) * (1.0 + s),
            ];
            x[k] = n.iter().zip(corners).map(|(ni, c)| ni * c[0]).sum();
            y[k] = n.iter().zip(corners).map(|(ni, c)| ni * c[1]).sum();
        }

        // Jacobian entries by differentiating the nodal coordinate fields.
        let xr = refq.diff_r(&x);
        let xs = refq.diff_s(&x);
        let yr = refq.diff_r(&y);
        let ys = refq.diff_s(&y);

        let mut jac = vec![0.0; nn];
        let (mut rx, mut ry, mut sx, mut sy) = (vec![0.0; nn], vec![0.0; nn], vec![0.0; nn], vec![0.0; nn]);
        let mut jw = vec![0.0; nn];
        for k in 0..nn {
            let j = xr[k] * ys[k] - xs[k] * yr[k];
            jac[k] = j;
            rx[k] = ys[k] / j;
            ry[k] = -xs[k] / j;
            sx[k] = -yr[k] / j;
            sy[k] = xr[k] / j;
            jw[k] = j * refq.mass[k];
        }

        Self { order: refq.order, x, y, rx, ry, sx, sy, jac, jw }
    }

    /// Physical mass diagonal (`detJ · w_ref`).
    pub fn mass_diagonal(&self) -> &[f64] {
        &self.jw
    }

    /// `∂f/∂x` at every node: `r_x ∂_r f + s_x ∂_s f` (chain rule).
    pub fn grad_x(&self, refq: &Reference2dQuad, f: &[f64]) -> Vec<f64> {
        let fr = refq.diff_r(f);
        let fs = refq.diff_s(f);
        (0..f.len()).map(|k| self.rx[k] * fr[k] + self.sx[k] * fs[k]).collect()
    }

    /// `∂f/∂y` at every node: `r_y ∂_r f + s_y ∂_s f`.
    pub fn grad_y(&self, refq: &Reference2dQuad, f: &[f64]) -> Vec<f64> {
        let fr = refq.diff_r(f);
        let fs = refq.diff_s(f);
        (0..f.len()).map(|k| self.ry[k] * fr[k] + self.sy[k] * fs[k]).collect()
    }

    /// Adjoint of [`grad_x`]: `Dxᵀ v = Drᵀ(r_x ⊙ v) + Dsᵀ(s_x ⊙ v)`. Building block
    /// of the stiffness action and the SIPG lift terms.
    pub fn gradx_t(&self, refq: &Reference2dQuad, v: &[f64]) -> Vec<f64> {
        let n = v.len();
        let rxv: Vec<f64> = (0..n).map(|k| self.rx[k] * v[k]).collect();
        let sxv: Vec<f64> = (0..n).map(|k| self.sx[k] * v[k]).collect();
        let a = refq.diff_r_t(&rxv);
        let b = refq.diff_s_t(&sxv);
        (0..n).map(|k| a[k] + b[k]).collect()
    }

    /// Adjoint of [`grad_y`]: `Dyᵀ v = Drᵀ(r_y ⊙ v) + Dsᵀ(s_y ⊙ v)`.
    pub fn grady_t(&self, refq: &Reference2dQuad, v: &[f64]) -> Vec<f64> {
        let n = v.len();
        let ryv: Vec<f64> = (0..n).map(|k| self.ry[k] * v[k]).collect();
        let syv: Vec<f64> = (0..n).map(|k| self.sy[k] * v[k]).collect();
        let a = refq.diff_r_t(&ryv);
        let b = refq.diff_s_t(&syv);
        (0..n).map(|k| a[k] + b[k]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX_P: usize = 6;

    /// Signed area of a CCW quad via the shoelace formula.
    fn shoelace(c: [[f64; 2]; 4]) -> f64 {
        let mut a = 0.0;
        for i in 0..4 {
            let j = (i + 1) % 4;
            a += c[i][0] * c[j][1] - c[j][0] * c[i][1];
        }
        0.5 * a
    }

    #[test]
    fn reference_identity_map() {
        // Corners = the reference square ⇒ x=r, y=s, detJ=1, identity metrics.
        let corners = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        for p in 1..=MAX_P {
            let refq = Reference2dQuad::new(p);
            let g = QuadGeometry::from_corners(&refq, corners);
            for k in 0..refq.n_nodes() {
                assert!((g.x[k] - refq.nodes[k][0]).abs() < 1e-13);
                assert!((g.y[k] - refq.nodes[k][1]).abs() < 1e-13);
                assert!((g.jac[k] - 1.0).abs() < 1e-12);
                assert!((g.rx[k] - 1.0).abs() < 1e-10 && g.ry[k].abs() < 1e-10);
                assert!(g.sx[k].abs() < 1e-10 && (g.sy[k] - 1.0).abs() < 1e-10);
                assert!((g.jw[k] - refq.mass[k]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn affine_rectangle_constant_jacobian_and_area() {
        // [-1,1]² → [0,2] × [1,4]: detJ = (2/2)(3/2) = 1.5 constant, area = 6.
        let corners = [[0.0, 1.0], [2.0, 1.0], [2.0, 4.0], [0.0, 4.0]];
        for p in 1..=MAX_P {
            let refq = Reference2dQuad::new(p);
            let g = QuadGeometry::from_corners(&refq, corners);
            for &j in &g.jac {
                assert!((j - 1.5).abs() < 1e-10);
            }
            let area: f64 = g.jw.iter().sum();
            assert!((area - 6.0).abs() < 1e-10, "p={p} area={area}");
        }
    }

    #[test]
    fn physical_gradient_exact_on_rectangle() {
        // On an affine cell, physical monomials are polynomials in (r,s) of the same
        // degree, so the discrete physical gradient is exact for a,b ≤ p.
        let corners = [[0.0, 1.0], [2.0, 1.0], [2.0, 4.0], [0.0, 4.0]];
        for p in 1..=MAX_P {
            let refq = Reference2dQuad::new(p);
            let g = QuadGeometry::from_corners(&refq, corners);
            for a in 0..=p {
                for b in 0..=p {
                    let f: Vec<f64> = (0..refq.n_nodes())
                        .map(|k| g.x[k].powi(a as i32) * g.y[k].powi(b as i32))
                        .collect();
                    let gx = g.grad_x(&refq, &f);
                    let gy = g.grad_y(&refq, &f);
                    for k in 0..refq.n_nodes() {
                        let (x, y) = (g.x[k], g.y[k]);
                        let ex = if a == 0 { 0.0 } else { a as f64 * x.powi(a as i32 - 1) * y.powi(b as i32) };
                        let ey = if b == 0 { 0.0 } else { b as f64 * x.powi(a as i32) * y.powi(b as i32 - 1) };
                        assert!((gx[k] - ex).abs() < 1e-7, "p={p} ∂x x^{a}y^{b}");
                        assert!((gy[k] - ey).abs() < 1e-7, "p={p} ∂y x^{a}y^{b}");
                    }
                }
            }
        }
    }

    #[test]
    fn metric_identities_hold_on_general_bilinear_quad() {
        // For ANY (even non-affine) quad, the chain rule must reproduce the physical
        // coordinates' own gradients exactly: ∇x = (1,0), ∇y = (0,1). This is the
        // strongest test of the metric inversion.
        let corners = [[0.0, 0.0], [2.0, 0.3], [2.4, 2.1], [0.2, 1.8]];
        for p in 1..=MAX_P {
            let refq = Reference2dQuad::new(p);
            let g = QuadGeometry::from_corners(&refq, corners);
            let gxx = g.grad_x(&refq, &g.x); // ∂x/∂x = 1
            let gyx = g.grad_y(&refq, &g.x); // ∂x/∂y = 0
            let gxy = g.grad_x(&refq, &g.y); // ∂y/∂x = 0
            let gyy = g.grad_y(&refq, &g.y); // ∂y/∂y = 1
            for k in 0..refq.n_nodes() {
                assert!((gxx[k] - 1.0).abs() < 1e-8, "p={p} ∂x/∂x");
                assert!(gyx[k].abs() < 1e-8, "p={p} ∂x/∂y");
                assert!(gxy[k].abs() < 1e-8, "p={p} ∂y/∂x");
                assert!((gyy[k] - 1.0).abs() < 1e-8, "p={p} ∂y/∂y");
            }
        }
    }

    #[test]
    fn bilinear_quad_area_matches_shoelace_and_is_positive() {
        // detJ is bilinear (degree ≤ 1 each direction) ⇒ LGL integrates it exactly,
        // so ∑ jw equals the planar quad area; CCW corners give detJ > 0.
        let corners = [[0.0, 0.0], [2.0, 0.3], [2.4, 2.1], [0.2, 1.8]];
        let area_exact = shoelace(corners);
        assert!(area_exact > 0.0);
        for p in 1..=MAX_P {
            let refq = Reference2dQuad::new(p);
            let g = QuadGeometry::from_corners(&refq, corners);
            assert!(g.jac.iter().all(|&j| j > 0.0), "p={p} positive Jacobian");
            let area: f64 = g.jw.iter().sum();
            assert!((area - area_exact).abs() < 1e-10, "p={p} area {area} vs {area_exact}");
        }
    }
}
