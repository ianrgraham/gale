//! Per-element geometry of a 3D hexahedron: physical node coordinates, the 3×3
//! curvilinear metric terms, the Jacobian determinant, and the physical mass
//! diagonal. The 3D analogue of [`QuadGeometry`](super::geometry::QuadGeometry).
//!
//! Metrics are the inverse of the 3×3 Jacobian `J = ∂(x,y,z)/∂(r,s,t)` computed by
//! cofactors. For **affine** hexes (boxes, parallelepipeds — the meshes the 3D
//! buildout starts with) `J` is constant and these metrics satisfy the discrete
//! geometric conservation law exactly, so free-stream is preserved to round-off.
//!
//! NOTE (curved elements): for genuinely curved (high-order) hexes the naive
//! cofactor metrics can violate the GCL and break free-stream; the **curl-form**
//! metric identities (Kopriva 2006) are required there. They reduce to the
//! cofactor form when `J` is constant, so this implementation is correct for the
//! affine meshes in use now; switching to curl-form metrics is a prerequisite
//! before curved elements are introduced (see `docs/3d-strategy.md` §5).

use super::hex::Reference3dHex;

/// Straight-sided/curvilinear hex geometry sampled at the reference nodes.
#[derive(Clone, Debug)]
pub struct HexGeometry {
    pub order: usize,
    /// Physical coordinates at the nodes, in reference node ordering.
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    /// Contravariant metric terms `∂(r,s,t)/∂(x,y,z)` at each node (9 components).
    pub rx: Vec<f64>,
    pub ry: Vec<f64>,
    pub rz: Vec<f64>,
    pub sx: Vec<f64>,
    pub sy: Vec<f64>,
    pub sz: Vec<f64>,
    pub tx: Vec<f64>,
    pub ty: Vec<f64>,
    pub tz: Vec<f64>,
    /// Jacobian determinant `detJ` at each node (> 0 for a right-handed hex).
    pub jac: Vec<f64>,
    /// Physical mass diagonal `detJ · w_ref` at each node.
    pub jw: Vec<f64>,
}

impl HexGeometry {
    /// Build the geometry of a hex from its 8 corner vertices via the trilinear map.
    ///
    /// Corner order: `corners[c]` sits at reference signs
    /// `(sx, sy, sz) = (bit0, bit1, bit2)` of `c` mapped `0→-1, 1→+1` — i.e.
    /// `c = cx + 2·cy + 4·cz`. So `corners[0] = (-1,-1,-1)`, `corners[7] = (1,1,1)`.
    pub fn from_corners(refh: &Reference3dHex, corners: [[f64; 3]; 8]) -> Self {
        let nn = refh.n_nodes();
        let mut x = vec![0.0; nn];
        let mut y = vec![0.0; nn];
        let mut z = vec![0.0; nn];
        for (k, c) in refh.nodes.iter().enumerate() {
            let (r, s, t) = (c[0], c[1], c[2]);
            // Trilinear shape function for each corner.
            let mut nx = 0.0;
            let mut ny = 0.0;
            let mut nz = 0.0;
            for (cc, corner) in corners.iter().enumerate() {
                let sr = if cc & 1 == 0 { -1.0 } else { 1.0 };
                let ss = if cc & 2 == 0 { -1.0 } else { 1.0 };
                let st = if cc & 4 == 0 { -1.0 } else { 1.0 };
                let n = 0.125 * (1.0 + sr * r) * (1.0 + ss * s) * (1.0 + st * t);
                nx += n * corner[0];
                ny += n * corner[1];
                nz += n * corner[2];
            }
            x[k] = nx;
            y[k] = ny;
            z[k] = nz;
        }

        // Jacobian columns ∂X/∂ξ by differentiating the nodal coordinate fields.
        let xr = refh.diff_r(&x);
        let xs = refh.diff_s(&x);
        let xt = refh.diff_t(&x);
        let yr = refh.diff_r(&y);
        let ys = refh.diff_s(&y);
        let yt = refh.diff_t(&y);
        let zr = refh.diff_r(&z);
        let zs = refh.diff_s(&z);
        let zt = refh.diff_t(&z);

        let (mut rx, mut ry, mut rz) = (vec![0.0; nn], vec![0.0; nn], vec![0.0; nn]);
        let (mut sx, mut sy, mut sz) = (vec![0.0; nn], vec![0.0; nn], vec![0.0; nn]);
        let (mut tx, mut ty, mut tz) = (vec![0.0; nn], vec![0.0; nn], vec![0.0; nn]);
        let mut jac = vec![0.0; nn];
        let mut jw = vec![0.0; nn];
        for k in 0..nn {
            // J = [a b c; d e f; g h i] with a=xr,b=xs,c=xt, d=yr,e=ys,f=yt, g=zr,h=zs,i=zt.
            let (a, b, c) = (xr[k], xs[k], xt[k]);
            let (d, e, f) = (yr[k], ys[k], yt[k]);
            let (g, h, i) = (zr[k], zs[k], zt[k]);
            let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
            jac[k] = det;
            // J⁻¹ = adj(J)/det, with J⁻¹[row][col] = ∂ξ_row/∂x_col.
            rx[k] = (e * i - f * h) / det;
            ry[k] = (c * h - b * i) / det;
            rz[k] = (b * f - c * e) / det;
            sx[k] = (f * g - d * i) / det;
            sy[k] = (a * i - c * g) / det;
            sz[k] = (c * d - a * f) / det;
            tx[k] = (d * h - e * g) / det;
            ty[k] = (b * g - a * h) / det;
            tz[k] = (a * e - b * d) / det;
            jw[k] = det * refh.mass[k];
        }

        Self { order: refh.order, x, y, z, rx, ry, rz, sx, sy, sz, tx, ty, tz, jac, jw }
    }

    /// Physical mass diagonal (`detJ · w_ref`).
    pub fn mass_diagonal(&self) -> &[f64] {
        &self.jw
    }

    /// `∂f/∂x` at every node: `r_x ∂_r f + s_x ∂_s f + t_x ∂_t f` (chain rule).
    pub fn grad_x(&self, refh: &Reference3dHex, f: &[f64]) -> Vec<f64> {
        let fr = refh.diff_r(f);
        let fs = refh.diff_s(f);
        let ft = refh.diff_t(f);
        (0..f.len()).map(|k| self.rx[k] * fr[k] + self.sx[k] * fs[k] + self.tx[k] * ft[k]).collect()
    }

    /// `∂f/∂y` at every node.
    pub fn grad_y(&self, refh: &Reference3dHex, f: &[f64]) -> Vec<f64> {
        let fr = refh.diff_r(f);
        let fs = refh.diff_s(f);
        let ft = refh.diff_t(f);
        (0..f.len()).map(|k| self.ry[k] * fr[k] + self.sy[k] * fs[k] + self.ty[k] * ft[k]).collect()
    }

    /// `∂f/∂z` at every node.
    pub fn grad_z(&self, refh: &Reference3dHex, f: &[f64]) -> Vec<f64> {
        let fr = refh.diff_r(f);
        let fs = refh.diff_s(f);
        let ft = refh.diff_t(f);
        (0..f.len()).map(|k| self.rz[k] * fr[k] + self.sz[k] * fs[k] + self.tz[k] * ft[k]).collect()
    }

    /// Adjoint of [`grad_x`]: `Dxᵀ v = Drᵀ(r_x⊙v) + Dsᵀ(s_x⊙v) + Dtᵀ(t_x⊙v)`.
    pub fn gradx_t(&self, refh: &Reference3dHex, v: &[f64]) -> Vec<f64> {
        let n = v.len();
        let a = refh.diff_r_t(&(0..n).map(|k| self.rx[k] * v[k]).collect::<Vec<_>>());
        let b = refh.diff_s_t(&(0..n).map(|k| self.sx[k] * v[k]).collect::<Vec<_>>());
        let c = refh.diff_t_t(&(0..n).map(|k| self.tx[k] * v[k]).collect::<Vec<_>>());
        (0..n).map(|k| a[k] + b[k] + c[k]).collect()
    }

    /// Adjoint of [`grad_y`].
    pub fn grady_t(&self, refh: &Reference3dHex, v: &[f64]) -> Vec<f64> {
        let n = v.len();
        let a = refh.diff_r_t(&(0..n).map(|k| self.ry[k] * v[k]).collect::<Vec<_>>());
        let b = refh.diff_s_t(&(0..n).map(|k| self.sy[k] * v[k]).collect::<Vec<_>>());
        let c = refh.diff_t_t(&(0..n).map(|k| self.ty[k] * v[k]).collect::<Vec<_>>());
        (0..n).map(|k| a[k] + b[k] + c[k]).collect()
    }

    /// Adjoint of [`grad_z`].
    pub fn gradz_t(&self, refh: &Reference3dHex, v: &[f64]) -> Vec<f64> {
        let n = v.len();
        let a = refh.diff_r_t(&(0..n).map(|k| self.rz[k] * v[k]).collect::<Vec<_>>());
        let b = refh.diff_s_t(&(0..n).map(|k| self.sz[k] * v[k]).collect::<Vec<_>>());
        let c = refh.diff_t_t(&(0..n).map(|k| self.tz[k] * v[k]).collect::<Vec<_>>());
        (0..n).map(|k| a[k] + b[k] + c[k]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX_P: usize = 4;

    /// Reference cube corners in `c = cx + 2cy + 4cz` order.
    fn unit_cube() -> [[f64; 3]; 8] {
        let mut c = [[0.0; 3]; 8];
        for (cc, slot) in c.iter_mut().enumerate() {
            *slot = [
                if cc & 1 == 0 { -1.0 } else { 1.0 },
                if cc & 2 == 0 { -1.0 } else { 1.0 },
                if cc & 4 == 0 { -1.0 } else { 1.0 },
            ];
        }
        c
    }

    /// Affine box `[x0,x1]×[y0,y1]×[z0,z1]` corners in the same order.
    fn box_corners(xr: [f64; 2], yr: [f64; 2], zr: [f64; 2]) -> [[f64; 3]; 8] {
        let mut c = [[0.0; 3]; 8];
        for (cc, slot) in c.iter_mut().enumerate() {
            *slot = [
                if cc & 1 == 0 { xr[0] } else { xr[1] },
                if cc & 2 == 0 { yr[0] } else { yr[1] },
                if cc & 4 == 0 { zr[0] } else { zr[1] },
            ];
        }
        c
    }

    #[test]
    fn reference_identity_map() {
        // Corners = reference cube ⇒ x=r, y=s, z=t, detJ=1, identity metrics.
        for p in 1..=MAX_P {
            let refh = Reference3dHex::new(p);
            let g = HexGeometry::from_corners(&refh, unit_cube());
            for k in 0..refh.n_nodes() {
                assert!((g.x[k] - refh.nodes[k][0]).abs() < 1e-13);
                assert!((g.y[k] - refh.nodes[k][1]).abs() < 1e-13);
                assert!((g.z[k] - refh.nodes[k][2]).abs() < 1e-13);
                assert!((g.jac[k] - 1.0).abs() < 1e-11);
                // Identity metric tensor.
                assert!((g.rx[k] - 1.0).abs() < 1e-9 && g.ry[k].abs() < 1e-9 && g.rz[k].abs() < 1e-9);
                assert!(g.sx[k].abs() < 1e-9 && (g.sy[k] - 1.0).abs() < 1e-9 && g.sz[k].abs() < 1e-9);
                assert!(g.tx[k].abs() < 1e-9 && g.ty[k].abs() < 1e-9 && (g.tz[k] - 1.0).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn affine_box_constant_jacobian_and_volume() {
        // [0,2]×[1,4]×[-1,2]: detJ = (1)(1.5)(1.5) = 2.25 constant; volume = 2·3·3 = 18.
        let corners = box_corners([0.0, 2.0], [1.0, 4.0], [-1.0, 2.0]);
        for p in 1..=MAX_P {
            let refh = Reference3dHex::new(p);
            let g = HexGeometry::from_corners(&refh, corners);
            for &j in &g.jac {
                assert!((j - 2.25).abs() < 1e-10, "p={p} detJ={j}");
            }
            let vol: f64 = g.jw.iter().sum();
            assert!((vol - 18.0).abs() < 1e-9, "p={p} volume={vol}");
        }
    }

    #[test]
    fn metric_inverse_is_consistent() {
        // J · J⁻¹ = I at every node, for a general (non-axis-aligned) parallelepiped.
        // Corners = affine map X = A·ξ + b with a sheared A; build via the 8 signs.
        let a = [[1.3, 0.4, -0.2], [0.1, 1.1, 0.3], [0.2, -0.3, 0.9]];
        let mut corners = [[0.0; 3]; 8];
        for (cc, slot) in corners.iter_mut().enumerate() {
            let s = [
                if cc & 1 == 0 { -1.0 } else { 1.0 },
                if cc & 2 == 0 { -1.0 } else { 1.0 },
                if cc & 4 == 0 { -1.0 } else { 1.0 },
            ];
            for (row, slot_r) in slot.iter_mut().enumerate() {
                *slot_r = a[row][0] * s[0] + a[row][1] * s[1] + a[row][2] * s[2];
            }
        }
        let refh = Reference3dHex::new(3);
        let g = HexGeometry::from_corners(&refh, corners);
        // Recompute J columns to multiply against the stored inverse.
        let xr = refh.diff_r(&g.x);
        let xs = refh.diff_s(&g.x);
        let xt = refh.diff_t(&g.x);
        let yr = refh.diff_r(&g.y);
        let ys = refh.diff_s(&g.y);
        let yt = refh.diff_t(&g.y);
        let zr = refh.diff_r(&g.z);
        let zs = refh.diff_s(&g.z);
        let zt = refh.diff_t(&g.z);
        for k in 0..refh.n_nodes() {
            // Rows of J⁻¹ are (rx,ry,rz),(sx,sy,sz),(tx,ty,tz); columns of J are
            // (xr,yr,zr),(xs,ys,zs),(xt,yt,zt). Product must be I.
            let jinv = [
                [g.rx[k], g.ry[k], g.rz[k]],
                [g.sx[k], g.sy[k], g.sz[k]],
                [g.tx[k], g.ty[k], g.tz[k]],
            ];
            let jcol = [[xr[k], xs[k], xt[k]], [yr[k], ys[k], yt[k]], [zr[k], zs[k], zt[k]]];
            // (J⁻¹ · J)[a][b] = Σ_m jinv[a][m] · J[m][b], J[m][b] = jcol[m][b].
            for ar in 0..3 {
                for br in 0..3 {
                    let mut s = 0.0;
                    for m in 0..3 {
                        s += jinv[ar][m] * jcol[m][br];
                    }
                    let want = if ar == br { 1.0 } else { 0.0 };
                    assert!((s - want).abs() < 1e-9, "J⁻¹J[{ar}][{br}]={s}");
                }
            }
        }
    }

    #[test]
    fn grad_exact_on_polynomials_affine() {
        // On an affine box the chain-rule gradient recovers ∂/∂x of a polynomial exactly.
        let corners = box_corners([0.0, 2.0], [1.0, 4.0], [-1.0, 2.0]);
        for p in 1..=MAX_P {
            let refh = Reference3dHex::new(p);
            let g = HexGeometry::from_corners(&refh, corners);
            for a in 0..=p {
                for b in 0..=p {
                    for c in 0..=p {
                        let f: Vec<f64> = (0..refh.n_nodes())
                            .map(|k| g.x[k].powi(a as i32) * g.y[k].powi(b as i32) * g.z[k].powi(c as i32))
                            .collect();
                        let dx = g.grad_x(&refh, &f);
                        let dy = g.grad_y(&refh, &f);
                        let dz = g.grad_z(&refh, &f);
                        for k in 0..refh.n_nodes() {
                            let (xx, yy, zz) = (g.x[k], g.y[k], g.z[k]);
                            let ex = if a == 0 { 0.0 } else { a as f64 * xx.powi(a as i32 - 1) * yy.powi(b as i32) * zz.powi(c as i32) };
                            let ey = if b == 0 { 0.0 } else { b as f64 * xx.powi(a as i32) * yy.powi(b as i32 - 1) * zz.powi(c as i32) };
                            let ez = if c == 0 { 0.0 } else { c as f64 * xx.powi(a as i32) * yy.powi(b as i32) * zz.powi(c as i32 - 1) };
                            assert!((dx[k] - ex).abs() < 1e-6, "p={p} ∂x");
                            assert!((dy[k] - ey).abs() < 1e-6, "p={p} ∂y");
                            assert!((dz[k] - ez).abs() < 1e-6, "p={p} ∂z");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn free_stream_constant_has_zero_gradient() {
        // A uniform field has zero gradient on a (non-trivial affine) element — the
        // geometry-level free-stream check.
        let corners = box_corners([-2.0, 3.0], [0.5, 1.5], [10.0, 11.0]);
        let refh = Reference3dHex::new(MAX_P);
        let g = HexGeometry::from_corners(&refh, corners);
        let f = vec![3.14159; refh.n_nodes()];
        for (v, comp) in [(g.grad_x(&refh, &f), "x"), (g.grad_y(&refh, &f), "y"), (g.grad_z(&refh, &f), "z")] {
            let m = v.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
            assert!(m < 1e-11, "∂{comp} of constant = {m}");
        }
    }
}
