//! Symmetric interior-penalty (SIPG) Poisson / Helmholtz on the 3D hex mesh — the
//! 3D analogue of [`Poisson`](super::poisson::Poisson). Matrix-free operator
//! `λM + A` with conjugate-gradient solves. Conforming meshes only for now (3D
//! octree/mortar AMR is a later step).
//!
//! `A u` = volume stiffness `Dxᵀ W Dx + Dyᵀ W Dy + Dzᵀ W Dz` plus the SIPG face
//! terms (consistency `−∮{∇u·n}[v]`, penalty `+∮τ[u][v]`, symmetry lift
//! `−∮{∇v·n}[u]`) over the 6 hex faces. Penalty `τ = α(p+1)²/h`, `h = vol^{1/3}`.

use super::face3d::Face;
use super::mesh3d::{Mesh3d, Neighbor3};

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn add3(a: Vec<f64>, b: Vec<f64>, c: Vec<f64>) -> Vec<f64> {
    (0..a.len()).map(|i| a[i] + b[i] + c[i]).collect()
}

/// Matrix-free SIPG Poisson/Helmholtz operator on a hex mesh.
pub struct Poisson3d<'m> {
    pub mesh: &'m Mesh3d,
    pub alpha: f64,
    pub reaction: f64,
    pub neumann_tags: Vec<u32>,
    /// Per-element length scale `h = vol^{1/3}`.
    h: Vec<f64>,
}

impl<'m> Poisson3d<'m> {
    pub fn new(mesh: &'m Mesh3d, alpha: f64) -> Self {
        Self::with_reaction(mesh, alpha, 0.0)
    }

    /// Helmholtz operator `λM + A` (reaction `λ ≥ 0`); `λ = 0` is pure Poisson.
    pub fn with_reaction(mesh: &'m Mesh3d, alpha: f64, reaction: f64) -> Self {
        Self::with_bc(mesh, alpha, reaction, Vec::new())
    }

    /// Full constructor: reaction `λ` and the set of Neumann boundary tags.
    pub fn with_bc(mesh: &'m Mesh3d, alpha: f64, reaction: f64, neumann_tags: Vec<u32>) -> Self {
        let h = mesh
            .elements
            .iter()
            .map(|e| e.geom.jw.iter().sum::<f64>().cbrt())
            .collect();
        Self { mesh, alpha, reaction, neumann_tags, h }
    }

    fn is_neumann(&self, tag: u32) -> bool {
        self.neumann_tags.contains(&tag)
    }

    fn penalty(&self, e: usize, neighbor: Option<usize>) -> f64 {
        let p1 = (self.mesh.order + 1) as f64;
        let h = match neighbor {
            Some(r) => self.h[e].min(self.h[r]),
            None => self.h[e],
        };
        self.alpha * p1 * p1 / h
    }

    pub fn ndof(&self) -> usize {
        self.mesh.n_elements() * self.mesh.refh.n_nodes()
    }

    /// Volume-stiffness action only (no face terms), fused `pr/ps/pt` form.
    pub fn apply_volume(&self, u: &[f64]) -> Vec<f64> {
        let m = self.mesh;
        let refh = &m.refh;
        let nn = refh.n_nodes();
        let mut r = vec![0.0; self.ndof()];
        for (e, el) in m.elements.iter().enumerate() {
            let ue = &u[e * nn..(e + 1) * nn];
            let gx = el.geom.grad_x(refh, ue);
            let gy = el.geom.grad_y(refh, ue);
            let gz = el.geom.grad_z(refh, ue);
            let mut pr = vec![0.0; nn];
            let mut ps = vec![0.0; nn];
            let mut pt = vec![0.0; nn];
            for k in 0..nn {
                let wx = el.geom.jw[k] * gx[k];
                let wy = el.geom.jw[k] * gy[k];
                let wz = el.geom.jw[k] * gz[k];
                pr[k] = el.geom.rx[k] * wx + el.geom.ry[k] * wy + el.geom.rz[k] * wz;
                ps[k] = el.geom.sx[k] * wx + el.geom.sy[k] * wy + el.geom.sz[k] * wz;
                pt[k] = el.geom.tx[k] * wx + el.geom.ty[k] * wy + el.geom.tz[k] * wz;
            }
            let a = refh.diff_r_t(&pr);
            let b = refh.diff_s_t(&ps);
            let c = refh.diff_t_t(&pt);
            for k in 0..nn {
                r[e * nn + k] = a[k] + b[k] + c[k];
            }
        }
        r
    }

    /// Matrix-free SIPG operator action `A u` (+ `λM u`).
    pub fn apply(&self, u: &[f64]) -> Vec<f64> {
        let m = self.mesh;
        let refh = &m.refh;
        let nn = refh.n_nodes();
        let ne = m.n_elements();

        // Per-element physical gradients (reused by the face terms).
        let mut gx = Vec::with_capacity(ne);
        let mut gy = Vec::with_capacity(ne);
        let mut gz = Vec::with_capacity(ne);
        for (e, el) in m.elements.iter().enumerate() {
            let ue = &u[e * nn..(e + 1) * nn];
            gx.push(el.geom.grad_x(refh, ue));
            gy.push(el.geom.grad_y(refh, ue));
            gz.push(el.geom.grad_z(refh, ue));
        }

        let mut r = self.apply_volume(u);

        for (e, el) in m.elements.iter().enumerate() {
            for face in Face::ALL {
                let f = &el.faces[face as usize];
                match &el.neighbors[face as usize] {
                    Neighbor3::Interior { elem: re, face: rface, perm } => {
                        if e >= *re {
                            continue; // process each interior face once (low side)
                        }
                        let rel = &m.elements[*re];
                        let rf = &rel.faces[*rface as usize];
                        let tau = self.penalty(e, Some(*re));
                        let mut hxl = vec![0.0; nn];
                        let mut hyl = vec![0.0; nn];
                        let mut hzl = vec![0.0; nn];
                        let mut hxr = vec![0.0; nn];
                        let mut hyr = vec![0.0; nn];
                        let mut hzr = vec![0.0; nn];
                        for a in 0..f.nodes.len() {
                            let b = perm[a];
                            let (vl, vr) = (f.nodes[a], rf.nodes[b]);
                            let (nx, ny, nz, sw) = (f.nx[a], f.ny[a], f.nz[a], f.sw[a]);
                            let dun_l = nx * gx[e][vl] + ny * gy[e][vl] + nz * gz[e][vl];
                            let dun_r = nx * gx[*re][vr] + ny * gy[*re][vr] + nz * gz[*re][vr];
                            let avg = 0.5 * (dun_l + dun_r);
                            let jump = u[e * nn + vl] - u[*re * nn + vr];
                            r[e * nn + vl] += -sw * avg;
                            r[*re * nn + vr] += sw * avg;
                            r[e * nn + vl] += tau * sw * jump;
                            r[*re * nn + vr] += -tau * sw * jump;
                            let g = 0.5 * sw * jump;
                            hxl[vl] += g * nx;
                            hyl[vl] += g * ny;
                            hzl[vl] += g * nz;
                            hxr[vr] += g * nx;
                            hyr[vr] += g * ny;
                            hzr[vr] += g * nz;
                        }
                        let ll = add3(
                            el.geom.gradx_t(refh, &hxl),
                            el.geom.grady_t(refh, &hyl),
                            el.geom.gradz_t(refh, &hzl),
                        );
                        let lr = add3(
                            rel.geom.gradx_t(refh, &hxr),
                            rel.geom.grady_t(refh, &hyr),
                            rel.geom.gradz_t(refh, &hzr),
                        );
                        for k in 0..nn {
                            r[e * nn + k] -= ll[k];
                            r[*re * nn + k] -= lr[k];
                        }
                    }
                    Neighbor3::Boundary { tag } => {
                        if self.is_neumann(*tag) {
                            continue; // natural BC: enters the RHS, not the operator
                        }
                        let tau = self.penalty(e, None);
                        let mut hx = vec![0.0; nn];
                        let mut hy = vec![0.0; nn];
                        let mut hz = vec![0.0; nn];
                        for a in 0..f.nodes.len() {
                            let v = f.nodes[a];
                            let (nx, ny, nz, sw) = (f.nx[a], f.ny[a], f.nz[a], f.sw[a]);
                            let dun = nx * gx[e][v] + ny * gy[e][v] + nz * gz[e][v];
                            let uval = u[e * nn + v];
                            r[e * nn + v] += -sw * dun;
                            r[e * nn + v] += tau * sw * uval;
                            hx[v] += sw * uval * nx;
                            hy[v] += sw * uval * ny;
                            hz[v] += sw * uval * nz;
                        }
                        let l = add3(
                            el.geom.gradx_t(refh, &hx),
                            el.geom.grady_t(refh, &hy),
                            el.geom.gradz_t(refh, &hz),
                        );
                        for k in 0..nn {
                            r[e * nn + k] -= l[k];
                        }
                    }
                }
            }
        }

        if self.reaction != 0.0 {
            for (e, el) in m.elements.iter().enumerate() {
                for k in 0..nn {
                    r[e * nn + k] += self.reaction * el.geom.jw[k] * u[e * nn + k];
                }
            }
        }
        r
    }

    /// RHS for `−∇²u = f` with all-Dirichlet data `u = g`.
    pub fn rhs(&self, f: &[f64], g: impl Fn(f64, f64, f64) -> f64) -> Vec<f64> {
        self.rhs_mixed(f, g, |_, _, _| 0.0)
    }

    /// RHS for mixed boundaries: value `g` on Dirichlet faces, flux `q = ∂u/∂n` on
    /// Neumann faces.
    pub fn rhs_mixed(
        &self,
        f: &[f64],
        g: impl Fn(f64, f64, f64) -> f64,
        q: impl Fn(f64, f64, f64) -> f64,
    ) -> Vec<f64> {
        self.rhs_tagged(f, |_, x, y, z| g(x, y, z), |_, x, y, z| q(x, y, z))
    }

    /// Like [`rhs_mixed`](Self::rhs_mixed) but the Dirichlet value `g(tag, x, y, z)` and
    /// Neumann flux `q(tag, x, y, z)` may depend on the boundary tag — the data side of
    /// per-region 3D boundary conditions.
    pub fn rhs_tagged(
        &self,
        f: &[f64],
        g: impl Fn(u32, f64, f64, f64) -> f64,
        q: impl Fn(u32, f64, f64, f64) -> f64,
    ) -> Vec<f64> {
        let m = self.mesh;
        let refh = &m.refh;
        let nn = refh.n_nodes();
        let mut b = vec![0.0; self.ndof()];

        for (e, el) in m.elements.iter().enumerate() {
            for k in 0..nn {
                b[e * nn + k] += el.geom.jw[k] * f[e * nn + k];
            }
        }
        for (e, el) in m.elements.iter().enumerate() {
            for face in Face::ALL {
                let Neighbor3::Boundary { tag } = el.neighbors[face as usize] else {
                    continue;
                };
                let fc = &el.faces[face as usize];
                if self.is_neumann(tag) {
                    for a in 0..fc.nodes.len() {
                        let v = fc.nodes[a];
                        b[e * nn + v] += fc.sw[a] * q(tag, el.geom.x[v], el.geom.y[v], el.geom.z[v]);
                    }
                } else {
                    let tau = self.penalty(e, None);
                    let mut hx = vec![0.0; nn];
                    let mut hy = vec![0.0; nn];
                    let mut hz = vec![0.0; nn];
                    for a in 0..fc.nodes.len() {
                        let v = fc.nodes[a];
                        let (nx, ny, nz, sw) = (fc.nx[a], fc.ny[a], fc.nz[a], fc.sw[a]);
                        let gv = g(tag, el.geom.x[v], el.geom.y[v], el.geom.z[v]);
                        b[e * nn + v] += tau * sw * gv;
                        hx[v] += sw * gv * nx;
                        hy[v] += sw * gv * ny;
                        hz[v] += sw * gv * nz;
                    }
                    let l = add3(
                        el.geom.gradx_t(refh, &hx),
                        el.geom.grady_t(refh, &hy),
                        el.geom.gradz_t(refh, &hz),
                    );
                    for k in 0..nn {
                        b[e * nn + k] -= l[k];
                    }
                }
            }
        }
        b
    }

    /// Conjugate gradient for the SPD operator. Returns `(u, iterations, residual)`.
    pub fn cg(&self, b: &[f64], tol: f64, maxit: usize) -> (Vec<f64>, usize, f64) {
        let n = b.len();
        let mut x = vec![0.0; n];
        let mut r = b.to_vec();
        let mut p = r.clone();
        let mut rs = dot(&r, &r);
        let bnorm = dot(b, b).sqrt().max(1e-300);
        for it in 0..maxit {
            let ap = self.apply(&p);
            let alpha = rs / dot(&p, &ap);
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            let rs_new = dot(&r, &r);
            let resid = rs_new.sqrt() / bnorm;
            if resid < tol {
                return (x, it + 1, resid);
            }
            let beta = rs_new / rs;
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
            rs = rs_new;
        }
        (x, maxit, (rs.sqrt() / bnorm))
    }

    /// CG for the singular (pure-Neumann) system: deflates the constant nullspace.
    pub fn cg_deflated(&self, b: &[f64], tol: f64, maxit: usize) -> (Vec<f64>, usize) {
        let n = b.len();
        let deflate = |v: &mut [f64]| {
            let mean = v.iter().sum::<f64>() / n as f64;
            v.iter_mut().for_each(|x| *x -= mean);
        };
        let mut x = vec![0.0; n];
        let mut r = b.to_vec();
        deflate(&mut r);
        let mut p = r.clone();
        let mut rs = dot(&r, &r);
        let bn = rs.sqrt().max(1e-300);
        for it in 0..maxit {
            let ap = self.apply(&p);
            let alpha = rs / dot(&p, &ap);
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            deflate(&mut r);
            let rs_new = dot(&r, &r);
            if rs_new.sqrt() / bn < tol {
                return (x, it + 1);
            }
            let beta = rs_new / rs;
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
            rs = rs_new;
        }
        (x, maxit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn nodal(mesh: &Mesh3d, func: impl Fn(f64, f64, f64) -> f64) -> Vec<f64> {
        let nn = mesh.refh.n_nodes();
        let mut u = vec![0.0; mesh.n_elements() * nn];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                u[e * nn + k] = func(el.geom.x[k], el.geom.y[k], el.geom.z[k]);
            }
        }
        u
    }

    #[test]
    fn operator_is_symmetric() {
        let mesh = Mesh3d::rectangular(3, 2, 2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
        let op = Poisson3d::new(&mesh, 5.0);
        // Two pseudo-random vectors.
        let n = op.ndof();
        let u: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 13) as f64 - 6.0).collect();
        let v: Vec<f64> = (0..n).map(|i| ((i * 5 + 1) % 11) as f64 - 5.0).collect();
        let au = op.apply(&u);
        let av = op.apply(&v);
        let uav = dot(&u, &av);
        let vau = dot(&v, &au);
        assert!((uav - vau).abs() / uav.abs().max(1.0) < 1e-10, "asymmetry {uav} vs {vau}");
    }

    #[test]
    fn spd_positive_on_nonzero() {
        let mesh = Mesh3d::rectangular(3, 2, 2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
        let op = Poisson3d::new(&mesh, 5.0);
        let u: Vec<f64> = (0..op.ndof()).map(|i| ((i * 3 + 2) % 7) as f64 - 3.0).collect();
        let quad = dot(&u, &op.apply(&u));
        assert!(quad > 0.0, "uᵀAu = {quad} not positive");
    }

    #[test]
    fn patch_test_exact_on_quadratic() {
        // u = x² + y² + z² ⇒ −∇²u = −6. SIPG is exact for degree ≤ p (p ≥ 2).
        let mesh = Mesh3d::rectangular(3, 2, 2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
        let op = Poisson3d::new(&mesh, 8.0);
        let exact = |x: f64, y: f64, z: f64| x * x + y * y + z * z;
        let f = nodal(&mesh, |_, _, _| -6.0);
        let b = op.rhs(&f, exact);
        let (u, _it, res) = op.cg(&b, 1e-12, 5000);
        assert!(res < 1e-10, "cg residual {res}");
        let ue = nodal(&mesh, exact);
        let err = u.iter().zip(&ue).fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
        assert!(err < 1e-8, "patch-test error {err}");
    }

    #[test]
    fn manufactured_solution_converges() {
        // u = sin(πx)sin(πy)sin(πz), zero Dirichlet on [0,1]³; −∇²u = 3π²u = f.
        let mesh = Mesh3d::rectangular(4, 3, 3, 3, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
        let op = Poisson3d::new(&mesh, 10.0);
        let exact = |x: f64, y: f64, z: f64| (PI * x).sin() * (PI * y).sin() * (PI * z).sin();
        let f = nodal(&mesh, |x, y, z| 3.0 * PI * PI * exact(x, y, z));
        let b = op.rhs(&f, exact);
        let (u, _it, res) = op.cg(&b, 1e-12, 20000);
        assert!(res < 1e-10, "cg residual {res}");
        let ue = nodal(&mesh, exact);
        let nn = mesh.refh.n_nodes();
        let mut e2 = 0.0;
        let mut n2 = 0.0;
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let d = u[e * nn + k] - ue[e * nn + k];
                e2 += el.geom.jw[k] * d * d;
                n2 += el.geom.jw[k] * ue[e * nn + k] * ue[e * nn + k];
            }
        }
        let rel = (e2 / n2).sqrt();
        assert!(rel < 1e-3, "MMS relative L2 error {rel:e}");
    }

    #[test]
    fn helmholtz_patch_test_exact_on_quadratic() {
        // (λM + A) with u = x²+y²+z²: residual f = λu − ∇²u = λu − 6. Recover u.
        let mesh = Mesh3d::rectangular(3, 2, 2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
        let lambda = 2.5;
        let op = Poisson3d::with_reaction(&mesh, 8.0, lambda);
        let exact = |x: f64, y: f64, z: f64| x * x + y * y + z * z;
        let f = nodal(&mesh, move |x, y, z| lambda * exact(x, y, z) - 6.0);
        let b = op.rhs(&f, exact);
        let (u, _it, res) = op.cg(&b, 1e-12, 5000);
        assert!(res < 1e-10, "cg residual {res}");
        let ue = nodal(&mesh, exact);
        let err = u.iter().zip(&ue).fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
        assert!(err < 1e-8, "helmholtz patch error {err}");
    }
}
