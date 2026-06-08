//! Viscoelastic constitutive models on the 3D hex mesh — the 3D analogue of
//! [`viscoelastic`](super::viscoelastic). The conformation tensor is now symmetric
//! **3×3 = 6 components** `[Cxx, Cxy, Cxz, Cyy, Cyz, Czz]`, and the log-conformation
//! form needs a **symmetric 3×3 eigensolver** (the 2D closed-form does not extend).
//!
//! The eigensolver is a fixed-sweep cyclic **Jacobi** rotation — robust at
//! degenerate eigenvalues, branch-light, and free of the `atan2`/`acos` device
//! intrinsics that the 2D log-conf GPU port had to avoid (so the same routine ports
//! to the GPU later; see `docs/3d-strategy.md` §8).

use super::face3d::Face;
use super::mesh3d::{Mesh3d, Neighbor3};

/// Inflow boundary data for the 3D conformation transport (6-component symmetric
/// tensor `[Cxx, Cxy, Cxz, Cyy, Cyz, Czz]`). 3D analogue of
/// [`ConformationInflow`](super::viscoelastic::ConformationInflow): at boundary nodes
/// whose tag is in `tags` and where flow enters (`u·n < 0`), the upwind trace is the
/// constant conformation `c`. Other boundaries are transparent.
#[derive(Clone, Debug)]
pub struct ConformationInflow3d {
    pub tags: Vec<u32>,
    pub c: [f64; 6],
}

impl ConformationInflow3d {
    /// Prescribe the incoming conformation `c` on boundary `tags`.
    pub fn new(tags: Vec<u32>, c: [f64; 6]) -> Self {
        Self { tags, c }
    }

    /// Relaxed (equilibrium, `C = I`) fluid entering on the given `tags`.
    pub fn equilibrium(tags: Vec<u32>) -> Self {
        Self { tags, c: [1.0, 0.0, 0.0, 1.0, 0.0, 1.0] }
    }
}

/// Matrix logarithm of an SPD 3×3 conformation `[Cxx,Cxy,Cxz,Cyy,Cyz,Czz]`, returning
/// `Ψ = log C` (6 components). Converts a conformation inflow datum into the
/// log-conformation variable for the upwind trace (host-side, incl. the GPU log path).
pub fn log_conformation3(c: [f64; 6]) -> [f64; 6] {
    sym_apply3(c, f64::ln)
}

/// Upwind DG **surface lift** for the advection of a 6-component (symmetric-tensor)
/// field `φ` by the divergence-free velocity `(ux,uy,uz)` on a hex mesh — the 3D
/// analogue of [`upwind_advection_lift`](super::viscoelastic::upwind_advection_lift).
/// Returns the correction to ADD to the collocation volume term `−(u·∇)φ` (which is
/// element-local and does not transport `φ` across hex faces). At a face node it is
/// `(sw/jw)(u·n)(φ⁻ − φ*)`, nonzero only at inflow nodes (`u·n < 0`), with `φ*` the
/// neighbour trace across interior faces, the `bdry` datum at boundary inflow nodes
/// (in `φ`'s own variable), or transparent otherwise. (3D meshes are conforming-only.)
pub fn upwind_advection_lift3(
    mesh: &Mesh3d,
    field: &[Vec<f64>; 6],
    ux: &[f64],
    uy: &[f64],
    uz: &[f64],
    bdry: impl Fn(u32) -> Option<[f64; 6]>,
) -> [Vec<f64>; 6] {
    let nn = mesh.refh.n_nodes();
    let n = mesh.n_elements() * nn;
    let mut out: [Vec<f64>; 6] = std::array::from_fn(|_| vec![0.0; n]);
    for (e, el) in mesh.elements.iter().enumerate() {
        for face_e in Face::ALL {
            let face = &el.faces[face_e as usize];
            let nb = &el.neighbors[face_e as usize];
            for a in 0..face.nodes.len() {
                let vl = face.nodes[a];
                let g = e * nn + vl;
                let un = ux[g] * face.nx[a] + uy[g] * face.ny[a] + uz[g] * face.nz[a];
                if un >= 0.0 {
                    continue; // outflow node ⇒ no correction
                }
                let ext: [f64; 6] = match nb {
                    Neighbor3::Interior { elem: re, face: rface, perm } => {
                        let rf = &mesh.elements[*re].faces[*rface as usize];
                        let vr = *re * nn + rf.nodes[perm[a]];
                        std::array::from_fn(|v| field[v][vr])
                    }
                    Neighbor3::Boundary { tag } => {
                        bdry(*tag).unwrap_or(std::array::from_fn(|v| field[v][g]))
                    }
                };
                let fac = face.sw[a] * un / el.geom.jw[vl];
                for comp in 0..6 {
                    out[comp][g] += fac * (field[comp][g] - ext[comp]);
                }
            }
        }
    }
    out
}

/// Symmetric-3×3 eigendecomposition by cyclic Jacobi. Input is the 6 upper entries
/// `[xx, xy, xz, yy, yz, zz]`; returns eigenvalues `λ` and the eigenvector matrix
/// `V` (columns are eigenvectors), so `A = V diag(λ) Vᵀ`.
pub fn sym_eig3(m: [f64; 6]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = [[m[0], m[1], m[2]], [m[1], m[3], m[4]], [m[2], m[4], m[5]]];
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _sweep in 0..12 {
        let mut off = 0.0f64;
        for &(p, q) in &[(0usize, 1usize), (0, 2), (1, 2)] {
            off = off.max(a[p][q].abs());
        }
        if off < 1e-300 {
            break;
        }
        for &(p, q) in &[(0usize, 1usize), (0, 2), (1, 2)] {
            let apq = a[p][q];
            if apq.abs() < 1e-300 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            // A ← Jᵀ A J (rotate columns then rows).
            for i in 0..3 {
                let aip = a[i][p];
                let aiq = a[i][q];
                a[i][p] = c * aip - s * aiq;
                a[i][q] = s * aip + c * aiq;
            }
            for i in 0..3 {
                let api = a[p][i];
                let aqi = a[q][i];
                a[p][i] = c * api - s * aqi;
                a[q][i] = s * api + c * aqi;
            }
            // V ← V J.
            for i in 0..3 {
                let vip = v[i][p];
                let viq = v[i][q];
                v[i][p] = c * vip - s * viq;
                v[i][q] = s * vip + c * viq;
            }
        }
    }
    ([a[0][0], a[1][1], a[2][2]], v)
}

/// Apply a scalar function to a symmetric 3×3 matrix via its eigendecomposition;
/// returns the 6 upper entries of `V diag(f(λ)) Vᵀ`.
pub fn sym_apply3(m: [f64; 6], f: impl Fn(f64) -> f64) -> [f64; 6] {
    let (lam, v) = sym_eig3(m);
    let fl = [f(lam[0]), f(lam[1]), f(lam[2])];
    let entry = |i: usize, j: usize| -> f64 { (0..3).map(|k| v[i][k] * fl[k] * v[j][k]).sum() };
    [entry(0, 0), entry(0, 1), entry(0, 2), entry(1, 1), entry(1, 2), entry(2, 2)]
}

// 3×3 helpers on full matrices (row-major [[f64;3];3]).
fn matmul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    c
}
fn transpose(a: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut t = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = a[j][i];
        }
    }
    t
}
fn expand(m: &[f64; 6]) -> [[f64; 3]; 3] {
    [[m[0], m[1], m[2]], [m[1], m[3], m[4]], [m[2], m[4], m[5]]]
}

/// Direct **Oldroyd-B** conformation transport in 3D. State `C = [Cxx, Cxy, Cxz,
/// Cyy, Cyz, Czz]`.
pub struct OldroydB3d<'m> {
    pub mesh: &'m Mesh3d,
    pub lambda: f64,
    pub eta_p: f64,
    /// Optional conformation inflow boundary data. `None` ⇒ transparent boundaries.
    pub inflow: Option<ConformationInflow3d>,
}

impl<'m> OldroydB3d<'m> {
    pub fn new(mesh: &'m Mesh3d, lambda: f64, eta_p: f64) -> Self {
        Self { mesh, lambda, eta_p, inflow: None }
    }

    /// Set the conformation inflow boundary data (builder style).
    pub fn with_inflow(mut self, inflow: ConformationInflow3d) -> Self {
        self.inflow = Some(inflow);
        self
    }
    fn ndof(&self) -> usize {
        self.mesh.n_elements() * self.mesh.refh.n_nodes()
    }
    /// `C = I` (equilibrium): `[1,0,0,1,0,1]`.
    pub fn identity(&self) -> [Vec<f64>; 6] {
        let n = self.ndof();
        [vec![1.0; n], vec![0.0; n], vec![0.0; n], vec![1.0; n], vec![0.0; n], vec![1.0; n]]
    }

    /// Velocity gradient `L = ∇u` (3×3) at every node, per element.
    fn velocity_gradient(&self, ux: &[f64], uy: &[f64], uz: &[f64]) -> Vec<[[f64; 3]; 3]> {
        let refh = &self.mesh.refh;
        let nn = refh.n_nodes();
        let mut l = vec![[[0.0; 3]; 3]; self.ndof()];
        for (e, el) in self.mesh.elements.iter().enumerate() {
            let sl = e * nn..(e + 1) * nn;
            let comps = [&ux[sl.clone()], &uy[sl.clone()], &uz[sl.clone()]];
            for (i, ui) in comps.iter().enumerate() {
                let gx = el.geom.grad_x(refh, ui);
                let gy = el.geom.grad_y(refh, ui);
                let gz = el.geom.grad_z(refh, ui);
                for k in 0..nn {
                    l[e * nn + k][i][0] = gx[k];
                    l[e * nn + k][i][1] = gy[k];
                    l[e * nn + k][i][2] = gz[k];
                }
            }
        }
        l
    }

    /// Per-node gradient of one scalar component over the mesh.
    fn comp_grad(&self, f: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let refh = &self.mesh.refh;
        let nn = refh.n_nodes();
        let (mut gx, mut gy, mut gz) = (vec![0.0; self.ndof()], vec![0.0; self.ndof()], vec![0.0; self.ndof()]);
        for (e, el) in self.mesh.elements.iter().enumerate() {
            let sl = e * nn..(e + 1) * nn;
            let a = el.geom.grad_x(refh, &f[sl.clone()]);
            let b = el.geom.grad_y(refh, &f[sl.clone()]);
            let c = el.geom.grad_z(refh, &f[sl]);
            for k in 0..nn {
                gx[e * nn + k] = a[k];
                gy[e * nn + k] = b[k];
                gz[e * nn + k] = c[k];
            }
        }
        (gx, gy, gz)
    }

    /// `∂C/∂t = −(u·∇)C + L·C + C·Lᵀ − (1/λ)(C − I)`.
    pub fn conformation_rhs(&self, c: &[Vec<f64>; 6], ux: &[f64], uy: &[f64], uz: &[f64]) -> [Vec<f64>; 6] {
        let n = self.ndof();
        let inv_lambda = 1.0 / self.lambda;
        let l = self.velocity_gradient(ux, uy, uz);
        let cg: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = (0..6).map(|v| self.comp_grad(&c[v])).collect();
        let mut out: [Vec<f64>; 6] = std::array::from_fn(|_| vec![0.0; n]);
        // Identity in 6-component form.
        let eye = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        for g in 0..n {
            let cm = expand(&std::array::from_fn(|v| c[v][g]));
            let lg = l[g];
            let lc = matmul(&lg, &cm);
            let clt = matmul(&cm, &transpose(&lg));
            let (u, v, w) = (ux[g], uy[g], uz[g]);
            // map full 3×3 (symmetric) → 6 entries.
            let idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
            for (o, &(i, j)) in idx.iter().enumerate() {
                let stretch = lc[i][j] + clt[i][j];
                let adv = u * cg[o].0[g] + v * cg[o].1[g] + w * cg[o].2[g];
                let relax = -inv_lambda * (c[o][g] - eye[o]);
                out[o][g] = -adv + stretch + relax;
            }
        }
        // Upwind DG surface lift — inter-element transport + inflow injection.
        let lift = upwind_advection_lift3(self.mesh, c, ux, uy, uz, |tag| {
            self.inflow.as_ref().filter(|i| i.tags.contains(&tag)).map(|i| i.c)
        });
        for o in 0..6 {
            for g in 0..n {
                out[o][g] += lift[o][g];
            }
        }
        out
    }

    /// One SSP-RK3 step of the conformation transport with a fixed velocity field.
    pub fn step_ssp_rk3(&self, c: &[Vec<f64>; 6], ux: &[f64], uy: &[f64], uz: &[f64], dt: f64) -> [Vec<f64>; 6] {
        step6(|s| self.conformation_rhs(s, ux, uy, uz), c, dt)
    }

    /// Polymer stress `τ_p = (η_p/λ)(C − I)` as 6 components.
    pub fn polymer_stress(&self, c: &[Vec<f64>; 6]) -> [Vec<f64>; 6] {
        let f = self.eta_p / self.lambda;
        let eye = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        std::array::from_fn(|o| c[o].iter().map(|&x| f * (x - eye[o])).collect())
    }

    /// `∇·τ_p` momentum body force `(fx, fy, fz)`.
    pub fn stress_divergence(&self, c: &[Vec<f64>; 6]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        stress_div6(self.mesh, &self.polymer_stress(c))
    }
}

/// `∇·τ` for a symmetric 6-component stress field: `fx = ∂xτxx+∂yτxy+∂zτxz`, etc.
fn stress_div6(mesh: &Mesh3d, tau: &[Vec<f64>; 6]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let refh = &mesh.refh;
    let nn = refh.n_nodes();
    let ndof = mesh.n_elements() * nn;
    let (mut fx, mut fy, mut fz) = (vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof]);
    for (e, el) in mesh.elements.iter().enumerate() {
        let sl = e * nn..(e + 1) * nn;
        // τ = [xx, xy, xz, yy, yz, zz].
        let txx_x = el.geom.grad_x(refh, &tau[0][sl.clone()]);
        let txy_x = el.geom.grad_x(refh, &tau[1][sl.clone()]);
        let txy_y = el.geom.grad_y(refh, &tau[1][sl.clone()]);
        let txz_x = el.geom.grad_x(refh, &tau[2][sl.clone()]);
        let txz_z = el.geom.grad_z(refh, &tau[2][sl.clone()]);
        let tyy_y = el.geom.grad_y(refh, &tau[3][sl.clone()]);
        let tyz_y = el.geom.grad_y(refh, &tau[4][sl.clone()]);
        let tyz_z = el.geom.grad_z(refh, &tau[4][sl.clone()]);
        let tzz_z = el.geom.grad_z(refh, &tau[5][sl]);
        for k in 0..nn {
            fx[e * nn + k] = txx_x[k] + txy_y[k] + txz_z[k];
            fy[e * nn + k] = txy_x[k] + tyy_y[k] + tyz_z[k];
            fz[e * nn + k] = txz_x[k] + tyz_y[k] + tzz_z[k];
        }
    }
    (fx, fy, fz)
}

/// SSP-RK3 over a 6-component state with rhs closure.
fn step6(rhs: impl Fn(&[Vec<f64>; 6]) -> [Vec<f64>; 6], c: &[Vec<f64>; 6], dt: f64) -> [Vec<f64>; 6] {
    let axpy = |a: &[Vec<f64>; 6], k: &[Vec<f64>; 6], sc: f64| -> [Vec<f64>; 6] {
        std::array::from_fn(|v| a[v].iter().zip(&k[v]).map(|(x, d)| x + sc * d).collect())
    };
    let combine = |a: &[Vec<f64>; 6], wa: f64, b: &[Vec<f64>; 6], wb: f64| -> [Vec<f64>; 6] {
        std::array::from_fn(|v| a[v].iter().zip(&b[v]).map(|(x, y)| wa * x + wb * y).collect())
    };
    let k0 = rhs(c);
    let u1 = axpy(c, &k0, dt);
    let k1 = rhs(&u1);
    let u2 = combine(c, 0.75, &axpy(&u1, &k1, dt), 0.25);
    let k2 = rhs(&u2);
    combine(c, 1.0 / 3.0, &axpy(&u2, &k2, dt), 2.0 / 3.0)
}

/// **Log-conformation** Oldroyd-B in 3D (Fattal–Kupferman): evolves `Ψ = log C`.
/// State `Ψ = [Ψxx, Ψxy, Ψxz, Ψyy, Ψyz, Ψzz]`; equilibrium `Ψ = 0`.
pub struct LogConfOldroydB3d<'m> {
    pub mesh: &'m Mesh3d,
    pub lambda: f64,
    pub eta_p: f64,
    /// Optional conformation inflow data (specified as `C`; converted to `Ψ = log C`).
    pub inflow: Option<ConformationInflow3d>,
}

impl<'m> LogConfOldroydB3d<'m> {
    pub fn new(mesh: &'m Mesh3d, lambda: f64, eta_p: f64) -> Self {
        Self { mesh, lambda, eta_p, inflow: None }
    }

    /// Set the conformation inflow boundary data (builder style).
    pub fn with_inflow(mut self, inflow: ConformationInflow3d) -> Self {
        self.inflow = Some(inflow);
        self
    }
    fn ndof(&self) -> usize {
        self.mesh.n_elements() * self.mesh.refh.n_nodes()
    }
    /// `Ψ = 0` (equilibrium, `C = I`).
    pub fn identity(&self) -> [Vec<f64>; 6] {
        let n = self.ndof();
        std::array::from_fn(|_| vec![0.0; n])
    }
    /// `Ψ = log C` componentwise from a conformation field.
    pub fn from_conformation(&self, c: &[Vec<f64>; 6]) -> [Vec<f64>; 6] {
        let n = self.ndof();
        let mut psi: [Vec<f64>; 6] = std::array::from_fn(|_| vec![0.0; n]);
        for g in 0..n {
            let p = sym_apply3(std::array::from_fn(|v| c[v][g]), |x| x.ln());
            for v in 0..6 {
                psi[v][g] = p[v];
            }
        }
        psi
    }
    /// `C = exp Ψ`.
    pub fn conformation(&self, psi: &[Vec<f64>; 6]) -> [Vec<f64>; 6] {
        let n = self.ndof();
        let mut c: [Vec<f64>; 6] = std::array::from_fn(|_| vec![0.0; n]);
        for g in 0..n {
            let cc = sym_apply3(std::array::from_fn(|v| psi[v][g]), |x| x.exp());
            for v in 0..6 {
                c[v][g] = cc[v];
            }
        }
        c
    }

    /// `∂Ψ/∂t = −(u·∇)Ψ + (ΩΨ − ΨΩ) + 2B + (1/λ)(e^{−Ψ} − I)`.
    pub fn psi_rhs(&self, psi: &[Vec<f64>; 6], ux: &[f64], uy: &[f64], uz: &[f64]) -> [Vec<f64>; 6] {
        let n = self.ndof();
        let inv_lambda = 1.0 / self.lambda;
        let ob = OldroydB3d::new(self.mesh, self.lambda, self.eta_p);
        let lvec = ob.velocity_gradient(ux, uy, uz);
        let pg: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = (0..6).map(|v| ob.comp_grad(&psi[v])).collect();
        let idx = [(0usize, 0usize), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
        let mut out: [Vec<f64>; 6] = std::array::from_fn(|_| vec![0.0; n]);
        for g in 0..n {
            let psi_m = std::array::from_fn(|v| psi[v][g]);
            let (mu, mut r) = sym_eig3(psi_m);
            let l = lvec[g];
            // Rate of strain D = ½(L + Lᵀ).
            let lt = transpose(&l);
            let mut d = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    d[i][j] = 0.5 * (l[i][j] + lt[i][j]);
                }
            }
            // Near-isotropic (all eigenvalues ~equal): the Ψ-eigenframe is
            // indeterminate; align with the rate-of-strain frame so 2B → L+Lᵀ.
            let spread = (mu[0] - mu[1]).abs().max((mu[0] - mu[2]).abs()).max((mu[1] - mu[2]).abs());
            if spread < 1e-7 {
                let (_, rd) = sym_eig3([d[0][0], d[0][1], d[0][2], d[1][1], d[1][2], d[2][2]]);
                r = rd;
            }
            let lam = [mu[0].exp(), mu[1].exp(), mu[2].exp()];
            let rt = transpose(&r);
            // M = Rᵀ L R.
            let m = matmul(&rt, &matmul(&l, &r));
            // B = R diag(M_ii) Rᵀ  (eigenframe-diagonal of the rate-of-strain).
            let mut bd = [[0.0; 3]; 3];
            for i in 0..3 {
                bd[i][i] = m[i][i];
            }
            let bmat = matmul(&r, &matmul(&bd, &rt));
            // Ω_eig (antisymmetric): ω_ij = (M_ij λ_j + M_ji λ_i)/(λ_j − λ_i).
            let mut om = [[0.0; 3]; 3];
            for (i, j) in [(0, 1), (0, 2), (1, 2)] {
                let denom = lam[j] - lam[i];
                let w = if denom.abs() > 1e-12 {
                    (m[i][j] * lam[j] + m[j][i] * lam[i]) / denom
                } else {
                    0.0
                };
                om[i][j] = w;
                om[j][i] = -w;
            }
            let omega = matmul(&r, &matmul(&om, &rt));
            // ΩΨ − ΨΩ.
            let psi_full = expand(&psi_m);
            let op = matmul(&omega, &psi_full);
            let po = matmul(&psi_full, &omega);
            // (1/λ)(e^{−Ψ} − I).
            let em = sym_apply3(psi_m, |x| (-x).exp());
            let eye = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
            for (o, &(i, j)) in idx.iter().enumerate() {
                let rot = op[i][j] - po[i][j];
                let adv = ux[g] * pg[o].0[g] + uy[g] * pg[o].1[g] + uz[g] * pg[o].2[g];
                let relax = inv_lambda * (em[o] - eye[o]);
                out[o][g] = -adv + rot + 2.0 * bmat[i][j] + relax;
            }
        }
        // Upwind DG surface lift for −(u·∇)Ψ; inflow C_in enters as Ψ_in = log C_in.
        let lift = upwind_advection_lift3(self.mesh, psi, ux, uy, uz, |tag| {
            self.inflow.as_ref().filter(|i| i.tags.contains(&tag)).map(|i| log_conformation3(i.c))
        });
        for o in 0..6 {
            for g in 0..n {
                out[o][g] += lift[o][g];
            }
        }
        out
    }

    pub fn step_ssp_rk3(&self, psi: &[Vec<f64>; 6], ux: &[f64], uy: &[f64], uz: &[f64], dt: f64) -> [Vec<f64>; 6] {
        step6(|s| self.psi_rhs(s, ux, uy, uz), psi, dt)
    }

    /// `∇·τ_p` from `Ψ`.
    pub fn stress_divergence(&self, psi: &[Vec<f64>; 6]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let c = self.conformation(psi);
        let f = self.eta_p / self.lambda;
        let eye = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let tau: [Vec<f64>; 6] = std::array::from_fn(|o| c[o].iter().map(|&x| f * (x - eye[o])).collect());
        stress_div6(self.mesh, &tau)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nodal(mesh: &Mesh3d, f: impl Fn(f64, f64, f64) -> f64) -> Vec<f64> {
        let nn = mesh.refh.n_nodes();
        let mut v = vec![0.0; mesh.n_elements() * nn];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                v[e * nn + k] = f(el.geom.x[k], el.geom.y[k], el.geom.z[k]);
            }
        }
        v
    }

    #[test]
    fn eig_reconstructs_and_is_orthonormal() {
        let m = [2.0, -0.3, 0.5, 1.4, 0.2, 3.1];
        let (lam, v) = sym_eig3(m);
        // V orthonormal.
        for i in 0..3 {
            for j in 0..3 {
                let dotc: f64 = (0..3).map(|k| v[k][i] * v[k][j]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dotc - want).abs() < 1e-10, "V not orthonormal");
            }
        }
        // V diag(λ) Vᵀ == A.
        let a = expand(&m);
        for i in 0..3 {
            for j in 0..3 {
                let r: f64 = (0..3).map(|k| v[i][k] * lam[k] * v[j][k]).sum();
                assert!((r - a[i][j]).abs() < 1e-10, "reconstruction off");
            }
        }
    }

    #[test]
    fn exp_log_roundtrip_spd() {
        // C SPD ⇒ exp(log C) = C.
        let c = [2.5, 0.4, -0.2, 1.8, 0.3, 2.1];
        let logc = sym_apply3(c, |x| x.ln());
        let back = sym_apply3(logc, |x| x.exp());
        for v in 0..6 {
            assert!((back[v] - c[v]).abs() < 1e-9, "roundtrip off at {v}");
        }
    }

    #[test]
    fn oldroyd_b_steady_simple_shear() {
        // u = (γ̇ y, 0, 0): analytic steady C has Cxx=1+2Wi², Cxy=Wi, Cyy=Czz=1.
        let mesh = Mesh3d::rectangular(3, 2, 2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
        let (lambda, gdot) = (1.0, 0.5);
        let wi = lambda * gdot;
        let ob = OldroydB3d::new(&mesh, lambda, 1.5);
        let ux = nodal(&mesh, |_, y, _| gdot * y);
        let uy = vec![0.0; ob.ndof()];
        let uz = vec![0.0; ob.ndof()];
        let mut c = ob.identity();
        for _ in 0..600 {
            c = ob.step_ssp_rk3(&c, &ux, &uy, &uz, 0.02);
        }
        // Check at an interior node.
        let g = ob.ndof() / 2;
        assert!((c[0][g] - (1.0 + 2.0 * wi * wi)).abs() < 1e-3, "Cxx={}", c[0][g]);
        assert!((c[1][g] - wi).abs() < 1e-3, "Cxy={}", c[1][g]);
        assert!(c[2][g].abs() < 1e-6, "Cxz={}", c[2][g]);
        assert!((c[3][g] - 1.0).abs() < 1e-3, "Cyy={}", c[3][g]);
        assert!(c[4][g].abs() < 1e-6, "Cyz={}", c[4][g]);
        assert!((c[5][g] - 1.0).abs() < 1e-3, "Czz={}", c[5][g]);
    }

    #[test]
    fn log_conf_rhs_vanishes_at_analytic_steady_shear() {
        // The log-conf rhs at Ψ = log(C_steady) (a constant field) must be ≈ 0 —
        // validating the full 3D eigenframe decomposition reproduces the steady state.
        let mesh = Mesh3d::rectangular(3, 2, 2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
        let (lambda, gdot) = (1.0, 0.5);
        let wi = lambda * gdot;
        let lc = LogConfOldroydB3d::new(&mesh, lambda, 1.5);
        let n = lc.ndof();
        // Constant C_steady field.
        let cxx = 1.0 + 2.0 * wi * wi;
        let c: [Vec<f64>; 6] =
            [vec![cxx; n], vec![wi; n], vec![0.0; n], vec![1.0; n], vec![0.0; n], vec![1.0; n]];
        let psi = lc.from_conformation(&c);
        let ux = nodal(&mesh, |_, y, _| gdot * y);
        let uy = vec![0.0; n];
        let uz = vec![0.0; n];
        let r = lc.psi_rhs(&psi, &ux, &uy, &uz);
        let m = r.iter().flat_map(|v| v.iter()).fold(0.0f64, |a, &x| a.max(x.abs()));
        assert!(m < 1e-9, "log-conf rhs at steady state = {m}");
    }

    #[test]
    fn log_conf_matches_direct_at_steady_shear() {
        // Integrating the log-conf form to steady recovers the same C as the direct
        // form (and the analytic value).
        let mesh = Mesh3d::rectangular(3, 2, 2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
        let (lambda, gdot) = (1.0, 0.5);
        let wi = lambda * gdot;
        let lc = LogConfOldroydB3d::new(&mesh, lambda, 1.5);
        let ux = nodal(&mesh, |_, y, _| gdot * y);
        let uy = vec![0.0; lc.ndof()];
        let uz = vec![0.0; lc.ndof()];
        let mut psi = lc.identity();
        for _ in 0..600 {
            psi = lc.step_ssp_rk3(&psi, &ux, &uy, &uz, 0.02);
        }
        let c = lc.conformation(&psi);
        let g = lc.ndof() / 2;
        assert!((c[0][g] - (1.0 + 2.0 * wi * wi)).abs() < 5e-3, "Cxx={}", c[0][g]);
        assert!((c[1][g] - wi).abs() < 5e-3, "Cxy={}", c[1][g]);
        assert!((c[3][g] - 1.0).abs() < 5e-3, "Cyy={}", c[3][g]);
        // C must stay SPD (the whole point of the log form).
        let cm = expand(&std::array::from_fn(|v| c[v][g]));
        let (lam, _) = sym_eig3([cm[0][0], cm[0][1], cm[0][2], cm[1][1], cm[1][2], cm[2][2]]);
        assert!(lam.iter().all(|&l| l > 0.0), "C not SPD: {lam:?}");
    }

    #[test]
    fn upwind_advection_transports_across_elements_3d() {
        // 3D analogue: pure x-advection of a smooth bump by u=(1,0,0) on a periodic hex
        // mesh; the upwind surface lift carries it across hex faces + the periodic seam.
        use std::f64::consts::PI;
        let mesh = Mesh3d::rectangular_periodic(3, 4, 1, 1, [0.0, 1.0], [0.0, 0.25], [0.0, 0.25]);
        let ob = OldroydB3d::new(&mesh, 1e6, 1.0);
        let nd = ob.ndof();
        let u = vec![1.0; nd];
        let z = vec![0.0; nd];
        let c0 = nodal(&mesh, |x, _, _| 1.0 + 0.5 * (2.0 * PI * x).sin());
        let mut c: [Vec<f64>; 6] =
            [c0.clone(), vec![0.0; nd], vec![0.0; nd], vec![1.0; nd], vec![0.0; nd], vec![1.0; nd]];
        let dt = 1e-3_f64;
        let t_end = 0.5_f64;
        for _ in 0..(t_end / dt).round() as usize {
            c = ob.step_ssp_rk3(&c, &u, &z, &z, dt);
        }
        let exact = nodal(&mesh, |x, _, _| 1.0 + 0.5 * (2.0 * PI * (x - t_end)).sin());
        let err = c[0].iter().zip(&exact).fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
        eprintln!("3D periodic conformation advection: max|Cxx − exact| = {err:.3e}");
        // Transport-correctness bound: element-local collocation would leave the bump
        // stuck (error ~0.5, the sine amplitude); the upwind lift advects it across
        // faces, leaving only upwind dissipation (~7e-3 at p=3, 4 elements).
        assert!(err < 2e-2, "3D upwind advection not transporting across elements: {err}");
    }

    #[test]
    fn conformation_inflow_fills_domain_3d() {
        // 3D analogue: stretched fluid enters at the WEST inlet (Face::ALL index 4 ⇒
        // tag 4) under uniform flow u=(1,0,0); carried downstream to fill the domain.
        let mesh = Mesh3d::rectangular(3, 4, 1, 1, [0.0, 2.0], [0.0, 1.0], [0.0, 1.0]);
        let c_in = [2.0, 0.5, 0.3, 1.0, 0.0, 1.0];
        let ob =
            OldroydB3d::new(&mesh, 1e6, 1.0).with_inflow(ConformationInflow3d::new(vec![4], c_in));
        let nn = mesh.refh.n_nodes();
        let nd = ob.ndof();
        let u = vec![1.0; nd];
        let z = vec![0.0; nd];
        let mut c = ob.identity();
        let dt = 4e-3_f64;
        for _ in 0..(8.0 / dt).round() as usize {
            c = ob.step_ssp_rk3(&c, &u, &z, &z, dt); // ~4 flow-throughs (L=2, U=1)
        }
        let mut err = 0.0f64;
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                if el.geom.x[k] > 0.6 {
                    let g = e * nn + k;
                    for o in 0..6 {
                        err = err.max((c[o][g] - c_in[o]).abs());
                    }
                }
            }
        }
        eprintln!("3D conformation inflow fill: downstream max|C − C_in| = {err:.3e}");
        assert!(err < 5e-3, "3D inflow conformation not carried downstream: {err}");
    }
}
