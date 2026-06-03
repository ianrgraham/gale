//! Generic DG operator for **hyperbolic conservation laws** `∂ₜu + ∇·F(u) = 0`,
//! parameterized by a pluggable [`ConservationLaw`]. This is the shared core for
//! both the incompressible-NS convection (advection law) and the compressible
//! entropy-stable track (Euler law) — see `docs/dg-gpu-fluid-simulation.md` §3.5.
//!
//! Weak nodal DG-SEM form with a Lax–Friedrichs (Rusanov) interface flux:
//! `M ∂ₜu = Dxᵀ(W Fx) + Dyᵀ(W Fy) − ∮ F*·n`, `∂ₜu = M⁻¹(…)`, `M = diag(Jw)`.
//! State is stored as `n_vars` scalar fields, each in the global `e·nn + k` layout.
//! A split-form volume option and an entropy-conserving two-point flux are layered
//! on in the next increment.

use super::face::Edge;
use super::mesh::{Mesh2d, Neighbor};

/// A hyperbolic conservation law: physical flux, interface wave speed, etc.
pub trait ConservationLaw {
    /// Number of conserved variables.
    fn n_vars(&self) -> usize;
    /// Physical flux at a state `u`: writes `fx`, `fy` (each length `n_vars`).
    fn flux(&self, u: &[f64], fx: &mut [f64], fy: &mut [f64]);
    /// Maximum signal speed in direction `(nx, ny)` at state `u` (for Rusanov).
    fn max_wave_speed(&self, u: &[f64], nx: f64, ny: f64) -> f64;

    /// Symmetric, consistent two-point volume flux for flux-differencing (split
    /// form). Default is the central average `½(F(uL)+F(uR))`; nonlinear laws
    /// override with an **entropy-conserving** flux for high-Re robustness.
    fn two_point_flux(&self, ul: &[f64], ur: &[f64], fx: &mut [f64], fy: &mut [f64]) {
        let nv = self.n_vars();
        let (mut axl, mut ayl, mut axr, mut ayr) =
            (vec![0.0; nv], vec![0.0; nv], vec![0.0; nv], vec![0.0; nv]);
        self.flux(ul, &mut axl, &mut ayl);
        self.flux(ur, &mut axr, &mut ayr);
        for v in 0..nv {
            fx[v] = 0.5 * (axl[v] + axr[v]);
            fy[v] = 0.5 * (ayl[v] + ayr[v]);
        }
    }
}

/// Inviscid **Burgers** equation `∂ₜu + ∂ₓ(u²/2) = 0` (1D flux in x). Nonlinear test
/// law exercising the split-form / entropy-conserving machinery.
pub struct Burgers;

impl ConservationLaw for Burgers {
    fn n_vars(&self) -> usize {
        1
    }
    fn flux(&self, u: &[f64], fx: &mut [f64], fy: &mut [f64]) {
        fx[0] = 0.5 * u[0] * u[0];
        fy[0] = 0.0;
    }
    fn max_wave_speed(&self, u: &[f64], nx: f64, _ny: f64) -> f64 {
        (u[0] * nx).abs()
    }
    /// Entropy-conserving two-point flux for Burgers: `(uL² + uL·uR + uR²)/6`.
    fn two_point_flux(&self, ul: &[f64], ur: &[f64], fx: &mut [f64], fy: &mut [f64]) {
        fx[0] = (ul[0] * ul[0] + ul[0] * ur[0] + ur[0] * ur[0]) / 6.0;
        fy[0] = 0.0;
    }
}

/// Numerically-stable logarithmic mean `(a−b)/(ln a − ln b)` (Ismail–Roe).
pub fn ln_mean(a: f64, b: f64) -> f64 {
    let d = (a - b) / (a + b);
    if d.abs() < 1e-4 {
        let d2 = d * d;
        0.5 * (a + b) / (1.0 + d2 / 3.0 + d2 * d2 / 5.0 + d2 * d2 * d2 / 7.0)
    } else {
        (a - b) / (a.ln() - b.ln())
    }
}

/// 2D compressible **Euler** equations, conserved variables `(ρ, ρu, ρv, E)`.
/// Provides the Chandrashekar entropy-conserving two-point flux ⇒ with the
/// [`VolumeForm::SplitForm`] volume this is an **entropy-stable DGSEM**.
pub struct Euler {
    pub gamma: f64,
}

impl Euler {
    fn primitives(&self, u: &[f64]) -> (f64, f64, f64, f64) {
        let (r, ux, uy) = (u[0], u[1] / u[0], u[2] / u[0]);
        let p = (self.gamma - 1.0) * (u[3] - 0.5 * r * (ux * ux + uy * uy));
        (r, ux, uy, p)
    }
}

impl ConservationLaw for Euler {
    fn n_vars(&self) -> usize {
        4
    }
    fn flux(&self, u: &[f64], fx: &mut [f64], fy: &mut [f64]) {
        let (r, ux, uy, p) = self.primitives(u);
        let e = u[3];
        fx[0] = r * ux;
        fx[1] = r * ux * ux + p;
        fx[2] = r * ux * uy;
        fx[3] = (e + p) * ux;
        fy[0] = r * uy;
        fy[1] = r * ux * uy;
        fy[2] = r * uy * uy + p;
        fy[3] = (e + p) * uy;
    }
    fn max_wave_speed(&self, u: &[f64], nx: f64, ny: f64) -> f64 {
        let (r, ux, uy, p) = self.primitives(u);
        let c = (self.gamma * p / r).sqrt();
        (ux * nx + uy * ny).abs() + c
    }
    /// Chandrashekar (2013) entropy-conserving flux.
    fn two_point_flux(&self, ul: &[f64], ur: &[f64], fx: &mut [f64], fy: &mut [f64]) {
        let g = self.gamma;
        let (rl, uxl, uyl, pl) = self.primitives(ul);
        let (rr, uxr, uyr, pr) = self.primitives(ur);
        let (bl, br) = (rl / (2.0 * pl), rr / (2.0 * pr));
        let rln = ln_mean(rl, rr);
        let bln = ln_mean(bl, br);
        let rbar = 0.5 * (rl + rr);
        let bbar = 0.5 * (bl + br);
        let ubar = 0.5 * (uxl + uxr);
        let vbar = 0.5 * (uyl + uyr);
        let phat = rbar / (2.0 * bbar);
        let vsq = 0.5 * (uxl * uxr + uyl * uyr);
        let h = 1.0 / (2.0 * (g - 1.0) * bln) - vsq;
        // x-direction.
        let f1 = rln * ubar;
        let f2 = f1 * ubar + phat;
        let f3 = f1 * vbar;
        fx[0] = f1;
        fx[1] = f2;
        fx[2] = f3;
        fx[3] = f1 * h + ubar * f2 + vbar * f3;
        // y-direction.
        let g1 = rln * vbar;
        let g2 = g1 * ubar;
        let g3 = g1 * vbar + phat;
        fy[0] = g1;
        fy[1] = g2;
        fy[2] = g3;
        fy[3] = g1 * h + ubar * g2 + vbar * g3;
    }
}

/// Incompressible momentum **convection** as a conservation law: state `(ux, uy)`,
/// flux `u⊗u` (so `∇·F = (u·∇)u` for divergence-free `u`). The kinetic-energy-
/// preserving two-point flux `{u_j}{u_i}` gives the energy-stable split form used
/// for high-Re robustness of the incompressible NS convection.
pub struct IncompressibleConvection;

impl ConservationLaw for IncompressibleConvection {
    fn n_vars(&self) -> usize {
        2
    }
    fn flux(&self, u: &[f64], fx: &mut [f64], fy: &mut [f64]) {
        fx[0] = u[0] * u[0];
        fx[1] = u[0] * u[1];
        fy[0] = u[0] * u[1];
        fy[1] = u[1] * u[1];
    }
    fn max_wave_speed(&self, u: &[f64], nx: f64, ny: f64) -> f64 {
        let vn = (u[0] * nx + u[1] * ny).abs();
        let vmag = (u[0] * u[0] + u[1] * u[1]).sqrt();
        vn + vmag
    }
    /// KEP two-point flux: `{u_j}{u_i}` (product of arithmetic means).
    fn two_point_flux(&self, ul: &[f64], ur: &[f64], fx: &mut [f64], fy: &mut [f64]) {
        let ub = 0.5 * (ul[0] + ur[0]);
        let vb = 0.5 * (ul[1] + ur[1]);
        fx[0] = ub * ub;
        fx[1] = ub * vb;
        fy[0] = vb * ub;
        fy[1] = vb * vb;
    }
}

/// Volume discretization: standard weak form, or split-form flux-differencing
/// (entropy/energy-stable for high-Re).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VolumeForm {
    Weak,
    SplitForm,
}

/// Constant-coefficient linear advection `∂ₜu + a·∇u = 0`.
pub struct LinearAdvection {
    pub ax: f64,
    pub ay: f64,
}

impl ConservationLaw for LinearAdvection {
    fn n_vars(&self) -> usize {
        1
    }
    fn flux(&self, u: &[f64], fx: &mut [f64], fy: &mut [f64]) {
        fx[0] = self.ax * u[0];
        fy[0] = self.ay * u[0];
    }
    fn max_wave_speed(&self, _u: &[f64], nx: f64, ny: f64) -> f64 {
        (self.ax * nx + self.ay * ny).abs()
    }
}

pub struct Hyperbolic<'m, L: ConservationLaw> {
    pub mesh: &'m Mesh2d,
    pub law: L,
    /// Volume discretization (Weak or SplitForm).
    pub form: VolumeForm,
    /// Whether the interface flux carries Rusanov dissipation (false ⇒ central,
    /// entropy-conserving; true ⇒ entropy-stable).
    pub dissipation: bool,
}

impl<'m, L: ConservationLaw> Hyperbolic<'m, L> {
    /// Standard weak-form operator with Rusanov dissipation.
    pub fn new(mesh: &'m Mesh2d, law: L) -> Self {
        Self { mesh, law, form: VolumeForm::Weak, dissipation: true }
    }

    /// Operator with an explicit volume form and dissipation choice.
    pub fn with_options(mesh: &'m Mesh2d, law: L, form: VolumeForm, dissipation: bool) -> Self {
        Self { mesh, law, form, dissipation }
    }

    pub fn n_vars(&self) -> usize {
        self.law.n_vars()
    }
    pub fn ndof(&self) -> usize {
        self.mesh.n_elements() * self.mesh.refq.n_nodes()
    }

    fn zeros(&self) -> Vec<Vec<f64>> {
        vec![vec![0.0; self.ndof()]; self.n_vars()]
    }

    /// Semi-discrete RHS `∂ₜu = L(u)`. `bc(x, y, t, out)` fills the exterior state at
    /// a boundary node (used in the interface flux for inflow/outflow).
    pub fn rhs(
        &self,
        state: &[Vec<f64>],
        t: f64,
        bc: &impl Fn(f64, f64, f64, &mut [f64]),
    ) -> Vec<Vec<f64>> {
        match self.form {
            VolumeForm::Weak => self.rhs_weak(state, t, bc),
            VolumeForm::SplitForm => self.rhs_split(state, t, bc),
        }
    }

    /// Rusanov (LLF) numerical flux `F*·n` per variable into `out`, with outward
    /// normal `(nx, ny)`. Factored so the conforming and mortar paths are identical.
    fn rusanov(&self, um: &[f64], up: &[f64], nx: f64, ny: f64, out: &mut [f64]) {
        let nv = self.n_vars();
        let (mut fxm, mut fym) = (vec![0.0; nv], vec![0.0; nv]);
        let (mut fxp, mut fyp) = (vec![0.0; nv], vec![0.0; nv]);
        self.law.flux(um, &mut fxm, &mut fym);
        self.law.flux(up, &mut fxp, &mut fyp);
        let lam = self
            .law
            .max_wave_speed(um, nx, ny)
            .max(self.law.max_wave_speed(up, nx, ny));
        let diss = if self.dissipation { lam } else { 0.0 };
        for v in 0..nv {
            let fnm = fxm[v] * nx + fym[v] * ny;
            let fnp = fxp[v] * nx + fyp[v] * ny;
            out[v] = 0.5 * (fnm + fnp) - 0.5 * diss * (up[v] - um[v]);
        }
    }

    fn rhs_weak(
        &self,
        state: &[Vec<f64>],
        t: f64,
        bc: &impl Fn(f64, f64, f64, &mut [f64]),
    ) -> Vec<Vec<f64>> {
        let mesh = self.mesh;
        let refq = &mesh.refq;
        let nn = refq.n_nodes();
        let nv = self.n_vars();
        let ndof = self.ndof();

        // Physical fluxes at every node.
        let mut fx = vec![vec![0.0; ndof]; nv];
        let mut fy = vec![vec![0.0; ndof]; nv];
        {
            let (mut u, mut a, mut b) = (vec![0.0; nv], vec![0.0; nv], vec![0.0; nv]);
            for i in 0..ndof {
                for v in 0..nv {
                    u[v] = state[v][i];
                }
                self.law.flux(&u, &mut a, &mut b);
                for v in 0..nv {
                    fx[v][i] = a[v];
                    fy[v][i] = b[v];
                }
            }
        }

        // Volume term: Dxᵀ(W Fx) + Dyᵀ(W Fy).
        let mut res = self.zeros();
        for v in 0..nv {
            for (e, el) in mesh.elements.iter().enumerate() {
                let wfx: Vec<f64> = (0..nn).map(|k| el.geom.jw[k] * fx[v][e * nn + k]).collect();
                let wfy: Vec<f64> = (0..nn).map(|k| el.geom.jw[k] * fy[v][e * nn + k]).collect();
                let a = el.geom.gradx_t(refq, &wfx);
                let b = el.geom.grady_t(refq, &wfy);
                for k in 0..nn {
                    res[v][e * nn + k] += a[k] + b[k];
                }
            }
        }

        // Interface term: − ∮ F*·n, Rusanov (LLF), with 2:1 mortar coupling at
        // non-conforming faces. Element-centric; non-conforming faces are processed
        // once from the coarse side (scattering to coarse + both fine neighbours).
        {
            let (mut um, mut up, mut fstar) = (vec![0.0; nv], vec![0.0; nv], vec![0.0; nv]);
            let mortar = super::amr::RefineQuad::new(mesh.order);
            let sorted = |e: usize, edge: Edge| -> Vec<usize> {
                let f = &mesh.elements[e].faces[edge as usize];
                let g = &mesh.elements[e].geom;
                let vert = matches!(edge, Edge::East | Edge::West);
                let mut idx: Vec<usize> = (0..f.nodes.len()).collect();
                idx.sort_by(|&a, &b| {
                    let ca = if vert { g.y[f.nodes[a]] } else { g.x[f.nodes[a]] };
                    let cb = if vert { g.y[f.nodes[b]] } else { g.x[f.nodes[b]] };
                    ca.partial_cmp(&cb).unwrap()
                });
                idx
            };
            for e in 0..mesh.elements.len() {
                for edge in Edge::ALL {
                    let face = &mesh.elements[e].faces[edge as usize];
                    match mesh.elements[e].neighbors[edge as usize].clone() {
                        Neighbor::FineToCoarse { .. } => {} // handled from the coarse side
                        Neighbor::Interior { elem: re, edge: redge, perm } => {
                            let rf = &mesh.elements[re].faces[redge as usize];
                            for ai in 0..face.nodes.len() {
                                let vl = face.nodes[ai];
                                let (nx, ny, sw) = (face.nx[ai], face.ny[ai], face.sw[ai]);
                                let rnode = rf.nodes[perm[ai]];
                                for v in 0..nv {
                                    um[v] = state[v][e * nn + vl];
                                    up[v] = state[v][re * nn + rnode];
                                }
                                self.rusanov(&um, &up, nx, ny, &mut fstar);
                                for v in 0..nv {
                                    res[v][e * nn + vl] -= sw * fstar[v];
                                }
                            }
                        }
                        Neighbor::Boundary { .. } => {
                            let g = &mesh.elements[e].geom;
                            for ai in 0..face.nodes.len() {
                                let vl = face.nodes[ai];
                                let (nx, ny, sw) = (face.nx[ai], face.ny[ai], face.sw[ai]);
                                for v in 0..nv {
                                    um[v] = state[v][e * nn + vl];
                                }
                                bc(g.x[vl], g.y[vl], t, &mut up);
                                self.rusanov(&um, &up, nx, ny, &mut fstar);
                                for v in 0..nv {
                                    res[v][e * nn + vl] -= sw * fstar[v];
                                }
                            }
                        }
                        Neighbor::CoarseToFine { fine } => {
                            let ce = sorted(e, edge);
                            let (ncx, ncy) = (face.nx[ce[0]], face.ny[ce[0]]); // axis-aligned
                            let uc: Vec<Vec<f64>> = (0..nv)
                                .map(|v| ce.iter().map(|&i| state[v][e * nn + face.nodes[i]]).collect())
                                .collect();
                            let mut fstar_half: [Vec<Vec<f64>>; 2] = [Vec::new(), Vec::new()];
                            for h in 0..2 {
                                let (re, redge) = fine[h];
                                let rw = sorted(re, redge);
                                let frw = &mesh.elements[re].faces[redge as usize];
                                let uc_h: Vec<Vec<f64>> =
                                    (0..nv).map(|v| mortar.mortar_to_fine(&uc[v], h)).collect();
                                let uf: Vec<Vec<f64>> = (0..nv)
                                    .map(|v| rw.iter().map(|&i| state[v][re * nn + frw.nodes[i]]).collect())
                                    .collect();
                                let mut fh: Vec<Vec<f64>> = (0..nv).map(|_| vec![0.0; rw.len()]).collect();
                                for m in 0..rw.len() {
                                    for v in 0..nv {
                                        um[v] = uc_h[v][m];
                                        up[v] = uf[v][m];
                                    }
                                    // Flux with the coarse-outward normal (for back-projection).
                                    self.rusanov(&um, &up, ncx, ncy, &mut fstar);
                                    for v in 0..nv {
                                        fh[v][m] = fstar[v];
                                    }
                                    // Fine element receives the flux with its own normal.
                                    let i = rw[m];
                                    let k = frw.nodes[i];
                                    self.rusanov(&up, &um, frw.nx[i], frw.ny[i], &mut fstar);
                                    for v in 0..nv {
                                        res[v][re * nn + k] -= frw.sw[i] * fstar[v];
                                    }
                                }
                                fstar_half[h] = fh;
                            }
                            // Coarse edge receives the back-projected mortar flux.
                            for v in 0..nv {
                                let fc = mortar
                                    .mortar_to_coarse(&[fstar_half[0][v].clone(), fstar_half[1][v].clone()]);
                                for m in 0..ce.len() {
                                    let i = ce[m];
                                    res[v][e * nn + face.nodes[i]] -= face.sw[i] * fc[m];
                                }
                            }
                        }
                    }
                }
            }
        }

        // ∂ₜu = M⁻¹ res.
        let mut dudt = self.zeros();
        for v in 0..nv {
            for (e, el) in mesh.elements.iter().enumerate() {
                for k in 0..nn {
                    dudt[v][e * nn + k] = res[v][e * nn + k] / el.geom.jw[k];
                }
            }
        }
        dudt
    }

    /// Split-form (flux-differencing) strong-form RHS — entropy/energy-stable.
    /// Volume `(∇·F)#ᵢ = (1/J)·2 Σⱼ D[i,j] F̃#(uᵢ,uⱼ)` (contravariant two-point flux,
    /// metric-averaged); strong-form surface `+ (1/Jw)(F·n − F*·n)`.
    /// Affine elements (constant per-element metrics) are assumed.
    fn rhs_split(
        &self,
        state: &[Vec<f64>],
        t: f64,
        bc: &impl Fn(f64, f64, f64, &mut [f64]),
    ) -> Vec<Vec<f64>> {
        let mesh = self.mesh;
        let refq = &mesh.refq;
        let n1 = refq.n_1d();
        let nn = refq.n_nodes();
        let nv = self.n_vars();
        let d = &refq.line.diff; // n1×n1
        let mut dudt = self.zeros();

        let (mut ui, mut um) = (vec![0.0; nv], vec![0.0; nv]);
        let (mut fpx, mut fpy) = (vec![0.0; nv], vec![0.0; nv]);

        for (e, el) in mesh.elements.iter().enumerate() {
            let g = &el.geom;
            // Volume: flux-differencing along each reference direction.
            for js in 0..n1 {
                for ir in 0..n1 {
                    let li = ir + js * n1;
                    let gi = e * nn + li;
                    for v in 0..nv {
                        ui[v] = state[v][gi];
                    }
                    let mut vol = vec![0.0; nv];
                    // r-direction line.
                    for mm in 0..n1 {
                        let lm = mm + js * n1;
                        for v in 0..nv {
                            um[v] = state[v][e * nn + lm];
                        }
                        self.law.two_point_flux(&ui, &um, &mut fpx, &mut fpy);
                        let jb = 0.5 * (g.jac[li] + g.jac[lm]);
                        let rxb = 0.5 * (g.rx[li] + g.rx[lm]);
                        let ryb = 0.5 * (g.ry[li] + g.ry[lm]);
                        let dr = 2.0 * d[ir * n1 + mm];
                        for v in 0..nv {
                            vol[v] += dr * jb * (rxb * fpx[v] + ryb * fpy[v]);
                        }
                    }
                    // s-direction line.
                    for mm in 0..n1 {
                        let lm = ir + mm * n1;
                        for v in 0..nv {
                            um[v] = state[v][e * nn + lm];
                        }
                        self.law.two_point_flux(&ui, &um, &mut fpx, &mut fpy);
                        let jb = 0.5 * (g.jac[li] + g.jac[lm]);
                        let sxb = 0.5 * (g.sx[li] + g.sx[lm]);
                        let syb = 0.5 * (g.sy[li] + g.sy[lm]);
                        let ds = 2.0 * d[js * n1 + mm];
                        for v in 0..nv {
                            vol[v] += ds * jb * (sxb * fpx[v] + syb * fpy[v]);
                        }
                    }
                    for v in 0..nv {
                        dudt[v][gi] = -vol[v] / g.jac[li];
                    }
                }
            }
            // Strong-form surface: + (1/Jw)(F(uᵢ)·n − F*·n).
            let (mut sm, mut sp) = (vec![0.0; nv], vec![0.0; nv]);
            let (mut fxm, mut fym) = (vec![0.0; nv], vec![0.0; nv]);
            let (mut fxp, mut fyp) = (vec![0.0; nv], vec![0.0; nv]);
            for edge in Edge::ALL {
                let face = &el.faces[edge as usize];
                let nb = &el.neighbors[edge as usize];
                for ai in 0..face.nodes.len() {
                    let vl = face.nodes[ai];
                    let (nx, ny, sw) = (face.nx[ai], face.ny[ai], face.sw[ai]);
                    for v in 0..nv {
                        sm[v] = state[v][e * nn + vl];
                    }
                    match nb {
                        Neighbor::Interior { elem: re, edge: redge, perm } => {
                            let rf = &mesh.elements[*re].faces[*redge as usize];
                            let rnode = rf.nodes[perm[ai]];
                            for v in 0..nv {
                                sp[v] = state[v][*re * nn + rnode];
                            }
                        }
                        Neighbor::Boundary { .. } => bc(g.x[vl], g.y[vl], t, &mut sp),
                        Neighbor::CoarseToFine { .. } | Neighbor::FineToCoarse { .. } => {
                            panic!("split-form does not support non-conforming meshes; use VolumeForm::Weak")
                        }
                    }
                    self.law.flux(&sm, &mut fxm, &mut fym);
                    self.law.flux(&sp, &mut fxp, &mut fyp);
                    let lam = self
                        .law
                        .max_wave_speed(&sm, nx, ny)
                        .max(self.law.max_wave_speed(&sp, nx, ny));
                    let diss = if self.dissipation { lam } else { 0.0 };
                    for v in 0..nv {
                        let fnm = fxm[v] * nx + fym[v] * ny;
                        let fnp = fxp[v] * nx + fyp[v] * ny;
                        let fstar = 0.5 * (fnm + fnp) - 0.5 * diss * (sp[v] - sm[v]);
                        dudt[v][e * nn + vl] += sw * (fnm - fstar) / g.jw[vl];
                    }
                }
            }
        }
        dudt
    }

    /// One SSP-RK3 (Shu–Osher) step from time `t` by `dt`.
    pub fn step_ssp_rk3(
        &self,
        state: &[Vec<f64>],
        t: f64,
        dt: f64,
        bc: &impl Fn(f64, f64, f64, &mut [f64]),
    ) -> Vec<Vec<f64>> {
        let nv = self.n_vars();
        let n = self.ndof();
        let comb = |coeffs: &[(f64, &[Vec<f64>])]| -> Vec<Vec<f64>> {
            let mut out = vec![vec![0.0; n]; nv];
            for &(c, s) in coeffs {
                for v in 0..nv {
                    for i in 0..n {
                        out[v][i] += c * s[v][i];
                    }
                }
            }
            out
        };
        let k1 = self.rhs(state, t, bc);
        let u1 = comb(&[(1.0, state), (dt, &k1)]);
        let k2 = self.rhs(&u1, t + dt, bc);
        let u2 = comb(&[(0.75, state), (0.25, &u1), (0.25 * dt, &k2)]);
        let k3 = self.rhs(&u2, t + 0.5 * dt, bc);
        comb(&[(1.0 / 3.0, state), (2.0 / 3.0, &u2), (2.0 / 3.0 * dt, &k3)])
    }

    /// L2 norm of variable `v`: `√(Σ Jw·u²)`.
    pub fn l2_norm(&self, state: &[Vec<f64>], v: usize) -> f64 {
        let nn = self.mesh.refq.n_nodes();
        let mut s = 0.0;
        for (e, el) in self.mesh.elements.iter().enumerate() {
            for k in 0..nn {
                s += el.geom.jw[k] * state[v][e * nn + k].powi(2);
            }
        }
        s.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn euler_cons(gamma: f64, r: f64, ux: f64, uy: f64, p: f64) -> [f64; 4] {
        let e = p / (gamma - 1.0) + 0.5 * r * (ux * ux + uy * uy);
        [r, r * ux, r * uy, e]
    }

    #[test]
    fn euler_free_stream_preserved() {
        // A uniform state has zero residual to round-off (flux consistency).
        let mesh = Mesh2d::rectangular_periodic(3, 3, 3, [0.0, 1.0], [0.0, 1.0]);
        let op = Hyperbolic::with_options(&mesh, Euler { gamma: 1.4 }, VolumeForm::SplitForm, true);
        let c = euler_cons(1.4, 1.2, 0.3, -0.1, 1.0);
        let state: Vec<Vec<f64>> = (0..4).map(|v| vec![c[v]; op.ndof()]).collect();
        let r = op.rhs(&state, 0.0, &|_, _, _, _: &mut [f64]| {});
        let md = (0..4)
            .flat_map(|v| r[v].iter().cloned())
            .fold(0.0f64, |a, x| a.max(x.abs()));
        assert!(md < 1e-9, "free-stream not preserved: {md}");
    }

    #[test]
    fn euler_free_stream_on_refined_mesh() {
        // Full integration: the shared weak-form Euler operator on a NON-CONFORMING
        // mesh (centre cell of a 3×3 refined → four 2:1 interfaces). A uniform state
        // must give zero residual everywhere across the hanging nodes.
        let mesh = Mesh2d::cartesian_refined(4, 3, 3, [0.0, 3.0], [0.0, 3.0], &[(1, 1)]);
        let op = Hyperbolic::new(&mesh, Euler { gamma: 1.4 });
        let c = euler_cons(1.4, 1.2, 0.3, -0.1, 1.0);
        let ndof = mesh.n_elements() * mesh.refq.n_nodes();
        let state: Vec<Vec<f64>> = (0..4).map(|v| vec![c[v]; ndof]).collect();
        let r = op.rhs(&state, 0.0, &move |_, _, _, out: &mut [f64]| out.copy_from_slice(&c));
        let md = (0..4).flat_map(|v| r[v].iter().cloned()).fold(0.0f64, |a, x| a.max(x.abs()));
        assert!(md < 1e-9, "Euler free-stream not preserved on refined mesh: {md}");
    }

    #[test]
    fn advection_linear_exact_on_refined_mesh() {
        // Linear advection of a globally-linear field on a refined mesh reproduces
        // ∂ₜu = −(aₓα + a_yβ) exactly — high order retained through every hanging node.
        let (ax, ay) = (0.8, -0.5);
        let mesh = Mesh2d::cartesian_refined(4, 3, 3, [0.0, 3.0], [0.0, 3.0], &[(1, 1)]);
        let op = Hyperbolic::new(&mesh, LinearAdvection { ax, ay });
        let (al, be, ga) = (0.7, -0.4, 0.2);
        let f = move |x: f64, y: f64| al * x + be * y + ga;
        let nn = mesh.refq.n_nodes();
        let mut u = vec![0.0; mesh.n_elements() * nn];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                u[e * nn + k] = f(el.geom.x[k], el.geom.y[k]);
            }
        }
        let r = op.rhs(&[u], 0.0, &move |x, y, _, out: &mut [f64]| out[0] = f(x, y));
        let expected = -(ax * al + ay * be);
        let md = r[0].iter().fold(0.0f64, |a, &x| a.max((x - expected).abs()));
        assert!(md < 1e-9, "linear advection not exact on refined Mesh2d: {md}");
    }

    #[test]
    fn euler_density_wave_converges() {
        // Exact Euler solution: ρ=1+0.2 sin(2π(x−t)), u=1, v=0, p=1 (constant p,u).
        let gamma = 1.4;
        let exact_rho = |x: f64, t: f64| 1.0 + 0.2 * (2.0 * PI * (x - t)).sin();
        let t_end = 0.1;
        let mut errs = Vec::new();
        for &nx in &[4usize, 8] {
            let mesh = Mesh2d::rectangular_periodic(3, nx, nx, [0.0, 1.0], [0.0, 1.0]);
            let op = Hyperbolic::with_options(&mesh, Euler { gamma }, VolumeForm::SplitForm, true);
            let nn = mesh.refq.n_nodes();
            let mut state: Vec<Vec<f64>> = (0..4).map(|_| vec![0.0; op.ndof()]).collect();
            for (e, el) in mesh.elements.iter().enumerate() {
                for k in 0..nn {
                    let c = euler_cons(gamma, exact_rho(el.geom.x[k], 0.0), 1.0, 0.0, 1.0);
                    for v in 0..4 {
                        state[v][e * nn + k] = c[v];
                    }
                }
            }
            let h = 1.0 / nx as f64;
            let dt = 0.1 * h / 7.0;
            let nsteps = (t_end / dt).ceil() as usize;
            let dt = t_end / nsteps as f64;
            let mut t = 0.0;
            for _ in 0..nsteps {
                state = op.step_ssp_rk3(&state, t, dt, &|_, _, _, _: &mut [f64]| {});
                t += dt;
            }
            let mut err = vec![vec![0.0; op.ndof()]; 4];
            for (e, el) in mesh.elements.iter().enumerate() {
                for k in 0..nn {
                    err[0][e * nn + k] = state[0][e * nn + k] - exact_rho(el.geom.x[k], t_end);
                }
            }
            errs.push(op.l2_norm(&err, 0));
        }
        eprintln!("Euler density-wave ρ error (nx=4,8): {errs:?}");
        assert!(errs[1] < errs[0] / 4.0, "not high-order: {errs:?}");
    }

    #[test]
    fn euler_split_form_is_entropy_conservative() {
        // Decisive EC test: split-form + central (no dissipation) on periodic Euler
        // ⇒ the semi-discrete entropy rate Σ Jw·(w·∂ₜu) must vanish to round-off,
        // which holds *iff* the Chandrashekar two-point flux is correct.
        let gamma = 1.4;
        let mesh = Mesh2d::rectangular_periodic(4, 4, 4, [0.0, 1.0], [0.0, 1.0]);
        let op = Hyperbolic::with_options(&mesh, Euler { gamma }, VolumeForm::SplitForm, false);
        let nn = mesh.refq.n_nodes();
        let mut state: Vec<Vec<f64>> = (0..4).map(|_| vec![0.0; op.ndof()]).collect();
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let (x, y) = (el.geom.x[k], el.geom.y[k]);
                // smooth, fully non-uniform state
                let r = 1.0 + 0.2 * (2.0 * PI * x).sin() * (2.0 * PI * y).cos();
                let ux = 0.3 + 0.1 * (2.0 * PI * y).sin();
                let uy = -0.2 + 0.1 * (2.0 * PI * x).cos();
                let p = 1.0 + 0.1 * (2.0 * PI * (x + y)).sin();
                let c = euler_cons(gamma, r, ux, uy, p);
                for v in 0..4 {
                    state[v][e * nn + k] = c[v];
                }
            }
        }
        let dudt = op.rhs(&state, 0.0, &|_, _, _, _: &mut [f64]| {});
        // entropy variables w = ∂U/∂u and the contraction Σ Jw w·∂ₜu.
        let mut rate = 0.0;
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let i = e * nn + k;
                let (r, mx, my, en) = (state[0][i], state[1][i], state[2][i], state[3][i]);
                let (ux, uy) = (mx / r, my / r);
                let p = (gamma - 1.0) * (en - 0.5 * r * (ux * ux + uy * uy));
                let s = p.ln() - gamma * r.ln();
                let rp = r / p;
                let w = [
                    (gamma - s) / (gamma - 1.0) - 0.5 * rp * (ux * ux + uy * uy),
                    rp * ux,
                    rp * uy,
                    -rp,
                ];
                let wd: f64 = (0..4).map(|v| w[v] * dudt[v][i]).sum();
                rate += el.geom.jw[k] * wd;
            }
        }
        eprintln!("Euler semi-discrete entropy rate = {rate:.3e}");
        assert!(rate.abs() < 1e-9, "not entropy-conservative: rate {rate}");
    }

    #[test]
    fn split_form_matches_weak_for_linear_advection() {
        // For a linear flux there is no aliasing, so split-form (central two-point)
        // and weak form are algebraically identical — a structural check of the
        // flux-differencing volume + strong-form surface.
        let p = 3;
        let mesh = Mesh2d::rectangular(p, 3, 3, [0.0, 1.0], [0.0, 1.0]);
        let nn = mesh.refq.n_nodes();
        let ndof = mesh.n_elements() * nn;
        let weak = Hyperbolic::with_options(&mesh, LinearAdvection { ax: 0.8, ay: -0.5 }, VolumeForm::Weak, true);
        let split = Hyperbolic::with_options(&mesh, LinearAdvection { ax: 0.8, ay: -0.5 }, VolumeForm::SplitForm, true);
        let mut state = vec![vec![0.0; ndof]];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                state[0][e * nn + k] = (2.0 * el.geom.x[k]).sin() + 0.3 * el.geom.y[k] * el.geom.y[k];
            }
        }
        let bc = |x: f64, y: f64, _t: f64, out: &mut [f64]| out[0] = x.cos() * y;
        let rw = weak.rhs(&state, 0.0, &bc);
        let rs = split.rhs(&state, 0.0, &bc);
        let mut md = 0.0f64;
        for i in 0..ndof {
            md = md.max((rw[0][i] - rs[0][i]).abs());
        }
        assert!(md < 1e-10, "split ≠ weak for linear advection: {md}");
    }

    #[test]
    fn burgers_two_point_flux_consistent_and_symmetric() {
        let b = Burgers;
        let (mut fx, mut fy) = (vec![0.0], vec![0.0]);
        b.two_point_flux(&[1.7], &[1.7], &mut fx, &mut fy);
        assert!((fx[0] - 0.5 * 1.7 * 1.7).abs() < 1e-13, "consistency F#(u,u)=u²/2");
        let (mut g1, mut g2, mut h1, mut h2) = (vec![0.0], vec![0.0], vec![0.0], vec![0.0]);
        b.two_point_flux(&[0.4], &[2.1], &mut g1, &mut h1);
        b.two_point_flux(&[2.1], &[0.4], &mut g2, &mut h2);
        assert!((g1[0] - g2[0]).abs() < 1e-14, "symmetry");
    }

    #[test]
    fn split_form_burgers_is_entropy_stable_through_a_shock() {
        // Periodic Burgers from u=sin(2πx) steepens into a shock at t=1/(2π)≈0.159.
        // Under-resolved, split-form + Rusanov is entropy-stable: ∫u²/2 is
        // non-increasing (the high-Re robustness guarantee).
        let p = 3;
        let nx = 8;
        let mesh = Mesh2d::rectangular_periodic(p, nx, nx, [0.0, 1.0], [0.0, 1.0]);
        let op = Hyperbolic::with_options(&mesh, Burgers, VolumeForm::SplitForm, true);
        let nn = mesh.refq.n_nodes();
        let mut state = vec![vec![0.0; op.ndof()]];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                state[0][e * nn + k] = (2.0 * PI * el.geom.x[k]).sin();
            }
        }
        let bc = |_x: f64, _y: f64, _t: f64, _out: &mut [f64]| {}; // periodic: never called
        let e0 = op.l2_norm(&state, 0);
        let h = 1.0 / nx as f64;
        let dt = 0.15 * h / (2 * p + 1) as f64;
        let mut t = 0.0;
        for _ in 0..((0.3 / dt).ceil() as usize) {
            state = op.step_ssp_rk3(&state, t, dt, &bc);
            t += dt;
        }
        let et = op.l2_norm(&state, 0);
        eprintln!("Burgers entropy ‖u‖: t=0 {e0:.6}  t=0.3 {et:.6}");
        assert!(et.is_finite(), "blew up");
        assert!(et <= e0 + 1e-9, "entropy increased: {e0} → {et}");
    }

    #[test]
    fn linear_advection_transports_a_traveling_wave() {
        // u(x,y,t) = sin(2π(x − t)) solves ∂ₜu + ∂ₓu = 0 (a=(1,0)). Inflow at x=0;
        // exterior state = exact. Check high-order spatial convergence at fixed T.
        let p = 3;
        let exact = |x: f64, t: f64| (2.0 * PI * (x - t)).sin();
        let t_end = 0.2;
        let mut errs = Vec::new();
        for &nx in &[4usize, 8] {
            let mesh = Mesh2d::rectangular(p, nx, nx, [0.0, 1.0], [0.0, 1.0]);
            let op = Hyperbolic::new(&mesh, LinearAdvection { ax: 1.0, ay: 0.0 });
            let nn = mesh.refq.n_nodes();
            let mut state = vec![vec![0.0; op.ndof()]];
            for (e, el) in mesh.elements.iter().enumerate() {
                for k in 0..nn {
                    state[0][e * nn + k] = exact(el.geom.x[k], 0.0);
                }
            }
            let bc = |x: f64, _y: f64, t: f64, out: &mut [f64]| out[0] = exact(x, t);
            let h = 1.0 / nx as f64;
            let dt = 0.3 * h / (1.0 * (2 * p + 1) as f64);
            let nsteps = (t_end / dt).ceil() as usize;
            let dt = t_end / nsteps as f64;
            let mut t = 0.0;
            for _ in 0..nsteps {
                state = op.step_ssp_rk3(&state, t, dt, &bc);
                t += dt;
            }
            let mut err = vec![vec![0.0; op.ndof()]];
            for (e, el) in mesh.elements.iter().enumerate() {
                for k in 0..nn {
                    err[0][e * nn + k] = state[0][e * nn + k] - exact(el.geom.x[k], t_end);
                }
            }
            errs.push(op.l2_norm(&err, 0));
        }
        eprintln!("advection L2 error (nx=4,8): {errs:?}");
        assert!(errs[1] < errs[0] / 4.0, "not high-order: {errs:?}");
        assert!(errs[1] < 1e-3, "final error {} too large", errs[1]);
    }
}
