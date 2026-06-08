//! Viscoelastic constitutive transport — the **Oldroyd-B** model via the
//! conformation tensor `C` (symmetric positive-definite, stored as `[Cxx, Cxy, Cyy]`).
//!
//! The conformation tensor obeys the upper-convected Maxwell equation
//! ```text
//!   ∂C/∂t + (u·∇)C = L·C + C·Lᵀ − (1/λ)(C − I),     L = ∇u  (Lᵢⱼ = ∂uᵢ/∂xⱼ)
//! ```
//! and the polymer contribution to the stress is `τ_p = (η_p/λ)(C − I)`. The
//! advection is treated by nodal collocation (element-local, like the NS
//! convection) — adequate for smooth/decoupled validation; the stretching and
//! relaxation terms are algebraic and pointwise. The headline target couples
//! `∇·τ_p` into the incompressible momentum balance; this module first validates
//! the constitutive transport on its own against the steady simple-shear solution
//! `Cxx = 1 + 2Wi², Cxy = Wi, Cyy = 1` (`Wi = λγ̇`).
//!
//! High-`Wi` robustness (the log-conformation reformulation that keeps `C` SPD) is
//! a follow-on; the direct form here is stable at moderate `Wi`.

use super::face::Edge;
use super::mesh::{Mesh2d, Neighbor};
use super::stokes::Stokes;

/// Inflow boundary data for the conformation transport: at boundary nodes whose tag is
/// in `tags` **and** where flow enters (`u·n < 0`), the upwind external trace is the
/// constant conformation `c = [Cxx, Cxy, Cyy]` (the incoming polymer state — e.g.
/// `[1, 0, 1]` for relaxed fluid). Boundaries not listed (walls, outflow) are
/// transparent (the field advects out / is set by the interior). Spatially- or
/// temporally-varying inflow is a follow-up.
#[derive(Clone, Debug)]
pub struct ConformationInflow {
    pub tags: Vec<u32>,
    pub c: [f64; 3],
}

impl ConformationInflow {
    /// Prescribe the incoming conformation `c = [Cxx, Cxy, Cyy]` on boundary `tags`.
    pub fn new(tags: Vec<u32>, c: [f64; 3]) -> Self {
        Self { tags, c }
    }

    /// Relaxed (equilibrium, `C = I`) fluid entering on the given `tags`.
    pub fn equilibrium(tags: Vec<u32>) -> Self {
        Self { tags, c: [1.0, 0.0, 1.0] }
    }
}

/// Upwind DG **surface lift** for the advection of a 3-component (symmetric-tensor)
/// field `φ` by the divergence-free velocity `(ux, uy)`. Returns the correction to ADD
/// to the collocation volume term `−(u·∇)φ`, upgrading it to full upwind DG transport
/// with inter-element coupling — the collocation term alone is element-local and does
/// not transport `φ` across element faces.
///
/// Strong form: `φ̇ = −∇·F + M⁻¹∮ v (F·n − F*·n)`, `F = uφ`. The volume term `−∇·F`
/// (incompressible ⇒ `−(u·∇)φ`) is the existing collocation term; this returns the
/// lift `M⁻¹∮ v (u·n)(φ⁻ − φ*)`, which at a collocated GLL face node `i` is
/// `(sw_i/jw_i)(u·n)_i (φ⁻_i − φ*_i)`. It is nonzero only at **inflow** nodes
/// (`u·n < 0`), with `φ*` the upwind trace: the neighbour across an interior face, the
/// `bdry` datum at a boundary inflow node (in `φ`'s own variable — `C` for the direct
/// form, `Ψ = log C` for log-conformation), or transparent (`φ⁻`, no correction) where
/// `bdry` returns `None`. Non-conforming faces are not handled (viscoelastic AMR is a
/// follow-up); they contribute no correction.
///
/// Public so the GPU conformation advance (`gale_gpu`) can add this cheap O(N) surface
/// correction host-side to its device-computed collocation volume term — the standard
/// "expensive solves on device, cheap element-local assembly on host" split.
pub fn upwind_advection_lift(
    mesh: &Mesh2d,
    field: &[Vec<f64>; 3],
    ux: &[f64],
    uy: &[f64],
    bdry: impl Fn(u32) -> Option<[f64; 3]>,
) -> [Vec<f64>; 3] {
    let nn = mesh.refq.n_nodes();
    let n = mesh.n_elements() * nn;
    let mut out = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    for (e, el) in mesh.elements.iter().enumerate() {
        for edge in Edge::ALL {
            let face = &el.faces[edge as usize];
            let nb = &el.neighbors[edge as usize];
            for a in 0..face.nodes.len() {
                let vl = face.nodes[a];
                let g = e * nn + vl;
                let un = ux[g] * face.nx[a] + uy[g] * face.ny[a];
                if un >= 0.0 {
                    continue; // outflow node: upwind = interior ⇒ no correction
                }
                // Inflow node: external (upwind) trace φ*.
                let ext: [f64; 3] = match nb {
                    Neighbor::Interior { elem: re, edge: redge, perm } => {
                        let rf = &mesh.elements[*re].faces[*redge as usize];
                        let vr = *re * nn + rf.nodes[perm[a]];
                        [field[0][vr], field[1][vr], field[2][vr]]
                    }
                    Neighbor::Boundary { tag } => {
                        bdry(*tag).unwrap_or([field[0][g], field[1][g], field[2][g]])
                    }
                    _ => [field[0][g], field[1][g], field[2][g]], // NC: no correction
                };
                let fac = face.sw[a] * un / el.geom.jw[vl];
                for comp in 0..3 {
                    out[comp][g] += fac * (field[comp][g] - ext[comp]);
                }
            }
        }
    }
    out
}

/// Oldroyd-B conformation-tensor transport on a DG mesh (velocity prescribed).
pub struct OldroydB<'m> {
    pub mesh: &'m Mesh2d,
    /// Polymer relaxation time `λ`.
    pub lambda: f64,
    /// Polymer viscosity `η_p` (used only for the stress map).
    pub eta_p: f64,
    /// Optional conformation inflow boundary data (the incoming polymer state at an
    /// inlet). `None` ⇒ all boundaries transparent.
    pub inflow: Option<ConformationInflow>,
}

impl<'m> OldroydB<'m> {
    pub fn new(mesh: &'m Mesh2d, lambda: f64, eta_p: f64) -> Self {
        Self { mesh, lambda, eta_p, inflow: None }
    }

    /// Set the conformation inflow boundary data (builder style).
    pub fn with_inflow(mut self, inflow: ConformationInflow) -> Self {
        self.inflow = Some(inflow);
        self
    }

    fn ndof(&self) -> usize {
        self.mesh.n_elements() * self.mesh.refq.n_nodes()
    }

    /// Conformation field initialized to the identity `C = I` (equilibrium).
    pub fn identity(&self) -> [Vec<f64>; 3] {
        let n = self.ndof();
        [vec![1.0; n], vec![0.0; n], vec![1.0; n]]
    }

    /// `∂C/∂t` from advection + upper-convected stretching + relaxation, at every
    /// node, given the (prescribed) velocity field. `c = [Cxx, Cxy, Cyy]`.
    pub fn conformation_rhs(&self, c: &[Vec<f64>; 3], ux: &[f64], uy: &[f64]) -> [Vec<f64>; 3] {
        let refq = &self.mesh.refq;
        let nn = refq.n_nodes();
        let n = self.ndof();
        let inv_lambda = 1.0 / self.lambda;
        let mut out = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
        for (e, el) in self.mesh.elements.iter().enumerate() {
            let sl = e * nn..(e + 1) * nn;
            // Velocity gradient L = ∇u (Lᵢⱼ = ∂uᵢ/∂xⱼ).
            let lxx = el.geom.grad_x(refq, &ux[sl.clone()]);
            let lxy = el.geom.grad_y(refq, &ux[sl.clone()]);
            let lyx = el.geom.grad_x(refq, &uy[sl.clone()]);
            let lyy = el.geom.grad_y(refq, &uy[sl.clone()]);
            // Gradients of each conformation component (for advection).
            let cxx_x = el.geom.grad_x(refq, &c[0][sl.clone()]);
            let cxx_y = el.geom.grad_y(refq, &c[0][sl.clone()]);
            let cxy_x = el.geom.grad_x(refq, &c[1][sl.clone()]);
            let cxy_y = el.geom.grad_y(refq, &c[1][sl.clone()]);
            let cyy_x = el.geom.grad_x(refq, &c[2][sl.clone()]);
            let cyy_y = el.geom.grad_y(refq, &c[2][sl]);
            for k in 0..nn {
                let g = e * nn + k;
                let (u, v) = (ux[g], uy[g]);
                let (cxx, cxy, cyy) = (c[0][g], c[1][g], c[2][g]);
                let (lxx, lxy, lyx, lyy) = (lxx[k], lxy[k], lyx[k], lyy[k]);

                // −(u·∇)C
                let adv_xx = u * cxx_x[k] + v * cxx_y[k];
                let adv_xy = u * cxy_x[k] + v * cxy_y[k];
                let adv_yy = u * cyy_x[k] + v * cyy_y[k];

                // L·C + C·Lᵀ (symmetric).
                let s_xx = 2.0 * (lxx * cxx + lxy * cxy);
                let s_xy = lxx * cxy + lxy * cyy + lyx * cxx + lyy * cxy;
                let s_yy = 2.0 * (lyx * cxy + lyy * cyy);

                // −(1/λ)(C − I).
                let r_xx = -inv_lambda * (cxx - 1.0);
                let r_xy = -inv_lambda * cxy;
                let r_yy = -inv_lambda * (cyy - 1.0);

                out[0][g] = -adv_xx + s_xx + r_xx;
                out[1][g] = -adv_xy + s_xy + r_xy;
                out[2][g] = -adv_yy + s_yy + r_yy;
            }
        }
        // Upwind DG surface lift — couples advection across element faces (the volume
        // term above is element-local) and injects the inflow conformation directly.
        let lift = upwind_advection_lift(self.mesh, c, ux, uy, |tag| {
            self.inflow.as_ref().filter(|i| i.tags.contains(&tag)).map(|i| i.c)
        });
        for comp in 0..3 {
            for g in 0..n {
                out[comp][g] += lift[comp][g];
            }
        }
        out
    }

    /// One SSP-RK3 step of the conformation transport with a fixed velocity field.
    pub fn step_ssp_rk3(&self, c: &[Vec<f64>; 3], ux: &[f64], uy: &[f64], dt: f64) -> [Vec<f64>; 3] {
        let axpy = |a: &[Vec<f64>; 3], k: &[Vec<f64>; 3], s: f64| -> [Vec<f64>; 3] {
            std::array::from_fn(|v| a[v].iter().zip(&k[v]).map(|(x, d)| x + s * d).collect())
        };
        let combine = |a: &[Vec<f64>; 3], wa: f64, b: &[Vec<f64>; 3], wb: f64| -> [Vec<f64>; 3] {
            std::array::from_fn(|v| {
                a[v].iter().zip(&b[v]).map(|(x, y)| wa * x + wb * y).collect()
            })
        };
        let k0 = self.conformation_rhs(c, ux, uy);
        let u1 = axpy(c, &k0, dt);
        let k1 = self.conformation_rhs(&u1, ux, uy);
        let u2a = axpy(&u1, &k1, dt);
        let u2 = combine(c, 0.75, &u2a, 0.25);
        let k2 = self.conformation_rhs(&u2, ux, uy);
        let u3a = axpy(&u2, &k2, dt);
        combine(c, 1.0 / 3.0, &u3a, 2.0 / 3.0)
    }

    /// Polymer stress `τ_p = (η_p/λ)(C − I)`, returned as `[τxx, τxy, τyy]`.
    pub fn polymer_stress(&self, c: &[Vec<f64>; 3]) -> [Vec<f64>; 3] {
        let f = self.eta_p / self.lambda;
        [
            c[0].iter().map(|&x| f * (x - 1.0)).collect(),
            c[1].iter().map(|&x| f * x).collect(),
            c[2].iter().map(|&x| f * (x - 1.0)).collect(),
        ]
    }

    /// Divergence of the polymer stress `∇·τ_p` (nodal collocation), returned as
    /// the momentum body force `(∂ₓτxx + ∂_yτxy, ∂ₓτxy + ∂_yτyy)`.
    pub fn stress_divergence(&self, c: &[Vec<f64>; 3]) -> (Vec<f64>, Vec<f64>) {
        let tau = self.polymer_stress(c);
        let refq = &self.mesh.refq;
        let nn = refq.n_nodes();
        let mut fx = vec![0.0; self.ndof()];
        let mut fy = vec![0.0; self.ndof()];
        for (e, el) in self.mesh.elements.iter().enumerate() {
            let sl = e * nn..(e + 1) * nn;
            let txx_x = el.geom.grad_x(refq, &tau[0][sl.clone()]);
            let txy_x = el.geom.grad_x(refq, &tau[1][sl.clone()]);
            let txy_y = el.geom.grad_y(refq, &tau[1][sl.clone()]);
            let tyy_y = el.geom.grad_y(refq, &tau[2][sl]);
            for k in 0..nn {
                fx[e * nn + k] = txx_x[k] + txy_y[k];
                fy[e * nn + k] = txy_x[k] + tyy_y[k];
            }
        }
        (fx, fy)
    }
}

/// A polymer constitutive model usable by the coupled solver. The state triple is
/// the model's evolved variable (`C` for the direct form, `Ψ = log C` for the
/// log-conformation form); `recover_c` maps it back to the conformation `C` for
/// diagnostics and the stress.
pub trait ConstitutiveModel {
    /// Equilibrium state (`C = I`, i.e. `Ψ = 0`).
    fn equilibrium(&self) -> [Vec<f64>; 3];
    /// Advance the state one step with a fixed velocity field.
    fn advance(&self, state: &[Vec<f64>; 3], ux: &[f64], uy: &[f64], dt: f64) -> [Vec<f64>; 3];
    /// Polymer-stress divergence `∇·τ_p` (the momentum body force).
    fn stress_div(&self, state: &[Vec<f64>; 3]) -> (Vec<f64>, Vec<f64>);
    /// Recover the conformation tensor `C` from the state.
    fn recover_c(&self, state: &[Vec<f64>; 3]) -> [Vec<f64>; 3];
}

impl ConstitutiveModel for OldroydB<'_> {
    fn equilibrium(&self) -> [Vec<f64>; 3] {
        self.identity()
    }
    fn advance(&self, s: &[Vec<f64>; 3], ux: &[f64], uy: &[f64], dt: f64) -> [Vec<f64>; 3] {
        self.step_ssp_rk3(s, ux, uy, dt)
    }
    fn stress_div(&self, s: &[Vec<f64>; 3]) -> (Vec<f64>, Vec<f64>) {
        self.stress_divergence(s)
    }
    fn recover_c(&self, s: &[Vec<f64>; 3]) -> [Vec<f64>; 3] {
        s.clone() // the state IS the conformation
    }
}

impl ConstitutiveModel for LogConfOldroydB<'_> {
    fn equilibrium(&self) -> [Vec<f64>; 3] {
        self.identity()
    }
    fn advance(&self, s: &[Vec<f64>; 3], ux: &[f64], uy: &[f64], dt: f64) -> [Vec<f64>; 3] {
        self.step_ssp_rk3(s, ux, uy, dt)
    }
    fn stress_div(&self, s: &[Vec<f64>; 3]) -> (Vec<f64>, Vec<f64>) {
        self.stress_divergence(s)
    }
    fn recover_c(&self, s: &[Vec<f64>; 3]) -> [Vec<f64>; 3] {
        self.conformation(s)
    }
}

/// Coupled **incompressible viscoelastic** solver, generic over the polymer
/// constitutive model: dual-splitting momentum (solvent viscosity `η_s`) forced by
/// the polymer-stress divergence `∇·τ_p`, with the constitutive state advected/
/// stretched by the resulting velocity. One step is the standard decoupled split:
/// momentum first with the current stress, then the constitutive update with the
/// new velocity. Use [`OldroydB`] for the direct form or [`LogConfOldroydB`] for
/// the high-Wi log-conformation form.
pub struct ViscoelasticFlow<'m, M: ConstitutiveModel> {
    pub stokes: Stokes<'m>,
    pub model: M,
}

impl<'m, M: ConstitutiveModel> ViscoelasticFlow<'m, M> {
    /// Build from an explicit constitutive model. `eta_s` solvent viscosity, `dt`
    /// time step, `alpha` SIPG penalty.
    pub fn with_model(mesh: &'m Mesh2d, eta_s: f64, dt: f64, alpha: f64, model: M) -> Self {
        Self { stokes: Stokes::new(mesh, alpha, eta_s, dt), model }
    }

    /// Advance `(u, state)` by one step under an external body force `(fx, fy)`
    /// (e.g. a constant pressure-gradient drive). Returns `(ux, uy, state)`.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        ux: &[f64],
        uy: &[f64],
        state: &[Vec<f64>; 3],
        t: f64,
        bc_u: impl Fn(f64, f64, f64) -> f64,
        bc_v: impl Fn(f64, f64, f64) -> f64,
        fx: impl Fn(f64, f64, f64) -> f64,
        fy: impl Fn(f64, f64, f64) -> f64,
    ) -> (Vec<f64>, Vec<f64>, [Vec<f64>; 3]) {
        // Total momentum body force = external drive + ∇·τ_p.
        let (mut bx, mut by) = self.model.stress_div(state);
        let mesh = self.stokes.mesh;
        let nn = mesh.refq.n_nodes();
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let (x, y) = (el.geom.x[k], el.geom.y[k]);
                bx[e * nn + k] += fx(x, y, t);
                by[e * nn + k] += fy(x, y, t);
            }
        }
        let (nux, nuy) = self.stokes.step_ns_forced(ux, uy, t, bc_u, bc_v, &bx, &by);
        let nstate = self.model.advance(state, &nux, &nuy, self.stokes.dt);
        (nux, nuy, nstate)
    }
}

impl<'m> ViscoelasticFlow<'m, OldroydB<'m>> {
    /// Convenience constructor using the direct Oldroyd-B form. Total (zero-shear)
    /// viscosity is `η₀ = η_s + η_p`.
    pub fn new(mesh: &'m Mesh2d, eta_s: f64, eta_p: f64, lambda: f64, dt: f64, alpha: f64) -> Self {
        Self::with_model(mesh, eta_s, dt, alpha, OldroydB::new(mesh, lambda, eta_p))
    }
}

/// Symmetric 2×2 eigendecomposition `[[a,b],[b,d]] = R diag(μ₁,μ₂) Rᵀ`,
/// `R = [[c,−s],[s,c]]`. Returns `(μ₁, μ₂, c, s)`.
fn sym_eig(a: f64, b: f64, d: f64) -> (f64, f64, f64, f64) {
    let tr = 0.5 * (a + d);
    let diff = a - d;
    let rad = (0.25 * diff * diff + b * b).sqrt();
    let mu1 = tr + rad;
    let mu2 = tr - rad;
    let theta = 0.5 * (2.0 * b).atan2(diff); // 0.5·atan2(2b, a−d)
    (mu1, mu2, theta.cos(), theta.sin())
}

/// Matrix logarithm of a symmetric-positive-definite 2×2 conformation `[Cxx,Cxy,Cyy]`,
/// returning `Ψ = log C` as `[Ψxx,Ψxy,Ψyy]`. Used to convert a conformation inflow
/// datum into the log-conformation variable for the upwind trace (host-side, including
/// the GPU log-conf path).
pub fn log_conformation(c: [f64; 3]) -> [f64; 3] {
    sym_apply(c[0], c[1], c[2], f64::ln)
}

/// Apply a scalar function to a symmetric 2×2 matrix via its eigendecomposition;
/// returns `[xx, xy, yy]` of `R diag(f(μ₁), f(μ₂)) Rᵀ`.
fn sym_apply(a: f64, b: f64, d: f64, f: impl Fn(f64) -> f64) -> [f64; 3] {
    let (mu1, mu2, c, s) = sym_eig(a, b, d);
    let (d1, d2) = (f(mu1), f(mu2));
    [c * c * d1 + s * s * d2, c * s * (d1 - d2), s * s * d1 + c * c * d2]
}

/// **Log-conformation** Oldroyd-B transport (Fattal–Kupferman): evolves `Ψ = log C`
/// so the recovered conformation `C = exp(Ψ)` is symmetric-positive-definite by
/// construction — removing the high-Weissenberg-number instability of the direct
/// form. State is `[Ψxx, Ψxy, Ψyy]`; equilibrium is `Ψ = 0` (`C = I`).
pub struct LogConfOldroydB<'m> {
    pub mesh: &'m Mesh2d,
    pub lambda: f64,
    pub eta_p: f64,
    /// Optional conformation inflow boundary data (specified as the conformation `C`;
    /// converted to `Ψ = log C` internally). `None` ⇒ all boundaries transparent.
    pub inflow: Option<ConformationInflow>,
}

impl<'m> LogConfOldroydB<'m> {
    pub fn new(mesh: &'m Mesh2d, lambda: f64, eta_p: f64) -> Self {
        Self { mesh, lambda, eta_p, inflow: None }
    }

    /// Set the conformation inflow boundary data (builder style).
    pub fn with_inflow(mut self, inflow: ConformationInflow) -> Self {
        self.inflow = Some(inflow);
        self
    }

    fn ndof(&self) -> usize {
        self.mesh.n_elements() * self.mesh.refq.n_nodes()
    }

    /// Equilibrium log-conformation `Ψ = log I = 0`.
    pub fn identity(&self) -> [Vec<f64>; 3] {
        let n = self.ndof();
        [vec![0.0; n], vec![0.0; n], vec![0.0; n]]
    }

    /// `Ψ = log C` from a conformation field (matrix logarithm, per node).
    pub fn from_conformation(&self, c: &[Vec<f64>; 3]) -> [Vec<f64>; 3] {
        let n = self.ndof();
        let mut psi = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
        for i in 0..n {
            let l = sym_apply(c[0][i], c[1][i], c[2][i], f64::ln);
            psi[0][i] = l[0];
            psi[1][i] = l[1];
            psi[2][i] = l[2];
        }
        psi
    }

    /// Recover the conformation `C = exp(Ψ)` (matrix exponential, per node).
    pub fn conformation(&self, psi: &[Vec<f64>; 3]) -> [Vec<f64>; 3] {
        let n = self.ndof();
        let mut c = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
        for i in 0..n {
            let e = sym_apply(psi[0][i], psi[1][i], psi[2][i], f64::exp);
            c[0][i] = e[0];
            c[1][i] = e[1];
            c[2][i] = e[2];
        }
        c
    }

    /// `∂Ψ/∂t` = −(u·∇)Ψ + (ΩΨ − ΨΩ) + 2B + (1/λ)(e^{−Ψ} − I), at every node.
    pub fn psi_rhs(&self, psi: &[Vec<f64>; 3], ux: &[f64], uy: &[f64]) -> [Vec<f64>; 3] {
        let refq = &self.mesh.refq;
        let nn = refq.n_nodes();
        let n = self.ndof();
        let inv_lambda = 1.0 / self.lambda;
        let mut out = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
        for (e, el) in self.mesh.elements.iter().enumerate() {
            let sl = e * nn..(e + 1) * nn;
            let lxx = el.geom.grad_x(refq, &ux[sl.clone()]);
            let lxy = el.geom.grad_y(refq, &ux[sl.clone()]);
            let lyx = el.geom.grad_x(refq, &uy[sl.clone()]);
            let lyy = el.geom.grad_y(refq, &uy[sl.clone()]);
            let pxx_x = el.geom.grad_x(refq, &psi[0][sl.clone()]);
            let pxx_y = el.geom.grad_y(refq, &psi[0][sl.clone()]);
            let pxy_x = el.geom.grad_x(refq, &psi[1][sl.clone()]);
            let pxy_y = el.geom.grad_y(refq, &psi[1][sl.clone()]);
            let pyy_x = el.geom.grad_x(refq, &psi[2][sl.clone()]);
            let pyy_y = el.geom.grad_y(refq, &psi[2][sl]);
            for k in 0..nn {
                let g = e * nn + k;
                let (u, v) = (ux[g], uy[g]);
                let (p, q, r) = (psi[0][g], psi[1][g], psi[2][g]); // Ψxx, Ψxy, Ψyy

                // Eigenframe of Ψ (shared with C): R = [[c,−s],[s,c]].
                let (mu1, mu2, mut c, mut s) = sym_eig(p, q, r);
                let (l1, l2) = (mu1.exp(), mu2.exp()); // eigenvalues of C
                let (lxx, lxy, lyx, lyy) = (lxx[k], lxy[k], lyx[k], lyy[k]);
                // Near the isotropic point (C ∝ I) the eigenframe is indeterminate and
                // the rotation rate ω is singular. Align the frame with the rate-of-
                // strain instead, so 2B → L+Lᵀ (the correct Ψ̇ ≈ 2D limit); ω → 0.
                if (mu1 - mu2).abs() < 1e-7 {
                    let (_, _, cc, ss) = sym_eig(lxx, 0.5 * (lxy + lyx), lyy);
                    c = cc;
                    s = ss;
                }

                // Velocity gradient in the eigenframe, M = RᵀLR.
                let a1 = c * lxx + s * lyx;
                let a2 = c * lxy + s * lyy;
                let b1 = -s * lxx + c * lyx;
                let b2 = -s * lxy + c * lyy;
                let m11 = a1 * c + a2 * s;
                let m12 = -a1 * s + a2 * c;
                let m21 = b1 * c + b2 * s;
                let m22 = -b1 * s + b2 * c;

                // 2B (B = R diag(m11,m22) Rᵀ).
                let bxx = c * c * m11 + s * s * m22;
                let bxy = c * s * (m11 - m22);
                let byy = s * s * m11 + c * c * m22;

                // Rotation rate ω (rotation-invariant in 2D ⇒ Ω = [[0,ω],[−ω,0]]).
                let denom = l2 - l1;
                let omega = if denom.abs() > 1e-12 {
                    (m12 * l2 + m21 * l1) / denom
                } else {
                    0.0
                };
                // ΩΨ − ΨΩ = [[2ωq, ω(r−p)],[·, −2ωq]].
                let rot_xx = 2.0 * omega * q;
                let rot_xy = omega * (r - p);
                let rot_yy = -2.0 * omega * q;

                // (1/λ)(e^{−Ψ} − I).
                let em = sym_apply(p, q, r, |x| (-x).exp());
                let relax_xx = inv_lambda * (em[0] - 1.0);
                let relax_xy = inv_lambda * em[1];
                let relax_yy = inv_lambda * (em[2] - 1.0);

                // −(u·∇)Ψ.
                let adv_xx = u * pxx_x[k] + v * pxx_y[k];
                let adv_xy = u * pxy_x[k] + v * pxy_y[k];
                let adv_yy = u * pyy_x[k] + v * pyy_y[k];

                out[0][g] = -adv_xx + rot_xx + 2.0 * bxx + relax_xx;
                out[1][g] = -adv_xy + rot_xy + 2.0 * bxy + relax_xy;
                out[2][g] = -adv_yy + rot_yy + 2.0 * byy + relax_yy;
            }
        }
        // Upwind DG surface lift for the −(u·∇)Ψ transport. Inflow is specified as the
        // conformation C; the upwind trace is in the Ψ variable, so convert Ψ = log C.
        let lift = upwind_advection_lift(self.mesh, psi, ux, uy, |tag| {
            self.inflow
                .as_ref()
                .filter(|i| i.tags.contains(&tag))
                .map(|i| sym_apply(i.c[0], i.c[1], i.c[2], f64::ln))
        });
        for comp in 0..3 {
            for g in 0..n {
                out[comp][g] += lift[comp][g];
            }
        }
        out
    }

    /// One SSP-RK3 step of the log-conformation transport with a fixed velocity.
    pub fn step_ssp_rk3(&self, psi: &[Vec<f64>; 3], ux: &[f64], uy: &[f64], dt: f64) -> [Vec<f64>; 3] {
        let axpy = |a: &[Vec<f64>; 3], k: &[Vec<f64>; 3], sc: f64| -> [Vec<f64>; 3] {
            std::array::from_fn(|v| a[v].iter().zip(&k[v]).map(|(x, d)| x + sc * d).collect())
        };
        let combine = |a: &[Vec<f64>; 3], wa: f64, b: &[Vec<f64>; 3], wb: f64| -> [Vec<f64>; 3] {
            std::array::from_fn(|v| a[v].iter().zip(&b[v]).map(|(x, y)| wa * x + wb * y).collect())
        };
        let k0 = self.psi_rhs(psi, ux, uy);
        let u1 = axpy(psi, &k0, dt);
        let k1 = self.psi_rhs(&u1, ux, uy);
        let u2 = combine(psi, 0.75, &axpy(&u1, &k1, dt), 0.25);
        let k2 = self.psi_rhs(&u2, ux, uy);
        combine(psi, 1.0 / 3.0, &axpy(&u2, &k2, dt), 2.0 / 3.0)
    }

    /// Polymer stress `τ_p = (η_p/λ)(C − I)` from `Ψ`, as `[τxx, τxy, τyy]`.
    pub fn polymer_stress(&self, psi: &[Vec<f64>; 3]) -> [Vec<f64>; 3] {
        let c = self.conformation(psi);
        let f = self.eta_p / self.lambda;
        [
            c[0].iter().map(|&x| f * (x - 1.0)).collect(),
            c[1].iter().map(|&x| f * x).collect(),
            c[2].iter().map(|&x| f * (x - 1.0)).collect(),
        ]
    }

    /// Divergence of the polymer stress `∇·τ_p` from `Ψ` (nodal collocation).
    pub fn stress_divergence(&self, psi: &[Vec<f64>; 3]) -> (Vec<f64>, Vec<f64>) {
        let tau = self.polymer_stress(psi);
        let refq = &self.mesh.refq;
        let nn = refq.n_nodes();
        let mut fx = vec![0.0; self.ndof()];
        let mut fy = vec![0.0; self.ndof()];
        for (e, el) in self.mesh.elements.iter().enumerate() {
            let sl = e * nn..(e + 1) * nn;
            let txx_x = el.geom.grad_x(refq, &tau[0][sl.clone()]);
            let txy_x = el.geom.grad_x(refq, &tau[1][sl.clone()]);
            let txy_y = el.geom.grad_y(refq, &tau[1][sl.clone()]);
            let tyy_y = el.geom.grad_y(refq, &tau[2][sl]);
            for k in 0..nn {
                fx[e * nn + k] = txx_x[k] + txy_y[k];
                fy[e * nn + k] = txy_x[k] + tyy_y[k];
            }
        }
        (fx, fy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nodal(mesh: &Mesh2d, f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
        let nn = mesh.refq.n_nodes();
        let mut v = vec![0.0; mesh.n_elements() * nn];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                v[e * nn + k] = f(el.geom.x[k], el.geom.y[k]);
            }
        }
        v
    }

    #[test]
    fn log_conf_exp_log_roundtrip() {
        // exp(log C) = C for an SPD field (the SPD-by-construction guarantee).
        let p = 3;
        let mesh = Mesh2d::rectangular(p, 2, 2, [0.0, 1.0], [0.0, 1.0]);
        let lc = LogConfOldroydB::new(&mesh, 1.0, 1.0);
        // A spatially varying SPD conformation.
        let c0 = [
            nodal(&mesh, |x, y| 2.0 + x + 0.5 * y),
            nodal(&mesh, |x, y| 0.3 * x - 0.2 * y),
            nodal(&mesh, |x, y| 1.5 + 0.4 * x * y),
        ];
        let psi = lc.from_conformation(&c0);
        let c = lc.conformation(&psi);
        for v in 0..3 {
            let err = c[v].iter().zip(&c0[v]).fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
            assert!(err < 1e-12, "roundtrip comp {v}: {err}");
        }
    }

    #[test]
    fn log_conf_high_wi_steady_shear() {
        // The decisive log-conformation test: Wi = 10 simple shear. The direct C-form
        // is fragile here (C can lose positive-definiteness); the log form must reach
        // the analytic steady state Cxx=1+2Wi²=201, Cxy=Wi=10, Cyy=1 — and C=exp(Ψ)
        // stays SPD throughout.
        let p = 4;
        let mesh = Mesh2d::rectangular(p, 3, 3, [0.0, 1.0], [0.0, 1.0]);
        let lambda = 1.0;
        let gdot = 10.0;
        let wi = lambda * gdot;
        let lc = LogConfOldroydB::new(&mesh, lambda, 1.5);
        let ux = nodal(&mesh, |_, y| gdot * y);
        let uy = vec![0.0; mesh.n_elements() * mesh.refq.n_nodes()];
        let mut psi = lc.identity();
        let dt = 0.002;
        let nsteps = (25.0_f64 / dt).round() as usize;
        for _ in 0..nsteps {
            psi = lc.step_ssp_rk3(&psi, &ux, &uy, dt);
            // C stays SPD: det = CxxCyy − Cxy² > 0 (guaranteed via exp).
        }
        let c = lc.conformation(&psi);
        // SPD check on the recovered conformation.
        for i in 0..c[0].len() {
            let det = c[0][i] * c[2][i] - c[1][i] * c[1][i];
            assert!(c[0][i] > 0.0 && det > 0.0, "C not SPD at {i}: det={det}");
        }
        let exact = [1.0 + 2.0 * wi * wi, wi, 1.0];
        for (v, &ex) in exact.iter().enumerate() {
            let err = c[v].iter().fold(0.0f64, |a, &x| a.max((x - ex).abs()));
            assert!(err < 1e-2 * ex.max(1.0), "comp {v}: max err {err}, want {ex}");
        }
    }

    #[test]
    fn relaxation_decays_to_identity_exponentially() {
        // No flow ⇒ pure relaxation: C(t) = I + (C₀ − I) e^{−t/λ}.
        let p = 3;
        let mesh = Mesh2d::rectangular(p, 2, 2, [0.0, 1.0], [0.0, 1.0]);
        let lambda = 0.7;
        let ob = OldroydB::new(&mesh, lambda, 1.0);
        let zero = vec![0.0; ob.ndof()];
        // Start stretched: Cxx=3, Cxy=0.5, Cyy=2.
        let mut c = [vec![3.0; ob.ndof()], vec![0.5; ob.ndof()], vec![2.0; ob.ndof()]];
        let dt = 0.001;
        let t_end = 1.0_f64;
        let nsteps = (t_end / dt).round() as usize;
        for _ in 0..nsteps {
            c = ob.step_ssp_rk3(&c, &zero, &zero, dt);
        }
        let decay = (-t_end / lambda).exp();
        let exact = [1.0 + 2.0 * decay, 0.5 * decay, 1.0 + 1.0 * decay];
        for (v, &ex) in exact.iter().enumerate() {
            let err = c[v].iter().fold(0.0f64, |a, &x| a.max((x - ex).abs()));
            assert!(err < 1e-7, "comp {v}: max err {err}, want {ex}");
        }
    }

    #[test]
    fn stress_divergence_exact_on_polynomials() {
        // τ from C = [Cxx=x², Cxy=xy, Cyy=y²] ⇒ with f=η_p/λ, τ=f(C−I) and
        // ∇·τ = f(∂ₓCxx+∂_yCxy, ∂ₓCxy+∂_yCyy) = f(2x+x, y+2y) = f(3x, 3y).
        let p = 4;
        let mesh = Mesh2d::rectangular(p, 2, 2, [0.0, 1.0], [0.0, 1.0]);
        let (lambda, eta_p) = (0.5, 1.3);
        let ob = OldroydB::new(&mesh, lambda, eta_p);
        let f = eta_p / lambda;
        let c = [
            nodal(&mesh, |x, _| x * x),
            nodal(&mesh, |x, y| x * y),
            nodal(&mesh, |_, y| y * y),
        ];
        let (dx, dy) = ob.stress_divergence(&c);
        let nn = mesh.refq.n_nodes();
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let (x, y) = (el.geom.x[k], el.geom.y[k]);
                assert!((dx[e * nn + k] - f * 3.0 * x).abs() < 1e-9, "∇·τ x");
                assert!((dy[e * nn + k] - f * 3.0 * y).abs() < 1e-9, "∇·τ y");
            }
        }
    }

    #[test]
    fn coupled_channel_recovers_total_viscosity() {
        // Body-force-driven planar channel of an Oldroyd-B fluid. The polymer enters
        // momentum ONLY through ∇·τ_p, so the steady velocity must be parabolic with
        // the TOTAL viscosity η₀ = η_s + η_p:  U(y) = (G/2η₀) y(1−y).  Recovering η₀
        // (not just η_s) is the proof that the coupling is wired correctly.
        let p = 4;
        let mesh = Mesh2d::rectangular(p, 3, 3, [0.0, 1.0], [0.0, 1.0]);
        let (eta_s, eta_p, lambda, g) = (0.5, 0.5, 0.5, 1.0);
        let eta0 = eta_s + eta_p;
        let dt = 0.02;
        let ve = ViscoelasticFlow::new(&mesh, eta_s, eta_p, lambda, dt, 5.0);
        let u_exact = |y: f64| (g / (2.0 * eta0)) * y * (1.0 - y);
        // Dirichlet velocity = the fully-developed profile on every boundary (0 on walls).
        let bc_u = move |_x: f64, y: f64, _t: f64| u_exact(y);
        let bc_v = |_: f64, _: f64, _: f64| 0.0;
        let drive_x = move |_: f64, _: f64, _: f64| g;
        let zero_f = |_: f64, _: f64, _: f64| 0.0;

        let mut ux = vec![0.0; ve.stokes.mesh.n_elements() * mesh.refq.n_nodes()];
        let mut uy = ux.clone();
        let mut c = ve.model.equilibrium();
        let mut t = 0.0;
        for _ in 0..600 {
            t += dt;
            let (nx, ny, nc) = ve.step(&ux, &uy, &c, t, bc_u, bc_v, drive_x, zero_f);
            ux = nx;
            uy = ny;
            c = nc;
        }

        // Velocity matches the total-viscosity parabola.
        let nn = mesh.refq.n_nodes();
        let mut uerr = 0.0f64;
        let mut n1err = 0.0f64; // interior column only (away from inlet/outlet corners)
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let g_ = e * nn + k;
                let (x, y) = (el.geom.x[k], el.geom.y[k]);
                uerr = uerr.max((ux[g_] - u_exact(y)).abs());
                // First normal-stress difference N₁ = τxx − τyy = 2 η_p λ γ̇², γ̇=U'(y).
                // Checked in the fully-developed interior; the inlet/outlet–wall corners
                // carry a stress singularity that this Dirichlet box can't represent.
                if x > 0.34 && x < 0.66 {
                    let gdot = (g / (2.0 * eta0)) * (1.0 - 2.0 * y);
                    let n1 = (eta_p / lambda) * (c[0][g_] - c[2][g_]);
                    n1err = n1err.max((n1 - 2.0 * eta_p * lambda * gdot * gdot).abs());
                }
            }
        }
        eprintln!("channel: max|u−U|={uerr:.3e} (Umax={:.4}), interior max N₁ err={n1err:.3e}", u_exact(0.5));
        assert!(uerr < 5e-3, "velocity not the η₀ parabola: {uerr}");
        assert!(n1err < 5e-3, "interior first normal-stress difference wrong: {n1err}");
    }

    #[test]
    fn coupled_channel_log_conformation() {
        // Same coupled channel through the GENERIC solver with the log-conformation
        // model: must still recover the total-viscosity parabola, and C = exp(Ψ) must
        // stay SPD. Validates that ViscoelasticFlow works with LogConfOldroydB.
        let p = 4;
        let mesh = Mesh2d::rectangular(p, 2, 2, [0.0, 1.0], [0.0, 1.0]);
        // Low-Wi params (Wi ≤ 0.25), matching the validated direct-form channel.
        let (eta_s, eta_p, lambda, g) = (0.5, 0.5, 0.5, 1.0);
        let eta0 = eta_s + eta_p;
        let dt = 0.02;
        let model = LogConfOldroydB::new(&mesh, lambda, eta_p);
        let ve = ViscoelasticFlow::with_model(&mesh, eta_s, dt, 5.0, model);
        let u_exact = |y: f64| (g / (2.0 * eta0)) * y * (1.0 - y);
        let bc_u = move |_x: f64, y: f64, _t: f64| u_exact(y);
        let bc_v = |_: f64, _: f64, _: f64| 0.0;
        let drive_x = move |_: f64, _: f64, _: f64| g;
        let zero_f = |_: f64, _: f64, _: f64| 0.0;

        let mut ux = vec![0.0; mesh.n_elements() * mesh.refq.n_nodes()];
        let mut uy = ux.clone();
        let mut psi = ve.model.equilibrium();
        let mut t = 0.0;
        for step in 0..400 {
            t += dt;
            let (nx, ny, np) = ve.step(&ux, &uy, &psi, t, bc_u, bc_v, drive_x, zero_f);
            ux = nx;
            uy = ny;
            psi = np;
            if step % 80 == 0 {
                let umax = ux.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
                let pmax = psi[0].iter().fold(0.0f64, |a, &v| a.max(v.abs()));
                eprintln!("  step {step}: max|u|={umax:.4e} max|Ψxx|={pmax:.4e}");
                assert!(umax.is_finite() && pmax.is_finite(), "diverged at step {step}");
            }
        }
        // C = exp(Ψ) is SPD everywhere.
        let c = ve.model.recover_c(&psi);
        for i in 0..c[0].len() {
            let det = c[0][i] * c[2][i] - c[1][i] * c[1][i];
            assert!(c[0][i] > 0.0 && det > 0.0, "C not SPD at {i}: det={det}");
        }
        let nn = mesh.refq.n_nodes();
        let mut uerr = 0.0f64;
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                uerr = uerr.max((ux[e * nn + k] - u_exact(el.geom.y[k])).abs());
            }
        }
        eprintln!("log-conf channel: max|u−U|={uerr:.3e} (Umax={:.4})", u_exact(0.5));
        assert!(uerr < 5e-3, "log-conf coupling: velocity not the η₀ parabola: {uerr}");
    }

    #[test]
    fn upwind_advection_transports_across_elements() {
        // Pure x-advection of a smooth conformation bump by uniform flow u=(1,0) on a
        // PERIODIC mesh: ∇u=0 (no stretching), λ huge (negligible relaxation). The exact
        // solution Cxx(x,t) = 1 + ½sin(2π(x−t)) advects across element faces and around
        // the periodic seam — which the element-local collocation term ALONE cannot do.
        // This is the test that the upwind surface lift actually transports across faces.
        use std::f64::consts::PI;
        let p = 4;
        let mesh = Mesh2d::rectangular_periodic(p, 6, 1, [0.0, 1.0], [0.0, 0.25]);
        let ob = OldroydB::new(&mesh, 1e6, 1.0);
        let nn = mesh.refq.n_nodes();
        let u = vec![1.0; mesh.n_elements() * nn];
        let v = vec![0.0; mesh.n_elements() * nn];
        let c0 = nodal(&mesh, |x, _| 1.0 + 0.5 * (2.0 * PI * x).sin());
        let mut c = [c0.clone(), vec![0.0; c0.len()], vec![1.0; c0.len()]];
        let dt = 1e-3_f64;
        let t_end = 0.5_f64;
        for _ in 0..(t_end / dt).round() as usize {
            c = ob.step_ssp_rk3(&c, &u, &v, dt);
        }
        let exact = nodal(&mesh, |x, _| 1.0 + 0.5 * (2.0 * PI * (x - t_end)).sin());
        let err = c[0].iter().zip(&exact).fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
        eprintln!("periodic conformation advection: max|Cxx − exact| = {err:.3e}");
        assert!(err < 5e-3, "upwind advection not transporting across elements: {err}");
    }

    #[test]
    fn conformation_inflow_fills_domain() {
        // Stretched fluid C_in = [2,0,1] enters at the west inlet (tag 3) under uniform
        // flow u=(1,0); no stretching (∇u=0), negligible relaxation (λ huge). The inflow
        // datum must be carried downstream across every element to fill the domain —
        // steady state C = C_in everywhere. Without the inflow flux the interior stays I.
        let p = 4;
        let mesh = Mesh2d::rectangular(p, 6, 2, [0.0, 3.0], [0.0, 1.0]);
        let c_in = [2.0, 0.0, 1.0];
        let ob = OldroydB::new(&mesh, 1e6, 1.0).with_inflow(ConformationInflow::new(vec![3], c_in));
        let nn = mesh.refq.n_nodes();
        let u = vec![1.0; mesh.n_elements() * nn];
        let v = vec![0.0; mesh.n_elements() * nn];
        let mut c = ob.identity(); // start relaxed, C = I
        let dt = 2e-3_f64;
        for _ in 0..(12.0 / dt).round() as usize {
            c = ob.step_ssp_rk3(&c, &u, &v, dt); // ~4 flow-throughs (L=3, U=1)
        }
        // Downstream of the inlet element, the domain is filled with C_in.
        let mut err = 0.0f64;
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                if el.geom.x[k] > 0.6 {
                    let g = e * nn + k;
                    for comp in 0..3 {
                        err = err.max((c[comp][g] - c_in[comp]).abs());
                    }
                }
            }
        }
        eprintln!("conformation inflow fill: downstream max|C − C_in| = {err:.3e}");
        assert!(err < 5e-3, "inflow conformation not carried downstream: {err}");
    }

    #[test]
    fn steady_simple_shear_matches_analytic() {
        // Prescribed homogeneous shear u = (γ̇ y, 0). Oldroyd-B steady state:
        //   Cxx = 1 + 2 Wi², Cxy = Wi, Cyy = 1,  Wi = λγ̇.
        let p = 4;
        let mesh = Mesh2d::rectangular(p, 3, 3, [0.0, 1.0], [0.0, 1.0]);
        let lambda = 1.0;
        let gdot = 2.0;
        let wi = lambda * gdot;
        let ob = OldroydB::new(&mesh, lambda, 1.5);
        let ux = nodal(&mesh, |_, y| gdot * y);
        let uy = vec![0.0; ob.ndof()];
        let mut c = ob.identity();
        let dt = 0.002;
        let t_end = 30.0_f64; // ≫ λ ⇒ fully relaxed
        let nsteps = (t_end / dt).round() as usize;
        for _ in 0..nsteps {
            c = ob.step_ssp_rk3(&c, &ux, &uy, dt);
        }
        let exact = [1.0 + 2.0 * wi * wi, wi, 1.0];
        for (v, &ex) in exact.iter().enumerate() {
            let err = c[v].iter().fold(0.0f64, |a, &x| a.max((x - ex).abs()));
            assert!(err < 1e-4, "comp {v}: max err {err}, want {ex}");
        }
        // Spot-check the polymer stress map: τxy = η_p γ̇ (Oldroyd-B shear stress).
        let tau = ob.polymer_stress(&c);
        let txy = tau[1][0];
        assert!((txy - ob.eta_p * gdot).abs() < 1e-4, "τxy={txy}, want {}", ob.eta_p * gdot);
    }
}
