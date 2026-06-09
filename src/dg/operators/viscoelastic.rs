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

    /// SSP-RK3 step with the **bound-preserving limiter** applied after each stage (Zhang–Shu
    /// stage limiting via [`limit_conformation_bounds`]). Keeps the direct-form conformation
    /// SPD (`det C ≥ ε`) and `tr C ≤ b_max` through high-order transport — the HWNP fix that
    /// the plain [`Self::step_ssp_rk3`] lacks. Otherwise identical (same Shu–Osher stages).
    pub fn step_ssp_rk3_bounded(
        &self,
        c: &[Vec<f64>; 3],
        ux: &[f64],
        uy: &[f64],
        dt: f64,
        eps: f64,
        b_max: f64,
    ) -> [Vec<f64>; 3] {
        let axpy = |a: &[Vec<f64>; 3], k: &[Vec<f64>; 3], s: f64| -> [Vec<f64>; 3] {
            std::array::from_fn(|v| a[v].iter().zip(&k[v]).map(|(x, d)| x + s * d).collect())
        };
        let combine = |a: &[Vec<f64>; 3], wa: f64, b: &[Vec<f64>; 3], wb: f64| -> [Vec<f64>; 3] {
            std::array::from_fn(|v| a[v].iter().zip(&b[v]).map(|(x, y)| wa * x + wb * y).collect())
        };
        let limit = |mut s: [Vec<f64>; 3]| -> [Vec<f64>; 3] {
            limit_conformation_bounds(self.mesh, &mut s, eps, b_max);
            s
        };
        let k0 = self.conformation_rhs(c, ux, uy);
        let u1 = limit(axpy(c, &k0, dt));
        let k1 = self.conformation_rhs(&u1, ux, uy);
        let u2 = limit(combine(c, 0.75, &axpy(&u1, &k1, dt), 0.25));
        let k2 = self.conformation_rhs(&u2, ux, uy);
        limit(combine(c, 1.0 / 3.0, &axpy(&u2, &k2, dt), 2.0 / 3.0))
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
    /// **Giesekus mobility** `α ∈ [0, 1]`. `α = 0` is Oldroyd-B; `α > 0` adds the quadratic
    /// relaxation `−(α/λ)(C − I)²`, giving shear-thinning and a bounded steady extension.
    /// Only the relaxation differs from Oldroyd-B (advection + upper-convected stretching are
    /// identical), so in the IMEX split it enters solely through the implicit relaxation
    /// physics ([`Self::relax_exact`] / [`Self::implicit_relax_solve`]).
    pub mobility: f64,
    /// **FENE-P extensibility** `b > 2` (`tr I = 2` in 2D). `b = ∞` (the default) disables the
    /// finite-extensibility limit (Oldroyd-B / Giesekus). When finite, the relaxation is the
    /// Peterlin form `−(1/λ)[f·C − I]`, `f = (1−2/b)/(1−tr C/b)`, whose barrier `f→∞ as tr C→b`
    /// keeps `tr C < b` — a **bound-preserving** implicit solve. Supported by the ARK stepper
    /// ([`Self::implicit_relax_solve`]); the Strang stepper's `relax_exact` is Oldroyd-B/Giesekus
    /// only (FENE-P's trace coupling has no per-eigenvalue closed form).
    pub extensibility: f64,
    /// Optional conformation inflow boundary data (specified as the conformation `C`;
    /// converted to `Ψ = log C` internally). `None` ⇒ all boundaries transparent.
    pub inflow: Option<ConformationInflow>,
}

impl<'m> LogConfOldroydB<'m> {
    pub fn new(mesh: &'m Mesh2d, lambda: f64, eta_p: f64) -> Self {
        Self { mesh, lambda, eta_p, mobility: 0.0, extensibility: f64::INFINITY, inflow: None }
    }

    /// Set the conformation inflow boundary data (builder style).
    pub fn with_inflow(mut self, inflow: ConformationInflow) -> Self {
        self.inflow = Some(inflow);
        self
    }

    /// Set the Giesekus mobility `α` (builder style). `α = 0` ⇒ Oldroyd-B (the default).
    pub fn with_mobility(mut self, alpha: f64) -> Self {
        self.mobility = alpha;
        self
    }

    /// Set the FENE-P extensibility `b` (builder style). `b = ∞` (default) ⇒ no finite limit.
    pub fn with_extensibility(mut self, b: f64) -> Self {
        self.extensibility = b;
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

    // ─── IMEX relaxation substep (Phase 1 of docs/plan-imex-relaxation-substep.md) ───
    //
    // The conformation RHS splits additively into a non-stiff transport part `E` and a
    // stiff relaxation part `S`:  ∂Ψ/∂t = E(Ψ,u) + S(Ψ),  with
    //   E = −(u·∇)Ψ + (ΩΨ−ΨΩ) + 2B        (advection + rotation + stretching, explicit)
    //   S = (1/λ)(e^{−Ψ} − I)              (relaxation, stiff as λ→0, treated implicitly)
    // Because `S` is local, pointwise, and isotropic in `C` (it commutes with `C`), the
    // implicit step decouples onto the conformation eigenvalues — and for Oldroyd-B the
    // eigenvalue ODE `dc_i/dt = −(c_i−1)/λ` has the *exact* closed-form flow used below.

    /// The stiff relaxation source `S(Ψ) = (1/λ)(e^{−Ψ} − I)` — the implicit operand of
    /// the IMEX split, identical to the relaxation contribution inside [`Self::psi_rhs`].
    pub fn relax_source(&self, psi: &[Vec<f64>; 3]) -> [Vec<f64>; 3] {
        let inv_lambda = 1.0 / self.lambda;
        let n = self.ndof();
        let mut out = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
        for i in 0..n {
            let em = sym_apply(psi[0][i], psi[1][i], psi[2][i], |x| (-x).exp());
            out[0][i] = inv_lambda * (em[0] - 1.0);
            out[1][i] = inv_lambda * em[1];
            out[2][i] = inv_lambda * (em[2] - 1.0);
        }
        out
    }

    /// Non-stiff transport part of the RHS (advection + rotation + stretching, *without*
    /// relaxation) — the explicit operand of the IMEX/Strang split. Computed as
    /// `psi_rhs − relax_source`, so it tracks [`Self::psi_rhs`] exactly by construction.
    pub fn psi_transport_rhs(&self, psi: &[Vec<f64>; 3], ux: &[f64], uy: &[f64]) -> [Vec<f64>; 3] {
        let full = self.psi_rhs(psi, ux, uy);
        let relax = self.relax_source(psi);
        std::array::from_fn(|v| full[v].iter().zip(&relax[v]).map(|(f, r)| f - r).collect())
    }

    /// Stiff relaxation substep integrated **exactly** on the conformation eigenvalues.
    /// The eigenvalue ODE `dc/dt = −(1/λ)[(c−1) + α(c−1)²]` (Oldroyd-B for `α = 0`, Giesekus
    /// for `α > 0`) is a Bernoulli equation with the closed-form flow
    /// `w(τ)/(1+αw(τ)) = w₀/(1+αw₀)·e^{−τ/λ}`, `w = c−1` ⇒ `c(τ) = 1 + R/(1−αR)`,
    /// `R = w₀/(1+αw₀)·e^{−τ/λ}`; in log-conformation `ψ ↦ log c(τ)` in the shared eigenframe.
    /// SPD-preserving and unconditionally stable for any `τ`. (`α = 0` recovers
    /// `c = 1 + (e^ψ−1)e^{−τ/λ}`.)
    pub fn relax_exact(&self, psi: &[Vec<f64>; 3], tau: f64) -> [Vec<f64>; 3] {
        let decay = (-tau / self.lambda).exp();
        let alpha = self.mobility;
        let n = self.ndof();
        let mut out = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
        for i in 0..n {
            let m = sym_apply(psi[0][i], psi[1][i], psi[2][i], |mu| {
                let w0 = mu.exp() - 1.0; // c₀ − 1
                let rr = (w0 / (1.0 + alpha * w0)) * decay;
                (1.0 + rr / (1.0 - alpha * rr)).ln()
            });
            out[0][i] = m[0];
            out[1][i] = m[1];
            out[2][i] = m[2];
        }
        out
    }

    /// SSP-RK3 on the transport-only RHS (the explicit half-steps of the Strang split).
    fn step_transport_ssp_rk3(
        &self,
        psi: &[Vec<f64>; 3],
        ux: &[f64],
        uy: &[f64],
        dt: f64,
    ) -> [Vec<f64>; 3] {
        let axpy = |a: &[Vec<f64>; 3], k: &[Vec<f64>; 3], sc: f64| -> [Vec<f64>; 3] {
            std::array::from_fn(|v| a[v].iter().zip(&k[v]).map(|(x, d)| x + sc * d).collect())
        };
        let combine = |a: &[Vec<f64>; 3], wa: f64, b: &[Vec<f64>; 3], wb: f64| -> [Vec<f64>; 3] {
            std::array::from_fn(|v| a[v].iter().zip(&b[v]).map(|(x, y)| wa * x + wb * y).collect())
        };
        let k0 = self.psi_transport_rhs(psi, ux, uy);
        let u1 = axpy(psi, &k0, dt);
        let k1 = self.psi_transport_rhs(&u1, ux, uy);
        let u2 = combine(psi, 0.75, &axpy(&u1, &k1, dt), 0.25);
        let k2 = self.psi_transport_rhs(&u2, ux, uy);
        combine(psi, 1.0 / 3.0, &axpy(&u2, &k2, dt), 2.0 / 3.0)
    }

    /// One **Strang-split IMEX** step: explicit transport over `dt/2`, *exact* relaxation
    /// over `dt`, explicit transport over `dt/2`. Second-order; because the stiff
    /// relaxation is integrated exactly (no `dt ≲ λ` limit) this is stable at `dt ≫ λ`,
    /// where the fully-explicit [`Self::step_ssp_rk3`] diverges. Targeted at the small-λ /
    /// high-elastic-modulus regime (see the plan's §1 scope note).
    pub fn step_strang_imex(
        &self,
        psi: &[Vec<f64>; 3],
        ux: &[f64],
        uy: &[f64],
        dt: f64,
    ) -> [Vec<f64>; 3] {
        let half = self.step_transport_ssp_rk3(psi, ux, uy, 0.5 * dt);
        let relaxed = self.relax_exact(&half, dt);
        self.step_transport_ssp_rk3(&relaxed, ux, uy, 0.5 * dt)
    }

    /// Solve one stiff implicit stage `Ψ − γ·S(Ψ) = B` for `Ψ`, where `S` is the (Oldroyd-B
    /// or Giesekus) relaxation. The source is isotropic in `C` (commutes with it), so the
    /// matrix solve decouples onto the eigenvalues of `B`: per eigenvalue `b`, Newton-solve
    /// `g(ψ) = ψ + (γ/λ)(1−e^{−ψ}) + (γα/λ)(e^ψ−1)²e^{−ψ} − b = 0`. The derivative
    /// `g′ = 1 + (γ/λ)e^{−ψ}[1 + α(e^{2ψ}−1)] > 0` for `α ∈ [0,1]`, so it is strictly monotone
    /// and converges globally from `ψ = b`. SPD-preserving (real `Ψ` ⇒ `C = expΨ` SPD).
    /// `α = 0` recovers the Oldroyd-B scalar `ψ − (γ/λ)(e^{−ψ}−1) = b`. (Plan §3/§7.)
    pub fn implicit_relax_solve(&self, b: &[Vec<f64>; 3], gamma: f64) -> [Vec<f64>; 3] {
        if self.extensibility.is_finite() {
            return self.implicit_relax_solve_fenep(b, gamma);
        }
        let gl = gamma / self.lambda; // γ/λ
        let alpha = self.mobility;
        let n = self.ndof();
        let mut out = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
        for i in 0..n {
            let m = sym_apply(b[0][i], b[1][i], b[2][i], |bi| {
                let mut psi = bi; // initial guess
                for _ in 0..60 {
                    let em = (-psi).exp(); // e^{−ψ}
                    let ep = psi.exp(); // e^{ψ}
                    let w = ep - 1.0;
                    let g = psi + gl * (1.0 - em) + gl * alpha * w * w * em - bi;
                    let gp = 1.0 + gl * em * (1.0 + alpha * (ep * ep - 1.0));
                    let step = g / gp;
                    psi -= step;
                    if step.abs() < 1e-14 {
                        break;
                    }
                }
                psi
            });
            out[0][i] = m[0];
            out[1][i] = m[1];
            out[2][i] = m[2];
        }
        out
    }

    /// FENE-P implicit relaxation solve. The Peterlin relaxation `−(1/λ)[f(T)C − I]`,
    /// `f = (1−2/b)/(1−T/b)`, couples the eigenvalues only through the scalar trace `T = tr C`.
    /// Given `T`, each eigenvalue solves the Oldroyd-B-type scalar `ψ − (γ/λ)e^{−ψ} = b_i − (γ/λ)f(T)`
    /// (monotone Newton); consistency `Σe^{ψ_i} = T` is a **monotone 1-D root** on `T ∈ (0, b)`,
    /// solved by bisection — which keeps `tr C < b` by construction (bound-preserving). (Plan §7.)
    fn implicit_relax_solve_fenep(&self, b: &[Vec<f64>; 3], gamma: f64) -> [Vec<f64>; 3] {
        let gl = gamma / self.lambda;
        let be = self.extensibility;
        let n = self.ndof();
        let mut out = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
        // Inner: solve ψ − (γ/λ)e^{−ψ} = μ − (γ/λ)f for ψ (monotone, globally convergent).
        let inner = |ff: f64, mu: f64| -> f64 {
            let beta = mu - gl * ff;
            let mut psi = mu;
            for _ in 0..50 {
                let e = (-psi).exp();
                let step = (psi - gl * e - beta) / (1.0 + gl * e);
                psi -= step;
                if step.abs() < 1e-14 {
                    break;
                }
            }
            psi
        };
        for i in 0..n {
            let (mu1, mu2, c, s) = sym_eig(b[0][i], b[1][i], b[2][i]);
            // Bisection on the trace T: G(T) = e^{ψ1(T)} + e^{ψ2(T)} − T is strictly decreasing,
            // G(0⁺) > 0, G(b⁻) < 0 ⇒ a unique root in (0, b).
            let (mut lo, mut hi) = (1e-12, be * (1.0 - 1e-12));
            let (mut p1, mut p2) = (mu1, mu2);
            for _ in 0..80 {
                let t = 0.5 * (lo + hi);
                let ff = (1.0 - 2.0 / be) / (1.0 - t / be);
                p1 = inner(ff, mu1);
                p2 = inner(ff, mu2);
                if p1.exp() + p2.exp() - t > 0.0 {
                    lo = t;
                } else {
                    hi = t;
                }
            }
            // Recompose Ψ = R diag(ψ1, ψ2) Rᵀ.
            out[0][i] = c * c * p1 + s * s * p2;
            out[1][i] = c * s * (p1 - p2);
            out[2][i] = s * s * p1 + c * c * p2;
        }
        out
    }

    /// One **ARK2 / ARS(2,2,2)** IMEX step (Ascher–Ruuth–Spiteri): L-stable, second-order,
    /// stiffly accurate, with **no operator-splitting error** (unlike [`Self::step_strang_imex`]).
    /// Transport (`E = psi_transport_rhs`) is explicit; relaxation (`S`) is implicit via
    /// [`Self::implicit_relax_solve`]. Stiffly accurate ⇒ the update equals the last implicit
    /// stage. Coefficients: `γ = 1 − √2/2`, `δ = 1 − 1/(2γ)`. (Plan §2.3 / Phase 2.)
    pub fn step_ark2_imex(
        &self,
        psi: &[Vec<f64>; 3],
        ux: &[f64],
        uy: &[f64],
        dt: f64,
    ) -> [Vec<f64>; 3] {
        let gamma = 1.0 - 0.5_f64.sqrt();
        let delta = 1.0 - 1.0 / (2.0 * gamma);
        let gdt = dt * gamma;
        // accumulate `psi + Σ c_t · term_t` (componentwise over the 3 tensor entries)
        let accum = |terms: &[(f64, &[Vec<f64>; 3])]| -> [Vec<f64>; 3] {
            std::array::from_fn(|v| {
                let mut o = psi[v].clone();
                for &(c, t) in terms {
                    for i in 0..o.len() {
                        o[i] += c * t[v][i];
                    }
                }
                o
            })
        };
        // Stage 1 (explicit, a^I_11 = 0): Ψ_1 = Ψⁿ.
        let e1 = self.psi_transport_rhs(psi, ux, uy);
        // Stage 2: B_2 = Ψⁿ + dt·γ·E_1 ; solve Ψ_2 − dt·γ·S(Ψ_2) = B_2.
        let b2 = accum(&[(dt * gamma, &e1)]);
        let psi2 = self.implicit_relax_solve(&b2, gdt);
        let e2 = self.psi_transport_rhs(&psi2, ux, uy);
        // Recover S_2 = (Ψ_2 − B_2)/(dt·γ) exactly from the stage equation (free).
        let s2: [Vec<f64>; 3] =
            std::array::from_fn(|v| psi2[v].iter().zip(&b2[v]).map(|(y, b)| (y - b) / gdt).collect());
        // Stage 3: B_3 = Ψⁿ + dt(δ·E_1 + (1−δ)·E_2) + dt(1−γ)·S_2 ; solve.
        let b3 = accum(&[(dt * delta, &e1), (dt * (1.0 - delta), &e2), (dt * (1.0 - gamma), &s2)]);
        // Stiffly accurate: Ψⁿ⁺¹ = Ψ_3 (= B_3 + dt·γ·S_3, the ARK update).
        self.implicit_relax_solve(&b3, gdt)
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

/// [`ImexSemi`](crate::sim::integrate::ImexSemi) adapter exposing [`LogConfOldroydB`] (with a
/// prescribed velocity field) to the generic [`ArkImex`](crate::sim::integrate::ArkImex)
/// driver. State layout is `[3][ndof]` = `[Ψxx, Ψxy, Ψyy]`; transport is the explicit part,
/// relaxation the (locally-solved) implicit part.
pub struct LogConfImex<'a> {
    pub model: &'a LogConfOldroydB<'a>,
    pub ux: &'a [f64],
    pub uy: &'a [f64],
}

impl crate::sim::integrate::ImexSemi for LogConfImex<'_> {
    fn n_vars(&self) -> usize {
        3
    }
    fn ndof(&self) -> usize {
        self.model.ndof()
    }
    fn rhs_explicit(&self, state: &[Vec<f64>], _t: f64) -> Vec<Vec<f64>> {
        let psi = [state[0].clone(), state[1].clone(), state[2].clone()];
        self.model.psi_transport_rhs(&psi, self.ux, self.uy).into()
    }
    fn rhs_implicit(&self, state: &[Vec<f64>], _t: f64) -> Vec<Vec<f64>> {
        let psi = [state[0].clone(), state[1].clone(), state[2].clone()];
        self.model.relax_source(&psi).into()
    }
    fn solve_implicit(&self, b: &[Vec<f64>], gamma: f64, _t: f64) -> Vec<Vec<f64>> {
        let bb = [b[0].clone(), b[1].clone(), b[2].clone()];
        self.model.implicit_relax_solve(&bb, gamma).into()
    }
}

/// Largest `θ ∈ [0, 1]` with `q(θ) = aθ² + bθ + c ≥ 0`, given `q(0) = c ≥ 0` and `q(1) < 0`
/// (a single down-crossing in `(0,1)`): the smallest positive root. Used by the det limiter.
fn theta_first_root(a: f64, b: f64, c: f64) -> f64 {
    if a.abs() < 1e-300 {
        // Linear `bθ + c`: with c ≥ 0 and b·1+c < 0 ⇒ b < 0 ⇒ root −c/b ∈ (0,1).
        return if b.abs() < 1e-300 { 1.0 } else { (-c / b).clamp(0.0, 1.0) };
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return 1.0; // no real crossing (shouldn't occur given the sign hypotheses)
    }
    let sq = disc.sqrt();
    let mut t = 1.0f64;
    for r in [(-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)] {
        if r > 0.0 && r <= 1.0 {
            t = t.min(r);
        }
    }
    t
}

/// **Scalar-surrogate bound-preserving limiter** for a direct-form conformation field
/// `C = [Cxx, Cxy, Cyy]` (Zhang–Shu / Christner–Chan style). Per element it scales every node's
/// deviation from the (quadrature-weighted) cell mean toward the mean by the largest common
/// `θ ∈ [0, 1]` that keeps all nodes in the admissible set, enforcing the scalar surrogates the
/// research (`docs/research-entropy-stable-methods.md` §9) identifies in place of the full SPD
/// cone: `det C ≥ ε` and `tr C ≥ 2√ε` (together ⇒ SPD), and optionally `tr C ≤ b` (FENE).
/// Conservative (the cell mean is unchanged) and high-order-accurate where the bound is slack.
/// This is the transport-side complement to the bound-preserving *implicit relaxation* solve:
/// the implicit solve keeps relaxation in-bounds; this keeps the high-order transport in-bounds
/// (the HWNP positivity failure). The cell mean is assumed admissible (a conservative scheme on
/// admissible data keeps the mean admissible); if it is not, the element is left untouched.
pub fn limit_conformation_bounds(mesh: &Mesh2d, c: &mut [Vec<f64>], eps: f64, b_max: f64) {
    let nn = mesh.refq.n_nodes();
    let tr_lo = 2.0 * eps.sqrt(); // AM–GM-consistent trace floor for SPD
    for (e, el) in mesh.elements.iter().enumerate() {
        let base = e * nn;
        // Quadrature-weighted cell mean.
        let (mut wsum, mut mxx, mut mxy, mut myy) = (0.0, 0.0, 0.0, 0.0);
        for k in 0..nn {
            let w = el.geom.jw[k];
            wsum += w;
            mxx += w * c[0][base + k];
            mxy += w * c[1][base + k];
            myy += w * c[2][base + k];
        }
        mxx /= wsum;
        mxy /= wsum;
        myy /= wsum;
        // Skip if the mean is itself inadmissible (nothing safe to limit toward).
        let trm = mxx + myy;
        if mxx * myy - mxy * mxy < eps || trm < tr_lo || (b_max.is_finite() && trm > b_max) {
            continue;
        }
        // Largest common θ keeping every node admissible.
        let mut theta = 1.0f64;
        for k in 0..nn {
            let (dxx, dxy, dyy) =
                (c[0][base + k] - mxx, c[1][base + k] - mxy, c[2][base + k] - myy);
            // det(M + θΔ) = a θ² + b θ + det(M) ≥ ε.
            let a = dxx * dyy - dxy * dxy;
            let b = mxx * dyy + myy * dxx - 2.0 * mxy * dxy;
            let det_m = mxx * myy - mxy * mxy;
            if det_m + b + a < eps {
                theta = theta.min(theta_first_root(a, b, det_m - eps));
            }
            // tr(M + θΔ) = trm + θ·(dxx+dyy) ∈ [tr_lo, b_max] (linear).
            let dtr = dxx + dyy;
            let trn = trm + dtr;
            if trn < tr_lo {
                theta = theta.min((tr_lo - trm) / dtr); // dtr < 0 here
            }
            if b_max.is_finite() && trn > b_max {
                theta = theta.min((b_max - trm) / dtr); // dtr > 0 here
            }
        }
        let theta = theta.clamp(0.0, 1.0);
        if theta < 1.0 {
            for k in 0..nn {
                c[0][base + k] = mxx + theta * (c[0][base + k] - mxx);
                c[1][base + k] = mxy + theta * (c[1][base + k] - mxy);
                c[2][base + k] = myy + theta * (c[2][base + k] - myy);
            }
        }
    }
}

/// [`StageHook`](crate::sim::integrate::StageHook) wrapper around [`limit_conformation_bounds`],
/// applying the scalar-surrogate bound-preserving limiter to the conformation state
/// `[Cxx, Cxy, Cyy]` after each Runge–Kutta stage.
pub struct ConformationBoundLimiter {
    /// Minimum `det C` (SPD floor). Typical `1e-8`–`1e-10`.
    pub eps: f64,
    /// Maximum `tr C` (FENE-P extensibility `b`); `f64::INFINITY` to disable.
    pub b_max: f64,
}

impl crate::sim::integrate::StageHook for ConformationBoundLimiter {
    fn after_stage(&self, mesh: &Mesh2d, state: &mut [Vec<f64>], _stage: usize) {
        limit_conformation_bounds(mesh, state, self.eps, self.b_max);
    }
}

/// **Quadratic-knapsack optimal blending** (Christner–Chan, arXiv:2507.14488). Given per-node
/// weights `w_i > 0`, deviations-from-mean `δ_i` (with `Σ w_i δ_i = 0`), and admissibility caps
/// `cap_i ∈ [0,1]` (the largest blend keeping node `i` in-bounds), returns the per-node
/// `θ_i ∈ [0, cap_i]` minimizing the added diffusion `Σ ½ w_i (1−θ_i)²` (i.e. staying closest to
/// the high-order `θ=1`) subject to the single conservation constraint `Σ w_i δ_i θ_i = 0`.
/// KKT ⇒ `θ_i(μ) = clamp(1 − μ δ_i, 0, cap_i)`; the multiplier `μ` is the unique root of the
/// monotone `F(μ) = Σ w_i δ_i θ_i(μ)`, found by bisection. The uniform Zhang–Shu `θ = min cap_i`
/// is a *feasible* point of the same program, so this is never more dissipative — and strictly
/// less when the non-capped nodes' deviations vary. Single-scalar-constraint case (research §9);
/// the full SPD-cone tensor case is multi-constraint (the open frontier — see the plan).
fn knapsack_theta(weights: &[f64], devs: &[f64], caps: &[f64]) -> Vec<f64> {
    let n = weights.len();
    let theta = |mu: f64| -> Vec<f64> {
        (0..n).map(|i| (1.0 - mu * devs[i]).clamp(0.0, caps[i])).collect()
    };
    let f = |mu: f64| -> f64 {
        (0..n).map(|i| weights[i] * devs[i] * (1.0 - mu * devs[i]).clamp(0.0, caps[i])).sum()
    };
    let f0 = f(0.0);
    if f0.abs() < 1e-300 {
        return theta(0.0); // caps already conservative (no limiting, or balanced)
    }
    // F(μ) is monotone non-increasing. Bracket the sign change, then bisect.
    let (mut lo, mut hi);
    if f0 > 0.0 {
        lo = 0.0;
        hi = 1.0;
        while f(hi) > 0.0 && hi < 1e12 {
            hi *= 2.0;
        }
    } else {
        hi = 0.0;
        lo = -1.0;
        while f(lo) < 0.0 && lo > -1e12 {
            lo *= 2.0;
        }
    }
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if f(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    theta(0.5 * (lo + hi))
}

/// **Log-conformation** bound-preserving limiter. In `Ψ`-space `C = exp(Ψ)` is SPD *by
/// construction*, so the only violable bound is the FENE-P trace `tr C = tr exp(Ψ) ≤ b`. Per
/// element it scales each node's `Ψ` deviation from the cell mean toward the mean by the largest
/// common `θ ∈ [0,1]` keeping every node's `tr exp(Ψ) ≤ b`. Because `tr exp(·)` is convex (and the
/// blend is affine in `θ`), `g(θ) = tr exp(Ψ̄ + θΔ)` is convex with `g(0) ≤ b < g(1)` at a violating
/// node ⇒ a single crossing, found by bisection. **Conserves the mean of `Ψ`** (not of `C` — the
/// usual log-conformation trade-off); leaves the field untouched where the bound is slack (incl.
/// `b = ∞`, the non-FENE case). Pairs with the FENE-P implicit solve (which bounds *relaxation*)
/// to keep the high-order *transport* under the extensibility limit.
pub fn limit_logconf_trace_bound(mesh: &Mesh2d, psi: &mut [Vec<f64>], b_max: f64) {
    if !b_max.is_finite() {
        return;
    }
    let nn = mesh.refq.n_nodes();
    let tr_exp = |a: f64, b: f64, d: f64| -> f64 {
        let (m1, m2, _, _) = sym_eig(a, b, d);
        m1.exp() + m2.exp()
    };
    for (e, el) in mesh.elements.iter().enumerate() {
        let base = e * nn;
        let (mut ws, mut mxx, mut mxy, mut myy) = (0.0, 0.0, 0.0, 0.0);
        for k in 0..nn {
            let w = el.geom.jw[k];
            ws += w;
            mxx += w * psi[0][base + k];
            mxy += w * psi[1][base + k];
            myy += w * psi[2][base + k];
        }
        mxx /= ws;
        mxy /= ws;
        myy /= ws;
        if tr_exp(mxx, mxy, myy) > b_max {
            continue; // mean already over the bound — nothing safe to limit toward
        }
        let mut theta = 1.0f64;
        for k in 0..nn {
            let (dxx, dxy, dyy) =
                (psi[0][base + k] - mxx, psi[1][base + k] - mxy, psi[2][base + k] - myy);
            if tr_exp(mxx + dxx, mxy + dxy, myy + dyy) > b_max {
                // g(θ) = tr exp(Ψ̄ + θΔ) convex, g(0) ≤ b < g(1): bisect the unique crossing.
                let (mut lo, mut hi) = (0.0f64, 1.0f64);
                for _ in 0..60 {
                    let mid = 0.5 * (lo + hi);
                    if tr_exp(mxx + mid * dxx, mxy + mid * dxy, myy + mid * dyy) <= b_max {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                theta = theta.min(lo);
            }
        }
        if theta < 1.0 {
            for k in 0..nn {
                psi[0][base + k] = mxx + theta * (psi[0][base + k] - mxx);
                psi[1][base + k] = mxy + theta * (psi[1][base + k] - mxy);
                psi[2][base + k] = myy + theta * (psi[2][base + k] - myy);
            }
        }
    }
}

/// [`StageHook`](crate::sim::integrate::StageHook) wrapper for [`limit_logconf_trace_bound`],
/// enforcing `tr exp(Ψ) ≤ b_max` on the log-conformation state after each Runge–Kutta stage.
pub struct LogConfTraceLimiter {
    /// FENE-P extensibility `b` (max `tr C`).
    pub b_max: f64,
}

impl crate::sim::integrate::StageHook for LogConfTraceLimiter {
    fn after_stage(&self, mesh: &Mesh2d, state: &mut [Vec<f64>], _stage: usize) {
        limit_logconf_trace_bound(mesh, state, self.b_max);
    }
}

/// Bound-preserving limiter for a **scalar** field `u` to `[lo, hi]`, using the
/// quadratic-knapsack optimal per-node blending ([`knapsack_theta`]) — conservative (cell mean
/// preserved), high-order where the bound is slack, and **less dissipative than the uniform
/// Zhang–Shu `min θ`**. Directly usable for any bounded transported scalar (e.g. a continuum
/// concentration / volume-fraction `φ`). The conformation-*tensor* analogue is multi-constraint
/// (3 conserved components) — the open frontier; `limit_conformation_bounds` stays on uniform `θ`.
pub fn limit_scalar_bounds(mesh: &Mesh2d, u: &mut [f64], lo: f64, hi: f64) {
    let nn = mesh.refq.n_nodes();
    for (e, el) in mesh.elements.iter().enumerate() {
        let base = e * nn;
        let (mut wsum, mut m) = (0.0, 0.0);
        for k in 0..nn {
            let w = el.geom.jw[k];
            wsum += w;
            m += w * u[base + k];
        }
        m /= wsum;
        if m < lo || m > hi {
            continue; // mean inadmissible — nothing safe to limit toward
        }
        let w: Vec<f64> = (0..nn).map(|k| el.geom.jw[k]).collect();
        let dev: Vec<f64> = (0..nn).map(|k| u[base + k] - m).collect();
        let caps: Vec<f64> = (0..nn)
            .map(|k| {
                let d = dev[k];
                let mut c = 1.0f64;
                if m + d > hi && d > 0.0 {
                    c = c.min((hi - m) / d);
                }
                if m + d < lo && d < 0.0 {
                    c = c.min((lo - m) / d);
                }
                c.clamp(0.0, 1.0)
            })
            .collect();
        if caps.iter().all(|&c| c >= 1.0) {
            continue; // all nodes in-bounds
        }
        let theta = knapsack_theta(&w, &dev, &caps);
        for k in 0..nn {
            u[base + k] = m + theta[k] * dev[k];
        }
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
    fn relax_exact_matches_analytic_for_any_dt() {
        // The exact eigenvalue relaxation reproduces C = I + (C₀−I)e^{−τ/λ} to machine
        // precision for ANY τ — including τ ≫ λ, where an explicit step is unstable.
        let p = 3;
        let mesh = Mesh2d::rectangular(p, 2, 2, [0.0, 1.0], [0.0, 1.0]);
        let lambda = 0.1;
        let lc = LogConfOldroydB::new(&mesh, lambda, 1.0);
        let c0 = [vec![3.0; lc.ndof()], vec![0.5; lc.ndof()], vec![2.0; lc.ndof()]];
        let psi0 = lc.from_conformation(&c0);
        for &dt in &[0.01_f64, 0.1, 1.0, 10.0] {
            let psi = lc.relax_exact(&psi0, dt);
            let c = lc.conformation(&psi);
            let g = (-dt / lambda).exp();
            let exact = [1.0 + 2.0 * g, 0.5 * g, 1.0 + 1.0 * g];
            for (v, &ex) in exact.iter().enumerate() {
                let err = c[v].iter().fold(0.0f64, |a, &x| a.max((x - ex).abs()));
                assert!(err < 1e-12, "dt={dt} comp {v}: err {err}, want {ex}");
            }
        }
    }

    #[test]
    fn imex_unlocks_large_timestep_where_explicit_fails() {
        // At dt = 10λ — far past the explicit stability limit dt ≲ 2.5λ — the IMEX/Strang
        // step is accurate and SPD, while fully-explicit SSP-RK3 is not. (The unlock.)
        let p = 3;
        let mesh = Mesh2d::rectangular(p, 2, 2, [0.0, 1.0], [0.0, 1.0]);
        let lambda = 0.1;
        let lc = LogConfOldroydB::new(&mesh, lambda, 1.0);
        let zero = vec![0.0; lc.ndof()];
        let c0 = [vec![3.0; lc.ndof()], vec![0.5; lc.ndof()], vec![2.0; lc.ndof()]];
        let psi0 = lc.from_conformation(&c0);
        let dt = 1.0; // = 10λ
        let g = (-dt / lambda).exp();
        let exact = [1.0 + 2.0 * g, 0.5 * g, 1.0 + 1.0 * g];

        // IMEX: accurate and SPD even at dt = 10λ.
        let psi_imex = lc.step_strang_imex(&psi0, &zero, &zero, dt);
        let c_imex = lc.conformation(&psi_imex);
        for i in 0..c_imex[0].len() {
            let det = c_imex[0][i] * c_imex[2][i] - c_imex[1][i] * c_imex[1][i];
            assert!(c_imex[0][i] > 0.0 && det > 0.0, "IMEX C not SPD at {i}");
        }
        for (v, &ex) in exact.iter().enumerate() {
            let err = c_imex[v].iter().fold(0.0f64, |a, &x| a.max((x - ex).abs()));
            assert!(err < 1e-10, "IMEX comp {v}: err {err}, want {ex}");
        }

        // Explicit SSP-RK3 at the same dt is past its stability limit and does NOT reach
        // the analytic state (here it diverges to non-finite values) — exactly the wall
        // IMEX removes. Check finiteness + accuracy explicitly: `f64::max` *absorbs* NaN
        // (returns the non-NaN arg), so a naive max-reduction would silently read 0.
        let psi_exp = lc.step_ssp_rk3(&psi0, &zero, &zero, dt);
        let c_exp = lc.conformation(&psi_exp);
        let explicit_accurate = c_exp.iter().all(|comp| comp.iter().all(|&x| x.is_finite()))
            && (0..3).all(|v| c_exp[v].iter().all(|&x| (x - exact[v]).abs() < 1e-2));
        assert!(!explicit_accurate, "explicit unexpectedly stable & accurate at dt=10λ");
    }

    #[test]
    fn strang_imex_recovers_high_wi_steady_shear() {
        // The Strang IMEX step (transport + exact relaxation) must reach the SAME Wi=10
        // analytic steady shear state as the fully-explicit path, staying SPD throughout.
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
            psi = lc.step_strang_imex(&psi, &ux, &uy, dt);
        }
        let c = lc.conformation(&psi);
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
    fn ark2_imex_recovers_high_wi_steady_shear() {
        // ARK2 (no splitting error) reaches the same Wi=10 analytic steady shear, SPD throughout.
        let p = 4;
        let mesh = Mesh2d::rectangular(p, 3, 3, [0.0, 1.0], [0.0, 1.0]);
        let lambda = 1.0;
        let gdot = 10.0;
        let wi = lambda * gdot;
        let lc = LogConfOldroydB::new(&mesh, lambda, 1.5);
        let ux = nodal(&mesh, |_, y| gdot * y);
        let uy = vec![0.0; mesh.n_elements() * mesh.refq.n_nodes()];
        let mut psi = lc.identity();
        let dt = 0.005;
        let nsteps = (25.0_f64 / dt).round() as usize;
        for _ in 0..nsteps {
            psi = lc.step_ark2_imex(&psi, &ux, &uy, dt);
        }
        let c = lc.conformation(&psi);
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
    fn ark2_imex_is_second_order_in_time() {
        // Startup of steady shear from C=I is spatially uniform ⇒ the exact transient is
        // Cxy(t) = Wi(1 − e^{−t/λ}). ARK2 must converge at 2nd order (no splitting error).
        let p = 3;
        let mesh = Mesh2d::rectangular(p, 2, 2, [0.0, 1.0], [0.0, 1.0]);
        let lambda = 1.0;
        let gdot = 1.0;
        let wi = lambda * gdot;
        let lc = LogConfOldroydB::new(&mesh, lambda, 1.0);
        let ux = nodal(&mesh, |_, y| gdot * y);
        let uy = vec![0.0; mesh.n_elements() * mesh.refq.n_nodes()];
        let t_end = 1.0_f64;
        let cxy_exact = wi * (1.0 - (-t_end / lambda).exp());
        let err_at = |dt: f64| -> f64 {
            let nsteps = (t_end / dt).round() as usize;
            let mut psi = lc.identity();
            for _ in 0..nsteps {
                psi = lc.step_ark2_imex(&psi, &ux, &uy, dt);
            }
            let c = lc.conformation(&psi);
            c[1].iter().fold(0.0f64, |a, &x| a.max((x - cxy_exact).abs()))
        };
        let (e1, e2) = (err_at(0.05), err_at(0.025));
        let rate = (e1 / e2).log2();
        assert!(rate > 1.8 && rate < 2.3, "ARK2 observed order {rate} (e1={e1}, e2={e2})");
    }

    #[test]
    fn ark2_imex_stable_and_spd_at_large_dt() {
        // Pure relaxation at dt = 10λ: the L-stable implicit relaxation stays finite & SPD
        // and damps toward equilibrium — where explicit SSP-RK3 would diverge.
        let p = 3;
        let mesh = Mesh2d::rectangular(p, 2, 2, [0.0, 1.0], [0.0, 1.0]);
        let lambda = 0.1;
        let lc = LogConfOldroydB::new(&mesh, lambda, 1.0);
        let zero = vec![0.0; lc.ndof()];
        let c0 = [vec![3.0; lc.ndof()], vec![0.5; lc.ndof()], vec![2.0; lc.ndof()]];
        let mut psi = lc.from_conformation(&c0);
        let dt = 1.0; // = 10λ
        for _ in 0..5 {
            psi = lc.step_ark2_imex(&psi, &zero, &zero, dt);
            let c = lc.conformation(&psi);
            for i in 0..c[0].len() {
                let det = c[0][i] * c[2][i] - c[1][i] * c[1][i];
                assert!(c[0][i].is_finite() && c[0][i] > 0.0 && det > 0.0, "not finite/SPD at {i}");
            }
        }
        let c = lc.conformation(&psi);
        assert!(
            (c[0][0] - 1.0).abs() < 0.2 && c[1][0].abs() < 0.2,
            "did not relax toward I: Cxx={}, Cxy={}",
            c[0][0],
            c[1][0]
        );
    }

    #[test]
    fn ark_imex_generic_matches_bespoke_ark2() {
        // The generic ArkImex(ars222) driver over the LogConfImex adapter must reproduce the
        // bespoke `step_ark2_imex` to round-off (Plan §4.5 / Validation 6).
        use crate::sim::integrate::{ArkImex, ArkTableau};
        let p = 3;
        let mesh = Mesh2d::rectangular(p, 2, 2, [0.0, 1.0], [0.0, 1.0]);
        let lc = LogConfOldroydB::new(&mesh, 0.5, 1.0);
        let ux = nodal(&mesh, |_, y| 2.0 * y);
        let uy = vec![0.0; mesh.n_elements() * mesh.refq.n_nodes()];
        let c0 = [
            nodal(&mesh, |x, _| 2.0 + 0.5 * x),
            nodal(&mesh, |x, y| 0.2 * x - 0.1 * y),
            nodal(&mesh, |_, y| 1.5 + 0.3 * y),
        ];
        let psi0 = lc.from_conformation(&c0);
        let dt = 0.03;
        let bespoke = lc.step_ark2_imex(&psi0, &ux, &uy, dt);
        let semi = LogConfImex { model: &lc, ux: &ux, uy: &uy };
        let ark = ArkImex::new(ArkTableau::ars222(), dt);
        let state = vec![psi0[0].clone(), psi0[1].clone(), psi0[2].clone()];
        let generic = ark.step(&semi, &state, 0.0);
        for v in 0..3 {
            let err =
                generic[v].iter().zip(&bespoke[v]).fold(0.0f64, |a, (g, b)| a.max((g - b).abs()));
            assert!(err < 1e-12, "generic vs bespoke comp {v}: {err}");
        }
    }

    #[test]
    fn giesekus_relax_exact_matches_ode_integration() {
        // The closed-form Giesekus relaxation flow (Bernoulli) must match a high-resolution
        // RK4 integration of dc/dt = −(1/λ)[(c−1) + α(c−1)²] per eigenvalue.
        let p = 3;
        let mesh = Mesh2d::rectangular(p, 2, 2, [0.0, 1.0], [0.0, 1.0]);
        let lambda = 0.7;
        let alpha = 0.4;
        let lc = LogConfOldroydB::new(&mesh, lambda, 1.0).with_mobility(alpha);
        // Diagonal conformation ⇒ eigenvalues are Cxx, Cyy directly.
        let c0 = [vec![3.0; lc.ndof()], vec![0.0; lc.ndof()], vec![2.0; lc.ndof()]];
        let psi0 = lc.from_conformation(&c0);
        let tau = 0.5_f64;
        let psi = lc.relax_exact(&psi0, tau);
        let c = lc.conformation(&psi);
        // RK4 reference for a scalar eigenvalue.
        let rk4 = |c_init: f64| -> f64 {
            let f = |cc: f64| -(1.0 / lambda) * ((cc - 1.0) + alpha * (cc - 1.0).powi(2));
            let n = 200_000;
            let h = tau / n as f64;
            let mut cc = c_init;
            for _ in 0..n {
                let k1 = f(cc);
                let k2 = f(cc + 0.5 * h * k1);
                let k3 = f(cc + 0.5 * h * k2);
                let k4 = f(cc + h * k3);
                cc += h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
            }
            cc
        };
        let (want_xx, want_yy) = (rk4(3.0), rk4(2.0));
        assert!((c[0][0] - want_xx).abs() < 1e-8, "Cxx {} vs {}", c[0][0], want_xx);
        assert!((c[2][0] - want_yy).abs() < 1e-8, "Cyy {} vs {}", c[2][0], want_yy);
        assert!(c[1][0].abs() < 1e-12, "Cxy stayed zero");
    }

    #[test]
    fn giesekus_imex_satisfies_steady_shear_equation() {
        // ARK2 IMEX with Giesekus mobility α>0 marched to steady shear. The converged
        // conformation must satisfy the Giesekus steady balance L·C+C·Lᵀ = (1/λ)[(C−I)+α(C−I)²],
        // and show bounded extension (Cxx < the Oldroyd-B value 1+2Wi²).
        let p = 4;
        let mesh = Mesh2d::rectangular(p, 3, 3, [0.0, 1.0], [0.0, 1.0]);
        let (lambda, gdot, alpha) = (1.0, 2.0, 0.4);
        let wi = lambda * gdot;
        let lc = LogConfOldroydB::new(&mesh, lambda, 1.0).with_mobility(alpha);
        let ux = nodal(&mesh, |_, y| gdot * y);
        let uy = vec![0.0; mesh.n_elements() * mesh.refq.n_nodes()];
        let mut psi = lc.identity();
        let dt = 0.01;
        for _ in 0..(25.0 / dt) as usize {
            psi = lc.step_ark2_imex(&psi, &ux, &uy, dt);
        }
        let c = lc.conformation(&psi);
        let mut max_res = 0.0f64;
        for i in 0..c[0].len() {
            let (cxx, cxy, cyy) = (c[0][i], c[1][i], c[2][i]);
            let (a, b, d) = (cxx - 1.0, cxy, cyy - 1.0);
            // L·C+C·Lᵀ − (1/λ)[(C−I) + α(C−I)²], with L = [[0,γ̇],[0,0]].
            let res_xx = 2.0 * gdot * cxy - (1.0 / lambda) * (a + alpha * (a * a + b * b));
            let res_xy = gdot * cyy - (1.0 / lambda) * (b + alpha * b * (a + d));
            let res_yy = -(1.0 / lambda) * (d + alpha * (b * b + d * d));
            max_res = max_res.max(res_xx.abs()).max(res_xy.abs()).max(res_yy.abs());
            assert!(cxx > 0.0 && cxx * cyy - cxy * cxy > 0.0, "C not SPD at {i}");
        }
        assert!(max_res < 1e-5, "Giesekus steady residual {max_res}");
        // Shear-thinning: bounded extension vs Oldroyd-B (1 + 2Wi²).
        assert!(c[0][0] < 1.0 + 2.0 * wi * wi, "Cxx={} not below Oldroyd-B {}", c[0][0], 1.0 + 2.0 * wi * wi);
    }

    #[test]
    fn fenep_implicit_solve_is_consistent_and_bounded() {
        // The FENE-P implicit solve must (a) satisfy the stage equation Ψ − γ·S(Ψ) = B with
        // S(Ψ) = (1/λ)(e^{−Ψ} − f·I), and (b) keep tr C < b (Peterlin bound preservation).
        let p = 3;
        let mesh = Mesh2d::rectangular(p, 2, 2, [0.0, 1.0], [0.0, 1.0]);
        let (lambda, b_ext) = (0.5, 10.0);
        let lc = LogConfOldroydB::new(&mesh, lambda, 1.0).with_extensibility(b_ext);
        let gamma = 0.05;
        let nn = mesh.refq.n_nodes();
        let mut bb = [vec![0.0; lc.ndof()], vec![0.0; lc.ndof()], vec![0.0; lc.ndof()]];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let g = e * nn + k;
                let (x, y) = (el.geom.x[k], el.geom.y[k]);
                bb[0][g] = 0.8 + 0.3 * (x + y).sin();
                bb[1][g] = 0.15 * x - 0.1 * y;
                bb[2][g] = 0.4 + 0.2 * (x * y).cos();
            }
        }
        let psi = lc.implicit_relax_solve(&bb, gamma);
        let c = lc.conformation(&psi);
        let mut max_res = 0.0f64;
        let mut max_tr = 0.0f64;
        for i in 0..c[0].len() {
            let trc = c[0][i] + c[2][i];
            max_tr = max_tr.max(trc);
            let f = (1.0 - 2.0 / b_ext) / (1.0 - trc / b_ext);
            let em = sym_apply(psi[0][i], psi[1][i], psi[2][i], |x| (-x).exp());
            // S = (1/λ)(e^{−Ψ} − f·I);  residual = Ψ − γ·S − B.
            let sxx = (em[0] - f) / lambda;
            let sxy = em[1] / lambda;
            let syy = (em[2] - f) / lambda;
            let rxx = psi[0][i] - gamma * sxx - bb[0][i];
            let rxy = psi[1][i] - gamma * sxy - bb[1][i];
            let ryy = psi[2][i] - gamma * syy - bb[2][i];
            max_res = max_res.max(rxx.abs()).max(rxy.abs()).max(ryy.abs());
        }
        assert!(max_res < 1e-9, "FENE-P stage residual {max_res}");
        assert!(max_tr < b_ext, "trace bound violated: {max_tr} ≥ {b_ext}");
    }

    #[test]
    fn fenep_imex_steady_shear_respects_trace_bound() {
        // ARK2 IMEX with FENE-P marched to steady shear: the conformation satisfies the
        // FENE-P steady balance L·C+C·Lᵀ = (1/λ)[f·C − I], stays SPD, and tr C < b throughout.
        let p = 4;
        let mesh = Mesh2d::rectangular(p, 3, 3, [0.0, 1.0], [0.0, 1.0]);
        let (lambda, gdot, b_ext) = (1.0, 4.0, 20.0);
        let wi = lambda * gdot;
        let lc = LogConfOldroydB::new(&mesh, lambda, 1.0).with_extensibility(b_ext);
        let ux = nodal(&mesh, |_, y| gdot * y);
        let uy = vec![0.0; mesh.n_elements() * mesh.refq.n_nodes()];
        let mut psi = lc.identity();
        let dt = 0.01;
        for _ in 0..(25.0 / dt) as usize {
            psi = lc.step_ark2_imex(&psi, &ux, &uy, dt);
        }
        let c = lc.conformation(&psi);
        let mut max_res = 0.0f64;
        let mut max_tr = 0.0f64;
        for i in 0..c[0].len() {
            let (cxx, cxy, cyy) = (c[0][i], c[1][i], c[2][i]);
            let trc = cxx + cyy;
            max_tr = max_tr.max(trc);
            let f = (1.0 - 2.0 / b_ext) / (1.0 - trc / b_ext);
            let res_xx = 2.0 * gdot * cxy - (1.0 / lambda) * (f * cxx - 1.0);
            let res_xy = gdot * cyy - (1.0 / lambda) * (f * cxy);
            let res_yy = -(1.0 / lambda) * (f * cyy - 1.0);
            max_res = max_res.max(res_xx.abs()).max(res_xy.abs()).max(res_yy.abs());
            assert!(cxx > 0.0 && cxx * cyy - cxy * cxy > 0.0, "C not SPD at {i}");
        }
        assert!(max_res < 1e-4, "FENE-P steady residual {max_res}");
        assert!(max_tr < b_ext, "trace bound violated: {max_tr} ≥ {b_ext}");
        // Finite extensibility ⇒ far below the Oldroyd-B extension 1 + 2Wi².
        assert!(c[0][0] < 1.0 + 2.0 * wi * wi, "Cxx={} not bounded below Oldroyd-B", c[0][0]);
    }

    fn elem_mean(mesh: &Mesh2d, c: &[Vec<f64>], comp: usize) -> f64 {
        let nn = mesh.refq.n_nodes();
        let el = &mesh.elements[0];
        let (mut ws, mut s) = (0.0, 0.0);
        for k in 0..nn {
            let w = el.geom.jw[k];
            ws += w;
            s += w * c[comp][k];
        }
        s / ws
    }

    #[test]
    fn limiter_restores_spd_and_conserves_mean() {
        // A field with an admissible cell mean but a non-SPD node: the limiter must pull every
        // node into `det ≥ ε` (SPD) while preserving the (conserved) cell mean exactly.
        let p = 3;
        let mesh = Mesh2d::rectangular(p, 1, 1, [0.0, 1.0], [0.0, 1.0]);
        let nn = mesh.refq.n_nodes();
        let ndof = mesh.n_elements() * nn;
        let mut c = [vec![1.0; ndof], vec![0.0; ndof], vec![1.0; ndof]]; // C = I everywhere
        c[1][0] = 2.0; // node 0: Cxy=2 ⇒ det = 1 − 4 = −3 (non-SPD)
        let m0 = [elem_mean(&mesh, &c, 0), elem_mean(&mesh, &c, 1), elem_mean(&mesh, &c, 2)];
        let eps = 1e-8;
        limit_conformation_bounds(&mesh, &mut c, eps, f64::INFINITY);
        for i in 0..ndof {
            let det = c[0][i] * c[2][i] - c[1][i] * c[1][i];
            assert!(det >= eps * (1.0 - 1e-9) && c[0][i] + c[2][i] > 0.0, "node {i} not SPD: det={det}");
        }
        let m1 = [elem_mean(&mesh, &c, 0), elem_mean(&mesh, &c, 1), elem_mean(&mesh, &c, 2)];
        for v in 0..3 {
            assert!((m1[v] - m0[v]).abs() < 1e-12, "mean comp {v} not conserved");
        }
    }

    #[test]
    fn limiter_enforces_fene_trace_bound() {
        // A node overshooting tr C > b is pulled back to tr C ≤ b; in-bounds nodes are kept.
        let p = 3;
        let mesh = Mesh2d::rectangular(p, 1, 1, [0.0, 1.0], [0.0, 1.0]);
        let nn = mesh.refq.n_nodes();
        let ndof = mesh.n_elements() * nn;
        let mut c = [vec![2.0; ndof], vec![0.0; ndof], vec![2.0; ndof]]; // tr = 4 everywhere
        c[0][0] = 15.0;
        c[2][0] = 5.0; // node 0: tr = 20
        let b = 8.0;
        limit_conformation_bounds(&mesh, &mut c, 1e-8, b);
        for i in 0..ndof {
            let tr = c[0][i] + c[2][i];
            assert!(tr <= b * (1.0 + 1e-9), "node {i} tr={tr} > b={b}");
        }
    }

    #[test]
    fn limiter_leaves_admissible_field_unchanged() {
        // A smooth SPD field well inside the bounds must be untouched (θ = 1, high-order intact).
        let p = 3;
        let mesh = Mesh2d::rectangular(p, 2, 2, [0.0, 1.0], [0.0, 1.0]);
        let nn = mesh.refq.n_nodes();
        let ndof = mesh.n_elements() * nn;
        let mut c = [vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof]];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let (x, y) = (el.geom.x[k], el.geom.y[k]);
                let g = e * nn + k;
                c[0][g] = 2.0 + 0.3 * (x + y).sin();
                c[1][g] = 0.1 * x;
                c[2][g] = 1.5 + 0.2 * y;
            }
        }
        let orig = c.clone();
        limit_conformation_bounds(&mesh, &mut c, 1e-8, 50.0);
        for v in 0..3 {
            for i in 0..ndof {
                assert!((c[v][i] - orig[v][i]).abs() < 1e-15, "admissible field altered at {v},{i}");
            }
        }
    }

    #[test]
    fn bounded_stepper_keeps_spd_where_plain_fails() {
        // End-to-end: advect a steep conformation front (Cxx=Cyy=1, Cxy a sharp tanh ⇒
        // det = 1 − Cxy²). The under-resolved high-order transport overshoots |Cxy| > 1 ⇒
        // det < 0 in the plain stepper; the bounded stepper holds det ≥ ε throughout.
        let p = 4;
        let mesh = Mesh2d::rectangular_periodic(p, 4, 1, [0.0, 1.0], [0.0, 0.25]);
        let nn = mesh.refq.n_nodes();
        let ndof = mesh.n_elements() * nn;
        let ob = OldroydB::new(&mesh, 100.0, 1.0); // large λ ⇒ transport-dominated
        let mut c0 = [vec![1.0; ndof], vec![0.0; ndof], vec![1.0; ndof]];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                c0[1][e * nn + k] = 0.95 * (60.0 * (el.geom.x[k] - 0.5)).tanh();
            }
        }
        let (ux, uy) = (vec![3.0; ndof], vec![0.0; ndof]);
        let (dt, nsteps, eps) = (0.002, 60usize, 1e-8);
        let min_det = |c: &[Vec<f64>; 3]| {
            c[0].iter()
                .zip(&c[1])
                .zip(&c[2])
                .map(|((&a, &b), &d)| a * d - b * b)
                .fold(f64::INFINITY, f64::min)
        };
        let mut cu = c0.clone();
        let mut worst_plain = f64::INFINITY;
        for _ in 0..nsteps {
            cu = ob.step_ssp_rk3(&cu, &ux, &uy, dt);
            worst_plain = worst_plain.min(min_det(&cu));
        }
        let mut cb = c0.clone();
        let mut worst_bounded = f64::INFINITY;
        for _ in 0..nsteps {
            cb = ob.step_ssp_rk3_bounded(&cb, &ux, &uy, dt, eps, f64::INFINITY);
            worst_bounded = worst_bounded.min(min_det(&cb));
        }
        assert!(worst_plain < eps, "premise: plain stepper should lose SPD (min det = {worst_plain})");
        assert!(worst_bounded >= eps * (1.0 - 1e-6), "bounded stepper kept SPD (min det = {worst_bounded})");
    }

    #[test]
    fn knapsack_theta_optimal_conservative_less_dissipative() {
        // Hand case with varying non-capped deviations: the knapsack θ must conserve
        // (Σ w·δ·θ = 0), respect the box [0, cap_i], and have strictly less added diffusion
        // Σ½w(1−θ)² than the uniform Zhang–Shu θ = min(cap) (a feasible point of the same QP).
        let w = vec![1.0, 1.0, 1.0, 1.0];
        let dev = vec![1.5, -1.0, -0.3, -0.2]; // Σ w·δ = 0
        let caps = vec![0.3, 1.0, 1.0, 1.0]; // node 0 caps low
        let theta = knapsack_theta(&w, &dev, &caps);
        let cons: f64 = (0..4).map(|i| w[i] * dev[i] * theta[i]).sum();
        assert!(cons.abs() < 1e-10, "not conservative: {cons}");
        for i in 0..4 {
            assert!(theta[i] >= -1e-12 && theta[i] <= caps[i] + 1e-12, "θ[{i}]={} out of box", theta[i]);
        }
        let j = |t: &[f64]| -> f64 { (0..4).map(|i| 0.5 * w[i] * (1.0 - t[i]).powi(2)).sum() };
        let uniform = vec![0.3; 4];
        assert!(j(&theta) <= j(&uniform) + 1e-12, "knapsack not optimal vs uniform");
        assert!(j(&theta) < j(&uniform) - 1e-6, "knapsack should strictly beat uniform here");
    }

    #[test]
    fn knapsack_scalar_limiter_bounds_and_conserves() {
        // A scalar field overshooting [0,1] at some nodes (admissible mean): the knapsack
        // limiter pulls every node into [0,1] while preserving the cell mean exactly.
        let p = 3;
        let mesh = Mesh2d::rectangular(p, 1, 1, [0.0, 1.0], [0.0, 1.0]);
        let nn = mesh.refq.n_nodes();
        let ndof = mesh.n_elements() * nn;
        let mut u = vec![0.5; ndof];
        u[0] = 1.4; // overshoot > 1
        u[1] = -0.2; // undershoot < 0
        let el = &mesh.elements[0];
        let mean = |u: &[f64]| -> f64 {
            let (mut ws, mut s) = (0.0, 0.0);
            for k in 0..nn {
                ws += el.geom.jw[k];
                s += el.geom.jw[k] * u[k];
            }
            s / ws
        };
        let m0 = mean(&u);
        limit_scalar_bounds(&mesh, &mut u, 0.0, 1.0);
        for (i, &v) in u.iter().enumerate() {
            assert!(v >= -1e-12 && v <= 1.0 + 1e-12, "node {i} out of [0,1]: {v}");
        }
        assert!((mean(&u) - m0).abs() < 1e-12, "mean not conserved");
    }

    #[test]
    fn knapsack_scalar_limiter_leaves_admissible_unchanged() {
        let p = 3;
        let mesh = Mesh2d::rectangular(p, 2, 2, [0.0, 1.0], [0.0, 1.0]);
        let nn = mesh.refq.n_nodes();
        let ndof = mesh.n_elements() * nn;
        let mut u = vec![0.0; ndof];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                u[e * nn + k] = 0.5 + 0.2 * (el.geom.x[k] - 0.5); // smooth, in (0.3, 0.7)
            }
        }
        let orig = u.clone();
        limit_scalar_bounds(&mesh, &mut u, 0.0, 1.0);
        for i in 0..ndof {
            assert!((u[i] - orig[i]).abs() < 1e-15, "admissible scalar altered at {i}");
        }
    }

    #[test]
    fn logconf_limiter_enforces_trace_bound_and_keeps_spd() {
        // A log-conf field with tr exp(Ψ) > b at a node (admissible mean): the limiter pulls
        // every node to tr C ≤ b; C = exp(Ψ) stays SPD for free; the Ψ-mean is conserved.
        let p = 3;
        let mesh = Mesh2d::rectangular(p, 1, 1, [0.0, 1.0], [0.0, 1.0]);
        let nn = mesh.refq.n_nodes();
        let ndof = mesh.n_elements() * nn;
        let lc = LogConfOldroydB::new(&mesh, 1.0, 1.0);
        let mut psi = [vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof]]; // Ψ=0 ⇒ C=I, tr=2
        psi[0][0] = 3.0; // node 0: eigenvalues 3,0 ⇒ tr C = e³ + 1 ≈ 21
        let b = 10.0;
        let el = &mesh.elements[0];
        let mean = |psi: &[Vec<f64>; 3], comp: usize| -> f64 {
            let (mut ws, mut s) = (0.0, 0.0);
            for k in 0..nn {
                ws += el.geom.jw[k];
                s += el.geom.jw[k] * psi[comp][k];
            }
            s / ws
        };
        let m0 = [mean(&psi, 0), mean(&psi, 1), mean(&psi, 2)];
        limit_logconf_trace_bound(&mesh, &mut psi, b);
        let c = lc.conformation(&psi);
        for i in 0..ndof {
            let tr = c[0][i] + c[2][i];
            assert!(tr <= b * (1.0 + 1e-9), "node {i} tr C = {tr} > b = {b}");
            assert!(c[0][i] * c[2][i] - c[1][i] * c[1][i] > 0.0, "C not SPD (impossible in log-conf)");
        }
        let m1 = [mean(&psi, 0), mean(&psi, 1), mean(&psi, 2)];
        for v in 0..3 {
            assert!((m1[v] - m0[v]).abs() < 1e-12, "Ψ-mean comp {v} not conserved");
        }
    }

    #[test]
    fn limiter_is_element_local_on_nonconforming_mesh() {
        // The limiter uses only per-element data (cell mean + θ, no neighbour/mortar), so on a
        // 2:1 non-conforming (AMR) mesh it enforces the bound and conserves *each* element's mean
        // exactly as on a conforming mesh — no mortar interaction, including on the refined cells.
        let p = 4;
        let mesh = Mesh2d::cartesian_refined(p, 3, 3, [0.0, 1.0], [0.0, 1.0], &[(1, 1)]); // centre refined
        let nn = mesh.refq.n_nodes();
        let ne = mesh.n_elements();
        let ndof = ne * nn;
        // C = I, with a non-SPD node 0 in every element (each element's mean stays SPD).
        let mut c = [vec![1.0; ndof], vec![0.0; ndof], vec![1.0; ndof]];
        for e in 0..ne {
            c[1][e * nn] = 2.0; // Cxy = 2 ⇒ det = 1 − 4 < 0
        }
        let elem_mean = |c: &[Vec<f64>; 3], e: usize, comp: usize| -> f64 {
            let el = &mesh.elements[e];
            let (mut ws, mut s) = (0.0, 0.0);
            for k in 0..nn {
                ws += el.geom.jw[k];
                s += el.geom.jw[k] * c[comp][e * nn + k];
            }
            s / ws
        };
        let m0: Vec<[f64; 3]> =
            (0..ne).map(|e| [elem_mean(&c, e, 0), elem_mean(&c, e, 1), elem_mean(&c, e, 2)]).collect();
        let eps = 1e-8;
        limit_conformation_bounds(&mesh, &mut c, eps, f64::INFINITY);
        for i in 0..ndof {
            let det = c[0][i] * c[2][i] - c[1][i] * c[1][i];
            // det is driven to ≈ ε (to ~machine-eps recomposition rounding) and stays SPD.
            assert!(det >= eps * (1.0 - 1e-6) && c[0][i] + c[2][i] > 0.0, "node {i} not SPD: det={det}");
        }
        for e in 0..ne {
            for comp in 0..3 {
                assert!(
                    (elem_mean(&c, e, comp) - m0[e][comp]).abs() < 1e-12,
                    "element {e} comp {comp} mean drifted (broken conservation on NC mesh)"
                );
            }
        }
    }

    #[test]
    fn logconf_limiter_leaves_admissible_unchanged() {
        let p = 3;
        let mesh = Mesh2d::rectangular(p, 2, 2, [0.0, 1.0], [0.0, 1.0]);
        let nn = mesh.refq.n_nodes();
        let ndof = mesh.n_elements() * nn;
        let mut psi = [vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof]];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let (x, y) = (el.geom.x[k], el.geom.y[k]);
                let g = e * nn + k;
                psi[0][g] = 0.3 * (x + y).sin(); // small Ψ ⇒ tr C ~ 2–3 ≪ b
                psi[1][g] = 0.1 * x;
                psi[2][g] = 0.2 * y;
            }
        }
        let orig = psi.clone();
        limit_logconf_trace_bound(&mesh, &mut psi, 50.0);
        for v in 0..3 {
            for i in 0..ndof {
                assert!((psi[v][i] - orig[v][i]).abs() < 1e-15, "admissible Ψ altered at {v},{i}");
            }
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
