//! Time integration: the `Semi` (method-of-lines spatial operator) seam and the
//! `Integrator` trait that advances a state through it.
//!
//! This is the central seam of `docs/api-design.md` §3.2. A [`Semi`] exposes a
//! semi-discrete right-hand side `∂ₜu = rhs(u, t)`; an [`Integrator`] consumes it
//! to advance the solution. Step 2 of the migration lifts the explicit SSP-RK3
//! stepper out of the per-regime structs (`Hyperbolic::step_ssp_rk3`,
//! `*::step_ssp_rk3`) into a single reusable [`SspRk3`] that reproduces them
//! **bit-for-bit** — the same Shu–Osher coefficients combined in the same order.
//!
//! Structured schemes (dual-splitting, IMEX) will land as further `Integrator`
//! implementors orchestrating their own stages while reusing the same operators
//! (api-design §3.2); they are not yet implemented.
//!
//! The integrator operates on the validated `Vec<Vec<f64>>` state-vector layout
//! (`[n_vars][ndof]`). Wiring it to drive [`crate::sim::State`] fields directly
//! comes with the `Simulation` orchestrator in a later step.

use crate::dg::mesh::Mesh2d;

/// A semi-discrete spatial operator in method-of-lines form: `∂ₜu = rhs(u, t)`.
///
/// `rhs` returns the time derivative in the same `[n_vars][ndof]` layout as the
/// input state — the shape every gale operator already produces.
pub trait Semi {
    fn n_vars(&self) -> usize;
    fn ndof(&self) -> usize;
    /// Evaluate `∂ₜu` at `(state, t)`.
    fn rhs(&self, state: &[Vec<f64>], t: f64) -> Vec<Vec<f64>>;
}

/// Adapter wrapping any rhs closure as a [`Semi`]. Lets a validated operator
/// (e.g. `|s, t| hyp.rhs(s, t, &bc)`) drive a generic [`Integrator`] without a
/// bespoke type.
pub struct ClosureSemi<F> {
    pub n_vars: usize,
    pub ndof: usize,
    pub f: F,
}

impl<F> ClosureSemi<F>
where
    F: Fn(&[Vec<f64>], f64) -> Vec<Vec<f64>>,
{
    pub fn new(n_vars: usize, ndof: usize, f: F) -> Self {
        Self { n_vars, ndof, f }
    }
}

impl<F> Semi for ClosureSemi<F>
where
    F: Fn(&[Vec<f64>], f64) -> Vec<Vec<f64>>,
{
    fn n_vars(&self) -> usize {
        self.n_vars
    }
    fn ndof(&self) -> usize {
        self.ndof
    }
    fn rhs(&self, state: &[Vec<f64>], t: f64) -> Vec<Vec<f64>> {
        (self.f)(state, t)
    }
}

/// A per-stage operation applied to the solution *between* Runge–Kutta stages —
/// the stage-callback granularity of `docs/api-design.md` §3.2/§3.4. This is the
/// home for operations of the form `u ← g(u)` that cannot be expressed as an
/// additive [`Term`](super::term::Term): limiters, positivity/entropy enforcement,
/// the SVV modal filter, and the implicit volume-penalization projection.
///
/// `stage` identifies which internal stage just completed (0-based), so a hook may
/// act selectively (e.g. only on the final stage).
pub trait StageHook {
    fn after_stage(&self, mesh: &Mesh2d, state: &mut [Vec<f64>], stage: usize);
}

/// The no-op stage hook (the implicit default of [`Integrator::step`]).
pub struct NoHook;

impl StageHook for NoHook {
    fn after_stage(&self, _mesh: &Mesh2d, _state: &mut [Vec<f64>], _stage: usize) {}
}

/// A time integrator: advances a state vector by one step starting at time `t`.
///
/// One trait, honored identically by every integrator family (explicit
/// method-of-lines here; structured/IMEX to come), so operators and callbacks
/// written once work across all of them.
pub trait Integrator {
    /// The fixed step size this integrator advances by.
    fn dt(&self) -> f64;
    /// Advance `state` (layout `[n_vars][ndof]`) by one step, returning the new
    /// state.
    fn step(&self, semi: &dyn Semi, state: &[Vec<f64>], t: f64) -> Vec<Vec<f64>>;
    /// Advance applying `hook` after each internal stage. The default applies the
    /// hook once to the completed step; multi-stage integrators override this to
    /// apply it between stages (the correct semantics for limiters/filters).
    fn step_with_hook(
        &self,
        semi: &dyn Semi,
        mesh: &Mesh2d,
        state: &[Vec<f64>],
        t: f64,
        hook: &dyn StageHook,
    ) -> Vec<Vec<f64>> {
        let mut u = self.step(semi, state, t);
        hook.after_stage(mesh, &mut u, 0);
        u
    }
}

/// Explicit three-stage, third-order strong-stability-preserving Runge–Kutta
/// (Shu–Osher). Byte-identical to the `step_ssp_rk3` previously inlined in each
/// regime.
#[derive(Clone, Copy, Debug)]
pub struct SspRk3 {
    pub dt: f64,
}

impl SspRk3 {
    pub fn new(dt: f64) -> Self {
        Self { dt }
    }
}

impl Integrator for SspRk3 {
    fn dt(&self) -> f64 {
        self.dt
    }

    fn step(&self, semi: &dyn Semi, state: &[Vec<f64>], t: f64) -> Vec<Vec<f64>> {
        let nv = semi.n_vars();
        let n = semi.ndof();
        let dt = self.dt;
        // Identical accumulation order to the original `comb` so results match to
        // the last bit.
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
        let k1 = semi.rhs(state, t);
        let u1 = comb(&[(1.0, state), (dt, &k1)]);
        let k2 = semi.rhs(&u1, t + dt);
        let u2 = comb(&[(0.75, state), (0.25, &u1), (0.25 * dt, &k2)]);
        let k3 = semi.rhs(&u2, t + 0.5 * dt);
        comb(&[(1.0 / 3.0, state), (2.0 / 3.0, &u2), (2.0 / 3.0 * dt, &k3)])
    }

    /// SSP-RK3 with `hook` applied after each of the three stage updates
    /// (`u1`, `u2`, final) — the Trixi stage-callback semantics.
    fn step_with_hook(
        &self,
        semi: &dyn Semi,
        mesh: &Mesh2d,
        state: &[Vec<f64>],
        t: f64,
        hook: &dyn StageHook,
    ) -> Vec<Vec<f64>> {
        let nv = semi.n_vars();
        let n = semi.ndof();
        let dt = self.dt;
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
        let k1 = semi.rhs(state, t);
        let mut u1 = comb(&[(1.0, state), (dt, &k1)]);
        hook.after_stage(mesh, &mut u1, 0);
        let k2 = semi.rhs(&u1, t + dt);
        let mut u2 = comb(&[(0.75, state), (0.25, &u1), (0.25 * dt, &k2)]);
        hook.after_stage(mesh, &mut u2, 1);
        let k3 = semi.rhs(&u2, t + 0.5 * dt);
        let mut u = comb(&[(1.0 / 3.0, state), (2.0 / 3.0, &u2), (2.0 / 3.0 * dt, &k3)]);
        hook.after_stage(mesh, &mut u, 2);
        u
    }
}

/// An **additively-split** semi-discrete operator for IMEX integration:
/// `∂ₜu = E(u) + S(u)`, with `E` non-stiff (explicit) and `S` stiff (implicit). The stiff
/// part is consumed through a *local solve* (`solve_implicit`) rather than a global system,
/// so stiffness never forces a global linear solve. Additive analogue of [`Semi`];
/// see `docs/plan-imex-relaxation-substep.md` §4.3.
pub trait ImexSemi {
    fn n_vars(&self) -> usize;
    fn ndof(&self) -> usize;
    /// Non-stiff (explicit) part `E(u)`.
    fn rhs_explicit(&self, state: &[Vec<f64>], t: f64) -> Vec<Vec<f64>>;
    /// Stiff source `S(u)` — needed only at explicit stages, where it cannot be recovered
    /// from a solve.
    fn rhs_implicit(&self, state: &[Vec<f64>], t: f64) -> Vec<Vec<f64>>;
    /// Solve the stiff stage `Y − γ·S(Y) = b` for `Y` (a local/elementwise solve).
    fn solve_implicit(&self, b: &[Vec<f64>], gamma: f64, t: f64) -> Vec<Vec<f64>>;
}

/// Butcher tableau pair for an additive (IMEX) Runge–Kutta method: an explicit table
/// `(a_e, b_e)` for the non-stiff part and a diagonally-implicit table `(a_i, b_i)` for the
/// stiff part, sharing abscissae. `a_i[i][i]` is the implicit-stage coefficient `γᵢ`.
#[derive(Clone, Debug)]
pub struct ArkTableau {
    pub a_e: Vec<Vec<f64>>,
    pub a_i: Vec<Vec<f64>>,
    pub b_e: Vec<f64>,
    pub b_i: Vec<f64>,
}

impl ArkTableau {
    /// ARS(2,2,2) (Ascher–Ruuth–Spiteri 1997): L-stable, 2nd-order, stiffly accurate.
    /// `γ = 1 − √2/2`, `δ = 1 − 1/(2γ)`.
    pub fn ars222() -> Self {
        let g = 1.0 - 0.5_f64.sqrt();
        let d = 1.0 - 1.0 / (2.0 * g);
        Self {
            a_e: vec![vec![0.0, 0.0, 0.0], vec![g, 0.0, 0.0], vec![d, 1.0 - d, 0.0]],
            a_i: vec![vec![0.0, 0.0, 0.0], vec![0.0, g, 0.0], vec![0.0, 1.0 - g, g]],
            b_e: vec![d, 1.0 - d, 0.0],
            b_i: vec![0.0, 1.0 - g, g],
        }
    }
    pub fn stages(&self) -> usize {
        self.b_e.len()
    }
}

/// Additive IMEX Runge–Kutta integrator over an [`ImexSemi`]. Drives the generic stage
/// recursion of `docs/plan-imex-relaxation-substep.md` §2.3, reusing the operator's local
/// implicit solve. For an implicit stage the stiff value is recovered exactly as
/// `S_i = (Y_i − B_i)/γ` (free); explicit stages fall back to `rhs_implicit`. (It is its own
/// integrator rather than an [`Integrator`] impl because it consumes an `ImexSemi`, not a
/// `Semi`.)
#[derive(Clone, Debug)]
pub struct ArkImex {
    pub tableau: ArkTableau,
    pub dt: f64,
}

impl ArkImex {
    pub fn new(tableau: ArkTableau, dt: f64) -> Self {
        Self { tableau, dt }
    }
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Advance `state` (layout `[n_vars][ndof]`) by one IMEX step.
    pub fn step(&self, semi: &dyn ImexSemi, state: &[Vec<f64>], t: f64) -> Vec<Vec<f64>> {
        let s = self.tableau.stages();
        let (nv, n, dt) = (semi.n_vars(), semi.ndof(), self.dt);
        let axpy = |dst: &mut [Vec<f64>], c: f64, src: &[Vec<f64>]| {
            if c != 0.0 {
                for v in 0..nv {
                    for k in 0..n {
                        dst[v][k] += c * src[v][k];
                    }
                }
            }
        };
        let mut e_stage: Vec<Vec<Vec<f64>>> = Vec::with_capacity(s);
        let mut s_stage: Vec<Vec<Vec<f64>>> = Vec::with_capacity(s);
        for i in 0..s {
            // Bᵢ = state + dt Σ_{j<i} (a_e[i][j] E_j + a_i[i][j] S_j)
            let mut b = state.to_vec();
            for j in 0..i {
                axpy(&mut b, dt * self.tableau.a_e[i][j], &e_stage[j]);
                axpy(&mut b, dt * self.tableau.a_i[i][j], &s_stage[j]);
            }
            let gamma = self.tableau.a_i[i][i];
            let (y, si) = if gamma == 0.0 {
                let si = semi.rhs_implicit(&b, t); // explicit stage
                (b, si)
            } else {
                let g = dt * gamma;
                let y = semi.solve_implicit(&b, g, t);
                let si: Vec<Vec<f64>> = (0..nv)
                    .map(|v| (0..n).map(|k| (y[v][k] - b[v][k]) / g).collect())
                    .collect();
                (y, si)
            };
            e_stage.push(semi.rhs_explicit(&y, t));
            s_stage.push(si);
        }
        // uⁿ⁺¹ = state + dt Σ_i (b_e[i] E_i + b_i[i] S_i)
        let mut out = state.to_vec();
        for i in 0..s {
            axpy(&mut out, dt * self.tableau.b_e[i], &e_stage[i]);
            axpy(&mut out, dt * self.tableau.b_i[i], &s_stage[i]);
        }
        out
    }
}

/// **Relaxation Runge–Kutta** parameter (Ranocha et al., SISC 2020): given the old functional value
/// `eta_old`, the per-step production estimate `prod` (so the target functional is `eta_old + γ·prod`),
/// and an evaluator `eval(γ)` of the functional at the relaxed state `γ·uₙ₊₁ + (1−γ)·uₙ`, returns the
/// relaxation `γ` solving `eval(γ) = eta_old + γ·prod` for the non-trivial root near `γ = 1`. Enforces
/// the discrete balance of **any convex functional** (energy, viscoelastic free energy, …) at the cost
/// of one scalar root-find. `γ = 0` is always a trivial root, so the search scans the window `(0, 2]`
/// (excluding 0); if no sign change is found (no admissible relaxation) it returns `1.0` (the plain step).
/// See `docs/plan-free-energy-compatible.md`. NOTE: relaxation only corrects *time-integration* entropy
/// leakage — it presupposes an entropy-stable *spatial* operator.
pub fn relaxation_gamma(eval: impl Fn(f64) -> f64, eta_old: f64, prod: f64) -> f64 {
    let r = |g: f64| eval(g) - (eta_old + g * prod);
    let (lo_bound, hi_bound, n) = (1e-3, 2.0, 256usize);
    let (mut prev_g, mut prev_r) = (lo_bound, r(lo_bound));
    let mut bracket = None;
    for i in 1..=n {
        let g = lo_bound + (hi_bound - lo_bound) * (i as f64) / (n as f64);
        let rg = r(g);
        if prev_r * rg <= 0.0 {
            bracket = Some((prev_g, g));
            break;
        }
        prev_g = g;
        prev_r = rg;
    }
    let (mut a, mut b) = match bracket {
        Some(x) => x,
        None => return 1.0,
    };
    let mut ra = r(a);
    for _ in 0..80 {
        let m = 0.5 * (a + b);
        let rm = r(m);
        if ra * rm <= 0.0 {
            b = m;
        } else {
            a = m;
            ra = rm;
        }
    }
    0.5 * (a + b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dg::hyperbolic::{Burgers, Hyperbolic};
    use crate::dg::mesh::Mesh2d;

    /// SspRk3 driven by a ClosureSemi wrapping `Hyperbolic::rhs` must reproduce
    /// `Hyperbolic::step_ssp_rk3` exactly, step after step.
    #[test]
    fn ssprk3_matches_inlined_stepper_bit_for_bit() {
        let mesh = Mesh2d::rectangular_periodic(3, 4, 4, [0.0, 1.0], [0.0, 1.0]);
        let hyp = Hyperbolic::new(&mesh, Burgers);
        let nn = mesh.refq.n_nodes();

        // A non-trivial smooth initial condition.
        let ndof = hyp.ndof();
        let mut s_ref: Vec<Vec<f64>> = vec![vec![0.0; ndof]];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let (x, y) = (el.geom.x[k], el.geom.y[k]);
                s_ref[0][e * nn + k] =
                    (2.0 * std::f64::consts::PI * x).sin() * (2.0 * std::f64::consts::PI * y).cos();
            }
        }
        let mut s_new = s_ref.clone();

        let bc = |_x: f64, _y: f64, _t: f64, _out: &mut [f64]| {};
        let semi = ClosureSemi::new(hyp.n_vars(), hyp.ndof(), |s: &[Vec<f64>], t: f64| {
            hyp.rhs(s, t, &bc)
        });
        let integ = SspRk3::new(1e-3);

        let mut t = 0.0;
        for _ in 0..25 {
            s_ref = hyp.step_ssp_rk3(&s_ref, t, integ.dt(), &bc);
            s_new = integ.step(&semi, &s_new, t);
            t += integ.dt();
            // Exact equality — same arithmetic, same order.
            assert_eq!(s_new, s_ref);
        }
        // Sanity: it actually evolved.
        assert!(s_ref[0].iter().any(|&v| v.abs() > 1e-6));
    }

    #[test]
    fn relaxation_gamma_matches_quadratic_closed_form() {
        // Quadratic functional η(u)=½u² along u(γ)=1+γ: r(γ)=η(u(γ))−½−γ·prod has non-trivial root
        // γ* = 2(prod−1). Verify the solver finds it, satisfies the equation, and returns 1 when
        // γ=1 is the root.
        let eval = |g: f64| 0.5 * (1.0 + g) * (1.0 + g);
        let eta_old = 0.5; // η(u(0)) = ½·1²
        for &prod in &[1.25_f64, 1.5, 1.75] {
            let g = relaxation_gamma(eval, eta_old, prod);
            let want = 2.0 * (prod - 1.0);
            assert!((g - want).abs() < 1e-9, "γ={g}, want {want} (prod={prod})");
            // residual of the relaxation equation
            assert!((eval(g) - (eta_old + g * prod)).abs() < 1e-9, "relaxation residual");
        }
        // prod = 1.5 ⇒ γ* = 1 exactly (no relaxation needed).
        assert!((relaxation_gamma(eval, eta_old, 1.5) - 1.0).abs() < 1e-9);
        // prod < 1 ⇒ γ* = 2(prod−1) < 0, outside (0,2] ⇒ no admissible relaxation ⇒ γ = 1.
        assert!((relaxation_gamma(eval, eta_old, 0.5) - 1.0).abs() < 1e-12);
    }
}
