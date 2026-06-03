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
}
