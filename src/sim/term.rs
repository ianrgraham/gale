//! Additive physics [`Term`]s and the [`Semidiscretization`] that sums them onto
//! a base spatial operator.
//!
//! A [`Term`] is gale's additive composability primitive (`docs/api-design.md`
//! §3.2): it *accumulates* (`+=`) its contribution into the rhs, never overwrites.
//! Convection stays in the base operator (the hyperbolic flux — see the four-homes
//! rule, api-design §3.4); `Term`s are the *additional* physics summed on top:
//! sources, body forces, and (once the State-level semidiscretization lands)
//! cross-field couplings.
//!
//! [`Semidiscretization`] bundles a base rhs (any [`Semi`]) with an ordered list
//! of `Term`s and itself implements [`Semi`], so an [`Integrator`] drives the
//! composed operator transparently.
//!
//! **Scope (this migration step).** `Term`s here act at the single-equation
//! state-vector level (`[n_vars][ndof]`): they read the equation's own state +
//! mesh and add to its rhs. That covers source/body-force terms (the clean
//! additive case). Cross-field coupling (e.g. polymer-stress divergence reading
//! the conformation field to force momentum) and the *implicit* volume-penalization
//! StageHook require the multi-field `State`-level semidiscretization and are
//! deferred to the next step, exactly as the four-homes rule (api-design §3.4)
//! dictates — penalization is a StageHook, not a `Term`.

use super::integrate::Semi;
use crate::dg::mesh::Mesh2d;

/// An additive contribution to a method-of-lines rhs.
///
/// Implementations **must** accumulate (`dudt[v][i] += …`) so that terms compose
/// by summation and order does not change the result.
pub trait Term {
    /// Add this term's contribution to `dudt`, given the current `state` (layout
    /// `[n_vars][ndof]`) on `mesh` at time `t`.
    fn accumulate(&self, mesh: &Mesh2d, state: &[Vec<f64>], t: f64, dudt: &mut [Vec<f64>]);
}

/// A space/time source term `s(x, y, t)` added directly to `∂ₜu`.
///
/// The closure fills the per-variable source at a node; it is sampled at every
/// collocation node and added to the rhs. Because the base operator already
/// returns `∂ₜu = M⁻¹(…)` (mass-inverted, collocated), the nodal source enters
/// the rhs directly with no extra mass-matrix application.
pub struct SourceTerm<F> {
    pub n_vars: usize,
    pub source: F,
}

impl<F> SourceTerm<F>
where
    F: Fn(f64, f64, f64, &mut [f64]),
{
    pub fn new(n_vars: usize, source: F) -> Self {
        Self { n_vars, source }
    }
}

impl<F> Term for SourceTerm<F>
where
    F: Fn(f64, f64, f64, &mut [f64]),
{
    fn accumulate(&self, mesh: &Mesh2d, _state: &[Vec<f64>], t: f64, dudt: &mut [Vec<f64>]) {
        let nn = mesh.refq.n_nodes();
        let mut s = vec![0.0; self.n_vars];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let (x, y) = (el.geom.x[k], el.geom.y[k]);
                for v in 0..self.n_vars {
                    s[v] = 0.0;
                }
                (self.source)(x, y, t, &mut s);
                for v in 0..self.n_vars {
                    dudt[v][e * nn + k] += s[v];
                }
            }
        }
    }
}

/// A spatial operator composed as `base rhs + Σ terms`, implementing [`Semi`].
///
/// `base` supplies the equation's flux divergence (convection, and for the
/// compressible/parabolic regimes its own diffusion); `terms` are the additive
/// extras. The composite is itself a [`Semi`], so it plugs into any [`Integrator`]
/// (`super::Integrator`).
pub struct Semidiscretization<'m, B: Semi> {
    pub mesh: &'m Mesh2d,
    pub base: B,
    pub terms: Vec<Box<dyn Term + 'm>>,
}

impl<'m, B: Semi> Semidiscretization<'m, B> {
    /// Start from a base operator with no extra terms.
    pub fn new(mesh: &'m Mesh2d, base: B) -> Self {
        Self { mesh, base, terms: Vec::new() }
    }

    /// Append an additive term (builder style).
    pub fn with_term(mut self, term: impl Term + 'm) -> Self {
        self.terms.push(Box::new(term));
        self
    }

    /// Append an additive term in place.
    pub fn add_term(&mut self, term: impl Term + 'm) {
        self.terms.push(Box::new(term));
    }
}

impl<'m, B: Semi> Semi for Semidiscretization<'m, B> {
    fn n_vars(&self) -> usize {
        self.base.n_vars()
    }
    fn ndof(&self) -> usize {
        self.base.ndof()
    }
    fn rhs(&self, state: &[Vec<f64>], t: f64) -> Vec<Vec<f64>> {
        // Base rhs, then accumulate each additive term onto it.
        let mut dudt = self.base.rhs(state, t);
        for term in &self.terms {
            term.accumulate(self.mesh, state, t, &mut dudt);
        }
        dudt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dg::hyperbolic::{Hyperbolic, LinearAdvection};
    use crate::dg::mesh::Mesh2d;
    use crate::sim::integrate::{ClosureSemi, Integrator, SspRk3};
    use std::f64::consts::PI;

    /// Method of manufactured solutions for `∂ₜu + a·∇u = s`, periodic on the unit
    /// square. With `u = φ(x,y)·g(t)`, `φ = sin(2πx)cos(2πy)`, `g = e^{-t}`, the
    /// source `s = φ g' + (aₓ φₓ + a_y φ_y) g` is supplied as a `SourceTerm`.
    /// The composed Semidiscretization integrated with SspRk3 must recover the
    /// exact solution to discretization accuracy — and a control run *without* the
    /// term must not.
    #[test]
    fn manufactured_source_term_recovers_exact_solution() {
        let (ax, ay) = (1.0, 0.5);
        let mesh = Mesh2d::rectangular_periodic(6, 8, 8, [0.0, 1.0], [0.0, 1.0]);
        let hyp = Hyperbolic::new(&mesh, LinearAdvection { ax, ay });
        let nn = mesh.refq.n_nodes();
        let ndof = hyp.ndof();

        let phi = |x: f64, y: f64| (2.0 * PI * x).sin() * (2.0 * PI * y).cos();
        let phix = |x: f64, y: f64| 2.0 * PI * (2.0 * PI * x).cos() * (2.0 * PI * y).cos();
        let phiy = |x: f64, y: f64| -2.0 * PI * (2.0 * PI * x).sin() * (2.0 * PI * y).sin();
        let g = |t: f64| (-t).exp();
        let gp = |t: f64| -(-t).exp();
        let exact = |x: f64, y: f64, t: f64| phi(x, y) * g(t);

        // Initial condition u(·,0) = φ.
        let mut u0 = vec![vec![0.0; ndof]];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                u0[0][e * nn + k] = exact(el.geom.x[k], el.geom.y[k], 0.0);
            }
        }

        let bc = |_x: f64, _y: f64, _t: f64, _o: &mut [f64]| {};

        // Composed operator: advection base + manufactured source term.
        let base = ClosureSemi::new(hyp.n_vars(), hyp.ndof(), |s: &[Vec<f64>], t: f64| {
            hyp.rhs(s, t, &bc)
        });
        let semi = Semidiscretization::new(&mesh, base).with_term(SourceTerm::new(
            1,
            move |x: f64, y: f64, t: f64, out: &mut [f64]| {
                out[0] = phi(x, y) * gp(t) + (ax * phix(x, y) + ay * phiy(x, y)) * g(t);
            },
        ));

        let integ = SspRk3::new(2e-4);
        let nsteps = 500; // T = 0.1
        let mut u = u0.clone();
        let mut t = 0.0;
        for _ in 0..nsteps {
            u = integ.step(&semi, &u, t);
            t += integ.dt();
        }
        let tf = integ.dt() * nsteps as f64;

        // L2 error of the composed run vs the exact manufactured solution.
        let mut err2 = 0.0;
        let mut nrm2 = 0.0;
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let ue = exact(el.geom.x[k], el.geom.y[k], tf);
                let d = u[0][e * nn + k] - ue;
                err2 += el.geom.jw[k] * d * d;
                nrm2 += el.geom.jw[k] * ue * ue;
            }
        }
        let rel = (err2 / nrm2).sqrt();
        assert!(rel < 1e-4, "MMS relative L2 error too large: {rel:e}");

        // Control: identical run WITHOUT the source term must be far off — proving
        // the term is both correct and necessary.
        let base2 = ClosureSemi::new(hyp.n_vars(), hyp.ndof(), |s: &[Vec<f64>], t: f64| {
            hyp.rhs(s, t, &bc)
        });
        let semi_ctrl = Semidiscretization::new(&mesh, base2);
        let mut uc = u0.clone();
        let mut t = 0.0;
        for _ in 0..nsteps {
            uc = integ.step(&semi_ctrl, &uc, t);
            t += integ.dt();
        }
        let mut cerr2 = 0.0;
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let ue = exact(el.geom.x[k], el.geom.y[k], tf);
                let d = uc[0][e * nn + k] - ue;
                cerr2 += el.geom.jw[k] * d * d;
            }
        }
        let crel = (cerr2 / nrm2).sqrt();
        assert!(crel > 1e-2, "control should be far from exact, got {crel:e}");
    }
}
