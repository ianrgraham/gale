//! Concrete [`StageHook`]s — per-stage `u ← g(u)` operations.
//!
//! These are the operations the four-homes rule (`docs/api-design.md` §3.4) places
//! between RK stages rather than in the additive rhs: the SVV modal filter and the
//! implicit volume-penalization projection. Each wraps a validated `dg` operator
//! and applies it through the [`StageHook`] seam, so any [`Integrator`] picks it up
//! via `step_with_hook` without bespoke loop code.

use super::integrate::StageHook;
use crate::dg::filter::ModalFilter;
use crate::dg::mesh::Mesh2d;

/// Applies a spectral-vanishing-viscosity [`ModalFilter`] to every component of
/// the solution after each stage. The filter is a linear, mass-conserving
/// `u ← Fu`, which is exactly why it is a stage hook and not a `Term`.
pub struct FilterHook {
    pub filter: ModalFilter,
    /// Apply only on this stage if `Some` (e.g. the final stage); apply on every
    /// stage if `None`.
    pub only_stage: Option<usize>,
}

impl FilterHook {
    /// Filter on every stage.
    pub fn every_stage(filter: ModalFilter) -> Self {
        Self { filter, only_stage: None }
    }

    /// Filter only after the given stage index.
    pub fn on_stage(filter: ModalFilter, stage: usize) -> Self {
        Self { filter, only_stage: Some(stage) }
    }
}

impl StageHook for FilterHook {
    fn after_stage(&self, _mesh: &Mesh2d, state: &mut [Vec<f64>], stage: usize) {
        if let Some(s) = self.only_stage {
            if s != stage {
                return;
            }
        }
        for comp in state.iter_mut() {
            *comp = self.filter.apply(comp);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dg::hyperbolic::{Burgers, Hyperbolic};
    use crate::dg::mesh::Mesh2d;
    use crate::sim::integrate::{ClosureSemi, Integrator, SspRk3};
    use std::f64::consts::PI;

    fn high_freq_ic(mesh: &Mesh2d, ndof: usize) -> Vec<Vec<f64>> {
        let nn = mesh.refq.n_nodes();
        let mut u = vec![vec![0.0; ndof]];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let (x, y) = (el.geom.x[k], el.geom.y[k]);
                // A smooth base plus a high-wavenumber wrinkle the filter should damp.
                u[0][e * nn + k] = (2.0 * PI * x).sin() * (2.0 * PI * y).cos()
                    + 0.25 * (8.0 * PI * x).sin() * (8.0 * PI * y).cos();
            }
        }
        u
    }

    /// `SspRk3::step_with_hook` with a `FilterHook` must reproduce a hand-written
    /// SSP-RK3 loop that filters after each stage — bit for bit.
    #[test]
    fn filter_hook_matches_manual_per_stage_filtering() {
        let mesh = Mesh2d::rectangular_periodic(6, 4, 4, [0.0, 1.0], [0.0, 1.0]);
        let hyp = Hyperbolic::new(&mesh, Burgers);
        let ndof = hyp.ndof();
        let bc = |_x: f64, _y: f64, _t: f64, _o: &mut [f64]| {};
        let semi =
            ClosureSemi::new(hyp.n_vars(), hyp.ndof(), |s: &[Vec<f64>], t: f64| hyp.rhs(s, t, &bc));
        let integ = SspRk3::new(1e-3);
        let filt = ModalFilter::new(mesh.order, mesh.order - 1, 36.0, 8.0);
        let hook = FilterHook::every_stage(filt.clone());

        // Manual reference: replicate the Shu–Osher stages and filter each.
        let manual_step = |state: &[Vec<f64>], t: f64| -> Vec<Vec<f64>> {
            let dt = integ.dt();
            let filt_all = |mut u: Vec<Vec<f64>>| {
                for c in u.iter_mut() {
                    *c = filt.apply(c);
                }
                u
            };
            let k1 = hyp.rhs(state, t, &bc);
            let u1 = filt_all(
                (0..1).map(|v| (0..ndof).map(|i| state[v][i] + dt * k1[v][i]).collect()).collect(),
            );
            let k2 = hyp.rhs(&u1, t + dt, &bc);
            let u2 = filt_all(
                (0..1)
                    .map(|v| {
                        (0..ndof)
                            .map(|i| 0.75 * state[v][i] + 0.25 * u1[v][i] + 0.25 * dt * k2[v][i])
                            .collect()
                    })
                    .collect(),
            );
            let k3 = hyp.rhs(&u2, t + 0.5 * dt, &bc);
            filt_all(
                (0..1)
                    .map(|v| {
                        (0..ndof)
                            .map(|i| {
                                (1.0 / 3.0) * state[v][i]
                                    + (2.0 / 3.0) * u2[v][i]
                                    + (2.0 / 3.0 * dt) * k3[v][i]
                            })
                            .collect()
                    })
                    .collect(),
            )
        };

        let mut a = high_freq_ic(&mesh, ndof);
        let mut b = a.clone();
        let mut t = 0.0;
        for _ in 0..20 {
            a = integ.step_with_hook(&semi, &mesh, &a, t, &hook);
            b = manual_step(&b, t);
            t += integ.dt();
            assert_eq!(a, b);
        }
    }

    /// Isolated hook property: with a *zero* rhs the step reduces to repeated
    /// application of the filter, which is a contraction (σ ≤ 1 eigenvalues). So
    /// energy must be non-increasing, and strictly decrease when a top mode is
    /// present — proving the hook actually damps and is wired into every stage.
    #[test]
    fn filter_hook_is_a_contraction_under_zero_dynamics() {
        let mesh = Mesh2d::rectangular_periodic(6, 4, 4, [0.0, 1.0], [0.0, 1.0]);
        let ndof = mesh.n_elements() * mesh.refq.n_nodes();
        // A semidiscretization whose rhs is identically zero: only the hook acts.
        let zero =
            ClosureSemi::new(1, ndof, |s: &[Vec<f64>], _t: f64| vec![vec![0.0; s[0].len()]]);
        let integ = SspRk3::new(1e-3);
        let filt = ModalFilter::new(mesh.order, mesh.order - 1, 36.0, 8.0);
        let hook = FilterHook::every_stage(filt);

        let energy = |u: &[Vec<f64>]| -> f64 {
            let nn = mesh.refq.n_nodes();
            let mut s = 0.0;
            for (e, el) in mesh.elements.iter().enumerate() {
                for k in 0..nn {
                    s += el.geom.jw[k] * u[0][e * nn + k].powi(2);
                }
            }
            s
        };

        let mut u = high_freq_ic(&mesh, ndof);
        let e0 = energy(&u);
        let mut prev = e0;
        let mut t = 0.0;
        for _ in 0..30 {
            u = integ.step_with_hook(&zero, &mesh, &u, t, &hook);
            let e = energy(&u);
            assert!(e <= prev + 1e-14, "energy must not increase: {e:e} > {prev:e}");
            prev = e;
            t += integ.dt();
        }
        // The top-mode wrinkle is damped, so total energy strictly fell.
        assert!(prev < e0, "filter should have removed top-mode energy: {prev:e} !< {e0:e}");
    }
}
