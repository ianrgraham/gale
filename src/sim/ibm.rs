//! Immersed-boundary operations for the framework.
//!
//! The four-homes rule (`docs/api-design.md` §3.4) places the *implicit*
//! volume-penalization (Brinkman) forcing in a [`StateStageHook`], not a `Term`,
//! because it is a relaxation `u ← (u + β u_s)/(1 + β)` rather than an additive
//! rhs contribution. [`PenalizationHook`] is that home; it is applied after each
//! integrator stage (e.g. the post-step hook of
//! [`DualSplitting`](super::dynamics::DualSplitting)). IBM diagnostics — the
//! hydrodynamic drag — are exposed as a [`Compute`].
//!
//! Both wrap the validated [`VolumePenalization`] operator unchanged.

use super::dynamics::StateStageHook;
use super::field::FieldId;
use super::simulation::Compute;
use super::state::State;
use crate::dg::immersed::VolumePenalization;
use crate::dg::immersed3d::VolumePenalization3d;
use crate::dg::mesh3d::Mesh3d;

/// Applies implicit volume penalization to the velocity field after a stage,
/// driving the fluid toward the solid velocity inside the immersed body. This is
/// the IBM no-slip enforcement as a per-stage `u ← g(u)` operation.
pub struct PenalizationHook {
    velocity: FieldId,
    penal: VolumePenalization,
    dt: f64,
}

impl PenalizationHook {
    /// Penalize the 2-component `velocity` field with `penal` over a step `dt`
    /// (must match the integrator's step size: β = (dt/η_b)·χ).
    pub fn new(velocity: FieldId, penal: VolumePenalization, dt: f64) -> Self {
        Self { velocity, penal, dt }
    }
}

impl StateStageHook for PenalizationHook {
    fn after_stage(&self, state: &mut State, _stage: usize) {
        let comps = state.fields.by_id_mut(self.velocity).components_mut();
        let (ux, uy) = comps.split_at_mut(1);
        self.penal.apply(&mut ux[0], &mut uy[0], self.dt);
    }
}

/// Reports one component of the hydrodynamic force the fluid exerts on an immersed
/// body, `F = ∫ (χ/η_b)(u − u_s) dV` — the penalization drag, consistent with
/// [`PenalizationHook`] when evaluated on the penalized velocity.
pub struct PenalizationDrag {
    name: String,
    velocity: FieldId,
    penal: VolumePenalization,
    /// 0 for `Fx`, 1 for `Fy`.
    axis: usize,
}

impl PenalizationDrag {
    pub fn new(name: impl Into<String>, velocity: FieldId, penal: VolumePenalization, axis: usize) -> Self {
        Self { name: name.into(), velocity, penal, axis }
    }
}

impl Compute for PenalizationDrag {
    fn name(&self) -> &str {
        &self.name
    }
    fn compute(&self, state: &State) -> f64 {
        let v = state.fields.by_id(self.velocity);
        let (fx, fy) = self.penal.force(v.component(0), v.component(1), &state.mesh);
        if self.axis == 0 {
            fx
        } else {
            fy
        }
    }
}

/// 3D volume-penalization stage hook (`StateStageHook<Mesh3d>`): the 3D analogue of
/// [`PenalizationHook`], damping the 3-component velocity field toward the solid.
pub struct Penalization3dHook {
    velocity: FieldId,
    penal: VolumePenalization3d,
    dt: f64,
}

impl Penalization3dHook {
    pub fn new(velocity: FieldId, penal: VolumePenalization3d, dt: f64) -> Self {
        Self { velocity, penal, dt }
    }
}

impl StateStageHook<Mesh3d> for Penalization3dHook {
    fn after_stage(&self, state: &mut State<Mesh3d>, _stage: usize) {
        let comps = state.fields.by_id_mut(self.velocity).components_mut();
        let (ux, rest) = comps.split_at_mut(1);
        let (uy, uz) = rest.split_at_mut(1);
        self.penal.apply(&mut ux[0], &mut uy[0], &mut uz[0], self.dt);
    }
}

/// 3D hydrodynamic-drag compute (`Compute<Mesh3d>`): one component of
/// `F = ∫(χ/η_b)(u − u_s)`.
pub struct Penalization3dDrag {
    name: String,
    velocity: FieldId,
    penal: VolumePenalization3d,
    /// 0 = Fx, 1 = Fy, 2 = Fz.
    axis: usize,
}

impl Penalization3dDrag {
    pub fn new(name: impl Into<String>, velocity: FieldId, penal: VolumePenalization3d, axis: usize) -> Self {
        Self { name: name.into(), velocity, penal, axis }
    }
}

impl Compute<Mesh3d> for Penalization3dDrag {
    fn name(&self) -> &str {
        &self.name
    }
    fn compute(&self, state: &State<Mesh3d>) -> f64 {
        let v = state.fields.by_id(self.velocity);
        let (fx, fy, fz) = self.penal.force(v.component(0), v.component(1), v.component(2), &state.mesh);
        [fx, fy, fz][self.axis]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dg::immersed::{Disk, VolumePenalization};
    use crate::dg::immersed3d::Sphere;
    use crate::dg::mesh::Mesh2d;
    use crate::sim::simulation::Compute;

    /// The 3D penalization hook damps the velocity field inside an immersed sphere
    /// through the `StateStageHook<Mesh3d>` seam, and the drag compute reports a
    /// finite positive Fx.
    #[test]
    fn penalization_3d_hook_damps_and_drag_reports() {
        let mesh = Mesh3d::rectangular(3, 4, 4, 4, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
        let sphere = Sphere::new(0.5, 0.5, 0.5, 0.2);
        let (eta_b, dt) = (1e-4, 0.01);
        let penal = VolumePenalization3d::new(&mesh, &sphere, eta_b);

        let mut st: State<Mesh3d> = State::new(mesh);
        let vid = st.add_field("velocity", 3);
        for u in st.field_mut("velocity").component_mut(0).iter_mut() {
            *u = 1.0;
        }
        let hook = Penalization3dHook::new(vid, penal.clone(), dt);
        hook.after_stage(&mut st, 0);

        let v = st.field("velocity");
        for i in 0..penal.mask.len() {
            if penal.mask[i] > 0.5 {
                assert!(v.component(0)[i].abs() < 0.05, "solid node not damped: {}", v.component(0)[i]);
            } else {
                assert!((v.component(0)[i] - 1.0).abs() < 1e-12, "exterior perturbed");
            }
        }

        let drag = Penalization3dDrag::new("drag_x", vid, penal, 0);
        let fx = drag.compute(&st);
        assert_eq!(drag.name(), "drag_x");
        assert!(fx.is_finite() && fx > 0.0, "drag {fx}");
    }

    /// The hook must reproduce a direct `VolumePenalization::apply` bit-for-bit
    /// (faithful wiring), strongly damp the velocity inside the solid, and leave
    /// the exterior untouched.
    #[test]
    fn penalization_hook_matches_apply_and_damps_solid() {
        let mesh = Mesh2d::rectangular(4, 4, 4, [0.0, 1.0], [0.0, 1.0]);
        let disk = Disk::new(0.5, 0.5, 0.2);
        let (eta_b, dt) = (1e-4, 0.01);

        let mut st = State::new(mesh.clone());
        let vid = st.add_field("velocity", 2);
        // Uniform stream u = (1, 0).
        for u in st.field_mut("velocity").component_mut(0).iter_mut() {
            *u = 1.0;
        }
        let ndof = st.ndof();

        let penal = VolumePenalization::new(&mesh, &disk, eta_b);
        let hook = PenalizationHook::new(vid, penal.clone(), dt);

        // Reference: direct apply on copies of the same initial state.
        let mut rux = vec![1.0; ndof];
        let mut ruy = vec![0.0; ndof];
        penal.apply(&mut rux, &mut ruy, dt);

        hook.after_stage(&mut st, 0);

        // Bit-for-bit faithful wrapping.
        assert_eq!(st.field("velocity").component(0), rux.as_slice());
        assert_eq!(st.field("velocity").component(1), ruy.as_slice());

        // Physics: solid interior strongly damped; exterior unchanged.
        let v = st.field("velocity");
        let mut solid_nodes = 0;
        let mut fluid_nodes = 0;
        for i in 0..ndof {
            if penal.mask[i] > 0.5 {
                solid_nodes += 1;
                assert!(v.component(0)[i].abs() < 0.05, "solid node not damped: {}", v.component(0)[i]);
            } else {
                fluid_nodes += 1;
                assert!((v.component(0)[i] - 1.0).abs() < 1e-12, "exterior perturbed");
            }
        }
        assert!(solid_nodes > 0 && fluid_nodes > 0, "mask degenerate");
    }

    /// End-to-end: a `DualSplitting` flow past a penalized disk, assembled in a
    /// `Simulation` with the `PenalizationHook` as its stage hook. Differential
    /// check — the with-hook run must damp the disk interior relative to an
    /// identical no-hook run — proving the integrator actually invokes the stage
    /// hook and the IBM enforces (partial) no-slip through the full assembly.
    #[test]
    fn penalization_hook_damps_through_dual_splitting_simulation() {
        use crate::dg::immersed::Disk;
        use crate::sim::dynamics::DualSplitting;
        use crate::sim::Simulation;

        let mesh = Mesh2d::rectangular(4, 6, 6, [0.0, 1.0], [0.0, 1.0]);
        let disk = Disk::new(0.5, 0.5, 0.18);
        let (eta_b, dt, nu, nsteps) = (1e-3, 5e-3, 0.1, 12u64);
        let penal = VolumePenalization::new(&mesh, &disk, eta_b);

        // Uniform stream u = (1, 0): driven on all walls, initialized in the interior.
        let build = || {
            let mut st = State::new(mesh.clone());
            let vid = st.add_field("velocity", 2);
            for u in st.field_mut("velocity").component_mut(0).iter_mut() {
                *u = 1.0;
            }
            let integ = DualSplitting::new(vid, dt, nu, 5.0)
                .boundary(|_x, _y, _t| 1.0, |_x, _y, _t| 0.0);
            let mut sim = Simulation::new(st);
            sim.set_integrator(integ);
            (sim, vid)
        };

        let interior_mean = |sim: &Simulation, vid: FieldId| -> f64 {
            let v = sim.state.fields.by_id(vid);
            let (mut s, mut n) = (0.0, 0.0);
            for i in 0..penal.mask.len() {
                if penal.mask[i] > 0.5 {
                    s += v.component(0)[i].abs();
                    n += 1.0;
                }
            }
            s / n
        };

        // With the penalization hook.
        let (mut sim_hook, vid) = build();
        sim_hook.set_stage_hook(PenalizationHook::new(vid, penal.clone(), dt));
        sim_hook.add_compute(PenalizationDrag::new("drag_x", vid, penal.clone(), 0));
        sim_hook.run(nsteps);

        // Without (default no-op hook), identical otherwise.
        let (mut sim_free, _) = build();
        sim_free.run(nsteps);

        let damped = interior_mean(&sim_hook, vid);
        let free = interior_mean(&sim_free, vid);
        assert!(damped.is_finite() && free.is_finite());
        assert!(damped < 0.5 * free, "hook did not damp the interior: {damped} vs free {free}");

        let drag = sim_hook.compute("drag_x").unwrap();
        assert!(drag.is_finite() && drag > 0.0, "expected positive drag, got {drag}");
    }

    /// The drag compute returns a finite, positive Fx for a uniform stream past a
    /// static disk (the fluid pushes the body downstream).
    #[test]
    fn penalization_drag_reports_positive_force() {
        let mesh = Mesh2d::rectangular(4, 4, 4, [0.0, 1.0], [0.0, 1.0]);
        let disk = Disk::new(0.5, 0.5, 0.2);
        let penal = VolumePenalization::new(&mesh, &disk, 1e-3);

        let mut st = State::new(mesh);
        let vid = st.add_field("velocity", 2);
        for u in st.field_mut("velocity").component_mut(0).iter_mut() {
            *u = 1.0;
        }

        let drag = PenalizationDrag::new("drag_x", vid, penal, 0);
        let fx = drag.compute(&st);
        assert_eq!(drag.name(), "drag_x");
        assert!(fx.is_finite() && fx > 0.0, "expected positive drag, got {fx}");
    }
}
