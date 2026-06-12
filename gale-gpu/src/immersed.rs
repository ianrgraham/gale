//! GPU immersed-boundary volume penalization — reusable library component.
//!
//! The implicit Brinkman relaxation `u ← (u + β·u_s)/(1+β)`, `β = χ·dt/η_b`, applied
//! per node. Embarrassingly parallel, libdevice-free. Validated bit-for-bit against
//! `gale::dg::VolumePenalization::apply`.
//!
//! This is a *second* `#[cuda_module]` in the gale-gpu crate (alongside
//! [`crate::operators::advection`]). cuda-oxide compiles every `#[kernel]` in the
//! crate into one device bundle keyed by the crate name, so multiple cuda_modules
//! coexist as long as their kernel export names are unique crate-wide.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice};
use cuda_host::cuda_module;
use gale::dg::{Mesh2d, Mesh3d, VolumePenalization, VolumePenalization3d};

#[cuda_module]
mod kernels {
    use super::*;

    /// In-place implicit penalization of both velocity components. One thread per dof.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn penalize(
        mut ux: DisjointSlice<f64>,
        mut uy: DisjointSlice<f64>,
        mask: &[f64],
        usx: &[f64],
        usy: &[f64],
        r: f64,
        nn: u32,
    ) {
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let g = e * (nn as usize) + m;
        let beta = r * mask[g];
        let denom = 1.0 + beta;
        let (bx, by) = (beta * usx[g], beta * usy[g]);
        if let Some(o) = ux.get_mut(thread::index_1d()) {
            *o = (*o + bx) / denom;
        }
        if let Some(o) = uy.get_mut(thread::index_1d()) {
            *o = (*o + by) / denom;
        }
    }
}

/// Apply implicit volume penalization to the velocity field `(ux, uy)` in place on
/// the GPU, using a precomputed [`VolumePenalization`] mask/solid-velocity field.
/// Bit-for-bit equal to `VolumePenalization::apply`.
pub fn penalize_apply(
    mesh: &Mesh2d,
    ux: &mut [f64],
    uy: &mut [f64],
    pen: &VolumePenalization,
    dt: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    assert_eq!(ux.len(), ndof, "ux length must be n_elements·n_nodes");
    assert_eq!(uy.len(), ndof, "uy length must be n_elements·n_nodes");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
    let mut ux_dev = up(ux)?;
    let mut uy_dev = up(uy)?;
    let mask_dev = up(&pen.mask)?;
    let usx_dev = up(&pen.us_x)?;
    let usy_dev = up(&pen.us_y)?;

    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig {
        grid_dim: (ne as u32, 1, 1),
        block_dim: (nn as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    module.penalize(
        &stream, cfg, &mut ux_dev, &mut uy_dev, &mask_dev, &usx_dev, &usy_dev,
        dt / pen.eta_b, nn as u32,
    )?;
    ux.copy_from_slice(&ux_dev.to_host_vec(&stream)?);
    uy.copy_from_slice(&uy_dev.to_host_vec(&stream)?);
    Ok(())
}

/// Implicit volume-penalization **stage hook** for the GPU flow integrators: applies
/// the IBM no-slip relaxation `u ← (u + β u_s)/(1 + β)` to the velocity field after
/// each integrator stage, on the GPU (via [`penalize_apply`]). The GPU analogue of
/// `gale::sim::PenalizationHook`; wire it with `Simulation::set_stage_hook` so a
/// [`crate::GpuDualSplitting`] / [`crate::GpuViscoelasticDualSplitting`] flow enforces
/// an immersed rigid body each step.
pub struct GpuPenalizationHook {
    velocity: gale::sim::FieldId,
    penal: VolumePenalization,
    dt: f64,
}

impl GpuPenalizationHook {
    /// Penalize the 2-component `velocity` field with `penal` over a step `dt` (must
    /// match the integrator's step size; β = (dt/η_b)·χ).
    pub fn new(velocity: gale::sim::FieldId, penal: VolumePenalization, dt: f64) -> Self {
        Self { velocity, penal, dt }
    }
}

impl gale::sim::StateStageHook for GpuPenalizationHook {
    fn after_stage(&self, state: &mut gale::sim::State, _stage: usize) {
        let mesh = state.mesh.clone();
        let comps = state.fields.by_id_mut(self.velocity).components_mut();
        let (ux, uy) = comps.split_at_mut(1);
        penalize_apply(&mesh, &mut ux[0], &mut uy[0], &self.penal, self.dt)
            .expect("gale-gpu: GpuPenalizationHook penalize_apply failed");
    }
}

/// **Moving** volume-penalization stage hook for the GPU flow integrators — a freely-
/// moving rigid body (M1, explicit Newton–Euler two-way coupling). The GPU twin of
/// `gale::sim::MovingPenalizationHook`: each step it penalizes the velocity on the GPU
/// ([`penalize_apply`]) to imprint the body at its current pose, recovers the
/// hydrodynamic force/torque, advances the body (`FreeBody::advance`), and rebuilds the
/// mask for the new pose. Only the penalize step runs on the GPU; the force/torque
/// recovery, Newton–Euler update and mask rebuild are the shared `gale` host code, so the
/// trajectory matches the CPU oracle bit-for-bit. Wire with `Simulation::set_stage_hook`.
///
/// For the single-stage dual-splitting integrators (`after_stage` once per step).
pub struct GpuMovingPenalizationHook {
    velocity: gale::sim::FieldId,
    dt: f64,
    body: gale::sim::BodyHandle,
    penal: std::cell::RefCell<VolumePenalization>,
    /// `true` ⇒ strong (implicit) coupling (M3, light/zero-mass stable); `false` ⇒
    /// explicit Newton–Euler (M1, heavy only).
    strong: bool,
}

impl GpuMovingPenalizationHook {
    /// Build the hook for a freely-moving `body` penalizing the 2-component `velocity`
    /// field over step `dt`. Returns the hook plus a `BodyHandle` clone for reading the
    /// trajectory after `Simulation::run`. `mesh` builds the initial mask. Defaults to
    /// EXPLICIT coupling; call [`strong`](Self::strong) for implicit (M3) coupling.
    pub fn new(
        velocity: gale::sim::FieldId,
        body: gale::dg::FreeBody,
        mesh: &Mesh2d,
        dt: f64,
    ) -> (Self, gale::sim::BodyHandle) {
        let penal = body.penalization(mesh);
        let handle: gale::sim::BodyHandle = std::rc::Rc::new(std::cell::RefCell::new(body));
        (Self { velocity, dt, body: handle.clone(), penal: std::cell::RefCell::new(penal), strong: false }, handle)
    }

    /// Enable strong (implicit) fluid–body coupling — required for light /
    /// neutrally-buoyant particles (removes the added-mass density-ratio limit). Builder.
    pub fn strong(mut self, strong: bool) -> Self {
        self.strong = strong;
        self
    }
}

impl gale::sim::StateStageHook for GpuMovingPenalizationHook {
    fn after_stage(&self, state: &mut gale::sim::State, stage: usize) {
        if stage != 0 {
            return; // advance once per step (single-stage dual-splitting)
        }
        let mesh = state.mesh.clone();
        let mut penal = self.penal.borrow_mut();
        let mut body = self.body.borrow_mut();
        if self.strong {
            // STRONG: implicit body-velocity solve against the penalization (host), using
            // the predictor field u*; then penalize on the GPU and advance the pose.
            let v = state.fields.by_id(self.velocity);
            let (u, vv, om) =
                body.strong_solve(v.component(0), v.component(1), &mesh, &penal, self.dt);
            body.body.u = u;
            body.body.v = vv;
            body.body.omega = om;
            *penal = body.penalization(&mesh); // current pose, NEW rigid velocity
            let comps = state.fields.by_id_mut(self.velocity).components_mut();
            let (ux, uy) = comps.split_at_mut(1);
            penalize_apply(&mesh, &mut ux[0], &mut uy[0], &penal, self.dt)
                .expect("gale-gpu: GpuMovingPenalizationHook penalize_apply (strong) failed");
            body.body.cx += self.dt * u;
            body.body.cy += self.dt * vv;
            body.body.phi += self.dt * om;
        } else {
            // EXPLICIT (M1): imprint at current pose, recover force/torque, advance.
            {
                let comps = state.fields.by_id_mut(self.velocity).components_mut();
                let (ux, uy) = comps.split_at_mut(1);
                penalize_apply(&mesh, &mut ux[0], &mut uy[0], &penal, self.dt)
                    .expect("gale-gpu: GpuMovingPenalizationHook penalize_apply failed");
            }
            let v = state.fields.by_id(self.velocity);
            let (fx, fy, tq) =
                penal.force_torque(v.component(0), v.component(1), &mesh, body.body.cx, body.body.cy);
            body.advance(fx, fy, tq, self.dt); // Newton–Euler (shared host code)
        }
        *penal = body.penalization(&mesh); // rebuild mask for the new pose (next step)
    }
}

/// **Many-body** moving volume-penalization stage hook for the GPU flow integrators
/// (M4 — particle-laden suspension). GPU twin of `gale::sim::MultiMovingPenalizationHook`:
/// drives a whole `gale::dg::Suspension`, imprinting all bodies through one combined mask
/// on the GPU each step, recovering per-body force/torque, applying short-range
/// repulsion/lubrication, and advancing every body (explicit or strong). Only the
/// combined-mask penalize runs on the GPU; force recovery, repulsion, the per-body
/// Newton-Euler / implicit solve, and pose updates are the shared `gale` host code ⇒ the
/// trajectory matches the CPU oracle bit-for-bit.
pub struct GpuMultiMovingPenalizationHook {
    velocity: gale::sim::FieldId,
    dt: f64,
    susp: gale::sim::SuspensionHandle,
    penal: std::cell::RefCell<VolumePenalization>,
    base_fext: Vec<(f64, f64)>,
    strong: bool,
}

impl GpuMultiMovingPenalizationHook {
    /// Build the hook for a `suspension` penalizing `velocity` over step `dt`. Returns the
    /// hook plus a `SuspensionHandle` for reading bodies / contact gap after the run.
    /// Defaults to EXPLICIT coupling; call [`strong`](Self::strong) for strong (implicit).
    pub fn new(
        velocity: gale::sim::FieldId,
        suspension: gale::dg::Suspension,
        mesh: &Mesh2d,
        dt: f64,
    ) -> (Self, gale::sim::SuspensionHandle) {
        let penal = suspension.combined_penalization(mesh);
        let base_fext = suspension.bodies.iter().map(|b| b.fext).collect();
        let handle: gale::sim::SuspensionHandle = std::rc::Rc::new(std::cell::RefCell::new(suspension));
        (Self { velocity, dt, susp: handle.clone(), penal: std::cell::RefCell::new(penal), base_fext, strong: false }, handle)
    }

    /// Enable strong (implicit) per-body coupling (light/neutrally-buoyant particles).
    pub fn strong(mut self, strong: bool) -> Self {
        self.strong = strong;
        self
    }
}

impl gale::sim::StateStageHook for GpuMultiMovingPenalizationHook {
    fn after_stage(&self, state: &mut gale::sim::State, stage: usize) {
        if stage != 0 {
            return;
        }
        let mesh = state.mesh.clone();
        let mut susp = self.susp.borrow_mut();
        let mut penal = self.penal.borrow_mut();
        let nb = susp.bodies.len();
        let rep = susp.repulsion();
        for b in 0..nb {
            susp.bodies[b].fext = (self.base_fext[b].0 + rep[b].0, self.base_fext[b].1 + rep[b].1);
        }
        if self.strong {
            let newvel: Vec<(f64, f64, f64)> = {
                let v = state.fields.by_id(self.velocity);
                let (ux, uy) = (v.component(0), v.component(1));
                (0..nb)
                    .map(|b| {
                        let pb = susp.bodies[b].penalization(&mesh);
                        susp.bodies[b].strong_solve(ux, uy, &mesh, &pb, self.dt)
                    })
                    .collect()
            };
            for b in 0..nb {
                susp.bodies[b].body.u = newvel[b].0;
                susp.bodies[b].body.v = newvel[b].1;
                susp.bodies[b].body.omega = newvel[b].2;
            }
            *penal = susp.combined_penalization(&mesh);
            {
                let comps = state.fields.by_id_mut(self.velocity).components_mut();
                let (ux, uy) = comps.split_at_mut(1);
                penalize_apply(&mesh, &mut ux[0], &mut uy[0], &penal, self.dt)
                    .expect("gale-gpu: GpuMultiMovingPenalizationHook penalize_apply (strong) failed");
            }
            for b in 0..nb {
                let (u, vv, om) = newvel[b];
                susp.bodies[b].body.cx += self.dt * u;
                susp.bodies[b].body.cy += self.dt * vv;
                susp.bodies[b].body.phi += self.dt * om;
            }
        } else {
            {
                let comps = state.fields.by_id_mut(self.velocity).components_mut();
                let (ux, uy) = comps.split_at_mut(1);
                penalize_apply(&mesh, &mut ux[0], &mut uy[0], &penal, self.dt)
                    .expect("gale-gpu: GpuMultiMovingPenalizationHook penalize_apply failed");
            }
            let ft: Vec<(f64, f64, f64)> = {
                let v = state.fields.by_id(self.velocity);
                let (ux, uy) = (v.component(0), v.component(1));
                (0..nb)
                    .map(|b| {
                        let pb = susp.bodies[b].penalization(&mesh);
                        let (cx, cy) = (susp.bodies[b].body.cx, susp.bodies[b].body.cy);
                        pb.force_torque(ux, uy, &mesh, cx, cy)
                    })
                    .collect()
            };
            for b in 0..nb {
                let (fx, fy, tq) = ft[b];
                susp.bodies[b].advance(fx, fy, tq, self.dt);
            }
        }
        susp.track_min_gap(); // record closest approach (poses final for this step)
        *penal = susp.combined_penalization(&mesh);
    }
}

// ===== 3D volume penalization ====================================================

#[cuda_module]
mod kernels3d {
    use super::*;

    /// In-place implicit penalization of all three velocity components. One thread
    /// per dof (`grid = ne`, `block = nn`).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn penalize3d(
        mut ux: DisjointSlice<f64>,
        mut uy: DisjointSlice<f64>,
        mut uz: DisjointSlice<f64>,
        mask: &[f64],
        usx: &[f64],
        usy: &[f64],
        usz: &[f64],
        r: f64,
        nn: u32,
    ) {
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let g = e * (nn as usize) + m;
        let beta = r * mask[g];
        let denom = 1.0 + beta;
        let (bx, by, bz) = (beta * usx[g], beta * usy[g], beta * usz[g]);
        if let Some(o) = ux.get_mut(thread::index_1d()) {
            *o = (*o + bx) / denom;
        }
        if let Some(o) = uy.get_mut(thread::index_1d()) {
            *o = (*o + by) / denom;
        }
        if let Some(o) = uz.get_mut(thread::index_1d()) {
            *o = (*o + bz) / denom;
        }
    }
}

/// Apply implicit 3D volume penalization to `(ux, uy, uz)` in place on the GPU, using
/// a precomputed [`VolumePenalization3d`]. Bit-for-bit equal to
/// `VolumePenalization3d::apply`. The 3D analogue of [`penalize_apply`].
#[allow(clippy::too_many_arguments)]
pub fn penalize3d_apply(
    mesh: &Mesh3d,
    ux: &mut [f64],
    uy: &mut [f64],
    uz: &mut [f64],
    pen: &VolumePenalization3d,
    dt: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let nn = mesh.refh.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    assert_eq!(ux.len(), ndof, "ux length must be n_elements·n_nodes");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
    let mut ux_dev = up(ux)?;
    let mut uy_dev = up(uy)?;
    let mut uz_dev = up(uz)?;
    let mask_dev = up(&pen.mask)?;
    let usx_dev = up(&pen.us_x)?;
    let usy_dev = up(&pen.us_y)?;
    let usz_dev = up(&pen.us_z)?;

    let module = kernels3d::load(&ctx)?;
    let cfg = LaunchConfig {
        grid_dim: (ne as u32, 1, 1),
        block_dim: (nn as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    module.penalize3d(
        &stream, cfg, &mut ux_dev, &mut uy_dev, &mut uz_dev, &mask_dev, &usx_dev, &usy_dev,
        &usz_dev, dt / pen.eta_b, nn as u32,
    )?;
    ux.copy_from_slice(&ux_dev.to_host_vec(&stream)?);
    uy.copy_from_slice(&uy_dev.to_host_vec(&stream)?);
    uz.copy_from_slice(&uz_dev.to_host_vec(&stream)?);
    Ok(())
}

/// 3D volume-penalization **stage hook** (`gale::sim::StateStageHook<Mesh3d>`): the 3D
/// analogue of [`GpuPenalizationHook`], damping the 3-component velocity toward the
/// solid on the GPU after each integrator stage. Wire with `Simulation::set_stage_hook`
/// alongside [`crate::GpuDualSplitting3d`] for GPU flow past an immersed body.
pub struct GpuPenalization3dHook {
    velocity: gale::sim::FieldId,
    penal: VolumePenalization3d,
    dt: f64,
}

impl GpuPenalization3dHook {
    pub fn new(velocity: gale::sim::FieldId, penal: VolumePenalization3d, dt: f64) -> Self {
        Self { velocity, penal, dt }
    }
}

impl gale::sim::StateStageHook<Mesh3d> for GpuPenalization3dHook {
    fn after_stage(&self, state: &mut gale::sim::State<Mesh3d>, _stage: usize) {
        let mesh = state.mesh.clone();
        let comps = state.fields.by_id_mut(self.velocity).components_mut();
        let (ux, rest) = comps.split_at_mut(1);
        let (uy, uz) = rest.split_at_mut(1);
        penalize3d_apply(&mesh, &mut ux[0], &mut uy[0], &mut uz[0], &self.penal, self.dt)
            .expect("gale-gpu: GpuPenalization3dHook penalize3d_apply failed");
    }
}
