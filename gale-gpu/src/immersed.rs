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
    let mut backend = PenalizeBackend::new(mesh)?;
    backend.apply(ux, uy, pen, dt)
}

/// **Persistent** 2D penalization backend: loads the device module **once** and keeps reusable
/// device buffers, so each [`apply`](Self::apply) only uploads the per-step fields (velocity + the
/// — possibly moving — mask/solid-velocity), launches, and downloads. This is what makes the
/// penalization stage hooks usable in a real time loop; the one-shot [`penalize_apply`] reloads +
/// recompiles the whole module (NVVM→nvJitLink→cubin, ~hundreds of ms) every call, which is
/// catastrophic when driven once per RK stage.
struct PenalizeBackend {
    _ctx: std::sync::Arc<CudaContext>,
    stream: std::sync::Arc<cuda_core::CudaStream>,
    module: kernels::LoadedModule,
    cfg: LaunchConfig,
    nn: u32,
    ndof: usize,
    ux_dev: DeviceBuffer<f64>,
    uy_dev: DeviceBuffer<f64>,
    mask_dev: DeviceBuffer<f64>,
    usx_dev: DeviceBuffer<f64>,
    usy_dev: DeviceBuffer<f64>,
}

impl PenalizeBackend {
    fn new(mesh: &Mesh2d) -> Result<Self, Box<dyn std::error::Error>> {
        let nn = mesh.refq.n_nodes();
        let ne = mesh.n_elements();
        let ndof = ne * nn;
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let z = || DeviceBuffer::<f64>::zeroed(&stream, ndof);
        let (ux_dev, uy_dev, mask_dev, usx_dev, usy_dev) = (z()?, z()?, z()?, z()?, z()?);
        let module = kernels::load(&ctx)?;
        let cfg = LaunchConfig {
            grid_dim: (ne as u32, 1, 1),
            block_dim: (nn as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        Ok(Self {
            _ctx: ctx, stream, module, cfg, nn: nn as u32, ndof,
            ux_dev, uy_dev, mask_dev, usx_dev, usy_dev,
        })
    }

    /// One in-place penalization on the persistent backend. Uploads `ux/uy` and the mask + solid
    /// velocity (re-uploaded every call so a *moving* body's rebuilt mask is honored), launches,
    /// downloads `ux/uy`. No module load — microseconds, not a per-call recompile.
    fn apply(
        &mut self,
        ux: &mut [f64],
        uy: &mut [f64],
        pen: &VolumePenalization,
        dt: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(ux.len(), self.ndof, "ux length must be n_elements·n_nodes");
        assert_eq!(uy.len(), self.ndof, "uy length must be n_elements·n_nodes");
        let s = self.stream.cu_stream();
        unsafe {
            use cuda_core::memory::memcpy_htod_async as h2d;
            h2d(self.ux_dev.cu_deviceptr(), ux.as_ptr(), std::mem::size_of_val(ux), s)?;
            h2d(self.uy_dev.cu_deviceptr(), uy.as_ptr(), std::mem::size_of_val(uy), s)?;
            h2d(self.mask_dev.cu_deviceptr(), pen.mask.as_ptr(), std::mem::size_of_val(&pen.mask[..]), s)?;
            h2d(self.usx_dev.cu_deviceptr(), pen.us_x.as_ptr(), std::mem::size_of_val(&pen.us_x[..]), s)?;
            h2d(self.usy_dev.cu_deviceptr(), pen.us_y.as_ptr(), std::mem::size_of_val(&pen.us_y[..]), s)?;
        }
        self.module.penalize(
            &self.stream, self.cfg, &mut self.ux_dev, &mut self.uy_dev,
            &self.mask_dev, &self.usx_dev, &self.usy_dev, dt / pen.eta_b, self.nn,
        )?;
        ux.copy_from_slice(&self.ux_dev.to_host_vec(&self.stream)?);
        uy.copy_from_slice(&self.uy_dev.to_host_vec(&self.stream)?);
        Ok(())
    }
}

/// Lazily build (on first call) and reuse a [`PenalizeBackend`] held in a hook's `RefCell`, then
/// apply it. Shared by all 2D penalization stage hooks so none reloads the module per stage.
fn run_penalize(
    backend: &std::cell::RefCell<Option<PenalizeBackend>>,
    mesh: &Mesh2d,
    ux: &mut [f64],
    uy: &mut [f64],
    pen: &VolumePenalization,
    dt: f64,
) {
    let mut slot = backend.borrow_mut();
    if slot.is_none() {
        *slot = Some(PenalizeBackend::new(mesh).expect("gale-gpu: penalize backend build failed"));
    }
    slot.as_mut().unwrap().apply(ux, uy, pen, dt).expect("gale-gpu: penalize_apply failed");
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
    backend: std::cell::RefCell<Option<PenalizeBackend>>,
}

impl GpuPenalizationHook {
    /// Penalize the 2-component `velocity` field with `penal` over a step `dt` (must
    /// match the integrator's step size; β = (dt/η_b)·χ).
    pub fn new(velocity: gale::sim::FieldId, penal: VolumePenalization, dt: f64) -> Self {
        Self { velocity, penal, dt, backend: std::cell::RefCell::new(None) }
    }
}

impl gale::sim::StateStageHook for GpuPenalizationHook {
    fn after_stage(&self, state: &mut gale::sim::State, _stage: usize) {
        let mesh = state.mesh.clone();
        let comps = state.fields.by_id_mut(self.velocity).components_mut();
        let (ux, uy) = comps.split_at_mut(1);
        run_penalize(&self.backend, &mesh, &mut ux[0], &mut uy[0], &self.penal, self.dt);
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
    backend: std::cell::RefCell<Option<PenalizeBackend>>,
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
        (Self { velocity, dt, body: handle.clone(), penal: std::cell::RefCell::new(penal), strong: false, backend: std::cell::RefCell::new(None) }, handle)
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
            run_penalize(&self.backend, &mesh, &mut ux[0], &mut uy[0], &penal, self.dt);
            body.body.cx += self.dt * u;
            body.body.cy += self.dt * vv;
            body.body.phi += self.dt * om;
        } else {
            // EXPLICIT (M1): imprint at current pose, recover force/torque, advance.
            {
                let comps = state.fields.by_id_mut(self.velocity).components_mut();
                let (ux, uy) = comps.split_at_mut(1);
                run_penalize(&self.backend, &mesh, &mut ux[0], &mut uy[0], &penal, self.dt);
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
    backend: std::cell::RefCell<Option<PenalizeBackend>>,
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
        (Self { velocity, dt, susp: handle.clone(), penal: std::cell::RefCell::new(penal), base_fext, strong: false, backend: std::cell::RefCell::new(None) }, handle)
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
                run_penalize(&self.backend, &mesh, &mut ux[0], &mut uy[0], &penal, self.dt);
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
                run_penalize(&self.backend, &mesh, &mut ux[0], &mut uy[0], &penal, self.dt);
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
    let mut backend = Penalize3dBackend::new(mesh)?;
    backend.apply(ux, uy, uz, pen, dt)
}

/// **Persistent** 3D penalization backend — the 3D analogue of [`PenalizeBackend`]: module loaded
/// once, reusable device buffers, each [`apply`](Self::apply) only uploads the per-step fields.
struct Penalize3dBackend {
    _ctx: std::sync::Arc<CudaContext>,
    stream: std::sync::Arc<cuda_core::CudaStream>,
    module: kernels3d::LoadedModule,
    cfg: LaunchConfig,
    nn: u32,
    ndof: usize,
    ux_dev: DeviceBuffer<f64>,
    uy_dev: DeviceBuffer<f64>,
    uz_dev: DeviceBuffer<f64>,
    mask_dev: DeviceBuffer<f64>,
    usx_dev: DeviceBuffer<f64>,
    usy_dev: DeviceBuffer<f64>,
    usz_dev: DeviceBuffer<f64>,
}

impl Penalize3dBackend {
    fn new(mesh: &Mesh3d) -> Result<Self, Box<dyn std::error::Error>> {
        let nn = mesh.refh.n_nodes();
        let ne = mesh.n_elements();
        let ndof = ne * nn;
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let z = || DeviceBuffer::<f64>::zeroed(&stream, ndof);
        let (ux_dev, uy_dev, uz_dev) = (z()?, z()?, z()?);
        let (mask_dev, usx_dev, usy_dev, usz_dev) = (z()?, z()?, z()?, z()?);
        let module = kernels3d::load(&ctx)?;
        let cfg = LaunchConfig {
            grid_dim: (ne as u32, 1, 1),
            block_dim: (nn as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        Ok(Self {
            _ctx: ctx, stream, module, cfg, nn: nn as u32, ndof,
            ux_dev, uy_dev, uz_dev,
            mask_dev, usx_dev, usy_dev, usz_dev,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn apply(
        &mut self,
        ux: &mut [f64],
        uy: &mut [f64],
        uz: &mut [f64],
        pen: &VolumePenalization3d,
        dt: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(ux.len(), self.ndof, "ux length must be n_elements·n_nodes");
        let s = self.stream.cu_stream();
        unsafe {
            use cuda_core::memory::memcpy_htod_async as h2d;
            h2d(self.ux_dev.cu_deviceptr(), ux.as_ptr(), std::mem::size_of_val(ux), s)?;
            h2d(self.uy_dev.cu_deviceptr(), uy.as_ptr(), std::mem::size_of_val(uy), s)?;
            h2d(self.uz_dev.cu_deviceptr(), uz.as_ptr(), std::mem::size_of_val(uz), s)?;
            h2d(self.mask_dev.cu_deviceptr(), pen.mask.as_ptr(), std::mem::size_of_val(&pen.mask[..]), s)?;
            h2d(self.usx_dev.cu_deviceptr(), pen.us_x.as_ptr(), std::mem::size_of_val(&pen.us_x[..]), s)?;
            h2d(self.usy_dev.cu_deviceptr(), pen.us_y.as_ptr(), std::mem::size_of_val(&pen.us_y[..]), s)?;
            h2d(self.usz_dev.cu_deviceptr(), pen.us_z.as_ptr(), std::mem::size_of_val(&pen.us_z[..]), s)?;
        }
        self.module.penalize3d(
            &self.stream, self.cfg, &mut self.ux_dev, &mut self.uy_dev, &mut self.uz_dev,
            &self.mask_dev, &self.usx_dev, &self.usy_dev, &self.usz_dev, dt / pen.eta_b, self.nn,
        )?;
        ux.copy_from_slice(&self.ux_dev.to_host_vec(&self.stream)?);
        uy.copy_from_slice(&self.uy_dev.to_host_vec(&self.stream)?);
        uz.copy_from_slice(&self.uz_dev.to_host_vec(&self.stream)?);
        Ok(())
    }
}

/// Lazily build + reuse a [`Penalize3dBackend`] held in a hook's `RefCell`, then apply it.
fn run_penalize3d(
    backend: &std::cell::RefCell<Option<Penalize3dBackend>>,
    mesh: &Mesh3d,
    ux: &mut [f64],
    uy: &mut [f64],
    uz: &mut [f64],
    pen: &VolumePenalization3d,
    dt: f64,
) {
    let mut slot = backend.borrow_mut();
    if slot.is_none() {
        *slot = Some(Penalize3dBackend::new(mesh).expect("gale-gpu: penalize3d backend build failed"));
    }
    slot.as_mut().unwrap().apply(ux, uy, uz, pen, dt).expect("gale-gpu: penalize3d_apply failed");
}

/// 3D volume-penalization **stage hook** (`gale::sim::StateStageHook<Mesh3d>`): the 3D
/// analogue of [`GpuPenalizationHook`], damping the 3-component velocity toward the
/// solid on the GPU after each integrator stage. Wire with `Simulation::set_stage_hook`
/// alongside [`crate::GpuDualSplitting3d`] for GPU flow past an immersed body.
pub struct GpuPenalization3dHook {
    velocity: gale::sim::FieldId,
    penal: VolumePenalization3d,
    dt: f64,
    backend: std::cell::RefCell<Option<Penalize3dBackend>>,
}

impl GpuPenalization3dHook {
    pub fn new(velocity: gale::sim::FieldId, penal: VolumePenalization3d, dt: f64) -> Self {
        Self { velocity, penal, dt, backend: std::cell::RefCell::new(None) }
    }
}

impl gale::sim::StateStageHook<Mesh3d> for GpuPenalization3dHook {
    fn after_stage(&self, state: &mut gale::sim::State<Mesh3d>, _stage: usize) {
        let mesh = state.mesh.clone();
        let comps = state.fields.by_id_mut(self.velocity).components_mut();
        let (ux, rest) = comps.split_at_mut(1);
        let (uy, uz) = rest.split_at_mut(1);
        run_penalize3d(&self.backend, &mesh, &mut ux[0], &mut uy[0], &mut uz[0], &self.penal, self.dt);
    }
}
