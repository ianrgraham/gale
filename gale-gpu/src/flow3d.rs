//! GPU incompressible Navier–Stokes on the 3D hex mesh — the 3D analogue of
//! [`crate::flow`]. BDF1 dual-splitting: explicit (nodal) convection + body force →
//! GPU pressure-Poisson projection → GPU viscous Helmholtz solve per velocity
//! component. The two elliptic solves run on the GPU ([`crate::pressure3d_cg_solve`],
//! [`crate::helmholtz3d_cg_solve`]); the cheap element-local assembly (convection,
//! divergence, gradient correction, SIPG RHS) reuses the validated host `gale::dg`
//! machinery. Faithful (to solver tolerance) to `gale::dg::Stokes3d`.

use crate::operators::poisson3d::{
    helmholtz3d_cg_solve, helmholtz3d_cg_solve_tags, pressure3d_cg_solve, GpuPoisson3d,
};
use crate::operators::poisson3d_nc::{helmholtz3d_nc_cg_solve_tags, pressure3d_nc_cg_solve};
use gale::dg::{BoundaryConditions3d, Mesh3d, Neighbor3, Poisson3d};
use std::cell::RefCell;

type StepResult = Result<(Vec<f64>, Vec<f64>, Vec<f64>), Box<dyn std::error::Error>>;

/// Whether the hex mesh has any 2:1 non-conforming (octree AMR) interface — if so the GPU elliptic
/// solves route through the non-conforming path (`poisson3d_nc`) instead of the conforming one.
fn mesh_is_nonconforming(mesh: &Mesh3d) -> bool {
    mesh.elements.iter().any(|el| {
        el.neighbors
            .iter()
            .any(|n| matches!(n, Neighbor3::CoarseToFine { .. } | Neighbor3::FineToCoarse { .. }))
    })
}

/// Lazily (re)build the persistent [`GpuPoisson3d`] handle in `slot` for `mesh`, so an
/// integrator's `step(&self, …)` can hold ONE handle across timesteps (P4) — the 3D
/// analogue of [`crate::flow::ensure_poisson_handle`]. The conforming handle is cleared for a
/// non-conforming mesh (the step then uses the one-shot `poisson3d_nc` solves); rebuilt when the
/// dof count changes (e.g. after an AMR remesh), so a static mesh pays the ~0.3 s setup once.
fn ensure_poisson3d_handle(slot: &RefCell<Option<GpuPoisson3d>>, mesh: &Mesh3d, alpha: f64) {
    let mut cur = slot.borrow_mut();
    if mesh_is_nonconforming(mesh) {
        *cur = None;
        return;
    }
    let ndof = mesh.n_elements() * mesh.refh.n_nodes();
    if cur.as_ref().map_or(true, |h| h.ndof() != ndof) {
        *cur = Some(GpuPoisson3d::new(mesh, alpha).expect("gale-gpu: GpuPoisson3d handle build failed"));
    }
}

/// GPU 3D unsteady-NS stepper. Host `Poisson3d` operators are used only for SIPG RHS
/// assembly; the operator applies / solves run on the GPU.
pub struct GpuStokes3d<'m, 'p> {
    pub mesh: &'m Mesh3d,
    pub nu: f64,
    pub dt: f64,
    alpha: f64,
    /// Optional **persistent** [`GpuPoisson3d`] handle (P4). When `Some`, the three
    /// per-step elliptic solves route through it — reusing the loaded module + uploaded
    /// mesh instead of paying ~0.3 s of context/upload setup per solve. Set via
    /// [`with_handle`](Self::with_handle); the owning integrator keeps it alive across
    /// timesteps. `None` ⇒ the one-shot solvers (legacy path).
    poisson: Option<&'p GpuPoisson3d>,
    /// Pressure-Poisson assembly (pure Neumann, singular — unless an outflow pins it).
    pressure: Poisson3d<'m>,
    /// Velocity Helmholtz assembly `(λM + A)`, one per component (differ only in their
    /// Neumann-tag set, for symmetry/slip faces; identical otherwise).
    velocity_x: Poisson3d<'m>,
    velocity_y: Poisson3d<'m>,
    velocity_z: Poisson3d<'m>,
    /// Whether any boundary is an outflow (pressure pinned ⇒ non-deflated solve).
    has_outflow: bool,
    /// Per-component velocity-Neumann tags for the GPU Helmholtz solve (outflow +
    /// symmetry-tangential faces).
    velx_neumann: Vec<u32>,
    vely_neumann: Vec<u32>,
    velz_neumann: Vec<u32>,
    /// Pressure-Poisson Neumann tags (everything except outflow) for the GPU solve.
    pres_neumann_tags: Vec<u32>,
    tol: f64,
    maxit: usize,
}

impl<'m, 'p> GpuStokes3d<'m, 'p> {
    /// Closed-box solver: all-Dirichlet velocity (data from the `bc_*` closures) +
    /// pure-Neumann (deflated) pressure. For per-region inflow/outflow/wall/symmetry
    /// conditions use [`with_bcs`](Self::with_bcs).
    pub fn new(mesh: &'m Mesh3d, alpha: f64, nu: f64, dt: f64) -> Self {
        let lambda = 1.0 / (nu * dt);
        Self {
            mesh,
            nu,
            dt,
            alpha,
            poisson: None,
            pressure: Poisson3d::with_bc(mesh, alpha, 0.0, mesh.boundary_tags()),
            velocity_x: Poisson3d::with_reaction(mesh, alpha, lambda),
            velocity_y: Poisson3d::with_reaction(mesh, alpha, lambda),
            velocity_z: Poisson3d::with_reaction(mesh, alpha, lambda),
            has_outflow: false,
            velx_neumann: Vec::new(),
            vely_neumann: Vec::new(),
            velz_neumann: Vec::new(),
            pres_neumann_tags: mesh.boundary_tags(),
            tol: 1e-10,
            maxit: 20000,
        }
    }

    /// Route the three per-step elliptic solves through the persistent [`GpuPoisson3d`]
    /// `handle` (P4) instead of the one-shot solvers, amortizing context/module/mesh setup
    /// across timesteps. The caller must ensure `handle` was built from a mesh matching
    /// `self.mesh` (same `ndof`). Bit-identical to the one-shot solver it replaces
    /// (validated by `poisson3d-handle-check`).
    pub fn with_handle(mut self, handle: &'p GpuPoisson3d) -> Self {
        self.poisson = Some(handle);
        self
    }

    /// Solver with **per-region** boundary conditions — the GPU analogue of
    /// `gale::dg::Stokes3d::with_bcs`. Each boundary tag is routed via `bcs` to the right
    /// pair of operator settings (no-slip/inflow ⇒ velocity-Dirichlet + pressure-Neumann;
    /// outflow ⇒ velocity-Neumann + pressure-Dirichlet `p=0`; symmetry ⇒ normal component
    /// Dirichlet `u·n=0` + tangential Neumann). The pinned pressure lets the deflated solve
    /// be replaced by a plain CG. Use the `*_bc` step methods.
    pub fn with_bcs(mesh: &'m Mesh3d, alpha: f64, nu: f64, dt: f64, bcs: &BoundaryConditions3d) -> Self {
        let lambda = 1.0 / (nu * dt);
        let velx_neumann = bcs.velocity_neumann_tags(mesh, 0);
        let vely_neumann = bcs.velocity_neumann_tags(mesh, 1);
        let velz_neumann = bcs.velocity_neumann_tags(mesh, 2);
        let pres_neumann = bcs.pressure_neumann_tags(mesh);
        Self {
            mesh,
            nu,
            dt,
            alpha,
            poisson: None,
            pressure: Poisson3d::with_bc(mesh, alpha, 0.0, pres_neumann.clone()),
            velocity_x: Poisson3d::with_bc(mesh, alpha, lambda, velx_neumann.clone()),
            velocity_y: Poisson3d::with_bc(mesh, alpha, lambda, vely_neumann.clone()),
            velocity_z: Poisson3d::with_bc(mesh, alpha, lambda, velz_neumann.clone()),
            has_outflow: bcs.has_outflow(mesh),
            velx_neumann,
            vely_neumann,
            velz_neumann,
            pres_neumann_tags: pres_neumann,
            tol: 1e-10,
            maxit: 20000,
        }
    }

    fn ndof(&self) -> usize {
        self.mesh.n_elements() * self.mesh.refh.n_nodes()
    }

    /// `∂x a + ∂y b + ∂z c` of a vector field, per element (host).
    fn divergence(&self, a: &[f64], b: &[f64], c: &[f64]) -> Vec<f64> {
        let refh = &self.mesh.refh;
        let nn = refh.n_nodes();
        let mut d = vec![0.0; self.ndof()];
        for (e, el) in self.mesh.elements.iter().enumerate() {
            let sl = e * nn..(e + 1) * nn;
            let ax = el.geom.grad_x(refh, &a[sl.clone()]);
            let by = el.geom.grad_y(refh, &b[sl.clone()]);
            let cz = el.geom.grad_z(refh, &c[sl]);
            for k in 0..nn {
                d[e * nn + k] = ax[k] + by[k] + cz[k];
            }
        }
        d
    }

    /// Nodal advective convection `(u·∇)u` per component (host).
    fn convection(&self, ux: &[f64], uy: &[f64], uz: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let refh = &self.mesh.refh;
        let nn = refh.n_nodes();
        let mut cx = vec![0.0; self.ndof()];
        let mut cy = vec![0.0; self.ndof()];
        let mut cz = vec![0.0; self.ndof()];
        for (e, el) in self.mesh.elements.iter().enumerate() {
            let sl = e * nn..(e + 1) * nn;
            let uxx = el.geom.grad_x(refh, &ux[sl.clone()]);
            let uxy = el.geom.grad_y(refh, &ux[sl.clone()]);
            let uxz = el.geom.grad_z(refh, &ux[sl.clone()]);
            let uyx = el.geom.grad_x(refh, &uy[sl.clone()]);
            let uyy = el.geom.grad_y(refh, &uy[sl.clone()]);
            let uyz = el.geom.grad_z(refh, &uy[sl.clone()]);
            let uzx = el.geom.grad_x(refh, &uz[sl.clone()]);
            let uzy = el.geom.grad_y(refh, &uz[sl.clone()]);
            let uzz = el.geom.grad_z(refh, &uz[sl]);
            for k in 0..nn {
                let (u, v, w) = (ux[e * nn + k], uy[e * nn + k], uz[e * nn + k]);
                cx[e * nn + k] = u * uxx[k] + v * uxy[k] + w * uxz[k];
                cy[e * nn + k] = u * uyx[k] + v * uyy[k] + w * uyz[k];
                cz[e * nn + k] = u * uzx[k] + v * uzy[k] + w * uzz[k];
            }
        }
        (cx, cy, cz)
    }

    /// One incompressible NS step with analytic forcing closures.
    #[allow(clippy::too_many_arguments)]
    pub fn step_ns(
        &self,
        ux: &[f64],
        uy: &[f64],
        uz: &[f64],
        t: f64,
        bc_u: impl Fn(f64, f64, f64, f64) -> f64,
        bc_v: impl Fn(f64, f64, f64, f64) -> f64,
        bc_w: impl Fn(f64, f64, f64, f64) -> f64,
        fx: impl Fn(f64, f64, f64, f64) -> f64,
        fy: impl Fn(f64, f64, f64, f64) -> f64,
        fz: impl Fn(f64, f64, f64, f64) -> f64,
    ) -> StepResult {
        let nn = self.mesh.refh.n_nodes();
        let (mut force_x, mut force_y, mut force_z) =
            (vec![0.0; self.ndof()], vec![0.0; self.ndof()], vec![0.0; self.ndof()]);
        for (e, el) in self.mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let (x, y, z) = (el.geom.x[k], el.geom.y[k], el.geom.z[k]);
                force_x[e * nn + k] = fx(x, y, z, t);
                force_y[e * nn + k] = fy(x, y, z, t);
                force_z[e * nn + k] = fz(x, y, z, t);
            }
        }
        self.step_ns_forced(ux, uy, uz, t, bc_u, bc_v, bc_w, &force_x, &force_y, &force_z)
    }

    /// One incompressible NS step under a precomputed nodal body force.
    #[allow(clippy::too_many_arguments)]
    pub fn step_ns_forced(
        &self,
        ux: &[f64],
        uy: &[f64],
        uz: &[f64],
        t: f64,
        bc_u: impl Fn(f64, f64, f64, f64) -> f64,
        bc_v: impl Fn(f64, f64, f64, f64) -> f64,
        bc_w: impl Fn(f64, f64, f64, f64) -> f64,
        fx: &[f64],
        fy: &[f64],
        fz: &[f64],
    ) -> StepResult {
        let dt = self.dt;
        let (cx, cy, cz) = self.convection(ux, uy, uz);
        let mut uhx = ux.to_vec();
        let mut uhy = uy.to_vec();
        let mut uhz = uz.to_vec();
        for i in 0..self.ndof() {
            uhx[i] += dt * (fx[i] - cx[i]);
            uhy[i] += dt * (fy[i] - cy[i]);
            uhz[i] += dt * (fz[i] - cz[i]);
        }
        self.project_and_diffuse(uhx, uhy, uhz, t, bc_u, bc_v, bc_w)
    }

    /// Stages 2–3: pressure projection (GPU) then viscous Helmholtz solve per
    /// component (GPU).
    #[allow(clippy::too_many_arguments)]
    fn project_and_diffuse(
        &self,
        mut uhx: Vec<f64>,
        mut uhy: Vec<f64>,
        mut uhz: Vec<f64>,
        t: f64,
        bc_u: impl Fn(f64, f64, f64, f64) -> f64,
        bc_v: impl Fn(f64, f64, f64, f64) -> f64,
        bc_w: impl Fn(f64, f64, f64, f64) -> f64,
    ) -> StepResult {
        let mesh = self.mesh;
        let refh = &mesh.refh;
        let nn = refh.n_nodes();
        let dt = self.dt;
        let lambda = 1.0 / (self.nu * dt);

        // Stage 2 — pressure projection: ∇²p = (1/Δt)∇·û (homogeneous Neumann), GPU.
        let div = self.divergence(&uhx, &uhy, &uhz);
        let fp: Vec<f64> = div.iter().map(|d| -d / dt).collect();
        let bp = self.pressure.rhs_mixed(&fp, |_, _, _| 0.0, |_, _, _| 0.0);
        let nc = mesh_is_nonconforming(mesh);
        let (p, _it) = if nc {
            pressure3d_nc_cg_solve(mesh, &bp, self.alpha, self.tol, self.maxit)?
        } else if let Some(h) = self.poisson {
            // Pure-Neumann pressure, deflated (closed box ⇒ all tags Neumann, no outflow).
            h.solve(&bp, 0.0, &self.pres_neumann_tags, true, self.tol, self.maxit)?
        } else {
            pressure3d_cg_solve(mesh, &bp, self.alpha, self.tol, self.maxit)?
        };
        for (e, el) in mesh.elements.iter().enumerate() {
            let sl = e * nn..(e + 1) * nn;
            let gpx = el.geom.grad_x(refh, &p[sl.clone()]);
            let gpy = el.geom.grad_y(refh, &p[sl.clone()]);
            let gpz = el.geom.grad_z(refh, &p[sl]);
            for k in 0..nn {
                uhx[e * nn + k] -= dt * gpx[k];
                uhy[e * nn + k] -= dt * gpy[k];
                uhz[e * nn + k] -= dt * gpz[k];
            }
        }

        // Stage 3 — viscous Helmholtz per component: (λM + A) uⁿ⁺¹ = λM û, GPU.
        let fxv: Vec<f64> = uhx.iter().map(|v| lambda * v).collect();
        let fyv: Vec<f64> = uhy.iter().map(|v| lambda * v).collect();
        let fzv: Vec<f64> = uhz.iter().map(|v| lambda * v).collect();
        let bx = self.velocity_x.rhs(&fxv, |x, y, z| bc_u(x, y, z, t));
        let by = self.velocity_y.rhs(&fyv, |x, y, z| bc_v(x, y, z, t));
        let bz = self.velocity_z.rhs(&fzv, |x, y, z| bc_w(x, y, z, t));
        // All-Dirichlet velocity Helmholtz (closed box ⇒ empty Neumann-tag sets).
        let solve_vel = |b: &[f64]| -> Result<Vec<f64>, Box<dyn std::error::Error>> {
            Ok(if nc {
                helmholtz3d_nc_cg_solve_tags(mesh, b, self.alpha, lambda, &self.velx_neumann, self.tol, self.maxit)?.0
            } else if let Some(h) = self.poisson {
                h.solve(b, lambda, &self.velx_neumann, false, self.tol, self.maxit)?.0
            } else {
                helmholtz3d_cg_solve(mesh, b, self.alpha, lambda, self.tol, self.maxit)?.0
            })
        };
        let uxn = solve_vel(&bx)?;
        let uyn = solve_vel(&by)?;
        let uzn = solve_vel(&bz)?;
        Ok((uxn, uyn, uzn))
    }

    /// BC-aware **Navier–Stokes** step with a precomputed nodal body force — the
    /// [`with_bcs`](Self::with_bcs) companion to [`step_ns_forced`](Self::step_ns_forced).
    /// Per-region velocity data comes from `bcs`; nodal convection only.
    #[allow(clippy::too_many_arguments)]
    pub fn step_ns_forced_bc(
        &self,
        ux: &[f64],
        uy: &[f64],
        uz: &[f64],
        t: f64,
        bcs: &BoundaryConditions3d,
        fx: &[f64],
        fy: &[f64],
        fz: &[f64],
    ) -> StepResult {
        let dt = self.dt;
        let (cx, cy, cz) = self.convection(ux, uy, uz);
        let mut uhx = ux.to_vec();
        let mut uhy = uy.to_vec();
        let mut uhz = uz.to_vec();
        for i in 0..self.ndof() {
            uhx[i] += dt * (fx[i] - cx[i]);
            uhy[i] += dt * (fy[i] - cy[i]);
            uhz[i] += dt * (fz[i] - cz[i]);
        }
        self.project_and_diffuse_bc(uhx, uhy, uhz, |tag, x, y, z| bcs.dirichlet(tag, x, y, z, t))
    }

    /// Stages 2–3 for the per-region BC path — the [`project_and_diffuse`] analogue
    /// using tag-aware host RHS assembly and the tag-aware GPU elliptic solves. An
    /// outflow pins the pressure (Dirichlet `p=0`) so the deflated solve is replaced by
    /// a plain CG.
    fn project_and_diffuse_bc(
        &self,
        mut uhx: Vec<f64>,
        mut uhy: Vec<f64>,
        mut uhz: Vec<f64>,
        vel_dir: impl Fn(u32, f64, f64, f64) -> (f64, f64, f64),
    ) -> StepResult {
        let mesh = self.mesh;
        let refh = &mesh.refh;
        let nn = refh.n_nodes();
        let dt = self.dt;
        let lambda = 1.0 / (self.nu * dt);

        // Stage 2 — pressure projection (homogeneous data either way). With an outflow
        // the operator is non-singular (Dirichlet p=0 there) ⇒ plain Poisson CG; with
        // no outflow it is the singular pure-Neumann system ⇒ deflated CG.
        let div = self.divergence(&uhx, &uhy, &uhz);
        let fp: Vec<f64> = div.iter().map(|d| -d / dt).collect();
        let bp = self.pressure.rhs_mixed(&fp, |_, _, _| 0.0, |_, _, _| 0.0);
        let nc = mesh_is_nonconforming(mesh);
        let (p, _it) = if nc {
            // Outflow ⇒ pressure pinned (non-deflated tag-CG); no outflow ⇒ singular ⇒ deflated.
            if self.has_outflow {
                helmholtz3d_nc_cg_solve_tags(mesh, &bp, self.alpha, 0.0, &self.pres_neumann_tags, self.tol, self.maxit)?
            } else {
                pressure3d_nc_cg_solve(mesh, &bp, self.alpha, self.tol, self.maxit)?
            }
        } else if let Some(h) = self.poisson {
            // Outflow ⇒ pressure pinned (non-deflated); no outflow ⇒ singular ⇒ deflated.
            h.solve(&bp, 0.0, &self.pres_neumann_tags, !self.has_outflow, self.tol, self.maxit)?
        } else if self.has_outflow {
            helmholtz3d_cg_solve_tags(mesh, &bp, self.alpha, 0.0, &self.pres_neumann_tags, self.tol, self.maxit)?
        } else {
            pressure3d_cg_solve(mesh, &bp, self.alpha, self.tol, self.maxit)?
        };
        for (e, el) in mesh.elements.iter().enumerate() {
            let sl = e * nn..(e + 1) * nn;
            let gpx = el.geom.grad_x(refh, &p[sl.clone()]);
            let gpy = el.geom.grad_y(refh, &p[sl.clone()]);
            let gpz = el.geom.grad_z(refh, &p[sl]);
            for k in 0..nn {
                uhx[e * nn + k] -= dt * gpx[k];
                uhy[e * nn + k] -= dt * gpy[k];
                uhz[e * nn + k] -= dt * gpz[k];
            }
        }

        // Stage 3 — viscous Helmholtz per component, per-region velocity-Neumann tags.
        let fxv: Vec<f64> = uhx.iter().map(|v| lambda * v).collect();
        let fyv: Vec<f64> = uhy.iter().map(|v| lambda * v).collect();
        let fzv: Vec<f64> = uhz.iter().map(|v| lambda * v).collect();
        let bx = self.velocity_x.rhs_tagged(&fxv, |tag, x, y, z| vel_dir(tag, x, y, z).0, |_, _, _, _| 0.0);
        let by = self.velocity_y.rhs_tagged(&fyv, |tag, x, y, z| vel_dir(tag, x, y, z).1, |_, _, _, _| 0.0);
        let bz = self.velocity_z.rhs_tagged(&fzv, |tag, x, y, z| vel_dir(tag, x, y, z).2, |_, _, _, _| 0.0);
        let solve_vel = |b: &[f64], neu: &[u32]| -> Result<Vec<f64>, Box<dyn std::error::Error>> {
            Ok(if nc {
                helmholtz3d_nc_cg_solve_tags(mesh, b, self.alpha, lambda, neu, self.tol, self.maxit)?.0
            } else if let Some(h) = self.poisson {
                h.solve(b, lambda, neu, false, self.tol, self.maxit)?.0
            } else {
                helmholtz3d_cg_solve_tags(mesh, b, self.alpha, lambda, neu, self.tol, self.maxit)?.0
            })
        };
        let uxn = solve_vel(&bx, &self.velx_neumann)?;
        let uyn = solve_vel(&by, &self.vely_neumann)?;
        let uzn = solve_vel(&bz, &self.velz_neumann)?;
        Ok((uxn, uyn, uzn))
    }
}

/// A GPU 3D incompressible Navier–Stokes [`gale::sim::StateIntegrator<Mesh3d>`]: the
/// GPU analogue of `gale::sim::DualSplitting3d`. Drives a 3-component velocity field
/// through `Simulation::run` with explicit convection + body force and the pressure +
/// viscous solves on the GPU each step. With [`crate::GpuPenalization3dHook`] it makes
/// GPU flow-past-a-sphere assemblable through the HOOMD API.
pub struct GpuDualSplitting3d {
    dt: f64,
    nu: f64,
    alpha: f64,
    velocity: gale::sim::FieldId,
    bc_u: Box<dyn Fn(f64, f64, f64, f64) -> f64>,
    bc_v: Box<dyn Fn(f64, f64, f64, f64) -> f64>,
    bc_w: Box<dyn Fn(f64, f64, f64, f64) -> f64>,
    #[allow(clippy::type_complexity)]
    body_force: Box<dyn Fn(&gale::sim::State<Mesh3d>, f64) -> (Vec<f64>, Vec<f64>, Vec<f64>)>,
    /// Persistent GPU elliptic handle (P4), built lazily on the first step and reused
    /// across timesteps so the 3 solves/step skip the ~0.3 s context/module/upload setup.
    poisson: RefCell<Option<GpuPoisson3d>>,
}

impl GpuDualSplitting3d {
    pub fn new(velocity: gale::sim::FieldId, dt: f64, nu: f64, alpha: f64) -> Self {
        Self {
            dt,
            nu,
            alpha,
            velocity,
            bc_u: Box::new(|_, _, _, _| 0.0),
            bc_v: Box::new(|_, _, _, _| 0.0),
            bc_w: Box::new(|_, _, _, _| 0.0),
            body_force: Box::new(|s: &gale::sim::State<Mesh3d>, _t: f64| {
                let n = s.ndof();
                (vec![0.0; n], vec![0.0; n], vec![0.0; n])
            }),
            poisson: RefCell::new(None),
        }
    }

    /// Set the Dirichlet velocity boundary conditions `(bc_u, bc_v, bc_w)`.
    pub fn boundary(
        mut self,
        bc_u: impl Fn(f64, f64, f64, f64) -> f64 + 'static,
        bc_v: impl Fn(f64, f64, f64, f64) -> f64 + 'static,
        bc_w: impl Fn(f64, f64, f64, f64) -> f64 + 'static,
    ) -> Self {
        self.bc_u = Box::new(bc_u);
        self.bc_v = Box::new(bc_v);
        self.bc_w = Box::new(bc_w);
        self
    }

    /// Set the nodal body force `(fx, fy, fz)` computed from the state and new time.
    pub fn body_force(
        mut self,
        f: impl Fn(&gale::sim::State<Mesh3d>, f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) + 'static,
    ) -> Self {
        self.body_force = Box::new(f);
        self
    }
}

impl gale::sim::StateIntegrator<Mesh3d> for GpuDualSplitting3d {
    fn dt(&self) -> f64 {
        self.dt
    }

    fn step(&self, state: &mut gale::sim::State<Mesh3d>, hook: &dyn gale::sim::StateStageHook<Mesh3d>) {
        let t_new = state.time.t + self.dt;
        ensure_poisson3d_handle(&self.poisson, &state.mesh, self.alpha);
        let handle = self.poisson.borrow();
        let mut stokes = GpuStokes3d::new(&state.mesh, self.alpha, self.nu, self.dt);
        if let Some(h) = handle.as_ref() {
            stokes = stokes.with_handle(h);
        }
        let (ux, uy, uz) = {
            let v = state.fields.by_id(self.velocity);
            (v.component(0).to_vec(), v.component(1).to_vec(), v.component(2).to_vec())
        };
        let (bx, by, bz) = (self.body_force)(state, t_new);
        let (nux, nuy, nuz) = stokes
            .step_ns_forced(&ux, &uy, &uz, t_new, &self.bc_u, &self.bc_v, &self.bc_w, &bx, &by, &bz)
            .expect("gale-gpu: GpuDualSplitting3d step failed");
        {
            let v = state.fields.by_id_mut(self.velocity);
            v.component_mut(0).copy_from_slice(&nux);
            v.component_mut(1).copy_from_slice(&nuy);
            v.component_mut(2).copy_from_slice(&nuz);
        }
        hook.after_stage(state, 0);
        state.time.t = t_new;
        state.time.step += 1;
    }
}

// ===== 3D viscoelastic coupling (Oldroyd-B) ======================================

use gale::dg::{
    log_conformation3, upwind_advection_lift3, ConformationInflow3d, LogConfOldroydB3d, OldroydB3d,
};
use gale::sim::ViscoModel;

fn axpy6(a: &[Vec<f64>; 6], k: &[Vec<f64>; 6], s: f64) -> [Vec<f64>; 6] {
    std::array::from_fn(|v| a[v].iter().zip(&k[v]).map(|(x, d)| x + s * d).collect())
}
fn combine6(a: &[Vec<f64>; 6], wa: f64, b: &[Vec<f64>; 6], wb: f64) -> [Vec<f64>; 6] {
    std::array::from_fn(|v| a[v].iter().zip(&b[v]).map(|(x, y)| wa * x + wb * y).collect())
}

/// One SSP-RK3 step of the 3D Oldroyd-B conformation transport with a fixed velocity,
/// evaluating the rhs on the **GPU** ([`crate::oldroyd3d_conf_rhs`]). Mirrors
/// `gale::dg::OldroydB3d::step_ssp_rk3` (host axpy/combine, GPU rhs).
#[allow(clippy::type_complexity)]
fn oldroyd3d_advance_gpu(
    mesh: &Mesh3d,
    c: &[Vec<f64>; 6],
    ux: &[f64],
    uy: &[f64],
    uz: &[f64],
    dt: f64,
    lambda: f64,
    inflow: Option<&ConformationInflow3d>,
) -> Result<[Vec<f64>; 6], Box<dyn std::error::Error>> {
    // Full rhs = GPU device kernel (collocation volume) + host upwind surface lift.
    let rhs = |cc: &[Vec<f64>; 6]| -> Result<[Vec<f64>; 6], Box<dyn std::error::Error>> {
        let mut k = crate::oldroyd3d_conf_rhs(mesh, cc, ux, uy, uz, lambda)?;
        let lift = upwind_advection_lift3(mesh, cc, ux, uy, uz, |tag| {
            inflow.filter(|i| i.tags.contains(&tag)).map(|i| i.c)
        });
        for o in 0..6 {
            for g in 0..k[o].len() {
                k[o][g] += lift[o][g];
            }
        }
        Ok(k)
    };
    let k0 = rhs(c)?;
    let u1 = axpy6(c, &k0, dt);
    let k1 = rhs(&u1)?;
    let u2a = axpy6(&u1, &k1, dt);
    let u2 = combine6(c, 0.75, &u2a, 0.25);
    let k2 = rhs(&u2)?;
    let u3a = axpy6(&u2, &k2, dt);
    Ok(combine6(c, 1.0 / 3.0, &u3a, 2.0 / 3.0))
}

/// One SSP-RK3 step of the 3D log-conformation transport with a fixed velocity,
/// evaluating the rhs on the **GPU** ([`crate::logconf3d_psi_rhs`]). Mirrors
/// `gale::dg::LogConfOldroydB3d::step_ssp_rk3`.
#[allow(clippy::too_many_arguments)]
fn logconf3d_advance_gpu(
    mesh: &Mesh3d,
    lc: &LogConfOldroydB3d,
    psi: &[Vec<f64>; 6],
    ux: &[f64],
    uy: &[f64],
    uz: &[f64],
    dt: f64,
    inflow: Option<&ConformationInflow3d>,
) -> Result<[Vec<f64>; 6], Box<dyn std::error::Error>> {
    // Full rhs = GPU device kernel + host upwind lift (inflow C_in enters as log C_in).
    let rhs = |pp: &[Vec<f64>; 6]| -> Result<[Vec<f64>; 6], Box<dyn std::error::Error>> {
        let mut k = crate::logconf3d_psi_rhs(mesh, lc, pp, ux, uy, uz)?;
        let lift = upwind_advection_lift3(mesh, pp, ux, uy, uz, |tag| {
            inflow.filter(|i| i.tags.contains(&tag)).map(|i| log_conformation3(i.c))
        });
        for o in 0..6 {
            for g in 0..k[o].len() {
                k[o][g] += lift[o][g];
            }
        }
        Ok(k)
    };
    let k0 = rhs(psi)?;
    let u1 = axpy6(psi, &k0, dt);
    let k1 = rhs(&u1)?;
    let u2a = axpy6(&u1, &k1, dt);
    let u2 = combine6(psi, 0.75, &u2a, 0.25);
    let k2 = rhs(&u2)?;
    let u3a = axpy6(&u2, &k2, dt);
    Ok(combine6(psi, 1.0 / 3.0, &u3a, 2.0 / 3.0))
}

/// **Coupled GPU 3D viscoelastic** integrator, the 3D analogue of
/// [`crate::GpuViscoelasticDualSplitting`]. One dual-split advance of the 3-component
/// velocity **and** the 6-component conformation: velocity updates on the GPU
/// ([`GpuStokes3d::step_ns_forced`]) using `∇·τ_p` from the *old* conformation, then
/// the conformation advances (GPU SSP-RK3) with the *new* velocity. Supports both the
/// direct Oldroyd-B and the log-conformation models (the latter via the on-device 3×3
/// eigensolver in [`crate::logconf3d_psi_rhs`]). `∇·τ_p` (cheap, element-local) and the
/// stress map use the validated host constitutive model.
pub struct GpuViscoelasticDualSplitting3d {
    pub dt: f64,
    pub eta_s: f64,
    pub eta_p: f64,
    pub lambda: f64,
    pub alpha: f64,
    pub model: ViscoModel,
    velocity: gale::sim::FieldId,
    conformation: gale::sim::FieldId,
    bc_u: Box<dyn Fn(f64, f64, f64, f64) -> f64>,
    bc_v: Box<dyn Fn(f64, f64, f64, f64) -> f64>,
    bc_w: Box<dyn Fn(f64, f64, f64, f64) -> f64>,
    fx: Box<dyn Fn(f64, f64, f64, f64) -> f64>,
    fy: Box<dyn Fn(f64, f64, f64, f64) -> f64>,
    fz: Box<dyn Fn(f64, f64, f64, f64) -> f64>,
    /// Optional conformation inflow boundary data (incoming polymer state at an inlet).
    inflow: Option<ConformationInflow3d>,
    /// Persistent GPU elliptic handle (P4), built lazily and reused across timesteps.
    poisson: RefCell<Option<GpuPoisson3d>>,
}

impl GpuViscoelasticDualSplitting3d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        velocity: gale::sim::FieldId,
        conformation: gale::sim::FieldId,
        dt: f64,
        eta_s: f64,
        eta_p: f64,
        lambda: f64,
        alpha: f64,
        model: ViscoModel,
    ) -> Self {
        Self {
            dt,
            eta_s,
            eta_p,
            lambda,
            alpha,
            model,
            velocity,
            conformation,
            bc_u: Box::new(|_, _, _, _| 0.0),
            bc_v: Box::new(|_, _, _, _| 0.0),
            bc_w: Box::new(|_, _, _, _| 0.0),
            fx: Box::new(|_, _, _, _| 0.0),
            fy: Box::new(|_, _, _, _| 0.0),
            fz: Box::new(|_, _, _, _| 0.0),
            inflow: None,
            poisson: RefCell::new(None),
        }
    }

    pub fn boundary(
        mut self,
        bc_u: impl Fn(f64, f64, f64, f64) -> f64 + 'static,
        bc_v: impl Fn(f64, f64, f64, f64) -> f64 + 'static,
        bc_w: impl Fn(f64, f64, f64, f64) -> f64 + 'static,
    ) -> Self {
        self.bc_u = Box::new(bc_u);
        self.bc_v = Box::new(bc_v);
        self.bc_w = Box::new(bc_w);
        self
    }

    /// Set the conformation inflow boundary data (incoming polymer state at an inlet).
    pub fn conformation_inflow(mut self, inflow: ConformationInflow3d) -> Self {
        self.inflow = Some(inflow);
        self
    }

    pub fn drive(
        mut self,
        fx: impl Fn(f64, f64, f64, f64) -> f64 + 'static,
        fy: impl Fn(f64, f64, f64, f64) -> f64 + 'static,
        fz: impl Fn(f64, f64, f64, f64) -> f64 + 'static,
    ) -> Self {
        self.fx = Box::new(fx);
        self.fy = Box::new(fy);
        self.fz = Box::new(fz);
        self
    }

    /// Equilibrium state over `state.mesh`: `C = I` (`[1,0,0,1,0,1]`) for Oldroyd-B,
    /// `Ψ = log I = 0` for log-conformation.
    pub fn equilibrium(&self, state: &gale::sim::State<Mesh3d>) -> [Vec<f64>; 6] {
        match self.model {
            ViscoModel::OldroydB => OldroydB3d::new(&state.mesh, self.lambda, self.eta_p).identity(),
            ViscoModel::LogConf => {
                LogConfOldroydB3d::new(&state.mesh, self.lambda, self.eta_p).identity()
            }
        }
    }
}

impl gale::sim::StateIntegrator<Mesh3d> for GpuViscoelasticDualSplitting3d {
    fn dt(&self) -> f64 {
        self.dt
    }

    fn step(&self, state: &mut gale::sim::State<Mesh3d>, hook: &dyn gale::sim::StateStageHook<Mesh3d>) {
        let t_new = state.time.t + self.dt;
        let (nux, nuy, nuz, nc) = {
            let v = state.fields.by_id(self.velocity);
            let (ux, uy, uz) = (v.component(0).to_vec(), v.component(1).to_vec(), v.component(2).to_vec());
            let cf = state.fields.by_id(self.conformation);
            let c: [Vec<f64>; 6] = std::array::from_fn(|o| cf.component(o).to_vec());

            // Momentum body force: ∇·τ_p (host, from OLD conformation) + drive.
            let (mut bx, mut by, mut bz) = match self.model {
                ViscoModel::OldroydB => {
                    OldroydB3d::new(&state.mesh, self.lambda, self.eta_p).stress_divergence(&c)
                }
                ViscoModel::LogConf => {
                    LogConfOldroydB3d::new(&state.mesh, self.lambda, self.eta_p).stress_divergence(&c)
                }
            };
            let nn = state.mesh.refh.n_nodes();
            for (e, el) in state.mesh.elements.iter().enumerate() {
                for k in 0..nn {
                    let (x, y, z) = (el.geom.x[k], el.geom.y[k], el.geom.z[k]);
                    bx[e * nn + k] += (self.fx)(x, y, z, t_new);
                    by[e * nn + k] += (self.fy)(x, y, z, t_new);
                    bz[e * nn + k] += (self.fz)(x, y, z, t_new);
                }
            }
            ensure_poisson3d_handle(&self.poisson, &state.mesh, self.alpha);
            let handle = self.poisson.borrow();
            let mut stokes = GpuStokes3d::new(&state.mesh, self.alpha, self.eta_s, self.dt);
            if let Some(h) = handle.as_ref() {
                stokes = stokes.with_handle(h);
            }
            let (nux, nuy, nuz) = stokes
                .step_ns_forced(&ux, &uy, &uz, t_new, &self.bc_u, &self.bc_v, &self.bc_w, &bx, &by, &bz)
                .expect("gale-gpu: 3D viscoelastic velocity step failed");
            let nc = match self.model {
                ViscoModel::OldroydB => {
                    oldroyd3d_advance_gpu(&state.mesh, &c, &nux, &nuy, &nuz, self.dt, self.lambda, self.inflow.as_ref())
                }
                ViscoModel::LogConf => {
                    let lc = LogConfOldroydB3d::new(&state.mesh, self.lambda, self.eta_p);
                    logconf3d_advance_gpu(&state.mesh, &lc, &c, &nux, &nuy, &nuz, self.dt, self.inflow.as_ref())
                }
            }
            .expect("gale-gpu: 3D viscoelastic conformation advance failed");
            (nux, nuy, nuz, nc)
        };
        {
            let v = state.fields.by_id_mut(self.velocity);
            v.component_mut(0).copy_from_slice(&nux);
            v.component_mut(1).copy_from_slice(&nuy);
            v.component_mut(2).copy_from_slice(&nuz);
        }
        {
            let cf = state.fields.by_id_mut(self.conformation);
            for o in 0..6 {
                cf.component_mut(o).copy_from_slice(&nc[o]);
            }
        }
        hook.after_stage(state, 0);
        state.time.t = t_new;
        state.time.step += 1;
    }
}
