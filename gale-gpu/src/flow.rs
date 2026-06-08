//! GPU unsteady **Stokes** solver — build-order step 2 (`docs/implicit-solver-strategy.md §7`).
//!
//! First-order (BDF1) dual-splitting (Chorin / Karniadakis–Israeli–Orszag), the GPU
//! analogue of `gale::dg::Stokes`. Per step (`∂ₜu = −∇p + ν∇²u + f`, `∇·u = 0`):
//! 1. explicit  `û  = uⁿ + Δt f`
//! 2. project   `∇²p = (1/Δt)∇·û` (Neumann) → `û̂ = û − Δt∇p`
//! 3. viscous   `(λM + A) uⁿ⁺¹ = λM û̂` (Helmholtz, Dirichlet), `λ = 1/(νΔt)`
//!
//! The two **elliptic solves are on the GPU** — the deflated pressure-Poisson
//! ([`crate::pressure_cg_solve`]) and the viscous-velocity Helmholtz
//! ([`crate::helmholtz_cg_solve`]), which are the bottleneck kernels per the strategy
//! doc. The cheap O(N) element-local assembly (explicit predictor, divergence,
//! gradient correction, SIPG RHS lifting) reuses the validated host `gale::dg`
//! machinery. Bit-for-bit (to solver tolerance) equal to `gale::dg::Stokes::step`.

use crate::operators::poisson::{helmholtz_cg_solve, helmholtz_cg_solve_tags, pressure_cg_solve};
use crate::operators::poisson_nc::{poisson_nc_cg_solve, pressure_nc_cg_solve};

/// True if the mesh has any 2:1 non-conforming interface (hanging nodes). The GPU
/// elliptic solves must then route through the mortar-capable NC path; conforming
/// meshes use the (faster) uniform path.
fn mesh_is_nonconforming(mesh: &Mesh2d) -> bool {
    use gale::dg::Neighbor;
    mesh.elements.iter().any(|el| {
        el.neighbors.iter().any(|n| matches!(n, Neighbor::CoarseToFine { .. } | Neighbor::FineToCoarse { .. }))
    })
}
use gale::dg::{
    log_conformation, upwind_advection_lift, BoundaryConditions, ConformationInflow,
    ConstitutiveModel, ConvectionScheme, Hyperbolic, IncompressibleConvection, LogConfOldroydB,
    Mesh2d, OldroydB, Poisson, VolumeForm,
};

type StepResult = Result<(Vec<f64>, Vec<f64>), Box<dyn std::error::Error>>;

/// GPU unsteady-Stokes stepper. Holds the host `Poisson` operators used only for SIPG
/// RHS assembly; the operator applies / solves run on the GPU.
pub struct GpuStokes<'m> {
    pub mesh: &'m Mesh2d,
    pub nu: f64,
    pub dt: f64,
    /// How the nonlinear convection `(u·∇)u` is discretized in [`step_ns`](Self::step_ns).
    pub convection_scheme: ConvectionScheme,
    alpha: f64,
    /// Pressure-Poisson assembly (pure Neumann, singular).
    pressure: Poisson<'m>,
    /// Velocity Helmholtz assembly `(λM + A)` (Dirichlet).
    velocity: Poisson<'m>,
    /// Whether any boundary is an outflow (pressure pinned ⇒ non-deflated solve).
    has_outflow: bool,
    /// Outflow boundary tags (velocity-Neumann faces for the GPU Helmholtz solve).
    outflow_tags: Vec<u32>,
    /// Pressure-Poisson Neumann tags (everything except outflow) for the GPU solve.
    pres_neumann_tags: Vec<u32>,
    tol: f64,
    maxit: usize,
}

impl<'m> GpuStokes<'m> {
    /// Closed-box solver: all-Dirichlet velocity (data from the `bc_u`/`bc_v` closures)
    /// + pure-Neumann (deflated) pressure. For per-region inflow/outflow/wall
    /// conditions use [`with_bcs`](Self::with_bcs).
    pub fn new(mesh: &'m Mesh2d, alpha: f64, nu: f64, dt: f64) -> Self {
        let lambda = 1.0 / (nu * dt);
        Self {
            mesh,
            nu,
            dt,
            convection_scheme: ConvectionScheme::Nodal,
            alpha,
            pressure: Poisson::with_bc(mesh, alpha, 0.0, mesh.boundary_tags()),
            velocity: Poisson::with_reaction(mesh, alpha, lambda),
            has_outflow: false,
            outflow_tags: Vec::new(),
            pres_neumann_tags: mesh.boundary_tags(),
            tol: 1e-10,
            maxit: 20000,
        }
    }

    /// Solver with **per-region** boundary conditions — the GPU analogue of
    /// `gale::dg::Stokes::with_bcs`. Each boundary tag is routed via `bcs` to the right
    /// pair of operator settings (no-slip/inflow ⇒ velocity-Dirichlet + pressure-Neumann;
    /// outflow ⇒ velocity-Neumann + pressure-Dirichlet `p=0`). The pinned pressure lets
    /// the deflated solve be replaced by a plain CG. Use the `*_bc` step methods.
    pub fn with_bcs(mesh: &'m Mesh2d, alpha: f64, nu: f64, dt: f64, bcs: &BoundaryConditions) -> Self {
        let lambda = 1.0 / (nu * dt);
        let outflow = bcs.outflow_tags(mesh);
        let pres_neumann: Vec<u32> = mesh
            .boundary_tags()
            .into_iter()
            .filter(|t| !outflow.contains(t))
            .collect();
        Self {
            mesh,
            nu,
            dt,
            convection_scheme: ConvectionScheme::Nodal,
            alpha,
            pressure: Poisson::with_bc(mesh, alpha, 0.0, pres_neumann.clone()),
            velocity: Poisson::with_bc(mesh, alpha, lambda, outflow.clone()),
            has_outflow: !outflow.is_empty(),
            outflow_tags: outflow,
            pres_neumann_tags: pres_neumann,
            tol: 1e-10,
            maxit: 20000,
        }
    }

    fn ndof(&self) -> usize {
        self.mesh.n_elements() * self.mesh.refq.n_nodes()
    }

    /// Per-element divergence `∂x a + ∂y b` of a vector field `(a, b)`.
    fn divergence(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        let refq = &self.mesh.refq;
        let nn = refq.n_nodes();
        let mut d = vec![0.0; self.ndof()];
        for (e, el) in self.mesh.elements.iter().enumerate() {
            let ax = el.geom.grad_x(refq, &a[e * nn..(e + 1) * nn]);
            let by = el.geom.grad_y(refq, &b[e * nn..(e + 1) * nn]);
            for k in 0..nn {
                d[e * nn + k] = ax[k] + by[k];
            }
        }
        d
    }

    /// Advance one BDF1 dual-splitting Stokes step to time `t` (the new level).
    /// `bc_u`,`bc_v` are Dirichlet velocity on the boundary; `fx`,`fy` the forcing.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        ux: &[f64],
        uy: &[f64],
        t: f64,
        bc_u: impl Fn(f64, f64, f64) -> f64,
        bc_v: impl Fn(f64, f64, f64) -> f64,
        fx: impl Fn(f64, f64, f64) -> f64,
        fy: impl Fn(f64, f64, f64) -> f64,
    ) -> StepResult {
        let mesh = self.mesh;
        let nn = mesh.refq.n_nodes();
        let dt = self.dt;

        // Stage 1 — explicit: û = uⁿ + Δt f.
        let mut uhx = ux.to_vec();
        let mut uhy = uy.to_vec();
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let (x, y) = (el.geom.x[k], el.geom.y[k]);
                uhx[e * nn + k] += dt * fx(x, y, t);
                uhy[e * nn + k] += dt * fy(x, y, t);
            }
        }
        self.project_and_diffuse(uhx, uhy, t, bc_u, bc_v)
    }

    /// One incompressible **Navier–Stokes** step: like [`step`](Self::step) but the
    /// explicit predictor includes the advection term, `û = uⁿ + Δt(f − (uⁿ·∇)uⁿ)`
    /// (explicit — non-stiff at low Reynolds number). The convection is computed on
    /// the host (cheap, element-local); the elliptic solves remain on the GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn step_ns(
        &self,
        ux: &[f64],
        uy: &[f64],
        t: f64,
        bc_u: impl Fn(f64, f64, f64) -> f64,
        bc_v: impl Fn(f64, f64, f64) -> f64,
        fx: impl Fn(f64, f64, f64) -> f64,
        fy: impl Fn(f64, f64, f64) -> f64,
    ) -> StepResult {
        let mesh = self.mesh;
        let nn = mesh.refq.n_nodes();
        let mut force_x = vec![0.0; self.ndof()];
        let mut force_y = vec![0.0; self.ndof()];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let (x, y) = (el.geom.x[k], el.geom.y[k]);
                force_x[e * nn + k] = fx(x, y, t);
                force_y[e * nn + k] = fy(x, y, t);
            }
        }
        self.step_ns_forced(ux, uy, t, bc_u, bc_v, &force_x, &force_y)
    }

    /// Like [`step_ns`](Self::step_ns) but with a **precomputed nodal** body force
    /// `(force_x, force_y)` — e.g. the polymer-stress divergence `∇·τ_p` in
    /// viscoelastic coupling. Mirrors `gale::dg::Stokes::step_ns_forced`.
    #[allow(clippy::too_many_arguments)]
    pub fn step_ns_forced(
        &self,
        ux: &[f64],
        uy: &[f64],
        t: f64,
        bc_u: impl Fn(f64, f64, f64) -> f64,
        bc_v: impl Fn(f64, f64, f64) -> f64,
        force_x: &[f64],
        force_y: &[f64],
    ) -> StepResult {
        let mesh = self.mesh;
        let dt = self.dt;
        let (cx, cy) = match self.convection_scheme {
            ConvectionScheme::Nodal => self.convection(ux, uy),
            ConvectionScheme::SplitFormDg => {
                // (u·∇)u = ∇·(u⊗u) = −rhs of the conservation-law operator (host).
                let op = Hyperbolic::with_options(mesh, IncompressibleConvection, VolumeForm::SplitForm, true);
                let state = vec![ux.to_vec(), uy.to_vec()];
                let conv_bc = |x: f64, y: f64, tt: f64, out: &mut [f64]| {
                    out[0] = bc_u(x, y, tt);
                    out[1] = bc_v(x, y, tt);
                };
                let r = op.rhs(&state, t, &conv_bc);
                (r[0].iter().map(|v| -v).collect(), r[1].iter().map(|v| -v).collect())
            }
        };
        let mut uhx = ux.to_vec();
        let mut uhy = uy.to_vec();
        for i in 0..self.ndof() {
            uhx[i] += dt * (force_x[i] - cx[i]);
            uhy[i] += dt * (force_y[i] - cy[i]);
        }
        self.project_and_diffuse(uhx, uhy, t, bc_u, bc_v)
    }

    /// Nodal (collocation) advective convection `((u·∇)u, (u·∇)v)` (host, element-local).
    fn convection(&self, ux: &[f64], uy: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let refq = &self.mesh.refq;
        let nn = refq.n_nodes();
        let mut cx = vec![0.0; self.ndof()];
        let mut cy = vec![0.0; self.ndof()];
        for (e, el) in self.mesh.elements.iter().enumerate() {
            let sl = e * nn..(e + 1) * nn;
            let dux_dx = el.geom.grad_x(refq, &ux[sl.clone()]);
            let dux_dy = el.geom.grad_y(refq, &ux[sl.clone()]);
            let duy_dx = el.geom.grad_x(refq, &uy[sl.clone()]);
            let duy_dy = el.geom.grad_y(refq, &uy[sl]);
            for k in 0..nn {
                let (u, v) = (ux[e * nn + k], uy[e * nn + k]);
                cx[e * nn + k] = u * dux_dx[k] + v * dux_dy[k];
                cy[e * nn + k] = u * duy_dx[k] + v * duy_dy[k];
            }
        }
        (cx, cy)
    }

    /// Stages 2–3: project the predictor `û` to divergence-free, then the implicit
    /// viscous solve — both elliptic solves on the GPU.
    fn project_and_diffuse(
        &self,
        mut uhx: Vec<f64>,
        mut uhy: Vec<f64>,
        t: f64,
        bc_u: impl Fn(f64, f64, f64) -> f64,
        bc_v: impl Fn(f64, f64, f64) -> f64,
    ) -> StepResult {
        let mesh = self.mesh;
        let refq = &mesh.refq;
        let nn = refq.n_nodes();
        let dt = self.dt;
        let lambda = 1.0 / (self.nu * dt);

        // Stage 2 — pressure projection: ∇²p = (1/Δt)∇·û (homogeneous Neumann).
        // Solver convention A ≈ −∇², so −∇²p = −(1/Δt)∇·û ⇒ fp = −div/dt.
        let div = self.divergence(&uhx, &uhy);
        let fp: Vec<f64> = div.iter().map(|d| -d / dt).collect();
        let bp = self.pressure.rhs_mixed(&fp, |_, _| 0.0, |_, _| 0.0);
        let nc = mesh_is_nonconforming(mesh);
        let (p, _it) = if nc {
            pressure_nc_cg_solve(mesh, &bp, self.alpha, self.tol, self.maxit)?
        } else {
            pressure_cg_solve(mesh, &bp, self.alpha, self.tol, self.maxit)?
        };
        for (e, el) in mesh.elements.iter().enumerate() {
            let gpx = el.geom.grad_x(refq, &p[e * nn..(e + 1) * nn]);
            let gpy = el.geom.grad_y(refq, &p[e * nn..(e + 1) * nn]);
            for k in 0..nn {
                uhx[e * nn + k] -= dt * gpx[k];
                uhy[e * nn + k] -= dt * gpy[k];
            }
        }

        // Stage 3 — viscous Helmholtz per component: (λM + A) uⁿ⁺¹ = λM û̂ + Dirichlet.
        let fxv: Vec<f64> = uhx.iter().map(|v| lambda * v).collect();
        let fyv: Vec<f64> = uhy.iter().map(|v| lambda * v).collect();
        let bx = self.velocity.rhs(&fxv, |x, y| bc_u(x, y, t));
        let by = self.velocity.rhs(&fyv, |x, y| bc_v(x, y, t));
        let (uxn, _) = if nc {
            poisson_nc_cg_solve(mesh, &bx, self.alpha, lambda, self.tol, self.maxit)?
        } else {
            helmholtz_cg_solve(mesh, &bx, self.alpha, lambda, self.tol, self.maxit)?
        };
        let (uyn, _) = if nc {
            poisson_nc_cg_solve(mesh, &by, self.alpha, lambda, self.tol, self.maxit)?
        } else {
            helmholtz_cg_solve(mesh, &by, self.alpha, lambda, self.tol, self.maxit)?
        };
        Ok((uxn, uyn))
    }

    /// BC-aware **Stokes** step (no convection) — the [`with_bcs`](Self::with_bcs)
    /// companion to [`step`](Self::step). Per-region velocity data comes from `bcs`;
    /// `fx`/`fy` are the analytic body force.
    pub fn step_bc(
        &self,
        ux: &[f64],
        uy: &[f64],
        t: f64,
        bcs: &BoundaryConditions,
        fx: impl Fn(f64, f64, f64) -> f64,
        fy: impl Fn(f64, f64, f64) -> f64,
    ) -> StepResult {
        let mesh = self.mesh;
        let nn = mesh.refq.n_nodes();
        let dt = self.dt;
        let mut uhx = ux.to_vec();
        let mut uhy = uy.to_vec();
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let (x, y) = (el.geom.x[k], el.geom.y[k]);
                uhx[e * nn + k] += dt * fx(x, y, t);
                uhy[e * nn + k] += dt * fy(x, y, t);
            }
        }
        self.project_and_diffuse_bc(uhx, uhy, |tag, x, y| bcs.dirichlet(tag, x, y, t))
    }

    /// BC-aware **Navier–Stokes** step with a precomputed nodal body force — the
    /// [`with_bcs`](Self::with_bcs) companion to [`step_ns_forced`](Self::step_ns_forced).
    /// Nodal convection only (the split-form convection BC hook is tag-agnostic).
    pub fn step_ns_forced_bc(
        &self,
        ux: &[f64],
        uy: &[f64],
        t: f64,
        bcs: &BoundaryConditions,
        force_x: &[f64],
        force_y: &[f64],
    ) -> StepResult {
        assert!(
            matches!(self.convection_scheme, ConvectionScheme::Nodal),
            "per-region BCs with split-form convection are not yet supported"
        );
        let dt = self.dt;
        let (cx, cy) = self.convection(ux, uy);
        let mut uhx = ux.to_vec();
        let mut uhy = uy.to_vec();
        for i in 0..self.ndof() {
            uhx[i] += dt * (force_x[i] - cx[i]);
            uhy[i] += dt * (force_y[i] - cy[i]);
        }
        self.project_and_diffuse_bc(uhx, uhy, |tag, x, y| bcs.dirichlet(tag, x, y, t))
    }

    /// Stages 2–3 for the per-region BC path — the [`project_and_diffuse`] analogue
    /// using tag-aware host RHS assembly and the tag-aware GPU elliptic solves. An
    /// outflow pins the pressure (Dirichlet `p=0`) so the deflated solve is replaced by
    /// a plain CG. Conforming meshes only (NC + per-region BCs is a follow-up).
    fn project_and_diffuse_bc(
        &self,
        mut uhx: Vec<f64>,
        mut uhy: Vec<f64>,
        vel_dir: impl Fn(u32, f64, f64) -> (f64, f64),
    ) -> StepResult {
        let mesh = self.mesh;
        let refq = &mesh.refq;
        let nn = refq.n_nodes();
        let dt = self.dt;
        let lambda = 1.0 / (self.nu * dt);
        assert!(
            !mesh_is_nonconforming(mesh),
            "per-region GPU BCs on non-conforming meshes are not yet supported"
        );

        // Stage 2 — pressure projection (homogeneous data either way). With an outflow
        // the operator is non-singular (Dirichlet p=0 there) ⇒ plain Poisson CG; with
        // no outflow it is the singular pure-Neumann system ⇒ deflated CG.
        let div = self.divergence(&uhx, &uhy);
        let fp: Vec<f64> = div.iter().map(|d| -d / dt).collect();
        let bp = self.pressure.rhs_mixed(&fp, |_, _| 0.0, |_, _| 0.0);
        let (p, _it) = if self.has_outflow {
            helmholtz_cg_solve_tags(mesh, &bp, self.alpha, 0.0, &self.pres_neumann_tags, self.tol, self.maxit)?
        } else {
            pressure_cg_solve(mesh, &bp, self.alpha, self.tol, self.maxit)?
        };
        for (e, el) in mesh.elements.iter().enumerate() {
            let gpx = el.geom.grad_x(refq, &p[e * nn..(e + 1) * nn]);
            let gpy = el.geom.grad_y(refq, &p[e * nn..(e + 1) * nn]);
            for k in 0..nn {
                uhx[e * nn + k] -= dt * gpx[k];
                uhy[e * nn + k] -= dt * gpy[k];
            }
        }

        // Stage 3 — viscous Helmholtz per component, outflow tags = velocity-Neumann.
        let fxv: Vec<f64> = uhx.iter().map(|v| lambda * v).collect();
        let fyv: Vec<f64> = uhy.iter().map(|v| lambda * v).collect();
        let bx = self.velocity.rhs_tagged(&fxv, |tag, x, y| vel_dir(tag, x, y).0, |_, _, _| 0.0);
        let by = self.velocity.rhs_tagged(&fyv, |tag, x, y| vel_dir(tag, x, y).1, |_, _, _| 0.0);
        let (uxn, _) = helmholtz_cg_solve_tags(mesh, &bx, self.alpha, lambda, &self.outflow_tags, self.tol, self.maxit)?;
        let (uyn, _) = helmholtz_cg_solve_tags(mesh, &by, self.alpha, lambda, &self.outflow_tags, self.tol, self.maxit)?;
        Ok((uxn, uyn))
    }

    /// Velocity-field L2 norm (uses the velocity operator's mass).
    pub fn l2_norm(&self, v: &[f64]) -> f64 {
        self.velocity.l2_norm(v)
    }
}

/// A GPU unsteady-Stokes [`gale::sim::StateIntegrator`]: drives a 2-component velocity
/// field through the HOOMD-style [`gale::sim::Simulation`] with **the pressure and
/// viscous solves on the GPU each step** (via [`GpuStokes`]). The GPU analogue of
/// `gale::dg::Stokes` wrapped as `gale::sim::DualSplitting` — but Stokes-only for now
/// (no explicit convection; NS convection is a follow-up). Like `DualSplitting`, it
/// builds the transient operator from `state.mesh` each step, so it stores no borrow.
pub struct GpuStokesIntegrator {
    dt: f64,
    nu: f64,
    alpha: f64,
    velocity: gale::sim::FieldId,
    bc_u: Box<dyn Fn(f64, f64, f64) -> f64>,
    bc_v: Box<dyn Fn(f64, f64, f64) -> f64>,
}

impl GpuStokesIntegrator {
    /// New GPU Stokes integrator advancing the 2-component `velocity` field, with
    /// zero-velocity walls by default.
    pub fn new(velocity: gale::sim::FieldId, dt: f64, nu: f64, alpha: f64) -> Self {
        Self {
            dt,
            nu,
            alpha,
            velocity,
            bc_u: Box::new(|_, _, _| 0.0),
            bc_v: Box::new(|_, _, _| 0.0),
        }
    }

    /// Set the Dirichlet velocity boundary conditions `(bc_u, bc_v)` of `(x, y, t)`.
    pub fn boundary(
        mut self,
        bc_u: impl Fn(f64, f64, f64) -> f64 + 'static,
        bc_v: impl Fn(f64, f64, f64) -> f64 + 'static,
    ) -> Self {
        self.bc_u = Box::new(bc_u);
        self.bc_v = Box::new(bc_v);
        self
    }
}

impl gale::sim::StateIntegrator for GpuStokesIntegrator {
    fn dt(&self) -> f64 {
        self.dt
    }

    fn step(&self, state: &mut gale::sim::State, hook: &dyn gale::sim::StateStageHook) {
        let t_new = state.time.t + self.dt;
        let stokes = GpuStokes::new(&state.mesh, self.alpha, self.nu, self.dt);
        let (ux, uy) = {
            let v = state.fields.by_id(self.velocity);
            (v.component(0).to_vec(), v.component(1).to_vec())
        };
        let (nux, nuy) = stokes
            .step(&ux, &uy, t_new, &self.bc_u, &self.bc_v, |_, _, _| 0.0, |_, _, _| 0.0)
            .expect("gale-gpu: GpuStokes step failed");
        {
            let v = state.fields.by_id_mut(self.velocity);
            v.component_mut(0).copy_from_slice(&nux);
            v.component_mut(1).copy_from_slice(&nuy);
        }
        // Per-stage hook: home for the implicit volume-penalization (IBM) projection.
        hook.after_stage(state, 0);
        state.time.t = t_new;
        state.time.step += 1;
    }
}

/// A GPU incompressible **Navier–Stokes** [`gale::sim::StateIntegrator`]: drives a
/// 2-component velocity field through [`gale::sim::Simulation`] with explicit
/// (low-Re) convection + an optional body force, and the pressure + viscous solves on
/// the GPU each step (via [`GpuStokes::step_ns_forced`]). The GPU analogue of
/// `gale::sim::DualSplitting`; builds the transient operator from `state.mesh` each
/// step, so it stores no borrow. The post-step hook is the home for the implicit
/// volume-penalization (IBM) projection.
pub struct GpuDualSplitting {
    dt: f64,
    nu: f64,
    alpha: f64,
    convection_scheme: ConvectionScheme,
    velocity: gale::sim::FieldId,
    bc_u: Box<dyn Fn(f64, f64, f64) -> f64>,
    bc_v: Box<dyn Fn(f64, f64, f64) -> f64>,
    /// Per-region BCs. `Some` ⇒ the [`GpuStokes::with_bcs`] path; `None` ⇒ the legacy
    /// `bc_u`/`bc_v` all-Dirichlet path.
    bcs: Option<BoundaryConditions>,
    #[allow(clippy::type_complexity)]
    body_force: Box<dyn Fn(&gale::sim::State, f64) -> (Vec<f64>, Vec<f64>)>,
}

impl GpuDualSplitting {
    /// New GPU NS integrator advancing the 2-component `velocity` field, with
    /// zero-velocity walls and no body force by default (nodal convection).
    pub fn new(velocity: gale::sim::FieldId, dt: f64, nu: f64, alpha: f64) -> Self {
        Self {
            dt,
            nu,
            alpha,
            convection_scheme: ConvectionScheme::Nodal,
            velocity,
            bc_u: Box::new(|_, _, _| 0.0),
            bc_v: Box::new(|_, _, _| 0.0),
            bcs: None,
            body_force: Box::new(|s: &gale::sim::State, _t: f64| {
                let n = s.ndof();
                (vec![0.0; n], vec![0.0; n])
            }),
        }
    }

    /// Set the Dirichlet velocity boundary conditions `(bc_u, bc_v)` of `(x, y, t)`.
    pub fn boundary(
        mut self,
        bc_u: impl Fn(f64, f64, f64) -> f64 + 'static,
        bc_v: impl Fn(f64, f64, f64) -> f64 + 'static,
    ) -> Self {
        self.bc_u = Box::new(bc_u);
        self.bc_v = Box::new(bc_v);
        self
    }

    /// Set **per-region** boundary conditions (inflow / outflow / walls by tag),
    /// overriding the single global [`boundary`](Self::boundary) closure. Routes the
    /// GPU flow step through [`GpuStokes::with_bcs`].
    pub fn boundary_conditions(mut self, bcs: BoundaryConditions) -> Self {
        self.bcs = Some(bcs);
        self
    }

    /// Set the nodal body force `(fx, fy)`, computed from the full state and the new
    /// time level — e.g. `∇·τ_p` plus an external drive.
    pub fn body_force(
        mut self,
        f: impl Fn(&gale::sim::State, f64) -> (Vec<f64>, Vec<f64>) + 'static,
    ) -> Self {
        self.body_force = Box::new(f);
        self
    }

    /// Use the energy-stable split-form DG convection (the high-Re path).
    pub fn split_form_convection(mut self) -> Self {
        self.convection_scheme = ConvectionScheme::SplitFormDg;
        self
    }
}

impl gale::sim::StateIntegrator for GpuDualSplitting {
    fn dt(&self) -> f64 {
        self.dt
    }

    fn step(&self, state: &mut gale::sim::State, hook: &dyn gale::sim::StateStageHook) {
        let t_new = state.time.t + self.dt;
        let (ux, uy) = {
            let v = state.fields.by_id(self.velocity);
            (v.component(0).to_vec(), v.component(1).to_vec())
        };
        let (bx, by) = (self.body_force)(state, t_new);
        let (nux, nuy) = if let Some(bcs) = &self.bcs {
            let mut stokes = GpuStokes::with_bcs(&state.mesh, self.alpha, self.nu, self.dt, bcs);
            stokes.convection_scheme = self.convection_scheme;
            stokes
                .step_ns_forced_bc(&ux, &uy, t_new, bcs, &bx, &by)
                .expect("gale-gpu: GpuDualSplitting BC step failed")
        } else {
            let mut stokes = GpuStokes::new(&state.mesh, self.alpha, self.nu, self.dt);
            stokes.convection_scheme = self.convection_scheme;
            stokes
                .step_ns_forced(&ux, &uy, t_new, &self.bc_u, &self.bc_v, &bx, &by)
                .expect("gale-gpu: GpuDualSplitting step failed")
        };
        {
            let v = state.fields.by_id_mut(self.velocity);
            v.component_mut(0).copy_from_slice(&nux);
            v.component_mut(1).copy_from_slice(&nuy);
        }
        hook.after_stage(state, 0);
        state.time.t = t_new;
        state.time.step += 1;
    }
}

// ===== Viscoelastic coupling (build-order step 4) ===============================

/// `a + s·k` componentwise over a symmetric-tensor triple.
fn axpy3(a: &[Vec<f64>; 3], k: &[Vec<f64>; 3], s: f64) -> [Vec<f64>; 3] {
    std::array::from_fn(|v| a[v].iter().zip(&k[v]).map(|(x, d)| x + s * d).collect())
}
/// `wa·a + wb·b` componentwise over a symmetric-tensor triple.
fn combine3(a: &[Vec<f64>; 3], wa: f64, b: &[Vec<f64>; 3], wb: f64) -> [Vec<f64>; 3] {
    std::array::from_fn(|v| a[v].iter().zip(&b[v]).map(|(x, y)| wa * x + wb * y).collect())
}

type ConfResult = Result<[Vec<f64>; 3], Box<dyn std::error::Error>>;

/// One SSP-RK3 step of the Oldroyd-B conformation transport with a fixed velocity,
/// evaluating the rhs on the **GPU** ([`crate::oldroyd_conf_rhs`]). Mirrors
/// `gale::dg::OldroydB::step_ssp_rk3` (host axpy/combine, GPU rhs).
fn oldroyd_advance_gpu(
    mesh: &Mesh2d,
    c: &[Vec<f64>; 3],
    ux: &[f64],
    uy: &[f64],
    dt: f64,
    lambda: f64,
    inflow: Option<&ConformationInflow>,
) -> ConfResult {
    // Full rhs = GPU device kernel (collocation volume term) + host upwind surface lift
    // (cheap O(N), the inter-element transport + inflow injection).
    let rhs = |cc: &[Vec<f64>; 3]| -> ConfResult {
        let mut k = crate::oldroyd_conf_rhs(mesh, cc, ux, uy, lambda)?;
        let lift = upwind_advection_lift(mesh, cc, ux, uy, |tag| {
            inflow.filter(|i| i.tags.contains(&tag)).map(|i| i.c)
        });
        for comp in 0..3 {
            for g in 0..k[comp].len() {
                k[comp][g] += lift[comp][g];
            }
        }
        Ok(k)
    };
    let k0 = rhs(c)?;
    let u1 = axpy3(c, &k0, dt);
    let k1 = rhs(&u1)?;
    let u2a = axpy3(&u1, &k1, dt);
    let u2 = combine3(c, 0.75, &u2a, 0.25);
    let k2 = rhs(&u2)?;
    let u3a = axpy3(&u2, &k2, dt);
    Ok(combine3(c, 1.0 / 3.0, &u3a, 2.0 / 3.0))
}

/// One SSP-RK3 step of the log-conformation transport with a fixed velocity,
/// evaluating the rhs on the **GPU** ([`crate::logconf_psi_rhs`]). Mirrors
/// `gale::dg::LogConfOldroydB::step_ssp_rk3`.
fn logconf_advance_gpu(
    mesh: &Mesh2d,
    lc: &LogConfOldroydB,
    psi: &[Vec<f64>; 3],
    ux: &[f64],
    uy: &[f64],
    dt: f64,
    inflow: Option<&ConformationInflow>,
) -> ConfResult {
    // Full rhs = GPU device kernel + host upwind lift. The advected variable is Ψ, so
    // the inflow conformation C_in enters as Ψ_in = log C_in.
    let rhs = |pp: &[Vec<f64>; 3]| -> ConfResult {
        let mut k = crate::logconf_psi_rhs(mesh, lc, pp, ux, uy)?;
        let lift = upwind_advection_lift(mesh, pp, ux, uy, |tag| {
            inflow.filter(|i| i.tags.contains(&tag)).map(|i| log_conformation(i.c))
        });
        for comp in 0..3 {
            for g in 0..k[comp].len() {
                k[comp][g] += lift[comp][g];
            }
        }
        Ok(k)
    };
    let k0 = rhs(psi)?;
    let u1 = axpy3(psi, &k0, dt);
    let k1 = rhs(&u1)?;
    let u2a = axpy3(&u1, &k1, dt);
    let u2 = combine3(psi, 0.75, &u2a, 0.25);
    let k2 = rhs(&u2)?;
    let u3a = axpy3(&u2, &k2, dt);
    Ok(combine3(psi, 1.0 / 3.0, &u3a, 2.0 / 3.0))
}

/// **Coupled GPU viscoelastic** integrator: one dual-splitting advance of velocity
/// **and** the conformation field, the GPU analogue of
/// `gale::sim::ViscoelasticDualSplitting` (which wraps `gale::dg::ViscoelasticFlow`).
/// Split order: the velocity updates (on the GPU, via [`GpuStokes::step_ns_forced`])
/// using `∇·τ_p` from the *old* conformation, then the conformation advances (GPU
/// SSP-RK3 rhs) with the *new* velocity. `∇·τ_p` (cheap, element-local) and the
/// stress map use the validated host constitutive model.
pub struct GpuViscoelasticDualSplitting {
    pub dt: f64,
    pub eta_s: f64,
    pub eta_p: f64,
    pub lambda: f64,
    pub alpha: f64,
    pub model: gale::sim::ViscoModel,
    velocity: gale::sim::FieldId,
    conformation: gale::sim::FieldId,
    bc_u: Box<dyn Fn(f64, f64, f64) -> f64>,
    bc_v: Box<dyn Fn(f64, f64, f64) -> f64>,
    fx: Box<dyn Fn(f64, f64, f64) -> f64>,
    fy: Box<dyn Fn(f64, f64, f64) -> f64>,
    /// Optional conformation inflow boundary data (incoming polymer state at an inlet).
    inflow: Option<ConformationInflow>,
}

impl GpuViscoelasticDualSplitting {
    /// New coupled integrator over the 2-component `velocity` and 3-component
    /// `conformation` (symmetric tensor `(xx, xy, yy)`) fields. Zero walls and no
    /// external drive by default.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        velocity: gale::sim::FieldId,
        conformation: gale::sim::FieldId,
        dt: f64,
        eta_s: f64,
        eta_p: f64,
        lambda: f64,
        alpha: f64,
        model: gale::sim::ViscoModel,
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
            bc_u: Box::new(|_, _, _| 0.0),
            bc_v: Box::new(|_, _, _| 0.0),
            fx: Box::new(|_, _, _| 0.0),
            fy: Box::new(|_, _, _| 0.0),
            inflow: None,
        }
    }

    /// Set the Dirichlet velocity boundary conditions `(bc_u, bc_v)`.
    pub fn boundary(
        mut self,
        bc_u: impl Fn(f64, f64, f64) -> f64 + 'static,
        bc_v: impl Fn(f64, f64, f64) -> f64 + 'static,
    ) -> Self {
        self.bc_u = Box::new(bc_u);
        self.bc_v = Box::new(bc_v);
        self
    }

    /// Set the conformation inflow boundary data (the incoming polymer state at an
    /// inlet), pinning the upwind trace there in the conformation transport.
    pub fn conformation_inflow(mut self, inflow: ConformationInflow) -> Self {
        self.inflow = Some(inflow);
        self
    }

    /// Set the external body force / drive `(fx, fy)` of `(x, y, t)`.
    pub fn drive(
        mut self,
        fx: impl Fn(f64, f64, f64) -> f64 + 'static,
        fy: impl Fn(f64, f64, f64) -> f64 + 'static,
    ) -> Self {
        self.fx = Box::new(fx);
        self.fy = Box::new(fy);
        self
    }

    /// The model's equilibrium conformation `[xx, xy, yy]` over `state.mesh` (identity
    /// for Oldroyd-B; `Ψ = log I = 0` for log-conformation). Use to initialize.
    pub fn equilibrium(&self, state: &gale::sim::State) -> [Vec<f64>; 3] {
        match self.model {
            gale::sim::ViscoModel::OldroydB => {
                OldroydB::new(&state.mesh, self.lambda, self.eta_p).equilibrium()
            }
            gale::sim::ViscoModel::LogConf => {
                LogConfOldroydB::new(&state.mesh, self.lambda, self.eta_p).equilibrium()
            }
        }
    }

    /// Total momentum body force `∇·τ_p + (fx, fy)` from the old conformation `c`.
    fn body_force(&self, mesh: &Mesh2d, c: &[Vec<f64>; 3], t: f64) -> (Vec<f64>, Vec<f64>) {
        let (mut bx, mut by) = match self.model {
            gale::sim::ViscoModel::OldroydB => {
                OldroydB::new(mesh, self.lambda, self.eta_p).stress_div(c)
            }
            gale::sim::ViscoModel::LogConf => {
                LogConfOldroydB::new(mesh, self.lambda, self.eta_p).stress_div(c)
            }
        };
        let nn = mesh.refq.n_nodes();
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let (x, y) = (el.geom.x[k], el.geom.y[k]);
                bx[e * nn + k] += (self.fx)(x, y, t);
                by[e * nn + k] += (self.fy)(x, y, t);
            }
        }
        (bx, by)
    }
}

impl gale::sim::StateIntegrator for GpuViscoelasticDualSplitting {
    fn dt(&self) -> f64 {
        self.dt
    }

    fn step(&self, state: &mut gale::sim::State, hook: &dyn gale::sim::StateStageHook) {
        let t_new = state.time.t + self.dt;
        let (nux, nuy, npsi) = {
            let (ux, uy) = {
                let v = state.fields.by_id(self.velocity);
                (v.component(0).to_vec(), v.component(1).to_vec())
            };
            let c = {
                let f = state.fields.by_id(self.conformation);
                [f.component(0).to_vec(), f.component(1).to_vec(), f.component(2).to_vec()]
            };
            // Momentum: GPU velocity using ∇·τ_p from the OLD conformation + drive.
            let (bx, by) = self.body_force(&state.mesh, &c, t_new);
            let stokes = GpuStokes::new(&state.mesh, self.alpha, self.eta_s, self.dt);
            let (nux, nuy) = stokes
                .step_ns_forced(&ux, &uy, t_new, &self.bc_u, &self.bc_v, &bx, &by)
                .expect("gale-gpu: viscoelastic velocity step failed");
            // Constitutive: GPU SSP-RK3 conformation transport with the NEW velocity.
            let npsi = match self.model {
                gale::sim::ViscoModel::OldroydB => {
                    oldroyd_advance_gpu(&state.mesh, &c, &nux, &nuy, self.dt, self.lambda, self.inflow.as_ref())
                }
                gale::sim::ViscoModel::LogConf => {
                    let lc = LogConfOldroydB::new(&state.mesh, self.lambda, self.eta_p);
                    logconf_advance_gpu(&state.mesh, &lc, &c, &nux, &nuy, self.dt, self.inflow.as_ref())
                }
            }
            .expect("gale-gpu: viscoelastic conformation advance failed");
            (nux, nuy, npsi)
        };
        {
            let v = state.fields.by_id_mut(self.velocity);
            v.component_mut(0).copy_from_slice(&nux);
            v.component_mut(1).copy_from_slice(&nuy);
        }
        {
            let cf = state.fields.by_id_mut(self.conformation);
            for j in 0..3 {
                cf.component_mut(j).copy_from_slice(&npsi[j]);
            }
        }
        hook.after_stage(state, 0);
        state.time.t = t_new;
        state.time.step += 1;
    }
}
