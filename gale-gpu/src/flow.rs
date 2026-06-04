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

use crate::operators::poisson::{helmholtz_cg_solve, pressure_cg_solve};
use gale::dg::{ConvectionScheme, Hyperbolic, IncompressibleConvection, Mesh2d, Poisson, VolumeForm};

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
    tol: f64,
    maxit: usize,
}

impl<'m> GpuStokes<'m> {
    pub fn new(mesh: &'m Mesh2d, alpha: f64, nu: f64, dt: f64) -> Self {
        let lambda = 1.0 / (nu * dt);
        Self {
            mesh,
            nu,
            dt,
            convection_scheme: ConvectionScheme::Nodal,
            alpha,
            pressure: Poisson::with_bc(mesh, alpha, 0.0, vec![0, 1, 2, 3]),
            velocity: Poisson::with_reaction(mesh, alpha, lambda),
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
        let (p, _it) = pressure_cg_solve(mesh, &bp, self.alpha, self.tol, self.maxit)?;
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
        let (uxn, _) = helmholtz_cg_solve(mesh, &bx, self.alpha, lambda, self.tol, self.maxit)?;
        let (uyn, _) = helmholtz_cg_solve(mesh, &by, self.alpha, lambda, self.tol, self.maxit)?;
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
        let mut stokes = GpuStokes::new(&state.mesh, self.alpha, self.nu, self.dt);
        stokes.convection_scheme = self.convection_scheme;
        let (ux, uy) = {
            let v = state.fields.by_id(self.velocity);
            (v.component(0).to_vec(), v.component(1).to_vec())
        };
        let (bx, by) = (self.body_force)(state, t_new);
        let (nux, nuy) = stokes
            .step_ns_forced(&ux, &uy, t_new, &self.bc_u, &self.bc_v, &bx, &by)
            .expect("gale-gpu: GpuDualSplitting step failed");
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
