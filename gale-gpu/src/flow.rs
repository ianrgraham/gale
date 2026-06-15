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

use crate::operators::poisson::{
    helmholtz_cg_solve, helmholtz_cg_solve_tags, pressure_cg_solve, GpuPoisson, GpuPoissonMg,
};
use crate::operators::poisson_nc::{
    helmholtz_nc_cg_solve_tags, poisson_nc_cg_solve, pressure_nc_cg_solve, GpuPoissonNc,
};
use std::cell::RefCell;

/// True if the mesh has any 2:1 non-conforming interface (hanging nodes). The GPU
/// elliptic solves must then route through the mortar-capable NC path; conforming
/// meshes use the (faster) uniform path.
fn mesh_is_nonconforming(mesh: &Mesh2d) -> bool {
    use gale::dg::Neighbor;
    mesh.elements.iter().any(|el| {
        el.neighbors.iter().any(|n| matches!(n, Neighbor::CoarseToFine { .. } | Neighbor::FineToCoarse { .. }))
    })
}

/// Lazily (re)build the persistent [`GpuPoisson`] handle in `slot` for `mesh`, so an
/// integrator's `step(&self, …)` can hold ONE handle across timesteps (P4). Conforming
/// meshes only: a non-conforming (2:1 AMR) mesh clears the slot, falling the step back to
/// the mortar NC one-shot path. The handle is rebuilt when the dof count changes (e.g.
/// after a remesh), so a static conforming mesh pays the ~0.3 s setup exactly once.
fn ensure_poisson_handle(slot: &RefCell<Option<GpuPoisson>>, mesh: &Mesh2d, alpha: f64) {
    let mut cur = slot.borrow_mut();
    if mesh_is_nonconforming(mesh) {
        *cur = None;
        return;
    }
    let ndof = mesh.n_elements() * mesh.refq.n_nodes();
    if cur.as_ref().map_or(true, |h| h.ndof() != ndof) {
        *cur = Some(GpuPoisson::new(mesh, alpha).expect("gale-gpu: GpuPoisson handle build failed"));
    }
}

/// Lazily (re)build the persistent **non-conforming** [`GpuPoissonNc`] handle in `slot`
/// (the AMR companion to [`ensure_poisson_handle`]): built only when the mesh is
/// non-conforming, cleared otherwise. So for any given mesh exactly one of the two handle
/// slots is `Some`, and the step uses whichever matches. Rebuilt when the dof count
/// changes (a remesh), so a steady adaptive mesh pays the NC setup once between remeshes.
fn ensure_poisson_nc_handle(slot: &RefCell<Option<GpuPoissonNc>>, mesh: &Mesh2d, alpha: f64) {
    let mut cur = slot.borrow_mut();
    if !mesh_is_nonconforming(mesh) {
        *cur = None;
        return;
    }
    let ndof = mesh.n_elements() * mesh.refq.n_nodes();
    if cur.as_ref().map_or(true, |h| h.ndof() != ndof) {
        *cur = Some(GpuPoissonNc::new(mesh, alpha).expect("gale-gpu: GpuPoissonNc handle build failed"));
    }
}
use gale::dg::{
    limit_logconf_trace_bound, log_conformation, upwind_advection_lift, BoundaryConditions,
    ConformationInflow, ConstitutiveModel, ConvectionScheme, Hyperbolic, IncompressibleConvection,
    LogConfOldroydB, Mesh2d, OldroydB, PMultigrid, Poisson, VolumeForm,
};

/// Lazily (re)build the persistent **p-multigrid-PCG pressure** handle in `slot`: built
/// (reaction 0, the given pressure `neumann_tags`) from a conforming **rectangular** flow
/// mesh via `PMultigrid::from_mesh`; left `None` for non-conforming or non-rectangular
/// meshes (the step then falls back to the deflated-CG pressure path). The MG-PCG is
/// mesh-independent where plain CG grows O(1/h) — the dominant pressure-solve win.
fn ensure_mg_pressure_handle(
    slot: &RefCell<Option<GpuPoissonMg>>,
    mesh: &Mesh2d,
    alpha: f64,
    neumann_tags: Vec<u32>,
) {
    let mut cur = slot.borrow_mut();
    if mesh_is_nonconforming(mesh) {
        *cur = None;
        return;
    }
    let ndof = mesh.n_elements() * mesh.refq.n_nodes();
    if cur.as_ref().map_or(true, |h| h.ndof() != ndof) {
        // None if `mesh` is not a uniform rectangular grid ⇒ stay on the CG pressure path.
        // CUDA-graph the V-cycle: the MG solve is launch-bound, so capturing its ~500-launch
        // sequence and replaying it with one cuGraphLaunch is a bit-exact 1.0–1.5× wall-clock win
        // (largest at the small/medium grids the flow runs). Validated end-to-end through ns-check.
        *cur = PMultigrid::from_mesh(mesh, alpha, 0.0, neumann_tags)
            .and_then(|mg| GpuPoissonMg::new(mg).and_then(|h| h.with_cuda_graph(true)).ok());
    }
}

/// Lazily (re)build a persistent **p-multigrid-PCG velocity-Helmholtz** handle in `slot`:
/// built with reaction `lambda = 1/(νΔt)` and the given per-component velocity-Neumann
/// tags. The companion to [`ensure_mg_pressure_handle`] for the non-singular viscous solve.
/// Rebuilt when the dof count changes (remesh) **or** when `lambda` changes (a Δt change),
/// since the reaction is baked into every level's diagonal/smoother; cleared (⇒ CG fallback)
/// for non-conforming or non-rectangular meshes.
fn ensure_mg_velocity_handle(
    slot: &RefCell<Option<GpuPoissonMg>>,
    mesh: &Mesh2d,
    alpha: f64,
    lambda: f64,
    neumann_tags: Vec<u32>,
) {
    let mut cur = slot.borrow_mut();
    if mesh_is_nonconforming(mesh) {
        *cur = None;
        return;
    }
    let ndof = mesh.n_elements() * mesh.refq.n_nodes();
    if cur.as_ref().map_or(true, |h| h.ndof() != ndof || h.reaction() != lambda) {
        // CUDA-graph the V-cycle (launch-bound remediation; bit-exact, see ensure_mg_pressure_handle).
        *cur = PMultigrid::from_mesh(mesh, alpha, lambda, neumann_tags)
            .and_then(|mg| GpuPoissonMg::new(mg).and_then(|h| h.with_cuda_graph(true)).ok());
    }
}

/// Ensure the pair of velocity-Helmholtz MG handles, **sharing a single hierarchy across
/// both components when their Neumann-tag sets are identical** — the common case, since the
/// closed-box all-Dirichlet path gives both components an empty tag set, and the per-region
/// path usually tags the same walls for `u` and `v`. Only when the tags genuinely differ
/// (e.g. a symmetry plane that is tangential to one component but normal to the other) does
/// the second component get its own hierarchy; otherwise `vely_slot` is left `None` and the
/// step reuses the `velx` handle for both, halving the velocity MG setup + device memory.
fn ensure_mg_velocity_handles(
    velx_slot: &RefCell<Option<GpuPoissonMg>>,
    vely_slot: &RefCell<Option<GpuPoissonMg>>,
    mesh: &Mesh2d,
    alpha: f64,
    lambda: f64,
    velx_tags: Vec<u32>,
    vely_tags: Vec<u32>,
) {
    let share = velx_tags == vely_tags;
    ensure_mg_velocity_handle(velx_slot, mesh, alpha, lambda, velx_tags);
    if share {
        // Both components solve the same operator ⇒ reuse the velx hierarchy (see the
        // `unwrap_or(vx)` at the apply site); keep the second slot empty.
        *vely_slot.borrow_mut() = None;
    } else {
        ensure_mg_velocity_handle(vely_slot, mesh, alpha, lambda, vely_tags);
    }
}

type StepResult = Result<(Vec<f64>, Vec<f64>), Box<dyn std::error::Error>>;

/// GPU unsteady-Stokes stepper. Holds the host `Poisson` operators used only for SIPG
/// RHS assembly; the operator applies / solves run on the GPU.
pub struct GpuStokes<'m, 'p> {
    pub mesh: &'m Mesh2d,
    pub nu: f64,
    pub dt: f64,
    /// How the nonlinear convection `(u·∇)u` is discretized in [`step_ns`](Self::step_ns).
    pub convection_scheme: ConvectionScheme,
    alpha: f64,
    /// Optional **persistent** [`GpuPoisson`] handle (P4). When `Some` (and the mesh is
    /// conforming), the three per-step elliptic solves route through it — reusing the
    /// loaded module + uploaded mesh instead of paying ~0.3 s of context/upload setup per
    /// solve. Set via [`with_handle`](Self::with_handle); the owning integrator keeps the
    /// handle alive across timesteps. `None` ⇒ the one-shot solvers (legacy path).
    poisson: Option<&'p GpuPoisson>,
    /// Optional persistent NC handle (P4 for AMR). Used when the mesh is non-conforming
    /// (then `poisson` is `None`); `None` ⇒ the one-shot mortar solvers. Set via
    /// [`with_nc_handle`](Self::with_nc_handle).
    poisson_nc: Option<&'p GpuPoissonNc>,
    /// Optional persistent **p-multigrid-PCG** handle for the PRESSURE solve (the most
    /// ill-conditioned: deflated CG iterations grow O(1/h), MG-PCG holds ~flat — 136× fewer
    /// at 64²). Built by the integrator from the (conforming, rectangular) flow mesh with
    /// the pressure neumann-tags; `None` ⇒ the CG path. Set via [`with_mg_pressure`].
    mg_pressure: Option<&'p GpuPoissonMg>,
    /// Optional persistent **p-multigrid-PCG** handles for the viscous velocity Helmholtz
    /// solves (`(λM + A)`, reaction `λ = 1/(νΔt)`), one per component since `velx`/`vely`
    /// can carry different Neumann-tag sets (symmetry/slip). Like [`mg_pressure`] but
    /// non-singular (no deflation); `None` ⇒ the persistent-`GpuPoisson`/CG path. Set via
    /// [`with_mg_velocity`].
    mg_velx: Option<&'p GpuPoissonMg>,
    mg_vely: Option<&'p GpuPoissonMg>,
    /// Pressure-Poisson assembly (pure Neumann, singular).
    pressure: Poisson<'m>,
    /// Velocity Helmholtz assembly `(λM + A)`, one per component (differ only in their
    /// Neumann-tag set, for symmetry/slip faces; identical otherwise).
    velocity_x: Poisson<'m>,
    velocity_y: Poisson<'m>,
    /// Whether any boundary is an outflow (pressure pinned ⇒ non-deflated solve).
    has_outflow: bool,
    /// Per-component velocity-Neumann tags for the GPU Helmholtz solve (outflow +
    /// symmetry-tangential faces).
    velx_neumann: Vec<u32>,
    vely_neumann: Vec<u32>,
    /// Pressure-Poisson Neumann tags (everything except outflow) for the GPU solve.
    pres_neumann_tags: Vec<u32>,
    tol: f64,
    maxit: usize,
}

impl<'m, 'p> GpuStokes<'m, 'p> {
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
            poisson: None,
            poisson_nc: None,
            mg_pressure: None,
            mg_velx: None,
            mg_vely: None,
            pressure: Poisson::with_bc(mesh, alpha, 0.0, mesh.boundary_tags()),
            velocity_x: Poisson::with_reaction(mesh, alpha, lambda),
            velocity_y: Poisson::with_reaction(mesh, alpha, lambda),
            has_outflow: false,
            velx_neumann: Vec::new(),
            vely_neumann: Vec::new(),
            pres_neumann_tags: mesh.boundary_tags(),
            tol: 1e-10,
            maxit: 20000,
        }
    }

    /// Route the three per-step elliptic solves through the persistent [`GpuPoisson`]
    /// `handle` (P4) instead of the one-shot solvers, amortizing context/module/mesh
    /// setup across timesteps. The caller must ensure `handle` was built from a mesh
    /// matching `self.mesh` (same connectivity ⇒ same `ndof`) and that the mesh is
    /// **conforming** — non-conforming meshes ignore the handle and stay on the mortar
    /// NC path. No effect on the numerics: the handle solve is bit-identical to the
    /// one-shot solver it replaces (validated by `poisson-handle-check`).
    pub fn with_handle(mut self, handle: &'p GpuPoisson) -> Self {
        self.poisson = Some(handle);
        self
    }

    /// Route the three per-step elliptic solves through the persistent **non-conforming**
    /// [`GpuPoissonNc`] `handle` (P4 for AMR) when the mesh has hanging nodes. Bit-identical
    /// to the one-shot mortar solvers it replaces (validated by `poisson-nc-handle-check`).
    pub fn with_nc_handle(mut self, handle: &'p GpuPoissonNc) -> Self {
        self.poisson_nc = Some(handle);
        self
    }

    /// Route the **pressure** projection solve through the persistent p-multigrid-PCG
    /// `handle` (built from this mesh with the pressure neumann-tags). The MG-PCG is
    /// mesh-independent (~flat iters) where plain CG grows O(1/h); for the singular
    /// pure-Neumann (closed-box) pressure it auto-deflates. Conforming rectangular meshes
    /// only; bit-equivalent to the CG it replaces (validated by pcg-pressure-check).
    pub fn with_mg_pressure(mut self, handle: &'p GpuPoissonMg) -> Self {
        self.mg_pressure = Some(handle);
        self
    }

    /// Route the two **viscous velocity Helmholtz** solves through persistent p-multigrid-PCG
    /// handles (`velx`/`vely`, each built with reaction `λ = 1/(νΔt)` and that component's
    /// velocity-Neumann tags). MG-PCG is p-robust and mesh-independent where the Helmholtz CG
    /// still grows with refinement; non-singular, so no deflation. Conforming rectangular
    /// meshes only; bit-equivalent to the CG it replaces (validated by pcg-helmholtz-check).
    pub fn with_mg_velocity(mut self, velx: &'p GpuPoissonMg, vely: &'p GpuPoissonMg) -> Self {
        self.mg_velx = Some(velx);
        self.mg_vely = Some(vely);
        self
    }

    /// Override the per-step elliptic-solve relative tolerance (default `1e-10`). In a
    /// time-accurate run the solve only needs to be as accurate as the time-discretization
    /// error, so a looser tol can cut iterations with no loss in the physical solution.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Solver with **per-region** boundary conditions — the GPU analogue of
    /// `gale::dg::Stokes::with_bcs`. Each boundary tag is routed via `bcs` to the right
    /// pair of operator settings (no-slip/inflow ⇒ velocity-Dirichlet + pressure-Neumann;
    /// outflow ⇒ velocity-Neumann + pressure-Dirichlet `p=0`). The pinned pressure lets
    /// the deflated solve be replaced by a plain CG. Use the `*_bc` step methods.
    pub fn with_bcs(mesh: &'m Mesh2d, alpha: f64, nu: f64, dt: f64, bcs: &BoundaryConditions) -> Self {
        let lambda = 1.0 / (nu * dt);
        let velx_neumann = bcs.velocity_neumann_tags(mesh, 0);
        let vely_neumann = bcs.velocity_neumann_tags(mesh, 1);
        let pres_neumann = bcs.pressure_neumann_tags(mesh);
        Self {
            mesh,
            nu,
            dt,
            convection_scheme: ConvectionScheme::Nodal,
            alpha,
            poisson: None,
            poisson_nc: None,
            mg_pressure: None,
            mg_velx: None,
            mg_vely: None,
            pressure: Poisson::with_bc(mesh, alpha, 0.0, pres_neumann.clone()),
            velocity_x: Poisson::with_bc(mesh, alpha, lambda, velx_neumann.clone()),
            velocity_y: Poisson::with_bc(mesh, alpha, lambda, vely_neumann.clone()),
            has_outflow: bcs.has_outflow(mesh),
            velx_neumann,
            vely_neumann,
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
        let (p, _it) = if !nc && self.mg_pressure.is_some() {
            // p-MG-PCG (mesh-independent iters; auto-deflated for the singular pure-Neumann op).
            self.mg_pressure.unwrap().solve(&bp, self.tol, self.maxit)?
        } else if nc {
            if let Some(h) = self.poisson_nc {
                h.solve(&bp, 0.0, &self.pres_neumann_tags, true, self.tol, self.maxit)?
            } else {
                pressure_nc_cg_solve(mesh, &bp, self.alpha, self.tol, self.maxit)?
            }
        } else if let Some(h) = self.poisson {
            // Pure-Neumann pressure, deflated (closed box ⇒ all tags Neumann, no outflow).
            h.solve(&bp, 0.0, &self.pres_neumann_tags, true, self.tol, self.maxit)?
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
        let bx = self.velocity_x.rhs(&fxv, |x, y| bc_u(x, y, t));
        let by = self.velocity_y.rhs(&fyv, |x, y| bc_v(x, y, t));
        let solve_vel = |b: &[f64], neu: &[u32], mg: Option<&GpuPoissonMg>| -> Result<Vec<f64>, Box<dyn std::error::Error>> {
            Ok(if nc {
                if let Some(h) = self.poisson_nc {
                    h.solve(b, lambda, neu, false, self.tol, self.maxit)?.0
                } else {
                    poisson_nc_cg_solve(mesh, b, self.alpha, lambda, self.tol, self.maxit)?.0
                }
            } else if let Some(h) = mg {
                // p-MG-PCG Helmholtz (reaction λ baked in; non-singular ⇒ no deflation).
                h.solve(b, self.tol, self.maxit)?.0
            } else if let Some(h) = self.poisson {
                // All-Dirichlet velocity Helmholtz (closed box ⇒ empty Neumann-tag set).
                h.solve(b, lambda, neu, false, self.tol, self.maxit)?.0
            } else {
                helmholtz_cg_solve(mesh, b, self.alpha, lambda, self.tol, self.maxit)?.0
            })
        };
        let uxn = solve_vel(&bx, &self.velx_neumann, self.mg_velx)?;
        let uyn = solve_vel(&by, &self.vely_neumann, self.mg_vely)?;
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
        let mesh = self.mesh;
        let dt = self.dt;
        // Convection is cheap O(N), assembled on the host (as for the closed-box path).
        // Split-form feeds the operator a per-region ghost state via Hyperbolic::rhs_ghost.
        let (cx, cy) = match self.convection_scheme {
            ConvectionScheme::Nodal => self.convection(ux, uy),
            ConvectionScheme::SplitFormDg => {
                let op = Hyperbolic::with_options(mesh, IncompressibleConvection, VolumeForm::SplitForm, true);
                let state = vec![ux.to_vec(), uy.to_vec()];
                let r = op.rhs_ghost(&state, t, &bcs.convection_ghost());
                (r[0].iter().map(|v| -v).collect(), r[1].iter().map(|v| -v).collect())
            }
        };
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
    /// a plain CG. **Non-conforming (2:1 AMR) meshes are supported**: when hanging nodes
    /// are present the three tag-aware solves route through the mortar SIPG NC path
    /// ([`helmholtz_nc_cg_solve_tags`] / [`pressure_nc_cg_solve`]); the conforming path is
    /// bit-identical otherwise.
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
        let nc = mesh_is_nonconforming(mesh);

        // Stage 2 — pressure projection (homogeneous data either way). With an outflow
        // the operator is non-singular (Dirichlet p=0 there) ⇒ plain Poisson CG; with
        // no outflow it is the singular pure-Neumann system ⇒ deflated CG.
        let div = self.divergence(&uhx, &uhy);
        let fp: Vec<f64> = div.iter().map(|d| -d / dt).collect();
        let bp = self.pressure.rhs_mixed(&fp, |_, _| 0.0, |_, _| 0.0);
        let (p, _it) = if !nc && self.mg_pressure.is_some() {
            // p-MG-PCG pressure (mesh-independent iters; auto-deflated for pure-Neumann,
            // plain for the outflow-pinned non-singular case).
            self.mg_pressure.unwrap().solve(&bp, self.tol, self.maxit)?
        } else { match (self.has_outflow, nc) {
            // Conforming + persistent handle: reaction 0, pressure-Neumann tags, and
            // deflate ⇔ no outflow (pinned pressure is non-singular ⇒ no nullspace removal).
            (_, false) if self.poisson.is_some() => {
                let h = self.poisson.unwrap();
                h.solve(&bp, 0.0, &self.pres_neumann_tags, !self.has_outflow, self.tol, self.maxit)?
            }
            (_, true) if self.poisson_nc.is_some() => {
                let h = self.poisson_nc.unwrap();
                h.solve(&bp, 0.0, &self.pres_neumann_tags, !self.has_outflow, self.tol, self.maxit)?
            }
            (true, false) => helmholtz_cg_solve_tags(mesh, &bp, self.alpha, 0.0, &self.pres_neumann_tags, self.tol, self.maxit)?,
            (true, true) => helmholtz_nc_cg_solve_tags(mesh, &bp, self.alpha, 0.0, &self.pres_neumann_tags, self.tol, self.maxit)?,
            (false, false) => pressure_cg_solve(mesh, &bp, self.alpha, self.tol, self.maxit)?,
            (false, true) => pressure_nc_cg_solve(mesh, &bp, self.alpha, self.tol, self.maxit)?,
        }};
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
        let bx = self.velocity_x.rhs_tagged(&fxv, |tag, x, y| vel_dir(tag, x, y).0, |_, _, _| 0.0);
        let by = self.velocity_y.rhs_tagged(&fyv, |tag, x, y| vel_dir(tag, x, y).1, |_, _, _| 0.0);
        let solve_vel = |b: &[f64], neu: &[u32], mg: Option<&GpuPoissonMg>| -> Result<Vec<f64>, Box<dyn std::error::Error>> {
            Ok(if nc {
                if let Some(h) = self.poisson_nc {
                    h.solve(b, lambda, neu, false, self.tol, self.maxit)?.0
                } else {
                    helmholtz_nc_cg_solve_tags(mesh, b, self.alpha, lambda, neu, self.tol, self.maxit)?.0
                }
            } else if let Some(h) = mg {
                // p-MG-PCG Helmholtz (reaction λ + this component's Neumann tags baked in).
                h.solve(b, self.tol, self.maxit)?.0
            } else if let Some(h) = self.poisson {
                h.solve(b, lambda, neu, false, self.tol, self.maxit)?.0
            } else {
                helmholtz_cg_solve_tags(mesh, b, self.alpha, lambda, neu, self.tol, self.maxit)?.0
            })
        };
        let uxn = solve_vel(&bx, &self.velx_neumann, self.mg_velx)?;
        let uyn = solve_vel(&by, &self.vely_neumann, self.mg_vely)?;
        Ok((uxn, uyn))
    }

    /// Velocity-field L2 norm (uses the velocity operator's mass).
    pub fn l2_norm(&self, v: &[f64]) -> f64 {
        self.velocity_x.l2_norm(v)
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
    /// Persistent GPU Poisson handle (P4), lazily built and reused across timesteps.
    poisson: RefCell<Option<GpuPoisson>>,
    poisson_nc: RefCell<Option<GpuPoissonNc>>,
    mg_pressure: RefCell<Option<GpuPoissonMg>>,
    mg_velx: RefCell<Option<GpuPoissonMg>>,
    mg_vely: RefCell<Option<GpuPoissonMg>>,
    solve_tol: f64,
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
            poisson: RefCell::new(None),
            poisson_nc: RefCell::new(None),
            mg_pressure: RefCell::new(None),
            mg_velx: RefCell::new(None),
            mg_vely: RefCell::new(None),
            solve_tol: 1e-10,
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

    /// Set the per-step elliptic-solve tolerance (default `1e-10`). For a time-accurate run set
    /// this ~1–2 orders below the time-discretization error: the physical solution is unchanged
    /// while iterations drop sharply (e.g. 1e-10→1e-4 cut a pressure solve 29→11 iters with the
    /// trajectory error flat — see the `solve-tol-sweep` bin).
    pub fn with_solve_tol(mut self, tol: f64) -> Self {
        self.solve_tol = tol;
        self
    }
}

impl gale::sim::StateIntegrator for GpuStokesIntegrator {
    fn dt(&self) -> f64 {
        self.dt
    }

    fn step(&self, state: &mut gale::sim::State, hook: &dyn gale::sim::StateStageHook) {
        let t_new = state.time.t + self.dt;
        ensure_poisson_handle(&self.poisson, &state.mesh, self.alpha);
        ensure_poisson_nc_handle(&self.poisson_nc, &state.mesh, self.alpha);
        ensure_mg_pressure_handle(&self.mg_pressure, &state.mesh, self.alpha, state.mesh.boundary_tags());
        // Closed box ⇒ all-Dirichlet velocity (empty Neumann-tag set, identical for both
        // components ⇒ one shared MG hierarchy); reaction λ = 1/(νΔt).
        let lambda = 1.0 / (self.nu * self.dt);
        ensure_mg_velocity_handles(&self.mg_velx, &self.mg_vely, &state.mesh, self.alpha, lambda, Vec::new(), Vec::new());
        let handle = self.poisson.borrow();
        let nc_handle = self.poisson_nc.borrow();
        let mg_handle = self.mg_pressure.borrow();
        let mg_vx = self.mg_velx.borrow();
        let mg_vy = self.mg_vely.borrow();
        let mut stokes = GpuStokes::new(&state.mesh, self.alpha, self.nu, self.dt).with_tol(self.solve_tol);
        if let Some(h) = handle.as_ref() {
            stokes = stokes.with_handle(h);
        }
        if let Some(h) = nc_handle.as_ref() {
            stokes = stokes.with_nc_handle(h);
        }
        if let Some(h) = mg_handle.as_ref() {
            stokes = stokes.with_mg_pressure(h);
        }
        if let Some(vx) = mg_vx.as_ref() {
            // vely shares the velx hierarchy unless its tags differed (then it has its own).
            stokes = stokes.with_mg_velocity(vx, mg_vy.as_ref().unwrap_or(vx));
        }
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
    /// Persistent GPU Poisson handle (P4), lazily built on the first step and reused
    /// across timesteps. `RefCell` because `step` is `&self`; conforming meshes only.
    poisson: RefCell<Option<GpuPoisson>>,
    poisson_nc: RefCell<Option<GpuPoissonNc>>,
    mg_pressure: RefCell<Option<GpuPoissonMg>>,
    mg_velx: RefCell<Option<GpuPoissonMg>>,
    mg_vely: RefCell<Option<GpuPoissonMg>>,
    solve_tol: f64,
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
            poisson: RefCell::new(None),
            poisson_nc: RefCell::new(None),
            mg_pressure: RefCell::new(None),
            mg_velx: RefCell::new(None),
            mg_vely: RefCell::new(None),
            solve_tol: 1e-10,
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

    /// Set the per-step elliptic-solve tolerance (default `1e-10`); see
    /// [`GpuStokesIntegrator::with_solve_tol`]. In a time-accurate run a looser tol (~1–2 orders
    /// below the time-discretization error) cuts iterations with no change to the trajectory.
    pub fn with_solve_tol(mut self, tol: f64) -> Self {
        self.solve_tol = tol;
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
        // Pressure-Neumann tags for the MG handle: per-region set (outflow excluded) with
        // BCs, else all boundary tags (the singular closed-box pressure).
        let pres_tags = match &self.bcs {
            Some(bcs) => bcs.pressure_neumann_tags(&state.mesh),
            None => state.mesh.boundary_tags(),
        };
        // Per-component velocity-Neumann tags for the MG Helmholtz handles (outflow +
        // symmetry/slip with BCs, empty for the closed-box all-Dirichlet path).
        let (velx_tags, vely_tags) = match &self.bcs {
            Some(bcs) => (bcs.velocity_neumann_tags(&state.mesh, 0), bcs.velocity_neumann_tags(&state.mesh, 1)),
            None => (Vec::new(), Vec::new()),
        };
        let lambda = 1.0 / (self.nu * self.dt);
        ensure_poisson_handle(&self.poisson, &state.mesh, self.alpha);
        ensure_poisson_nc_handle(&self.poisson_nc, &state.mesh, self.alpha);
        ensure_mg_pressure_handle(&self.mg_pressure, &state.mesh, self.alpha, pres_tags);
        ensure_mg_velocity_handles(&self.mg_velx, &self.mg_vely, &state.mesh, self.alpha, lambda, velx_tags, vely_tags);
        let handle = self.poisson.borrow();
        let nc_handle = self.poisson_nc.borrow();
        let mg_handle = self.mg_pressure.borrow();
        let mg_vx = self.mg_velx.borrow();
        let mg_vy = self.mg_vely.borrow();
        let (nux, nuy) = if let Some(bcs) = &self.bcs {
            let mut stokes = GpuStokes::with_bcs(&state.mesh, self.alpha, self.nu, self.dt, bcs).with_tol(self.solve_tol);
            stokes.convection_scheme = self.convection_scheme;
            if let Some(h) = handle.as_ref() {
                stokes = stokes.with_handle(h);
            }
            if let Some(h) = nc_handle.as_ref() {
                stokes = stokes.with_nc_handle(h);
            }
            if let Some(h) = mg_handle.as_ref() {
                stokes = stokes.with_mg_pressure(h);
            }
            if let Some(vx) = mg_vx.as_ref() {
                stokes = stokes.with_mg_velocity(vx, mg_vy.as_ref().unwrap_or(vx));
            }
            stokes
                .step_ns_forced_bc(&ux, &uy, t_new, bcs, &bx, &by)
                .expect("gale-gpu: GpuDualSplitting BC step failed")
        } else {
            let mut stokes = GpuStokes::new(&state.mesh, self.alpha, self.nu, self.dt).with_tol(self.solve_tol);
            stokes.convection_scheme = self.convection_scheme;
            if let Some(h) = handle.as_ref() {
                stokes = stokes.with_handle(h);
            }
            if let Some(h) = nc_handle.as_ref() {
                stokes = stokes.with_nc_handle(h);
            }
            if let Some(h) = mg_handle.as_ref() {
                stokes = stokes.with_mg_pressure(h);
            }
            if let Some(vx) = mg_vx.as_ref() {
                stokes = stokes.with_mg_velocity(vx, mg_vy.as_ref().unwrap_or(vx));
            }
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
    trace_bound: Option<f64>,
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
    // Optional bound-preserving trace cap (tr C ≤ b_max): the log-conformation analogue of FENE-P
    // finite extensibility, applied after each SSP-RK3 stage. Keeps the high-Wi extensional growth
    // bounded so the unbounded Oldroyd-B stress can't blow up (needed for elastic-turbulence runs).
    let clip = |mut s: [Vec<f64>; 3]| -> [Vec<f64>; 3] {
        if let Some(b) = trace_bound {
            limit_logconf_trace_bound(mesh, &mut s, b);
        }
        s
    };
    let k0 = rhs(psi)?;
    let u1 = clip(axpy3(psi, &k0, dt));
    let k1 = rhs(&u1)?;
    let u2 = clip(combine3(psi, 0.75, &axpy3(&u1, &k1, dt), 0.25));
    let k2 = rhs(&u2)?;
    Ok(clip(combine3(psi, 1.0 / 3.0, &axpy3(&u2, &k2, dt), 2.0 / 3.0)))
}

/// One **ARK2 / ARS(2,2,2)** IMEX step of the log-conformation transport with a fixed
/// velocity, on the **GPU**: explicit transport via [`crate::logconf_psi_rhs`] (+ host upwind
/// lift), implicit relaxation via the device solve [`crate::logconf_implicit_relax`]. This is
/// the GPU analogue of `gale::dg::LogConfOldroydB::step_ark2_imex` (Phases 2 & 3): L-stable,
/// 2nd-order, stiffly accurate, no operator-splitting error — and stable at `dt ≫ λ`.
pub fn logconf_ark2_advance_gpu(
    mesh: &Mesh2d,
    lc: &LogConfOldroydB,
    psi: &[Vec<f64>; 3],
    ux: &[f64],
    uy: &[f64],
    dt: f64,
    inflow: Option<&ConformationInflow>,
) -> ConfResult {
    let gamma = 1.0 - 0.5_f64.sqrt();
    let delta = 1.0 - 1.0 / (2.0 * gamma);
    let gdt = dt * gamma;
    // Explicit transport E = (GPU psi_rhs + host upwind lift) − relaxation source S, so the
    // relaxation is handled only by the implicit solve (no double counting).
    let transport = |pp: &[Vec<f64>; 3]| -> ConfResult {
        let mut k = crate::logconf_psi_rhs(mesh, lc, pp, ux, uy)?;
        let lift = upwind_advection_lift(mesh, pp, ux, uy, |tag| {
            inflow.filter(|i| i.tags.contains(&tag)).map(|i| log_conformation(i.c))
        });
        let relax = lc.relax_source(pp);
        for comp in 0..3 {
            for g in 0..k[comp].len() {
                k[comp][g] += lift[comp][g] - relax[comp][g];
            }
        }
        Ok(k)
    };
    // ARS(2,2,2); stiffly accurate ⇒ Ψⁿ⁺¹ = the last implicit stage.
    let e1 = transport(psi)?;
    let b2 = axpy3(psi, &e1, dt * gamma);
    let psi2 = crate::logconf_implicit_relax(mesh, lc, &b2, gdt)?;
    let e2 = transport(&psi2)?;
    // S₂ = (Ψ₂ − B₂)/(dt·γ), recovered exactly from the stage equation.
    let s2: [Vec<f64>; 3] =
        std::array::from_fn(|v| psi2[v].iter().zip(&b2[v]).map(|(y, b)| (y - b) / gdt).collect());
    // B₃ = Ψⁿ + dt(δ·E₁ + (1−δ)·E₂) + dt(1−γ)·S₂.
    let mut b3 = psi.clone();
    for comp in 0..3 {
        for g in 0..b3[comp].len() {
            b3[comp][g] += dt * (delta * e1[comp][g] + (1.0 - delta) * e2[comp][g])
                + dt * (1.0 - gamma) * s2[comp][g];
        }
    }
    crate::logconf_implicit_relax(mesh, lc, &b3, gdt)
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
    /// Optional bound-preserving cap on `tr C` (log-conf model only) — the FENE-P-like finite-
    /// extensibility limiter applied after each conformation RK stage. `None` ⇒ unbounded (Oldroyd-B).
    trace_bound: Option<f64>,
    /// Persistent GPU Poisson handle (P4) for the momentum (velocity) solves, lazily built
    /// on the first step and reused across timesteps. Conforming meshes only.
    poisson: RefCell<Option<GpuPoisson>>,
    poisson_nc: RefCell<Option<GpuPoissonNc>>,
    mg_pressure: RefCell<Option<GpuPoissonMg>>,
    mg_velx: RefCell<Option<GpuPoissonMg>>,
    mg_vely: RefCell<Option<GpuPoissonMg>>,
    solve_tol: f64,
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
            trace_bound: None,
            poisson: RefCell::new(None),
            poisson_nc: RefCell::new(None),
            mg_pressure: RefCell::new(None),
            mg_velx: RefCell::new(None),
            mg_vely: RefCell::new(None),
            solve_tol: 1e-10,
        }
    }

    /// Cap `tr C ≤ b_max` after each conformation RK stage (log-conf model) — the FENE-P-like
    /// finite-extensibility bound that keeps high-Wi extensional growth from blowing up. `b_max` is
    /// the maximum polymer stretch (≈ FENE-P L²). Default off (unbounded Oldroyd-B).
    pub fn with_trace_bound(mut self, b_max: f64) -> Self {
        self.trace_bound = Some(b_max);
        self
    }

    /// Set the per-step elliptic-solve tolerance (default `1e-10`); see
    /// [`GpuStokesIntegrator::with_solve_tol`]. Looser (matched to the time-discretization error)
    /// cuts the velocity/pressure iterations with no change to the trajectory.
    pub fn with_solve_tol(mut self, tol: f64) -> Self {
        self.solve_tol = tol;
        self
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
            ensure_poisson_handle(&self.poisson, &state.mesh, self.alpha);
            ensure_poisson_nc_handle(&self.poisson_nc, &state.mesh, self.alpha);
            ensure_mg_pressure_handle(&self.mg_pressure, &state.mesh, self.alpha, state.mesh.boundary_tags());
            // Closed box ⇒ all-Dirichlet velocity (both components share one MG hierarchy);
            // the viscous reaction uses the SOLVENT viscosity η_s (the coefficient given to
            // `GpuStokes::new` below): λ = 1/(η_s Δt).
            let lambda = 1.0 / (self.eta_s * self.dt);
            ensure_mg_velocity_handles(&self.mg_velx, &self.mg_vely, &state.mesh, self.alpha, lambda, Vec::new(), Vec::new());
            let handle = self.poisson.borrow();
            let nc_handle = self.poisson_nc.borrow();
            let mg_handle = self.mg_pressure.borrow();
            let mg_vx = self.mg_velx.borrow();
            let mg_vy = self.mg_vely.borrow();
            let mut stokes = GpuStokes::new(&state.mesh, self.alpha, self.eta_s, self.dt).with_tol(self.solve_tol);
            if let Some(h) = handle.as_ref() {
                stokes = stokes.with_handle(h);
            }
            if let Some(h) = nc_handle.as_ref() {
                stokes = stokes.with_nc_handle(h);
            }
            if let Some(h) = mg_handle.as_ref() {
                stokes = stokes.with_mg_pressure(h);
            }
            if let Some(vx) = mg_vx.as_ref() {
                stokes = stokes.with_mg_velocity(vx, mg_vy.as_ref().unwrap_or(vx));
            }
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
                    logconf_advance_gpu(&state.mesh, &lc, &c, &nux, &nuy, self.dt, self.inflow.as_ref(), self.trace_bound)
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
