//! **Device-resident, GPU-self-driving flow integrators** — Stages 1–2 of the production
//! device-residency plan (`docs/plan-device-resident-production.md`).
//!
//! All simulation state (velocity, pressure, later conformation + mesh) stays **resident on the
//! GPU** across the whole time loop. The host uploads initial conditions once and reads results
//! back only for occasional I/O. There is **no per-step host work and no per-step host↔device
//! synchronization** — every stage (convection, projection, RHS assembly, all elliptic solves)
//! runs as device kernels on resident buffers. This is the RULE ZERO requirement for production
//! (sweep-ready) sims — see the `gale-gpu-perf` skill and memory `gpu-sync-host-device-copies`.
//!
//! The per-step sequence is the validated `device_resident_flow_check` dual-splitting step, extended
//! with a (steady) body force, promoted into a reusable integrator. Stage 2 will capture the step +
//! time loop as a self-driving CUDA graph; Stage 3 adds GPU-resident AMR.

use crate::operators::logconf::GpuLogConf;
use crate::operators::poisson::GpuPoissonMg;
use cuda_core::{CudaContext, DeviceBuffer};
use gale::dg::{Edge, Mesh2d, Neighbor, PMultigrid, Poisson};

type Err = Box<dyn std::error::Error>;
type Buf = DeviceBuffer<f64>;

/// Device-resident incompressible **Navier–Stokes** (BDF1 dual-splitting / Chorin) integrator.
/// Closed-box all-Dirichlet velocity + singular all-Neumann (deflated) pressure, with an optional
/// **steady** body force and steady Dirichlet velocity data (both precomputed once, so the whole
/// step stays on the GPU). The first production-residency building block; the viscoelastic step
/// layers the polymer-stress divergence + conformation transport on top of this.
pub struct GpuResidentNs {
    hp: GpuPoissonMg, // pressure (singular ⇒ deflated)
    hv: GpuPoissonMg, // velocity Helmholtz (λM + A)
    // Resident state.
    ux: DeviceBuffer<f64>,
    uy: DeviceBuffer<f64>,
    // Resident per-step scratch (allocated once, reused every step).
    uhx: DeviceBuffer<f64>,
    uhy: DeviceBuffer<f64>,
    ga: DeviceBuffer<f64>,
    gb: DeviceBuffer<f64>,
    gc: DeviceBuffer<f64>,
    gd: DeviceBuffer<f64>,
    cx: DeviceBuffer<f64>,
    cy: DeviceBuffer<f64>,
    div: DeviceBuffer<f64>,
    pp: DeviceBuffer<f64>,
    rhs: DeviceBuffer<f64>,
    // Precomputed device constants (uploaded once).
    jw_d: DeviceBuffer<f64>,      // diagonal mass M·1
    lift_vx_d: DeviceBuffer<f64>, // velocity Nitsche lift for the steady Dirichlet data (x,y comps)
    lift_vy_d: DeviceBuffer<f64>,
    lift_p_d: DeviceBuffer<f64>, // pressure lift (zero for homogeneous Neumann)
    bx_d: DeviceBuffer<f64>,     // steady body force, nodal (x, y comps)
    by_d: DeviceBuffer<f64>,
    dt: f64,
    lambda: f64,
    tol: f64,
    maxit: usize,
    ndof: usize,
}

impl GpuResidentNs {
    /// Build the integrator for `mesh` (must be a uniform rectangular grid — the MG hierarchy
    /// requirement). `nu` kinematic viscosity, `dt` step, `alpha` SIPG penalty. `fx`/`fy` are the
    /// **steady** body force and `bc_u`/`bc_v` the **steady** Dirichlet velocity data, each a
    /// function of `(x, y)`; all four are sampled/assembled once on the host and uploaded.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mesh: &Mesh2d,
        nu: f64,
        dt: f64,
        alpha: f64,
        fx: impl Fn(f64, f64) -> f64,
        fy: impl Fn(f64, f64) -> f64,
        bc_u: impl Fn(f64, f64) -> f64,
        bc_v: impl Fn(f64, f64) -> f64,
    ) -> Result<Self, Err> {
        let nn = mesh.refq.n_nodes();
        let ndof = mesh.n_elements() * nn;
        let lambda = 1.0 / (nu * dt);
        let tags = mesh.boundary_tags();

        // GPU MG handles: pressure (all-Neumann, reaction 0 ⇒ singular/deflated) and velocity
        // Helmholtz (all-Dirichlet ⇒ empty Neumann set, reaction λ).
        let hp = GpuPoissonMg::new(
            PMultigrid::from_mesh(mesh, alpha, 0.0, tags.clone())
                .ok_or("GpuResidentNs: pressure MG needs a uniform rectangular mesh")?,
        )?;
        let hv = GpuPoissonMg::new(
            PMultigrid::from_mesh(mesh, alpha, lambda, Vec::new())
                .ok_or("GpuResidentNs: velocity MG needs a uniform rectangular mesh")?,
        )?;

        // Host operators only for one-time constant assembly (diagonal mass + boundary lifts).
        let vel_op = Poisson::with_reaction(mesh, alpha, lambda);
        let jw = vel_op.rhs(&vec![1.0; ndof], |_, _| 0.0); // M·1
        let lift_vx = vel_op.rhs(&vec![0.0; ndof], &bc_u); // Nitsche lift of the Dirichlet data
        let lift_vy = vel_op.rhs(&vec![0.0; ndof], &bc_v);

        // Steady body force sampled at the nodes (nodal acceleration, added in the predictor).
        let mut bx = vec![0.0; ndof];
        let mut by = vec![0.0; ndof];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let (x, y) = (el.geom.x[k], el.geom.y[k]);
                bx[e * nn + k] = fx(x, y);
                by[e * nn + k] = fy(x, y);
            }
        }

        let alloc = || hv.alloc_field();
        Ok(Self {
            jw_d: hv.upload_field(&jw)?,
            lift_vx_d: hv.upload_field(&lift_vx)?,
            lift_vy_d: hv.upload_field(&lift_vy)?,
            lift_p_d: hv.upload_field(&vec![0.0; ndof])?,
            bx_d: hv.upload_field(&bx)?,
            by_d: hv.upload_field(&by)?,
            ux: alloc()?,
            uy: alloc()?,
            uhx: alloc()?,
            uhy: alloc()?,
            ga: alloc()?,
            gb: alloc()?,
            gc: alloc()?,
            gd: alloc()?,
            cx: alloc()?,
            cy: alloc()?,
            div: alloc()?,
            pp: alloc()?,
            rhs: alloc()?,
            hp,
            hv,
            dt,
            lambda,
            tol: 1e-9,
            maxit: 2000,
            ndof,
        })
    }

    /// Set the solver tolerance / iteration cap for the elliptic solves.
    pub fn with_tol(mut self, tol: f64, maxit: usize) -> Self {
        self.tol = tol;
        self.maxit = maxit;
        self
    }

    /// Upload the initial velocity (host → device, one-time). Lengths must be `n_elements·n_nodes`.
    pub fn set_velocity(&mut self, ux: &[f64], uy: &[f64]) -> Result<(), Err> {
        assert_eq!(ux.len(), self.ndof);
        assert_eq!(uy.len(), self.ndof);
        // One-time IC upload: replace the resident buffers (reallocation is fine here — it never
        // happens in the step loop).
        self.ux = self.hv.upload_field(ux)?;
        self.uy = self.hv.upload_field(uy)?;
        Ok(())
    }

    /// Advance ONE BDF1 dual-splitting step, **entirely on the device** — no host work, no field
    /// transfers. Sequence: predictor (convection + body force) → pressure projection (deflated) →
    /// gradient correction → viscous Helmholtz (per component). Mirrors the validated
    /// `device_resident_flow_check` step plus the steady body force.
    pub fn step(&mut self) -> Result<(), Err> {
        let dt = self.dt;
        // Stage 1 — predictor û = u + Δt(b − (u·∇)u).
        self.hv.gradient_dev(&self.ux, &mut self.ga, &mut self.gb)?; // ∂ux/∂x, ∂ux/∂y
        self.hv.gradient_dev(&self.uy, &mut self.gc, &mut self.gd)?; // ∂uy/∂x, ∂uy/∂y
        self.hv.fma2_dev(&mut self.cx, &self.ux, &self.ga, &self.uy, &self.gb)?; // (u·∇)ux
        self.hv.fma2_dev(&mut self.cy, &self.ux, &self.gc, &self.uy, &self.gd)?; // (u·∇)uy
        self.hv.copy_dev(&mut self.uhx, &self.ux)?;
        self.hv.copy_dev(&mut self.uhy, &self.uy)?;
        self.hv.axpy_dev(&mut self.uhx, &self.cx, -dt)?;
        self.hv.axpy_dev(&mut self.uhy, &self.cy, -dt)?;
        self.hv.axpy_dev(&mut self.uhx, &self.bx_d, dt)?; // + Δt body force
        self.hv.axpy_dev(&mut self.uhy, &self.by_d, dt)?;
        // Stage 2 — pressure projection −∇²p = (1/Δt)∇·û (singular ⇒ deflated).
        self.hv.gradient_dev(&self.uhx, &mut self.ga, &mut self.gb)?; // ∂ûx/∂x in ga
        self.hv.gradient_dev(&self.uhy, &mut self.gc, &mut self.gd)?; // ∂ûy/∂y in gd
        self.hv.copy_dev(&mut self.div, &self.ga)?;
        self.hv.axpy_dev(&mut self.div, &self.gd, 1.0)?; // ∇·û
        self.hv.scal_dev(&mut self.div, -1.0 / dt)?; // fp = −∇·û/Δt
        self.hv.rhs_madd_dev(&mut self.rhs, &self.jw_d, &self.div, &self.lift_p_d, 1.0)?;
        self.hp.solve_dev(&self.rhs, None, &mut self.pp, self.tol, self.maxit)?;
        // Gradient correction u* = û − Δt ∇p.
        self.hv.gradient_dev(&self.pp, &mut self.ga, &mut self.gb)?;
        self.hv.axpy_dev(&mut self.uhx, &self.ga, -dt)?;
        self.hv.axpy_dev(&mut self.uhy, &self.gb, -dt)?;
        // Stage 3 — viscous Helmholtz (λM + A) uⁿ⁺¹ = λM u* + lift.
        self.hv.rhs_madd_dev(&mut self.rhs, &self.jw_d, &self.uhx, &self.lift_vx_d, self.lambda)?;
        self.hv.solve_dev(&self.rhs, None, &mut self.ux, self.tol, self.maxit)?;
        self.hv.rhs_madd_dev(&mut self.rhs, &self.jw_d, &self.uhy, &self.lift_vy_d, self.lambda)?;
        self.hv.solve_dev(&self.rhs, None, &mut self.uy, self.tol, self.maxit)?;
        Ok(())
    }

    /// Advance `n` steps. (Stage 1: a host-driven loop over the device-resident step; Stage 2 will
    /// replace this with a single self-driving CUDA-graph launch.)
    pub fn run(&mut self, n: usize) -> Result<(), Err> {
        for _ in 0..n {
            self.step()?;
        }
        Ok(())
    }

    /// Download the current velocity (device → host) — for I/O / diagnostics only.
    pub fn velocity(&self) -> Result<(Vec<f64>, Vec<f64>), Err> {
        Ok((self.hv.download_field(&self.ux)?, self.hv.download_field(&self.uy)?))
    }
}

// ===== Device-resident VISCOELASTIC dual-splitting (Stage 1 of the VE port) ====================

/// `y[i] ← y[i] + s·x[i]` over a symmetric-tensor triple (device).
fn axpy3(hv: &GpuPoissonMg, y: &mut [Buf; 3], x: &[Buf; 3], s: f64) -> Result<(), Err> {
    for i in 0..3 {
        hv.axpy_dev(&mut y[i], &x[i], s)?;
    }
    Ok(())
}

/// `dst[i] ← wa·a[i] + wb·b[i]` over a triple (device): copy a, scale, axpy b.
fn comb3(hv: &GpuPoissonMg, dst: &mut [Buf; 3], wa: f64, a: &[Buf; 3], wb: f64, b: &[Buf; 3]) -> Result<(), Err> {
    for i in 0..3 {
        hv.copy_dev(&mut dst[i], &a[i])?;
        hv.scal_dev(&mut dst[i], wa)?;
        hv.axpy_dev(&mut dst[i], &b[i], wb)?;
    }
    Ok(())
}

/// FENE-P trace clip `dst ← clip(src)` (device): the limiter if a bound is set, else a plain copy.
#[allow(clippy::too_many_arguments)]
fn clip3(
    lcg: &GpuLogConf, hv: &GpuPoissonMg, ne: usize, n1: u32, jw: &Buf, tb: Option<f64>,
    src: &[Buf; 3], dst: &mut [Buf; 3],
) -> Result<(), Err> {
    match tb {
        Some(b) => {
            let [d0, d1, d2] = dst;
            lcg.limit_trace_dev(ne, n1, &src[0], &src[1], &src[2], jw, b, d0, d1, d2)?;
        }
        None => {
            for i in 0..3 {
                hv.copy_dev(&mut dst[i], &src[i])?;
            }
        }
    }
    Ok(())
}

/// Device-resident **viscoelastic** (Oldroyd-B log-conformation) BDF1 dual-splitting integrator.
/// Closed box (zero-velocity walls), steady drive. The momentum step is the resident NS dual
/// splitting with body force = drive + ∇·τ_p(Ψ); the constitutive step is the device SSP-RK3 of Ψ
/// (volume `psi_rhs` + upwind surface lift + FENE-P trace clip). Every stage runs on the GPU; the
/// host only uploads ICs and reads back tr C for I/O. No polymer stress diffusion yet (κ=0).
pub struct GpuResidentVe {
    hp: GpuPoissonMg,
    hv: GpuPoissonMg,
    /// Polymer stress-diffusion Helmholtz handle `(λ_d M + A)`, all-Neumann, `λ_d = 1/(Δt·κ)`.
    /// `Some` iff `κ > 0` (Sureshkumar–Beris regularization for elastic-turbulence runs).
    hdiff: Option<GpuPoissonMg>,
    lcg: GpuLogConf,
    // Velocity state + scratch.
    ux: Buf,
    uy: Buf,
    uhx: Buf,
    uhy: Buf,
    ga: Buf,
    gb: Buf,
    gc: Buf,
    gd: Buf,
    cvx: Buf,
    cvy: Buf,
    div: Buf,
    pp: Buf,
    rhs: Buf,
    bxt: Buf,
    byt: Buf,
    // Conformation Ψ state + SSP-RK3 scratch + recovered C.
    psi: [Buf; 3],
    sa: [Buf; 3],
    sb: [Buf; 3],
    sk: [Buf; 3],
    slt: [Buf; 3],
    cc: [Buf; 3],
    // Precomputed device constants.
    jw_d: Buf,
    lift_vx_d: Buf,
    lift_vy_d: Buf,
    lift_p_d: Buf,
    bx0_d: Buf,
    by0_d: Buf,
    d_dev: Buf,
    rxm: Buf,
    rym: Buf,
    sxm: Buf,
    sym: Buf,
    // Face connectivity for the upwind lift.
    fvl: DeviceBuffer<u32>,
    fnx: Buf,
    fny: Buf,
    fsw: Buf,
    fnbr: DeviceBuffer<u32>,
    // Params.
    dt: f64,
    lambda: f64,        // velocity Helmholtz reaction 1/(η_s Δt)
    inv_lambda_p: f64,  // 1/relaxation-time
    stress_coeff: f64,  // η_p/λ_p
    lam_d: f64,         // stress-diffusion reaction 1/(Δt·κ) (only used when hdiff is Some)
    trace_bound: Option<f64>,
    tol: f64,
    maxit: usize,
    ne: usize,
    n1: u32,
}

impl GpuResidentVe {
    /// Build the integrator. `eta_s`/`eta_p` solvent/polymer viscosities, `lambda_p` relaxation
    /// time, `dt` step, `alpha` SIPG penalty, `fx`/`fy` the steady body force. Closed box (zero
    /// velocity walls); conformation initialised to equilibrium (Ψ = 0).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mesh: &Mesh2d,
        eta_s: f64,
        eta_p: f64,
        lambda_p: f64,
        dt: f64,
        alpha: f64,
        kappa: f64,
        fx: impl Fn(f64, f64) -> f64,
        fy: impl Fn(f64, f64) -> f64,
    ) -> Result<Self, Err> {
        let nn = mesh.refq.n_nodes();
        let ne = mesh.n_elements();
        let ndof = ne * nn;
        let n1 = (mesh.order + 1) as u32;
        let lambda = 1.0 / (eta_s * dt);
        let tags = mesh.boundary_tags();

        // ONE shared non-legacy stream for ALL handles (pressure, velocity, diffusion, log-conf):
        // their kernels then chain on the same ordered stream, and the whole step is CUDA-graph
        // capturable (illegal on the legacy stream). `with_while_graph` runs each solve's whole PCG
        // loop as a device-side conditional graph — the convergence test is on the GPU, so there is
        // NO per-iteration host residual readback (the last per-step host sync).
        let g_stream = CudaContext::new(0)?.new_stream()?;
        // Each solve's PCG loop runs as a device-side conditional WHILE graph by default (zero
        // per-iter host sync — the production path). `RVP_NOGRAPH=1` disables capture so the solve
        // kernels launch normally: this is the PROFILING ESCAPE HATCH — ncu cannot profile kernels
        // inside conditional graphs (per-node profiling is "unsupported", whole-graph profiling
        // mis-attributes), so an out-of-graph run is the only way to get real per-kernel SOL/BW.
        // Functionally identical (same kernels/data, just not captured); never use it for sweeps.
        let use_graph = std::env::var("RVP_NOGRAPH").is_err();
        let mk = |mg: PMultigrid| -> Result<GpuPoissonMg, Err> {
            Ok(GpuPoissonMg::new_on_stream(mg, g_stream.clone())?.with_while_graph(use_graph)?)
        };
        let hp = mk(PMultigrid::from_mesh(mesh, alpha, 0.0, tags.clone()).ok_or("GpuResidentVe: needs uniform rect mesh")?)?;
        let hv = mk(PMultigrid::from_mesh(mesh, alpha, lambda, Vec::new()).ok_or("GpuResidentVe: needs uniform rect mesh")?)?;
        // Implicit polymer stress diffusion (operator split): (λ_d M + A) Ψⁿ⁺¹ = λ_d M Ψ*, all-Neumann
        // (zero stress flux), λ_d = 1/(Δt·κ). Non-singular (λ_d M ≻ 0) ⇒ no deflation. Built only if κ>0.
        let lam_d = if kappa > 0.0 { 1.0 / (dt * kappa) } else { 0.0 };
        let hdiff = if kappa > 0.0 {
            Some(mk(PMultigrid::from_mesh(mesh, alpha, lam_d, tags).ok_or("GpuResidentVe: needs uniform rect mesh")?)?)
        } else {
            None
        };
        let lcg = GpuLogConf::new_on_stream(g_stream.clone())?;

        // One-time host constants.
        let vel_op = Poisson::with_reaction(mesh, alpha, lambda);
        let jw = vel_op.rhs(&vec![1.0; ndof], |_, _| 0.0);
        let (mut bx0, mut by0) = (vec![0.0; ndof], vec![0.0; ndof]);
        let (mut rxm, mut rym, mut sxm, mut sym) =
            (vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof]);
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let g = e * nn + k;
                bx0[g] = fx(el.geom.x[k], el.geom.y[k]);
                by0[g] = fy(el.geom.x[k], el.geom.y[k]);
                rxm[g] = el.geom.rx[k];
                rym[g] = el.geom.ry[k];
                sxm[g] = el.geom.sx[k];
                sym[g] = el.geom.sy[k];
            }
        }
        // Face connectivity (interior + boundary; boundary ⇒ face_nbr = self ⇒ zero lift).
        let nfc = ne * 4 * (n1 as usize);
        let (mut fvl, mut fnx, mut fny, mut fsw, mut fnbr) =
            (vec![0u32; nfc], vec![0.0; nfc], vec![0.0; nfc], vec![0.0; nfc], vec![0u32; nfc]);
        for (e, el) in mesh.elements.iter().enumerate() {
            for (t, edge) in Edge::ALL.iter().enumerate() {
                let face = &el.faces[*edge as usize];
                for a in 0..n1 as usize {
                    let idx = (e * 4 + t) * n1 as usize + a;
                    let vl = face.nodes[a];
                    fvl[idx] = vl as u32;
                    fnx[idx] = face.nx[a];
                    fny[idx] = face.ny[a];
                    fsw[idx] = face.sw[a];
                    fnbr[idx] = match &el.neighbors[*edge as usize] {
                        Neighbor::Interior { elem: re, edge: redge, perm } => {
                            let rf = &mesh.elements[*re].faces[*redge as usize];
                            (*re * nn + rf.nodes[perm[a]]) as u32
                        }
                        _ => (e * nn + vl) as u32,
                    };
                }
            }
        }

        let f = |v: &[f64]| hv.upload_field(v);
        let fu = |v: &[u32]| -> Result<DeviceBuffer<u32>, Err> { Ok(DeviceBuffer::from_host(hv.stream(), v)?) };
        let z = || hv.alloc_field();
        let triple = || -> Result<[Buf; 3], Err> { Ok([z()?, z()?, z()?]) };
        Ok(Self {
            jw_d: f(&jw)?,
            lift_vx_d: f(&vec![0.0; ndof])?,
            lift_vy_d: f(&vec![0.0; ndof])?,
            lift_p_d: f(&vec![0.0; ndof])?,
            bx0_d: f(&bx0)?,
            by0_d: f(&by0)?,
            d_dev: f(&mesh.refq.line.diff)?,
            rxm: f(&rxm)?,
            rym: f(&rym)?,
            sxm: f(&sxm)?,
            sym: f(&sym)?,
            fvl: fu(&fvl)?,
            fnx: f(&fnx)?,
            fny: f(&fny)?,
            fsw: f(&fsw)?,
            fnbr: fu(&fnbr)?,
            ux: z()?,
            uy: z()?,
            uhx: z()?,
            uhy: z()?,
            ga: z()?,
            gb: z()?,
            gc: z()?,
            gd: z()?,
            cvx: z()?,
            cvy: z()?,
            div: z()?,
            pp: z()?,
            rhs: z()?,
            bxt: z()?,
            byt: z()?,
            psi: triple()?,
            sa: triple()?,
            sb: triple()?,
            sk: triple()?,
            slt: triple()?,
            cc: triple()?,
            hp,
            hv,
            hdiff,
            lcg,
            dt,
            lambda,
            inv_lambda_p: 1.0 / lambda_p,
            stress_coeff: eta_p / lambda_p,
            lam_d,
            trace_bound: None,
            tol: 1e-9,
            maxit: 4000,
            ne,
            n1,
        })
    }

    pub fn with_trace_bound(mut self, b: f64) -> Self {
        self.trace_bound = Some(b);
        self
    }
    pub fn with_tol(mut self, tol: f64, maxit: usize) -> Self {
        self.tol = tol;
        self.maxit = maxit;
        self
    }

    /// Upload the initial velocity (Ψ starts at equilibrium 0).
    pub fn set_velocity(&mut self, ux: &[f64], uy: &[f64]) -> Result<(), Err> {
        self.ux = self.hv.upload_field(ux)?;
        self.uy = self.hv.upload_field(uy)?;
        Ok(())
    }

    /// Conformation transport rhs `k ← psi_rhs(pin) + upwind_lift(pin)` with the current velocity.
    fn conf_rhs(&mut self, pin_is_psi: bool, stage: usize) -> Result<(), Err> {
        // `pin` selects which Ψ buffer set the rhs reads (psi for k0, else sa). Outputs into sk.
        let _ = stage;
        let pin: &[Buf; 3] = if pin_is_psi { &self.psi } else { &self.sa };
        {
            let [k0, k1, k2] = &mut self.sk;
            self.lcg.psi_rhs_dev(
                self.ne, self.n1, &self.d_dev, &self.ux, &self.uy, &pin[0], &pin[1], &pin[2],
                &self.rxm, &self.rym, &self.sxm, &self.sym, self.inv_lambda_p, k0, k1, k2,
            )?;
        }
        {
            let [l0, l1, l2] = &mut self.slt;
            self.lcg.upwind_lift_dev(
                self.ne, self.n1, &self.ux, &self.uy, &pin[0], &pin[1], &pin[2], &self.jw_d,
                &self.fvl, &self.fnx, &self.fny, &self.fsw, &self.fnbr, l0, l1, l2,
            )?;
        }
        axpy3(&self.hv, &mut self.sk, &self.slt, 1.0)?; // k = volume + lift
        Ok(())
    }

    /// Advance ONE viscoelastic dual-splitting step, entirely on the device.
    pub fn step(&mut self) -> Result<(), Err> {
        let dt = self.dt;
        let sc = self.stress_coeff;
        // --- Stress divergence ∇·τ_p = (η_p/λ_p) ∇·C, C = exp(Ψ) ---
        {
            let [c0, c1, c2] = &mut self.cc;
            self.lcg.conformation_dev(self.ne, self.n1, &self.psi[0], &self.psi[1], &self.psi[2], c0, c1, c2)?;
        }
        self.hv.gradient_dev(&self.cc[0], &mut self.ga, &mut self.gb)?; // ∂x Cxx
        self.hv.gradient_dev(&self.cc[1], &mut self.gc, &mut self.gd)?; // ∂x Cxy, ∂y Cxy
        self.hv.gradient_dev(&self.cc[2], &mut self.cvx, &mut self.cvy)?; // ∂y Cyy in cvy
        // bxt = drive + sc(∂x Cxx + ∂y Cxy); byt = drive + sc(∂x Cxy + ∂y Cyy)
        self.hv.copy_dev(&mut self.bxt, &self.bx0_d)?;
        self.hv.axpy_dev(&mut self.bxt, &self.ga, sc)?;
        self.hv.axpy_dev(&mut self.bxt, &self.gd, sc)?;
        self.hv.copy_dev(&mut self.byt, &self.by0_d)?;
        self.hv.axpy_dev(&mut self.byt, &self.gc, sc)?;
        self.hv.axpy_dev(&mut self.byt, &self.cvy, sc)?;
        // --- Momentum: dual-splitting with body force bxt/byt (uses the OLD Ψ stress) ---
        self.hv.gradient_dev(&self.ux, &mut self.ga, &mut self.gb)?;
        self.hv.gradient_dev(&self.uy, &mut self.gc, &mut self.gd)?;
        self.hv.fma2_dev(&mut self.cvx, &self.ux, &self.ga, &self.uy, &self.gb)?;
        self.hv.fma2_dev(&mut self.cvy, &self.ux, &self.gc, &self.uy, &self.gd)?;
        self.hv.copy_dev(&mut self.uhx, &self.ux)?;
        self.hv.copy_dev(&mut self.uhy, &self.uy)?;
        self.hv.axpy_dev(&mut self.uhx, &self.cvx, -dt)?;
        self.hv.axpy_dev(&mut self.uhy, &self.cvy, -dt)?;
        self.hv.axpy_dev(&mut self.uhx, &self.bxt, dt)?;
        self.hv.axpy_dev(&mut self.uhy, &self.byt, dt)?;
        self.hv.gradient_dev(&self.uhx, &mut self.ga, &mut self.gb)?;
        self.hv.gradient_dev(&self.uhy, &mut self.gc, &mut self.gd)?;
        self.hv.copy_dev(&mut self.div, &self.ga)?;
        self.hv.axpy_dev(&mut self.div, &self.gd, 1.0)?;
        self.hv.scal_dev(&mut self.div, -1.0 / dt)?;
        self.hv.rhs_madd_dev(&mut self.rhs, &self.jw_d, &self.div, &self.lift_p_d, 1.0)?;
        self.hp.solve_dev(&self.rhs, None, &mut self.pp, self.tol, self.maxit)?;
        self.hv.gradient_dev(&self.pp, &mut self.ga, &mut self.gb)?;
        self.hv.axpy_dev(&mut self.uhx, &self.ga, -dt)?;
        self.hv.axpy_dev(&mut self.uhy, &self.gb, -dt)?;
        self.hv.rhs_madd_dev(&mut self.rhs, &self.jw_d, &self.uhx, &self.lift_vx_d, self.lambda)?;
        self.hv.solve_dev(&self.rhs, None, &mut self.ux, self.tol, self.maxit)?;
        self.hv.rhs_madd_dev(&mut self.rhs, &self.jw_d, &self.uhy, &self.lift_vy_d, self.lambda)?;
        self.hv.solve_dev(&self.rhs, None, &mut self.uy, self.tol, self.maxit)?;
        // --- Constitutive SSP-RK3 of Ψ with the NEW velocity (volume + lift + FENE-P clip) ---
        let tb = self.trace_bound;
        // Stage 1: sa = clip(psi + dt·rhs(psi))
        self.conf_rhs(true, 0)?;
        comb3(&self.hv, &mut self.sb, 1.0, &self.psi, dt, &self.sk)?; // sb = psi + dt k0
        clip3(&self.lcg, &self.hv, self.ne, self.n1, &self.jw_d, tb, &self.sb, &mut self.sa)?;
        // Stage 2: sa = clip(0.75 psi + 0.25(sa + dt·rhs(sa)))
        self.conf_rhs(false, 1)?;
        axpy3(&self.hv, &mut self.sa, &self.sk, dt)?; // sa += dt k1
        comb3(&self.hv, &mut self.sb, 0.75, &self.psi, 0.25, &self.sa)?;
        clip3(&self.lcg, &self.hv, self.ne, self.n1, &self.jw_d, tb, &self.sb, &mut self.sa)?;
        // Stage 3: psi = clip(psi/3 + 2/3(sa + dt·rhs(sa)))
        self.conf_rhs(false, 2)?;
        axpy3(&self.hv, &mut self.sa, &self.sk, dt)?; // sa += dt k2
        comb3(&self.hv, &mut self.sb, 1.0 / 3.0, &self.psi, 2.0 / 3.0, &self.sa)?;
        clip3(&self.lcg, &self.hv, self.ne, self.n1, &self.jw_d, tb, &self.sb, &mut self.psi)?;
        // --- Implicit polymer stress diffusion (operator split, all on device) ---
        // (λ_d M + A) Ψⁿ⁺¹ = λ_d M Ψ*, warm-started from Ψ* (mass-dominated ⇒ a few CG iters), then
        // the FENE-P clip is re-applied. Removes no parabolic CFL — κ is free. Skipped if κ = 0.
        if let Some(hd) = &self.hdiff {
            for i in 0..3 {
                // rhs = λ_d · jw ⊙ Ψ*  (lift_p_d is zero ⇒ pure mass-weighted source).
                self.hv.rhs_madd_dev(&mut self.rhs, &self.jw_d, &self.psi[i], &self.lift_p_d, self.lam_d)?;
                hd.solve_dev(&self.rhs, Some(&self.psi[i]), &mut self.sb[i], self.tol, self.maxit)?;
            }
            clip3(&self.lcg, &self.hv, self.ne, self.n1, &self.jw_d, tb, &self.sb, &mut self.psi)?;
        }
        Ok(())
    }

    pub fn run(&mut self, n: usize) -> Result<(), Err> {
        for _ in 0..n {
            self.step()?;
        }
        Ok(())
    }

    pub fn velocity(&self) -> Result<(Vec<f64>, Vec<f64>), Err> {
        Ok((self.hv.download_field(&self.ux)?, self.hv.download_field(&self.uy)?))
    }

    /// Download the log-conformation Ψ (three components).
    pub fn psi(&self) -> Result<[Vec<f64>; 3], Err> {
        Ok([
            self.hv.download_field(&self.psi[0])?,
            self.hv.download_field(&self.psi[1])?,
            self.hv.download_field(&self.psi[2])?,
        ])
    }
}
