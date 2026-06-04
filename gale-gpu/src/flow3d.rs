//! GPU incompressible Navier–Stokes on the 3D hex mesh — the 3D analogue of
//! [`crate::flow`]. BDF1 dual-splitting: explicit (nodal) convection + body force →
//! GPU pressure-Poisson projection → GPU viscous Helmholtz solve per velocity
//! component. The two elliptic solves run on the GPU ([`crate::pressure3d_cg_solve`],
//! [`crate::helmholtz3d_cg_solve`]); the cheap element-local assembly (convection,
//! divergence, gradient correction, SIPG RHS) reuses the validated host `gale::dg`
//! machinery. Faithful (to solver tolerance) to `gale::dg::Stokes3d`.

use crate::operators::poisson3d::{helmholtz3d_cg_solve, pressure3d_cg_solve};
use gale::dg::{Mesh3d, Poisson3d};

type StepResult = Result<(Vec<f64>, Vec<f64>, Vec<f64>), Box<dyn std::error::Error>>;

/// GPU 3D unsteady-NS stepper. Host `Poisson3d` operators are used only for SIPG RHS
/// assembly; the operator applies / solves run on the GPU.
pub struct GpuStokes3d<'m> {
    pub mesh: &'m Mesh3d,
    pub nu: f64,
    pub dt: f64,
    alpha: f64,
    pressure: Poisson3d<'m>,
    velocity: Poisson3d<'m>,
    tol: f64,
    maxit: usize,
}

impl<'m> GpuStokes3d<'m> {
    pub fn new(mesh: &'m Mesh3d, alpha: f64, nu: f64, dt: f64) -> Self {
        let lambda = 1.0 / (nu * dt);
        Self {
            mesh,
            nu,
            dt,
            alpha,
            pressure: Poisson3d::with_bc(mesh, alpha, 0.0, vec![0, 1, 2, 3, 4, 5]),
            velocity: Poisson3d::with_reaction(mesh, alpha, lambda),
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
        let (p, _it) = pressure3d_cg_solve(mesh, &bp, self.alpha, self.tol, self.maxit)?;
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
        let bx = self.velocity.rhs(&fxv, |x, y, z| bc_u(x, y, z, t));
        let by = self.velocity.rhs(&fyv, |x, y, z| bc_v(x, y, z, t));
        let bz = self.velocity.rhs(&fzv, |x, y, z| bc_w(x, y, z, t));
        let (uxn, _) = helmholtz3d_cg_solve(mesh, &bx, self.alpha, lambda, self.tol, self.maxit)?;
        let (uyn, _) = helmholtz3d_cg_solve(mesh, &by, self.alpha, lambda, self.tol, self.maxit)?;
        let (uzn, _) = helmholtz3d_cg_solve(mesh, &bz, self.alpha, lambda, self.tol, self.maxit)?;
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
        let stokes = GpuStokes3d::new(&state.mesh, self.alpha, self.nu, self.dt);
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
