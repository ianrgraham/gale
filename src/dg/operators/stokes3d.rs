//! Incompressible Navier–Stokes on the 3D hex mesh by BDF1 dual-splitting — the 3D
//! analogue of [`Stokes`](super::stokes::Stokes). Three stages per step: explicit
//! convection + body force, pressure-Poisson projection to divergence-free, then an
//! implicit viscous Helmholtz solve per velocity component. Uses [`Poisson3d`] for
//! the pressure (all-Neumann) and velocity (Helmholtz) operators.
//!
//! Nodal (collocation) convection only for now; the energy-stable split-form path
//! (the high-Re option in 2D) is a later addition.

use super::bc::BoundaryConditions3d;
use super::mesh3d::Mesh3d;
use super::poisson3d::Poisson3d;

/// BDF1 dual-splitting incompressible NS solver on a hex mesh.
pub struct Stokes3d<'m> {
    pub mesh: &'m Mesh3d,
    pub nu: f64,
    pub dt: f64,
    pressure: Poisson3d<'m>,
    /// Per-component velocity Helmholtz operators (differ only in Neumann-tag set, for
    /// symmetry/slip faces; identical otherwise).
    velocity_x: Poisson3d<'m>,
    velocity_y: Poisson3d<'m>,
    velocity_z: Poisson3d<'m>,
    /// Whether any boundary is an outflow (pressure pinned ⇒ plain CG, not deflated).
    has_outflow: bool,
    tol: f64,
    maxit: usize,
}

impl<'m> Stokes3d<'m> {
    /// Closed-box solver: all-Dirichlet velocity + pure-Neumann (deflated) pressure.
    /// For per-region inflow/outflow/wall/symmetry conditions use [`with_bcs`](Self::with_bcs).
    pub fn new(mesh: &'m Mesh3d, alpha: f64, nu: f64, dt: f64) -> Self {
        let lambda = 1.0 / (nu * dt);
        Self {
            mesh,
            nu,
            dt,
            // Pressure: pure-Neumann on all boundary faces.
            pressure: Poisson3d::with_bc(mesh, alpha, 0.0, mesh.boundary_tags()),
            // Velocity: Helmholtz (λM + A), all-Dirichlet (per-component, identical here).
            velocity_x: Poisson3d::with_reaction(mesh, alpha, lambda),
            velocity_y: Poisson3d::with_reaction(mesh, alpha, lambda),
            velocity_z: Poisson3d::with_reaction(mesh, alpha, lambda),
            has_outflow: false,
            tol: 1e-10,
            maxit: 20000,
        }
    }

    /// Solver with **per-region** boundary conditions (the 3D analogue of
    /// `Stokes::with_bcs`): each boundary tag routed via `bcs` to per-component velocity
    /// + pressure operator settings. Outflow ⇒ velocity-Neumann + pressure-Dirichlet
    /// `p=0`; symmetry ⇒ normal-component Dirichlet + tangential Neumann.
    pub fn with_bcs(mesh: &'m Mesh3d, alpha: f64, nu: f64, dt: f64, bcs: &BoundaryConditions3d) -> Self {
        let lambda = 1.0 / (nu * dt);
        Self {
            mesh,
            nu,
            dt,
            pressure: Poisson3d::with_bc(mesh, alpha, 0.0, bcs.pressure_neumann_tags(mesh)),
            velocity_x: Poisson3d::with_bc(mesh, alpha, lambda, bcs.velocity_neumann_tags(mesh, 0)),
            velocity_y: Poisson3d::with_bc(mesh, alpha, lambda, bcs.velocity_neumann_tags(mesh, 1)),
            velocity_z: Poisson3d::with_bc(mesh, alpha, lambda, bcs.velocity_neumann_tags(mesh, 2)),
            has_outflow: bcs.has_outflow(mesh),
            tol: 1e-10,
            maxit: 20000,
        }
    }

    fn ndof(&self) -> usize {
        self.mesh.n_elements() * self.mesh.refh.n_nodes()
    }

    /// `∂x a + ∂y b + ∂z c` of a vector field, per element.
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

    /// Nodal advective convection `(u·∇)u` for each component.
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
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
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
        self.project_and_diffuse(uhx, uhy, uhz, |_, x, y, z| {
            (bc_u(x, y, z, t), bc_v(x, y, z, t), bc_w(x, y, z, t))
        })
    }

    /// BC-aware NS step with a precomputed nodal body force — the [`with_bcs`](Self::with_bcs)
    /// companion to [`step_ns_forced`](Self::step_ns_forced). Per-region velocity data
    /// (inflow / walls / symmetry) comes from `bcs`.
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
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
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
        self.project_and_diffuse(uhx, uhy, uhz, |tag, x, y, z| bcs.dirichlet(tag, x, y, z, t))
    }

    /// One NS step with analytic forcing closures.
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
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
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

    /// Stages 2–3: pressure projection to divergence-free, then implicit viscous
    /// solve per component.
    fn project_and_diffuse(
        &self,
        mut uhx: Vec<f64>,
        mut uhy: Vec<f64>,
        mut uhz: Vec<f64>,
        vel_dir: impl Fn(u32, f64, f64, f64) -> (f64, f64, f64),
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mesh = self.mesh;
        let refh = &mesh.refh;
        let nn = refh.n_nodes();
        let dt = self.dt;
        let lambda = 1.0 / (self.nu * dt);

        // Stage 2 — pressure projection: ∇²p = (1/Δt)∇·û. Outflow pins p ⇒ plain CG;
        // otherwise the pure-Neumann system is singular ⇒ deflated CG.
        let div = self.divergence(&uhx, &uhy, &uhz);
        let fp: Vec<f64> = div.iter().map(|d| -d / dt).collect();
        let bp = self.pressure.rhs_mixed(&fp, |_, _, _| 0.0, |_, _, _| 0.0);
        let p = if self.has_outflow {
            let (p, _, _) = self.pressure.cg(&bp, self.tol, self.maxit);
            p
        } else {
            let (p, _it) = self.pressure.cg_deflated(&bp, self.tol, self.maxit);
            p
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

        // Stage 3 — viscous Helmholtz per component: (λM + A) uⁿ⁺¹ = λM û + Dirichlet.
        let fxv: Vec<f64> = uhx.iter().map(|v| lambda * v).collect();
        let fyv: Vec<f64> = uhy.iter().map(|v| lambda * v).collect();
        let fzv: Vec<f64> = uhz.iter().map(|v| lambda * v).collect();
        let bx = self.velocity_x.rhs_tagged(&fxv, |tag, x, y, z| vel_dir(tag, x, y, z).0, |_, _, _, _| 0.0);
        let by = self.velocity_y.rhs_tagged(&fyv, |tag, x, y, z| vel_dir(tag, x, y, z).1, |_, _, _, _| 0.0);
        let bz = self.velocity_z.rhs_tagged(&fzv, |tag, x, y, z| vel_dir(tag, x, y, z).2, |_, _, _, _| 0.0);
        let (uxn, _, _) = self.velocity_x.cg(&bx, self.tol, self.maxit);
        let (uyn, _, _) = self.velocity_y.cg(&by, self.tol, self.maxit);
        let (uzn, _, _) = self.velocity_z.cg(&bz, self.tol, self.maxit);
        (uxn, uyn, uzn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn nodal(mesh: &Mesh3d, f: impl Fn(f64, f64, f64) -> f64) -> Vec<f64> {
        let nn = mesh.refh.n_nodes();
        let mut v = vec![0.0; mesh.n_elements() * nn];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                v[e * nn + k] = f(el.geom.x[k], el.geom.y[k], el.geom.z[k]);
            }
        }
        v
    }

    #[test]
    fn per_region_bcs_preserve_uniform_flow_3d() {
        // 3D per-region BCs together: west inlet (tag 4) u=(1,0,0), east outflow (tag 5),
        // and free-slip on the 4 lateral faces (bottom/top z-normal, south/north
        // y-normal). u=(1,0,0) is the exact steady state and must be preserved —
        // exercising inflow + outflow + symmetry and the per-component routing.
        use crate::dg::bc::{BoundaryConditions3d, FlowBc3d};
        let nu = 1.0;
        let p = 3;
        let mesh = Mesh3d::rectangular(p, 3, 2, 2, [0.0, 2.0], [0.0, 1.0], [0.0, 1.0]);
        let bcs = BoundaryConditions3d::no_slip()
            .set(4, FlowBc3d::velocity(|_, _, _, _| (1.0, 0.0, 0.0)))
            .set(5, FlowBc3d::Outflow)
            .set(0, FlowBc3d::Symmetry)
            .set(1, FlowBc3d::Symmetry)
            .set(2, FlowBc3d::Symmetry)
            .set(3, FlowBc3d::Symmetry);
        let dt = 0.05;
        let st = Stokes3d::with_bcs(&mesh, 5.0, nu, dt, &bcs);
        let nd = mesh.n_elements() * mesh.refh.n_nodes();
        let (mut ux, mut uy, mut uz) = (vec![1.0; nd], vec![0.0; nd], vec![0.0; nd]);
        let z = vec![0.0; nd];
        let mut t = 0.0;
        for _ in 0..15 {
            t += dt;
            let (nx, ny, nz) = st.step_ns_forced_bc(&ux, &uy, &uz, t, &bcs, &z, &z, &z);
            ux = nx;
            uy = ny;
            uz = nz;
        }
        let err_u = ux.iter().fold(0.0f64, |a, &v| a.max((v - 1.0).abs()));
        let err_t = uy.iter().chain(uz.iter()).fold(0.0f64, |a, &v| a.max(v.abs()));
        eprintln!("3D per-region BC uniform flow: max|u−1|={err_u:.3e}, max|v,w|={err_t:.3e}");
        assert!(err_u < 1e-6, "uniform flow not preserved: {err_u}");
        assert!(err_t < 1e-6, "spurious transverse velocity: {err_t}");
    }

    #[test]
    fn navier_stokes_taylor_green_3d() {
        // (x,z)-plane Taylor–Green embedded in 3D (uniform in y, v=0): an exact
        // decaying NS solution. Divergence-free with nonzero w, so it exercises the
        // 3-component divergence, grad_z, and all three viscous solves.
        let nu = 1.0;
        let p = 3;
        let mesh = Mesh3d::rectangular(p, 2, 2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
        let decay = move |t: f64| (-2.0 * PI * PI * nu * t).exp();
        let eu = move |x: f64, _y: f64, z: f64, t: f64| -(PI * x).cos() * (PI * z).sin() * decay(t);
        let ev = move |_x: f64, _y: f64, _z: f64, _t: f64| 0.0;
        let ew = move |x: f64, _y: f64, z: f64, t: f64| (PI * x).sin() * (PI * z).cos() * decay(t);
        let zero = |_: f64, _: f64, _: f64, _: f64| 0.0;

        let t_end = 0.05;
        let nsteps = 10;
        let dt = t_end / nsteps as f64;
        let st = Stokes3d::new(&mesh, 5.0, nu, dt);

        let mut ux = nodal(&mesh, |x, y, z| eu(x, y, z, 0.0));
        let mut uy = nodal(&mesh, |x, y, z| ev(x, y, z, 0.0));
        let mut uz = nodal(&mesh, |x, y, z| ew(x, y, z, 0.0));
        let mut t = 0.0;
        for _ in 0..nsteps {
            t += dt;
            let (nx, ny, nz) = st.step_ns(&ux, &uy, &uz, t, eu, ev, ew, zero, zero, zero);
            ux = nx;
            uy = ny;
            uz = nz;
        }

        let nn = mesh.refh.n_nodes();
        let mut e2 = 0.0;
        let mut n2 = 0.0;
        let mut vmax = 0.0f64;
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let (x, y, z) = (el.geom.x[k], el.geom.y[k], el.geom.z[k]);
                let du = ux[e * nn + k] - eu(x, y, z, t_end);
                let dv = uy[e * nn + k]; // exact v ≡ 0
                let dw = uz[e * nn + k] - ew(x, y, z, t_end);
                e2 += el.geom.jw[k] * (du * du + dv * dv + dw * dw);
                n2 += el.geom.jw[k] * (eu(x, y, z, t_end).powi(2) + ew(x, y, z, t_end).powi(2));
                vmax = vmax.max(uy[e * nn + k].abs());
            }
        }
        let rel = (e2 / n2).sqrt();
        // v stays negligible vs the O(0.37) velocity scale (pure iterative-solver noise).
        assert!(vmax < 1e-3, "v drifted from 0: max|v|={vmax:e}");
        assert!(rel < 2e-2, "3D Taylor–Green relative L2 error {rel:e}");
    }
}
