//! Incompressible Navier–Stokes on the 3D hex mesh by BDF1 dual-splitting — the 3D
//! analogue of [`Stokes`](super::stokes::Stokes). Three stages per step: explicit
//! convection + body force, pressure-Poisson projection to divergence-free, then an
//! implicit viscous Helmholtz solve per velocity component. Uses [`Poisson3d`] for
//! the pressure (all-Neumann) and velocity (Helmholtz) operators.
//!
//! Nodal (collocation) convection only for now; the energy-stable split-form path
//! (the high-Re option in 2D) is a later addition.

use super::mesh3d::Mesh3d;
use super::poisson3d::Poisson3d;

/// BDF1 dual-splitting incompressible NS solver on a hex mesh.
pub struct Stokes3d<'m> {
    pub mesh: &'m Mesh3d,
    pub nu: f64,
    pub dt: f64,
    pressure: Poisson3d<'m>,
    velocity: Poisson3d<'m>,
    tol: f64,
    maxit: usize,
}

impl<'m> Stokes3d<'m> {
    pub fn new(mesh: &'m Mesh3d, alpha: f64, nu: f64, dt: f64) -> Self {
        let lambda = 1.0 / (nu * dt);
        Self {
            mesh,
            nu,
            dt,
            // Pressure: pure-Neumann on all 6 faces (tags 0..6).
            pressure: Poisson3d::with_bc(mesh, alpha, 0.0, vec![0, 1, 2, 3, 4, 5]),
            // Velocity: Helmholtz (λM + A), Dirichlet.
            velocity: Poisson3d::with_reaction(mesh, alpha, lambda),
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
        self.project_and_diffuse(uhx, uhy, uhz, t, bc_u, bc_v, bc_w)
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
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mesh = self.mesh;
        let refh = &mesh.refh;
        let nn = refh.n_nodes();
        let dt = self.dt;
        let lambda = 1.0 / (self.nu * dt);

        // Stage 2 — pressure projection: ∇²p = (1/Δt)∇·û (homogeneous Neumann).
        let div = self.divergence(&uhx, &uhy, &uhz);
        let fp: Vec<f64> = div.iter().map(|d| -d / dt).collect();
        let bp = self.pressure.rhs_mixed(&fp, |_, _, _| 0.0, |_, _, _| 0.0);
        let (p, _it) = self.pressure.cg_deflated(&bp, self.tol, self.maxit);
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
        let bx = self.velocity.rhs(&fxv, |x, y, z| bc_u(x, y, z, t));
        let by = self.velocity.rhs(&fyv, |x, y, z| bc_v(x, y, z, t));
        let bz = self.velocity.rhs(&fzv, |x, y, z| bc_w(x, y, z, t));
        let (uxn, _, _) = self.velocity.cg(&bx, self.tol, self.maxit);
        let (uyn, _, _) = self.velocity.cg(&by, self.tol, self.maxit);
        let (uzn, _, _) = self.velocity.cg(&bz, self.tol, self.maxit);
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
