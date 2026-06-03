//! Unsteady incompressible **Stokes** solver via the dual-splitting (Chorin /
//! Karniadakis–Israeli–Orszag) scheme, first-order (BDF1) in time. Reuses the SIPG
//! elliptic machinery: a pure-Neumann **pressure-Poisson** projection and a
//! Dirichlet **viscous Helmholtz** solve per velocity component.
//!
//! Per step (`∂ₜu = −∇p + ν∇²u + f`, `∇·u = 0`):
//! 1. explicit       `û  = uⁿ + Δt f`
//! 2. project        `∇²p = (1/Δt)∇·û`  (Neumann) → `û̂ = û − Δt∇p`
//! 3. viscous        `(1/(νΔt) I − ∇²)uⁿ⁺¹ = (1/(νΔt))û̂`  (Helmholtz, Dirichlet)
//!
//! See `docs/implicit-solver-strategy.md` §1,§3. Velocity components are stored as
//! two scalar nodal fields in the global `e·nn + k` layout.

use super::hyperbolic::{Hyperbolic, IncompressibleConvection, VolumeForm};
use super::mesh::Mesh2d;
use super::poisson::Poisson;

/// How the nonlinear convection `(u·∇)u` is discretized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvectionScheme {
    /// Nodal/collocation advective form (element-local; smooth, low-Re).
    Nodal,
    /// Split-form (KEP) DG with a Rusanov interface flux (energy-stable, high-Re).
    SplitFormDg,
}

pub struct Stokes<'m> {
    pub mesh: &'m Mesh2d,
    pub nu: f64,
    pub dt: f64,
    /// Convection discretization for [`step_ns`](Stokes::step_ns).
    pub convection_scheme: ConvectionScheme,
    /// Pressure-Poisson operator (pure Neumann, singular).
    pressure: Poisson<'m>,
    /// Velocity Helmholtz operator `(1/(νΔt))M + A` (Dirichlet).
    velocity: Poisson<'m>,
    tol: f64,
    maxit: usize,
}

impl<'m> Stokes<'m> {
    pub fn new(mesh: &'m Mesh2d, alpha: f64, nu: f64, dt: f64) -> Self {
        let lambda = 1.0 / (nu * dt);
        Self {
            mesh,
            nu,
            dt,
            convection_scheme: ConvectionScheme::Nodal,
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

    /// Advance one BDF1 dual-splitting step to time `t` (the new time level).
    /// `bc_u`,`bc_v` give Dirichlet velocity on the boundary; `fx`,`fy` the forcing.
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
    ) -> (Vec<f64>, Vec<f64>) {
        let mesh = self.mesh;
        let refq = &mesh.refq;
        let nn = refq.n_nodes();
        let dt = self.dt;
        let lambda = 1.0 / (self.nu * dt);

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
        let _ = lambda;
        self.project_and_diffuse(uhx, uhy, t, bc_u, bc_v)
    }

    /// One incompressible **Navier–Stokes** step: like [`step`], but the explicit
    /// predictor includes the advection term, `û = uⁿ + Δt(f − (uⁿ·∇)uⁿ)`
    /// (extrapolated/explicit — non-stiff at low Reynolds number).
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
    ) -> (Vec<f64>, Vec<f64>) {
        let mesh = self.mesh;
        let nn = mesh.refq.n_nodes();
        // Sample the analytic forcing to a nodal field and delegate.
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
    /// viscoelastic coupling, which is a field rather than a closure.
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
    ) -> (Vec<f64>, Vec<f64>) {
        let mesh = self.mesh;
        let dt = self.dt;
        let (cx, cy) = match self.convection_scheme {
            ConvectionScheme::Nodal => self.convection(ux, uy),
            ConvectionScheme::SplitFormDg => {
                // (u·∇)u = ∇·(u⊗u) = −rhs of the conservation-law operator.
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

    /// Nodal (collocation) advective convection `((u·∇)u, (u·∇)v)`.
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

    /// Stages 2–3 shared by Stokes and NS: project the predictor to divergence-free,
    /// then implicit viscous solve. `uhx`,`uhy` are the explicit predictor `û`.
    fn project_and_diffuse(
        &self,
        mut uhx: Vec<f64>,
        mut uhy: Vec<f64>,
        t: f64,
        bc_u: impl Fn(f64, f64, f64) -> f64,
        bc_v: impl Fn(f64, f64, f64) -> f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let mesh = self.mesh;
        let refq = &mesh.refq;
        let nn = refq.n_nodes();
        let dt = self.dt;
        let lambda = 1.0 / (self.nu * dt);

        // Stage 2 — pressure projection: ∇²p = (1/Δt)∇·û (homogeneous Neumann).
        // Solver convention: A ≈ −∇², so −∇²p = −(1/Δt)∇·û gives ∇²p = (1/Δt)∇·û.
        let div = self.divergence(&uhx, &uhy);
        let fp: Vec<f64> = div.iter().map(|d| -d / dt).collect();
        let bp = self.pressure.rhs_mixed(&fp, |_, _| 0.0, |_, _| 0.0);
        let (p, _it) = self.pressure.cg_deflated(&bp, self.tol, self.maxit);
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
        let (uxn, _, _) = self.velocity.cg(&bx, self.tol, self.maxit);
        let (uyn, _, _) = self.velocity.cg(&by, self.tol, self.maxit);
        (uxn, uyn)
    }

    /// Velocity-field L2 error norm (uses the velocity operator's mass).
    pub fn l2_norm(&self, v: &[f64]) -> f64 {
        self.velocity.l2_norm(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn nodal(mesh: &Mesh2d, f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
        let nn = mesh.refq.n_nodes();
        let mut v = vec![0.0; mesh.n_elements() * nn];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                v[e * nn + k] = f(el.geom.x[k], el.geom.y[k]);
            }
        }
        v
    }

    #[test]
    fn convection_operator_is_exact_on_polynomials() {
        // (u·∇)u for u=(x², y³): N_u = x²·2x = 2x³, N_v = y³·3y² = 3y⁵. Nodal/pointwise
        // ⇒ exact to round-off (grads exact for p≥3).
        let p = 5;
        let mesh = Mesh2d::rectangular(p, 2, 2, [0.0, 1.0], [0.0, 1.0]);
        let st = Stokes::new(&mesh, 5.0, 1.0, 0.01);
        let ux = nodal(&mesh, |x, _| x * x);
        let uy = nodal(&mesh, |_, y| y * y * y);
        let (cx, cy) = st.convection(&ux, &uy);
        let nn = mesh.refq.n_nodes();
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let (x, y) = (el.geom.x[k], el.geom.y[k]);
                assert!((cx[e * nn + k] - 2.0 * x.powi(3)).abs() < 1e-9, "N_u");
                assert!((cy[e * nn + k] - 3.0 * y.powi(5)).abs() < 1e-9, "N_v");
            }
        }
    }

    #[test]
    fn navier_stokes_taylor_green_converges() {
        // Taylor–Green is an exact NS solution: same decaying velocity as the Stokes
        // vortex, with p = −(F²/4)(cos2πx+cos2πy), f=0. The advection here is a pure
        // gradient (−∇p), so the projection must absorb it and leave the TG velocity.
        // (The convection *magnitude* is pinned by the unit test above.)
        let nu = 1.0;
        let p = 4;
        let mesh = Mesh2d::rectangular(p, 4, 4, [0.0, 1.0], [0.0, 1.0]);
        let decay = |t: f64| (-2.0 * PI * PI * nu * t).exp();
        let eu = |x: f64, y: f64, t: f64| -(PI * x).cos() * (PI * y).sin() * decay(t);
        let ev = |x: f64, y: f64, t: f64| (PI * x).sin() * (PI * y).cos() * decay(t);
        let zero = |_: f64, _: f64, _: f64| 0.0;
        let t_end = 0.1;
        let mut errs = Vec::new();
        for &nsteps in &[5usize, 10, 20] {
            let dt = t_end / nsteps as f64;
            let st = Stokes::new(&mesh, 5.0, nu, dt);
            let mut ux = nodal(&mesh, |x, y| eu(x, y, 0.0));
            let mut uy = nodal(&mesh, |x, y| ev(x, y, 0.0));
            let mut t = 0.0;
            for _ in 0..nsteps {
                t += dt;
                let (nx, ny) = st.step_ns(&ux, &uy, t, eu, ev, zero, zero);
                ux = nx;
                uy = ny;
            }
            let exu = nodal(&mesh, |x, y| eu(x, y, t_end));
            let exv = nodal(&mesh, |x, y| ev(x, y, t_end));
            let ex: Vec<f64> = ux.iter().zip(&exu).map(|(a, b)| a - b).collect();
            let ey: Vec<f64> = uy.iter().zip(&exv).map(|(a, b)| a - b).collect();
            errs.push((st.l2_norm(&ex).powi(2) + st.l2_norm(&ey).powi(2)).sqrt());
        }
        eprintln!("Navier–Stokes Taylor–Green L2 error (Δt halving): {errs:?}");
        assert!(errs[1] < errs[0] && errs[2] < errs[1], "NS not converging: {errs:?}");
        assert!(errs[2] < 5e-3, "final NS error {}", errs[2]);
    }

    #[test]
    fn navier_stokes_taylor_green_split_form_convection() {
        // Same Taylor–Green NS check, but with the energy-stable split-form DG
        // convection (the high-Re path) — must still reproduce the exact vortex.
        let nu = 1.0;
        let p = 4;
        let mesh = Mesh2d::rectangular(p, 4, 4, [0.0, 1.0], [0.0, 1.0]);
        let decay = |t: f64| (-2.0 * PI * PI * nu * t).exp();
        let eu = |x: f64, y: f64, t: f64| -(PI * x).cos() * (PI * y).sin() * decay(t);
        let ev = |x: f64, y: f64, t: f64| (PI * x).sin() * (PI * y).cos() * decay(t);
        let zero = |_: f64, _: f64, _: f64| 0.0;
        let t_end = 0.1;
        let mut errs = Vec::new();
        for &nsteps in &[5usize, 10, 20] {
            let dt = t_end / nsteps as f64;
            let mut st = Stokes::new(&mesh, 5.0, nu, dt);
            st.convection_scheme = ConvectionScheme::SplitFormDg;
            let mut ux = nodal(&mesh, |x, y| eu(x, y, 0.0));
            let mut uy = nodal(&mesh, |x, y| ev(x, y, 0.0));
            let mut t = 0.0;
            for _ in 0..nsteps {
                t += dt;
                let (nx, ny) = st.step_ns(&ux, &uy, t, eu, ev, zero, zero);
                ux = nx;
                uy = ny;
            }
            let exu = nodal(&mesh, |x, y| eu(x, y, t_end));
            let exv = nodal(&mesh, |x, y| ev(x, y, t_end));
            let ex: Vec<f64> = ux.iter().zip(&exu).map(|(a, b)| a - b).collect();
            let ey: Vec<f64> = uy.iter().zip(&exv).map(|(a, b)| a - b).collect();
            errs.push((st.l2_norm(&ex).powi(2) + st.l2_norm(&ey).powi(2)).sqrt());
        }
        eprintln!("NS (split-form DG convection) TG error: {errs:?}");
        assert!(errs[1] < errs[0] && errs[2] < errs[1], "not converging: {errs:?}");
        assert!(errs[2] < 5e-3, "final error {} too large", errs[2]);
    }

    #[test]
    fn decaying_stokes_vortex_first_order_in_time() {
        // Exact Stokes solution on [0,1]²: u = −cos(πx)sin(πy)e^{−2π²νt},
        // v = sin(πx)cos(πy)e^{−2π²νt}, with p ≡ 0 and f ≡ 0. Divergence-free.
        let nu = 1.0;
        let p = 4;
        let mesh = Mesh2d::rectangular(p, 4, 4, [0.0, 1.0], [0.0, 1.0]);
        let decay = |t: f64| (-2.0 * PI * PI * nu * t).exp();
        let eu = |x: f64, y: f64, t: f64| -(PI * x).cos() * (PI * y).sin() * decay(t);
        let ev = |x: f64, y: f64, t: f64| (PI * x).sin() * (PI * y).cos() * decay(t);
        let zero = |_: f64, _: f64, _: f64| 0.0;
        let t_end = 0.1;

        let mut errs = Vec::new();
        for &nsteps in &[5usize, 10, 20] {
            let dt = t_end / nsteps as f64;
            let stokes = Stokes::new(&mesh, 5.0, nu, dt);
            let mut ux = nodal(&mesh, |x, y| eu(x, y, 0.0));
            let mut uy = nodal(&mesh, |x, y| ev(x, y, 0.0));
            let mut t = 0.0;
            for _ in 0..nsteps {
                t += dt;
                let (nx, ny) = stokes.step(&ux, &uy, t, eu, ev, zero, zero);
                ux = nx;
                uy = ny;
            }
            let exu = nodal(&mesh, |x, y| eu(x, y, t_end));
            let exv = nodal(&mesh, |x, y| ev(x, y, t_end));
            let ex: Vec<f64> = ux.iter().zip(&exu).map(|(a, b)| a - b).collect();
            let ey: Vec<f64> = uy.iter().zip(&exv).map(|(a, b)| a - b).collect();
            let err = (stokes.l2_norm(&ex).powi(2) + stokes.l2_norm(&ey).powi(2)).sqrt();
            errs.push(err);
        }
        eprintln!("Stokes velocity L2 error (Δt halving): {errs:?}");
        // Monotone decrease and roughly first-order temporal convergence.
        assert!(errs[1] < errs[0] && errs[2] < errs[1], "not converging: {errs:?}");
        let rate = (errs[0] / errs[1]).log2();
        assert!(rate > 0.7, "temporal rate {rate} too low (errs {errs:?})");
        assert!(errs[2] < 5e-3, "final error {} too large", errs[2]);
    }
}
