//! Immersed boundaries — Stage 1: **volume penalization** (Brinkman) for rigid
//! bodies on a fixed mesh. See `docs/immersed-boundary-strategy.md`.
//!
//! A solid occupying region `Ω_s` is represented by an Eulerian indicator
//! `χ(x) ∈ [0,1]` (1 inside the solid). The momentum equation gains a porous-drag
//! body force `−(χ/η_b)(u − u_s)` that drives the fluid velocity toward the solid
//! velocity `u_s` inside the body; as the penalization parameter `η_b → 0` the
//! no-slip/rigid constraint is enforced ever more strongly. We apply it as an
//! **implicit** per-step relaxation (backward Euler of the drag ODE):
//! ```text
//!   u ← (u + β·u_s)/(1 + β),     β = χ · dt/η_b
//! ```
//! which is a convex combination (unconditionally stable) and lets `η_b` be small
//! without a time-step restriction. This is low-order at the interface (≈1st-order
//! velocity), but integrated forces converge — adequate for the rigid-body PoC.

use super::mesh::Mesh2d;

/// An immersed solid: its Eulerian indicator and (rigid-body) velocity field.
pub trait ImmersedSolid {
    /// Solid indicator `χ ∈ [0,1]` at a point (1 inside, 0 in the fluid).
    fn indicator(&self, x: f64, y: f64) -> f64;
    /// Solid velocity `u_s = (uₛₓ, uₛᵧ)` at a point (0 for a stationary body).
    fn velocity(&self, x: f64, y: f64) -> (f64, f64);
}

/// A circular disk. `smooth` is the half-width of a `tanh` interface band; set it
/// to `0` for a sharp indicator. A stationary disk unless `vel` is set.
#[derive(Clone, Copy, Debug)]
pub struct Disk {
    pub cx: f64,
    pub cy: f64,
    pub r: f64,
    pub smooth: f64,
    /// Rigid translation velocity `(uₛₓ, uₛᵧ)`.
    pub vel: (f64, f64),
}

impl Disk {
    /// Stationary sharp-interface disk.
    pub fn new(cx: f64, cy: f64, r: f64) -> Self {
        Self { cx, cy, r, smooth: 0.0, vel: (0.0, 0.0) }
    }
}

impl ImmersedSolid for Disk {
    fn indicator(&self, x: f64, y: f64) -> f64 {
        // Signed distance to the surface (negative inside).
        let d = ((x - self.cx).powi(2) + (y - self.cy).powi(2)).sqrt() - self.r;
        if self.smooth <= 0.0 {
            if d < 0.0 {
                1.0
            } else {
                0.0
            }
        } else {
            // Smoothed Heaviside: 1 deep inside, 0 far outside.
            0.5 * (1.0 - (d / self.smooth).tanh())
        }
    }
    fn velocity(&self, _x: f64, _y: f64) -> (f64, f64) {
        self.vel
    }
}

/// Shape of a rigid body, defined in its own body frame (centered at the origin,
/// major axis along body-`x`).
#[derive(Clone, Copy, Debug)]
pub enum Shape {
    Disk { r: f64 },
    Ellipse { a: f64, b: f64 },
}

/// A rigid body with a position, orientation `φ`, and rigid velocity `(u, v, ω)`.
/// Its velocity field is `u_s(x) = (u − ω(y−c_y), v + ω(x−c_x))`. Used for
/// freely-suspended particles (Jeffery orbits): the velocity is set each step by an
/// L2 projection of the fluid onto rigid motions over the body.
#[derive(Clone, Copy, Debug)]
pub struct RigidBody {
    pub shape: Shape,
    pub cx: f64,
    pub cy: f64,
    pub phi: f64,
    pub u: f64,
    pub v: f64,
    pub omega: f64,
    pub smooth: f64,
}

impl RigidBody {
    pub fn disk(cx: f64, cy: f64, r: f64) -> Self {
        Self { shape: Shape::Disk { r }, cx, cy, phi: 0.0, u: 0.0, v: 0.0, omega: 0.0, smooth: 0.0 }
    }
    pub fn ellipse(cx: f64, cy: f64, a: f64, b: f64, phi: f64) -> Self {
        Self { shape: Shape::Ellipse { a, b }, cx, cy, phi, u: 0.0, v: 0.0, omega: 0.0, smooth: 0.0 }
    }
}

impl ImmersedSolid for RigidBody {
    fn indicator(&self, x: f64, y: f64) -> f64 {
        // Rotate the point into the body frame (by −φ).
        let (dx, dy) = (x - self.cx, y - self.cy);
        let (c, s) = (self.phi.cos(), self.phi.sin());
        let xb = c * dx + s * dy;
        let yb = -s * dx + c * dy;
        // Signed-distance-like scalar (negative inside).
        let d = match self.shape {
            Shape::Disk { r } => (xb * xb + yb * yb).sqrt() - r,
            Shape::Ellipse { a, b } => {
                let q = ((xb / a).powi(2) + (yb / b).powi(2)).sqrt();
                (q - 1.0) * a.min(b) // approx distance; sign is exact
            }
        };
        if self.smooth <= 0.0 {
            if d < 0.0 {
                1.0
            } else {
                0.0
            }
        } else {
            0.5 * (1.0 - (d / self.smooth).tanh())
        }
    }
    fn velocity(&self, x: f64, y: f64) -> (f64, f64) {
        (self.u - self.omega * (y - self.cy), self.v + self.omega * (x - self.cx))
    }
}

/// Precomputed nodal volume-penalization operator for one (rigid) solid.
#[derive(Clone, Debug)]
pub struct VolumePenalization {
    /// Nodal indicator `χ`.
    pub mask: Vec<f64>,
    /// Nodal solid velocity components.
    pub us_x: Vec<f64>,
    pub us_y: Vec<f64>,
    /// Penalization parameter `η_b` (smaller ⇒ stiffer no-slip).
    pub eta_b: f64,
}

impl VolumePenalization {
    /// Sample a solid's indicator and velocity onto the mesh nodes.
    pub fn new(mesh: &Mesh2d, solid: &impl ImmersedSolid, eta_b: f64) -> Self {
        let nn = mesh.refq.n_nodes();
        let ndof = mesh.n_elements() * nn;
        let mut mask = vec![0.0; ndof];
        let mut us_x = vec![0.0; ndof];
        let mut us_y = vec![0.0; ndof];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let (x, y) = (el.geom.x[k], el.geom.y[k]);
                mask[e * nn + k] = solid.indicator(x, y);
                let (sx, sy) = solid.velocity(x, y);
                us_x[e * nn + k] = sx;
                us_y[e * nn + k] = sy;
            }
        }
        Self { mask, us_x, us_y, eta_b }
    }

    /// Apply the implicit Brinkman relaxation in place over a step `dt`.
    pub fn apply(&self, ux: &mut [f64], uy: &mut [f64], dt: f64) {
        let r = dt / self.eta_b;
        for i in 0..ux.len() {
            let beta = r * self.mask[i];
            let denom = 1.0 + beta;
            ux[i] = (ux[i] + beta * self.us_x[i]) / denom;
            uy[i] = (uy[i] + beta * self.us_y[i]) / denom;
        }
    }

    /// L2-project the fluid velocity onto rigid motions over the body region,
    /// returning the freely-suspended rigid velocity `(U, V, ω)`:
    /// `U = ∫χu/∫χ`, `V = ∫χv/∫χ`, `ω = ∫χ(r×u)/∫χ|r|²` about `(cx, cy)`.
    /// For a body in ambient shear this yields exactly the Jeffery angular velocity.
    pub fn project_rigid(&self, ux: &[f64], uy: &[f64], mesh: &Mesh2d, cx: f64, cy: f64) -> (f64, f64, f64) {
        let nn = mesh.refq.n_nodes();
        let (mut su, mut sv, mut sm, mut sl, mut sr2) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let i = e * nn + k;
                let w = el.geom.jw[k] * self.mask[i];
                let (rx, ry) = (el.geom.x[k] - cx, el.geom.y[k] - cy);
                su += w * ux[i];
                sv += w * uy[i];
                sm += w;
                sl += w * (rx * uy[i] - ry * ux[i]);
                sr2 += w * (rx * rx + ry * ry);
            }
        }
        if sm <= 0.0 || sr2 <= 0.0 {
            return (0.0, 0.0, 0.0); // no resolved solid nodes
        }
        (su / sm, sv / sm, sl / sr2)
    }

    /// Number of nodes flagged as solid (`χ > 0.5`) — handy for diagnostics.
    pub fn n_solid_nodes(&self) -> usize {
        self.mask.iter().filter(|&&c| c > 0.5).count()
    }

    /// Hydrodynamic force the fluid exerts on the body (the penalization drag),
    /// `F = ∫ (χ/η_b)(u − u_s) dV`. This is exactly the momentum the implicit
    /// relaxation removes from the fluid per unit time, so it is consistent with
    /// [`apply`](Self::apply) when evaluated on the penalized velocity. Returns
    /// `(Fx, Fy)`.
    pub fn force(&self, ux: &[f64], uy: &[f64], mesh: &Mesh2d) -> (f64, f64) {
        let nn = mesh.refq.n_nodes();
        let inv = 1.0 / self.eta_b;
        let (mut fx, mut fy) = (0.0, 0.0);
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let i = e * nn + k;
                let c = self.mask[i] * inv * el.geom.jw[k];
                fx += c * (ux[i] - self.us_x[i]);
                fy += c * (uy[i] - self.us_y[i]);
            }
        }
        (fx, fy)
    }

    /// Hydrodynamic force AND torque the fluid exerts on the body, about the center
    /// `(cx, cy)`: `(Fx, Fy, T)` with `F = ∫ (χ/η_b)(u − u_s) dV` and
    /// `T = ∫ (χ/η_b) [(x−cx)(u_y−u_{s,y}) − (y−cy)(u_x−u_{s,x})] dV`. The torque is
    /// the `z`-component of `∫ r × (χ/η_b)(u − u_s)`. Consistent with [`force`](Self::force)
    /// (same per-node weight) — the additional moment arm gives the angular reaction for
    /// the Newton–Euler update of a freely-moving body. See [`FreeBody::advance`].
    pub fn force_torque(&self, ux: &[f64], uy: &[f64], mesh: &Mesh2d, cx: f64, cy: f64) -> (f64, f64, f64) {
        let nn = mesh.refq.n_nodes();
        let inv = 1.0 / self.eta_b;
        let (mut fx, mut fy, mut tq) = (0.0, 0.0, 0.0);
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let i = e * nn + k;
                let c = self.mask[i] * inv * el.geom.jw[k];
                let dux = ux[i] - self.us_x[i];
                let duy = uy[i] - self.us_y[i];
                fx += c * dux;
                fy += c * duy;
                tq += c * ((el.geom.x[k] - cx) * duy - (el.geom.y[k] - cy) * dux);
            }
        }
        (fx, fy, tq)
    }
}

/// A freely-moving rigid body: a [`RigidBody`] (shape + pose + rigid velocity)
/// carrying mass, moment of inertia, and an optional constant external body force
/// (e.g. buoyancy-corrected gravity). It owns the **explicit Newton–Euler** update
/// driven by the hydrodynamic force/torque recovered from the volume penalization —
/// the "M1" two-way coupling (research: `docs/research-moving-particle-coupling.md`).
///
/// Stability note: explicit (weak) coupling is only stable above a critical
/// solid/fluid density ratio (the added-mass limit), so this is for **heavy**
/// particles. Light / neutrally-buoyant particles need strong coupling (M3).
///
/// Each step the driver ([`crate::sim::MovingPenalizationHook`] and its GPU twin):
/// 1. applies penalization to imprint the body at its current pose/velocity,
/// 2. recovers `(Fx, Fy, T)` via [`VolumePenalization::force_torque`],
/// 3. calls [`advance`](Self::advance) (Newton–Euler),
/// 4. rebuilds the penalization mask via [`penalization`](Self::penalization).
#[derive(Clone, Copy, Debug)]
pub struct FreeBody {
    /// Shape + pose (`cx, cy, phi`) + rigid velocity (`u, v, omega`); also supplies
    /// the penalization indicator and the rigid-body velocity target field.
    pub body: RigidBody,
    /// Mass `m` (Newton: `m·dU/dt = F`).
    pub mass: f64,
    /// Moment of inertia `I` about the center (Euler: `I·dω/dt = T`).
    pub inertia: f64,
    /// Constant external body force `(Fx, Fy)` — e.g. `((ρ_s−ρ_f)·V·g)` for gravity
    /// with buoyancy already removed. Zero by default.
    pub fext: (f64, f64),
    /// Penalization parameter `η_b` used to rebuild the mask as the body moves.
    pub eta_b: f64,
}

impl FreeBody {
    /// A freely-moving body from a [`RigidBody`] with `mass`, `inertia`, penalization
    /// `eta_b`, and no external force.
    pub fn new(body: RigidBody, mass: f64, inertia: f64, eta_b: f64) -> Self {
        Self { body, mass, inertia, fext: (0.0, 0.0), eta_b }
    }

    /// A solid **disk** of radius `r` and uniform density `rho`: sets mass `ρ·πr²` and
    /// inertia `½·m·r²` automatically. The classic M1 test particle.
    pub fn disk(cx: f64, cy: f64, r: f64, rho: f64, eta_b: f64) -> Self {
        let mass = rho * std::f64::consts::PI * r * r;
        let inertia = 0.5 * mass * r * r;
        Self::new(RigidBody::disk(cx, cy, r), mass, inertia, eta_b)
    }

    /// Set a constant external body force (e.g. gravity/buoyancy). Builder-style.
    pub fn with_external_force(mut self, fx: f64, fy: f64) -> Self {
        self.fext = (fx, fy);
        self
    }

    /// Rebuild the volume-penalization operator for the body's current pose/velocity.
    pub fn penalization(&self, mesh: &Mesh2d) -> VolumePenalization {
        VolumePenalization::new(mesh, &self.body, self.eta_b)
    }

    /// Explicit **Newton–Euler** advance over `dt` given the hydrodynamic force/torque
    /// `(fx, fy, torque)` from [`VolumePenalization::force_torque`]. Semi-implicit
    /// (symplectic) Euler: update the rigid velocity first, then advect the pose with
    /// the new velocity — better momentum behavior than fully-explicit Euler. The
    /// external force is added to the hydrodynamic one.
    ///
    /// EXPLICIT (weak) coupling: the force is computed before the body responds, so it is
    /// unstable below a critical solid/fluid density ratio (added-mass effect). For light
    /// / neutrally-buoyant particles use [`strong_solve`](Self::strong_solve) (M3).
    pub fn advance(&mut self, fx: f64, fy: f64, torque: f64, dt: f64) {
        self.body.u += dt * (fx + self.fext.0) / self.mass;
        self.body.v += dt * (fy + self.fext.1) / self.mass;
        self.body.omega += dt * torque / self.inertia;
        self.body.cx += dt * self.body.u;
        self.body.cy += dt * self.body.v;
        self.body.phi += dt * self.body.omega;
    }

    /// **Strong (implicit) coupling** — solve the new rigid velocity `(U, V, ω)`
    /// *simultaneously* with the implicit penalization constraint, removing the
    /// added-mass density-ratio limit so light / neutrally-buoyant / zero-mass particles
    /// are stable (M3; the discrete form of Lee/Lee's simultaneous Newton–Euler + IB-force
    /// solve, localized to gale's volume penalization).
    ///
    /// The implicit penalization sets `u = (u* + β·u_s)/(1+β)`, `β = χ·dt/η_b`, so the
    /// hydrodynamic force `F = ∫(χ/η_b)(u − u_s) = ∫ w·(u* − u_s)`, `w = (χ/η_b)·jw/(1+β)`,
    /// is LINEAR in `u_s(x) = (U − ω(y−c_y), V + ω(x−c_x))`. Inserting `F`/`T` into the
    /// backward-Euler Newton–Euler relations `m(U−U₀)/dt = F_x+F_x^{ext}` etc. gives the
    /// symmetric 3×3 system (in moments `S₀=Σw`, `S_x=Σw r_x`, `S_y=Σw r_y`,
    /// `S_{r²}=Σw|r|²`, and predictor loads `A_u=Σw u*_x`, `A_v=Σw u*_y`,
    /// `A_t=Σw(r_x u*_y − r_y u*_x)`):
    /// ```text
    ///   [ a   0  −S_y ] [U]   [ (m/dt)U₀ + A_u + F_x^ext ]
    ///   [ 0   a   S_x ] [V] = [ (m/dt)V₀ + A_v + F_y^ext ]
    ///   [−S_y S_x  b  ] [ω]   [ (I/dt)ω₀ + A_t          ]
    /// ```
    /// with `a = m/dt + S₀`, `b = I/dt + S_{r²}`. The determinant `a(ab − S_x² − S_y²)`
    /// is positive for any body of finite extent (Cauchy–Schwarz: `S_x²+S_y² ≤ S₀ S_{r²}
    /// ≤ ab`) — well-posed even at `m = I = 0`. `(ux, uy)` is the predictor field `u*`
    /// (post-fluid-solve, pre-penalization); `pen` supplies `χ`/`η_b` at the body's
    /// current pose. Returns the implicit `(U, V, ω)`; does not mutate the body.
    pub fn strong_solve(&self, ux: &[f64], uy: &[f64], mesh: &Mesh2d, pen: &VolumePenalization, dt: f64) -> (f64, f64, f64) {
        let nn = mesh.refq.n_nodes();
        let inv = 1.0 / pen.eta_b;
        let r = dt * inv; // dt/η_b, so β = r·χ
        let (cx, cy) = (self.body.cx, self.body.cy);
        let (mut s0, mut sx, mut sy, mut sr2) = (0.0, 0.0, 0.0, 0.0);
        let (mut au, mut av, mut at) = (0.0, 0.0, 0.0);
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let i = e * nn + k;
                let chi = pen.mask[i];
                if chi == 0.0 {
                    continue;
                }
                let w = chi * inv * el.geom.jw[k] / (1.0 + r * chi);
                let rx = el.geom.x[k] - cx;
                let ry = el.geom.y[k] - cy;
                s0 += w;
                sx += w * rx;
                sy += w * ry;
                sr2 += w * (rx * rx + ry * ry);
                au += w * ux[i];
                av += w * uy[i];
                at += w * (rx * uy[i] - ry * ux[i]);
            }
        }
        let a = self.mass / dt + s0;
        let b = self.inertia / dt + sr2;
        let r0 = self.mass / dt * self.body.u + au + self.fext.0;
        let r1 = self.mass / dt * self.body.v + av + self.fext.1;
        let r2 = self.inertia / dt * self.body.omega + at;
        // Block elimination: U = (r0 + S_y ω)/a, V = (r1 − S_x ω)/a, then solve for ω.
        let denom = b - (sx * sx + sy * sy) / a;
        if a.abs() < 1e-300 || denom.abs() < 1e-300 {
            return (self.body.u, self.body.v, self.body.omega); // degenerate (no resolved body)
        }
        let omega = (r2 + (sy * r0 - sx * r1) / a) / denom;
        let u = (r0 + sy * omega) / a;
        let v = (r1 - sx * omega) / a;
        (u, v, omega)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dg::{ConstitutiveModel, LogConfOldroydB, Stokes, ViscoelasticFlow};
    use std::f64::consts::PI;

    #[test]
    fn disk_indicator_inside_outside() {
        let d = Disk::new(0.5, 0.5, 0.2);
        assert_eq!(d.indicator(0.5, 0.5), 1.0); // center inside
        assert_eq!(d.indicator(0.5, 0.65), 1.0); // r=0.15 < 0.2 inside
        assert_eq!(d.indicator(0.5, 0.8), 0.0); // r=0.3 > 0.2 outside
        assert_eq!(d.indicator(0.9, 0.9), 0.0);
    }

    #[test]
    fn penalization_drives_solid_to_us_exactly() {
        // Implicit relaxation is exact algebra: inside (χ=1) u → (u+β uₛ)/(1+β);
        // outside (χ=0) u is untouched. Residual shrinks like η_b/(η_b+dt).
        let mesh = Mesh2d::rectangular(3, 3, 3, [0.0, 1.0], [0.0, 1.0]);
        let disk = Disk::new(0.5, 0.5, 0.2);
        let dt = 0.02;
        let nn = mesh.refq.n_nodes();
        let ndof = mesh.n_elements() * nn;

        let u0 = 1.3;
        let mut residuals = Vec::new();
        for &eta_b in &[1e-2, 1e-3, 1e-4] {
            let pen = VolumePenalization::new(&mesh, &disk, eta_b);
            let mut ux = vec![u0; ndof];
            let mut uy = vec![-0.7; ndof];
            pen.apply(&mut ux, &mut uy, dt);
            let beta = dt / eta_b; // χ=1 inside
            let inside_expected = u0 / (1.0 + beta);
            for i in 0..ndof {
                if pen.mask[i] > 0.5 {
                    assert!((ux[i] - inside_expected).abs() < 1e-12, "in-solid x");
                    assert!((uy[i] - (-0.7 / (1.0 + beta))).abs() < 1e-12, "in-solid y");
                } else if pen.mask[i] == 0.0 {
                    assert!((ux[i] - u0).abs() < 1e-12, "fluid untouched x");
                    assert!((uy[i] - (-0.7)).abs() < 1e-12, "fluid untouched y");
                }
            }
            residuals.push(inside_expected);
        }
        // Single-step residual ∝ 1/(1+dt/η_b): strictly decreasing with η_b,
        // and → 0 in the stiff limit (η_b=1e-4 ⇒ u0/201 < 1% u0).
        assert!(residuals[1] < residuals[0] && residuals[2] < residuals[1], "not converging: {residuals:?}");
        assert!(residuals[2] < 0.01 * u0, "stiff-limit residual too large: {}", residuals[2]);
    }

    #[test]
    fn penalized_disk_suppresses_flow_inside() {
        // Body-force-driven channel (periodic x, walls y) past a penalized disk:
        // the fluid velocity inside the disk must be driven far below the
        // free-stream Poiseuille peak, and smaller η_b ⇒ stronger suppression.
        let p = 3;
        let mesh = Mesh2d::channel_x(p, 3, 3, [0.0, 1.0], [0.0, 1.0]);
        let nu = 1.0;
        let g = 1.0;
        let dt = 0.02;
        let umax = g / (8.0 * nu); // Poiseuille peak without the obstacle
        let disk = Disk::new(0.5, 0.5, 0.2);
        let bc = |_: f64, _: f64, _: f64| 0.0; // no-slip walls (tags 0,2)
        let drive = move |_: f64, _: f64, _: f64| g;
        let zero = |_: f64, _: f64, _: f64| 0.0;

        let run = |eta_b: f64, nsteps: usize| -> f64 {
            let st = Stokes::new(&mesh, 5.0, nu, dt);
            let pen = VolumePenalization::new(&mesh, &disk, eta_b);
            let mut ux = vec![0.0; mesh.n_elements() * mesh.refq.n_nodes()];
            let mut uy = ux.clone();
            let mut t = 0.0;
            for _ in 0..nsteps {
                t += dt;
                let (nx, ny) = st.step(&ux, &uy, t, bc, bc, drive, zero);
                ux = nx;
                uy = ny;
                pen.apply(&mut ux, &mut uy, dt);
            }
            // Max fluid speed inside the disk.
            let mut inside = 0.0f64;
            for i in 0..ux.len() {
                if pen.mask[i] > 0.5 {
                    inside = inside.max((ux[i] * ux[i] + uy[i] * uy[i]).sqrt());
                }
            }
            assert!(inside.is_finite(), "diverged");
            inside
        };

        let loose = run(1e-2, 150);
        let tight = run(1e-3, 150);
        eprintln!("in-disk |u|: η_b=1e-2 → {loose:.3e}, η_b=1e-3 → {tight:.3e} (Umax={umax:.3e})");
        assert!(loose < 0.25 * umax, "disk did not suppress flow: {loose} vs Umax {umax}");
        assert!(tight < loose, "smaller η_b should suppress more: {tight} !< {loose}");
    }

    #[test]
    fn hydrodynamic_drag_direction_symmetry_linearity() {
        // Drag on a centered disk in a body-force-driven Stokes channel: must point
        // downstream (Fx>0), have ~zero lift (Fy≈0 by symmetry), and be LINEAR in the
        // drive G (Stokes regime: doubling G doubles the drag).
        let p = 3;
        let mesh = Mesh2d::channel_x(p, 3, 3, [0.0, 1.0], [0.0, 1.0]);
        let nu = 1.0;
        let dt = 0.02;
        let eta_b = 1e-3;
        let disk = Disk::new(0.5, 0.5, 0.2);
        let bc = |_: f64, _: f64, _: f64| 0.0;
        let zero = |_: f64, _: f64, _: f64| 0.0;

        let drag = |g: f64| -> (f64, f64) {
            let st = Stokes::new(&mesh, 5.0, nu, dt);
            let pen = VolumePenalization::new(&mesh, &disk, eta_b);
            let drive = move |_: f64, _: f64, _: f64| g;
            let mut ux = vec![0.0; mesh.n_elements() * mesh.refq.n_nodes()];
            let mut uy = ux.clone();
            let mut t = 0.0;
            for _ in 0..200 {
                t += dt;
                let (nx, ny) = st.step(&ux, &uy, t, bc, bc, drive, zero);
                ux = nx;
                uy = ny;
                pen.apply(&mut ux, &mut uy, dt);
            }
            pen.force(&ux, &uy, &mesh)
        };

        let (fx1, fy1) = drag(1.0);
        let (fx2, _fy2) = drag(2.0);
        eprintln!("drag: G=1 → (Fx={fx1:.4e}, Fy={fy1:.2e}); G=2 → Fx={fx2:.4e}; ratio={:.3}", fx2 / fx1);
        assert!(fx1 > 0.0, "drag not downstream: Fx={fx1}");
        assert!(fy1.abs() < 0.02 * fx1, "lift not ~0 by symmetry: Fy={fy1}, Fx={fx1}");
        assert!((fx2 / fx1 - 2.0).abs() < 0.1, "drag not linear in G (Stokes): ratio={}", fx2 / fx1);
    }

    // Ambient simple shear u = (γ̇(y−0.5), 0) sampled at the mesh nodes.
    fn shear_field(mesh: &Mesh2d, gdot: f64) -> (Vec<f64>, Vec<f64>) {
        let nn = mesh.refq.n_nodes();
        let ndof = mesh.n_elements() * nn;
        let mut ux = vec![0.0; ndof];
        let uy = vec![0.0; ndof];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                ux[e * nn + k] = gdot * (el.geom.y[k] - 0.5);
            }
        }
        (ux, uy)
    }

    #[test]
    fn disk_in_shear_rotates_at_half_vorticity() {
        // A freely-suspended disk in simple shear rotates at the ambient half-
        // vorticity ω = −γ̇/2 (the r=1 Jeffery limit). The rigid-motion projection of
        // the exact shear field gives this to round-off (disk symmetry ⇒ Iₓₓ=I_yy).
        let mesh = Mesh2d::rectangular(4, 4, 4, [0.0, 1.0], [0.0, 1.0]);
        let gdot = 2.0;
        let (ux, uy) = shear_field(&mesh, gdot);
        let body = RigidBody::disk(0.5, 0.5, 0.2);
        let pen = VolumePenalization::new(&mesh, &body, 1.0);
        let (u, v, omega) = pen.project_rigid(&ux, &uy, &mesh, 0.5, 0.5);
        assert!(u.abs() < 1e-12 && v.abs() < 1e-12, "translation should vanish: ({u},{v})");
        assert!((omega - (-gdot / 2.0)).abs() < 1e-10, "ω={omega}, want {}", -gdot / 2.0);
    }

    #[test]
    fn ellipse_projection_matches_jeffery() {
        // Projecting ambient shear onto an ellipse's rigid rotation yields the 2D
        // Jeffery angular velocity ω(φ) = −γ̇(r²sin²φ + cos²φ)/(r²+1). Verified at
        // several orientations (discrete moments ⇒ a few-percent tolerance), plus the
        // signature that |ω| is larger with the major axis along the gradient (φ=π/2).
        let mesh = Mesh2d::rectangular(4, 8, 8, [0.0, 1.0], [0.0, 1.0]);
        let gdot = 2.0;
        let (ux, uy) = shear_field(&mesh, gdot);
        let (a, b) = (0.25, 0.125);
        let r = a / b;
        let jeffery = |phi: f64| -gdot * (r * r * phi.sin().powi(2) + phi.cos().powi(2)) / (r * r + 1.0);
        let mut om = Vec::new();
        for &phi in &[0.0, PI / 4.0, PI / 2.0] {
            let body = RigidBody::ellipse(0.5, 0.5, a, b, phi);
            let pen = VolumePenalization::new(&mesh, &body, 1.0);
            let (_, _, omega) = pen.project_rigid(&ux, &uy, &mesh, 0.5, 0.5);
            let exact = jeffery(phi);
            eprintln!("φ={phi:.3}: ω={omega:.4} (Jeffery {exact:.4})");
            assert!((omega - exact).abs() < 0.12 * exact.abs().max(0.1), "φ={phi}: ω={omega} vs {exact}");
            om.push(omega);
        }
        // Faster rotation with the major axis across the flow (φ=π/2) than aligned (φ=0).
        assert!(om[2].abs() > 1.5 * om[0].abs(), "Jeffery modulation wrong: {om:?}");
    }

    #[test]
    fn ellipse_jeffery_orbit_period() {
        // Integrate the orientation φ of a freely-suspended ellipse in ambient shear,
        // with the angular velocity from the rigid-motion projection at each (rotated)
        // orientation. The tumbling period — time for φ to advance by π — must match
        // Jeffery's T = (π/γ̇)(r + 1/r). This exercises the rotating mask and the full
        // orbit; the fluid is the analytic shear (the projection IS the freely-
        // suspended velocity), sidestepping the fragile coupled singular-pressure solve.
        let mesh = Mesh2d::rectangular(4, 8, 8, [0.0, 1.0], [0.0, 1.0]);
        let gdot = 2.0;
        let (su, sv) = shear_field(&mesh, gdot);
        let (a, b) = (0.25, 0.125);
        let r = a / b;
        let dt = 0.001;
        let mut phi = 0.0_f64;
        let mut t = 0.0;
        // ω < 0 (clockwise): integrate until φ has advanced by −π.
        while phi > -PI {
            let body = RigidBody::ellipse(0.5, 0.5, a, b, phi);
            let pen = VolumePenalization::new(&mesh, &body, 1.0);
            let (_, _, omega) = pen.project_rigid(&su, &sv, &mesh, 0.5, 0.5);
            phi += omega * dt;
            t += dt;
            assert!(t < 100.0 && phi.is_finite(), "no tumble (t={t}, φ={phi})");
        }
        let jeffery_t = PI * (r + 1.0 / r) / gdot;
        eprintln!("Jeffery orbit: period={t:.3}, theory T={jeffery_t:.3} (r={r})");
        assert!((t - jeffery_t).abs() < 0.15 * jeffery_t, "period {t} vs Jeffery {jeffery_t}");
    }

    #[test]
    fn force_torque_force_matches_force_and_newton_euler_arithmetic() {
        // (1) The (Fx,Fy) returned by force_torque must equal force() exactly (same
        // per-node weight); the torque is the added moment-arm integral. (2) FreeBody::
        // advance must apply semi-implicit (symplectic) Euler: velocity first, then pose
        // with the NEW velocity, hydrodynamic + external force summed.
        let mesh = Mesh2d::rectangular(3, 4, 4, [0.0, 1.0], [0.0, 1.0]);
        let disk = Disk::new(0.5, 0.5, 0.2);
        let pen = VolumePenalization::new(&mesh, &disk, 1e-3);
        let nn = mesh.refq.n_nodes();
        let ndof = mesh.n_elements() * nn;
        // An asymmetric field so the torque is genuinely nonzero.
        let mut ux = vec![0.0; ndof];
        let uy = vec![0.0; ndof];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                ux[e * nn + k] = el.geom.y[k]; // shear-like ⇒ net spin
            }
        }
        let (fx, fy) = pen.force(&ux, &uy, &mesh);
        let (fxt, fyt, tq) = pen.force_torque(&ux, &uy, &mesh, 0.5, 0.5);
        assert_eq!((fx, fy), (fxt, fyt), "force_torque force components must match force()");
        assert!(tq.abs() > 0.0 && tq.is_finite(), "expected nonzero finite torque, got {tq}");

        // Newton–Euler arithmetic on a disk with a known external force.
        let mut fb = FreeBody::disk(0.5, 0.5, 0.2, 10.0, 1e-3).with_external_force(0.0, -1.0);
        let (m, inertia) = (fb.mass, fb.inertia);
        let (dt, fxh, fyh, th) = (0.1, 3.0, 0.0, 0.5);
        fb.advance(fxh, fyh, th, dt);
        let want_u = dt * (fxh + 0.0) / m;
        let want_v = dt * (fyh - 1.0) / m;
        let want_w = dt * th / inertia;
        assert!((fb.body.u - want_u).abs() < 1e-15, "u {} vs {want_u}", fb.body.u);
        assert!((fb.body.v - want_v).abs() < 1e-15, "v {} vs {want_v}", fb.body.v);
        assert!((fb.body.omega - want_w).abs() < 1e-15, "ω {} vs {want_w}", fb.body.omega);
        // Pose advected with the NEW velocity (symplectic).
        assert!((fb.body.cx - (0.5 + dt * want_u)).abs() < 1e-15);
        assert!((fb.body.cy - (0.5 + dt * want_v)).abs() < 1e-15);
        assert!((fb.body.phi - dt * want_w).abs() < 1e-15);
    }

    #[test]
    fn strong_solve_massless_disk_tracks_uniform_flow_and_is_wellposed_at_zero_mass() {
        // Strong (implicit) coupling on a MASSLESS disk in a uniform predictor field
        // u* = (1, 0): the implicit body-velocity solve must return U=1, V=0, ω=0 — a
        // massless body simply tracks the ambient flow — and must be well-posed at m=I=0
        // (the determinant a(ab−Sx²−Sy²) stays positive by Cauchy–Schwarz). This is the
        // property that makes strong coupling stable where explicit blows up.
        let mesh = Mesh2d::rectangular(4, 4, 4, [0.0, 1.0], [0.0, 1.0]);
        let nn = mesh.refq.n_nodes();
        let ndof = mesh.n_elements() * nn;
        let ux = vec![1.0; ndof];
        let uy = vec![0.0; ndof];
        // Zero mass and inertia ⇒ the added-mass-dominated limit.
        let fb = FreeBody::new(RigidBody::disk(0.5, 0.5, 0.2), 0.0, 0.0, 1e-3);
        let pen = fb.penalization(&mesh);
        let (u, v, om) = fb.strong_solve(&ux, &uy, &mesh, &pen, 0.01);
        assert!((u - 1.0).abs() < 1e-9, "massless disk should track flow: U={u}");
        assert!(v.abs() < 1e-9, "V={v}");
        assert!(om.abs() < 1e-9, "ω={om}");

        // Heavier body in the same uniform flow lags it (0 < U < 1) but stays finite/bounded.
        let heavy = FreeBody::disk(0.5, 0.5, 0.2, 50.0, 1e-3);
        let (uh, vh, omh) = heavy.strong_solve(&ux, &uy, &mesh, &pen, 0.01);
        assert!(uh > 0.0 && uh < 1.0 && uh.is_finite(), "heavy U={uh}");
        assert!(vh.abs() < 1e-9 && omh.abs() < 1e-9, "symmetry broken: V={vh} ω={omh}");
    }

    #[test]
    fn rigid_particle_in_viscoelastic_matrix_is_stable() {
        // Capstone integration: a penalized rigid disk in a log-conformation
        // Oldroyd-B channel flow. Exercises the full stack at once and checks the
        // research-flagged risk — stress pathology / loss of SPD at the immersed
        // surface. Requirements: flow suppressed inside the disk, C = exp(Ψ) stays
        // SPD EVERYWHERE (incl. right at the body), and nothing blows up.
        let p = 3;
        let mesh = Mesh2d::channel_x(p, 3, 3, [0.0, 1.0], [0.0, 1.0]);
        let (eta_s, eta_p, lambda, g) = (0.5, 0.5, 0.5, 1.0);
        let eta0 = eta_s + eta_p;
        let umax = g / (8.0 * eta0);
        let dt = 0.01;
        let eta_b = 1e-3;
        let model = LogConfOldroydB::new(&mesh, lambda, eta_p);
        let ve = ViscoelasticFlow::with_model(&mesh, eta_s, dt, 5.0, model);
        let disk = Disk::new(0.5, 0.5, 0.2);
        let pen = VolumePenalization::new(&mesh, &disk, eta_b);

        let bc = |_: f64, _: f64, _: f64| 0.0;
        let drive = move |_: f64, _: f64, _: f64| g;
        let zero = |_: f64, _: f64, _: f64| 0.0;

        let ndof = mesh.n_elements() * mesh.refq.n_nodes();
        let mut ux = vec![0.0; ndof];
        let mut uy = vec![0.0; ndof];
        let mut psi = ve.model.equilibrium();
        let mut t = 0.0;
        for _ in 0..250 {
            t += dt;
            let (nx, ny, np) = ve.step(&ux, &uy, &psi, t, bc, bc, drive, zero);
            ux = nx;
            uy = ny;
            psi = np;
            pen.apply(&mut ux, &mut uy, dt);
        }

        // C = exp(Ψ) SPD everywhere, including at the immersed surface.
        let c = ve.model.recover_c(&psi);
        for i in 0..ndof {
            let det = c[0][i] * c[2][i] - c[1][i] * c[1][i];
            assert!(c[0][i] > 0.0 && det > 0.0 && det.is_finite(), "C not SPD at node {i}: det={det}");
        }
        // Flow suppressed inside the disk.
        let mut inside = 0.0f64;
        for i in 0..ndof {
            if pen.mask[i] > 0.5 {
                inside = inside.max((ux[i] * ux[i] + uy[i] * uy[i]).sqrt());
            }
        }
        let (fx, fy) = pen.force(&ux, &uy, &mesh);
        eprintln!("VE+IBM: in-disk |u|={inside:.3e} (Umax={umax:.3e}), drag Fx={fx:.4e} Fy={fy:.2e}");
        assert!(inside.is_finite() && inside < 0.3 * umax, "flow not suppressed in disk: {inside}");
        assert!(fx > 0.0 && fx.is_finite(), "drag wrong: Fx={fx}");
    }
}
