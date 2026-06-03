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
