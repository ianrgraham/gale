//! Immersed boundaries in 3D — volume penalization (Brinkman) for rigid bodies on
//! a fixed hex mesh. The 3D analogue of [`immersed`](super::immersed): the implicit
//! per-step relaxation `u ← (u + β u_s)/(1+β)`, `β = χ·dt/η_b`, now over three
//! velocity components, plus the 3D rigid-body L2 projection
//! `ω = I⁻¹ ∫χ(r×u)` (the 3D generalization of the 2D Jeffery projection).

use super::mesh3d::Mesh3d;

/// An immersed solid in 3D: its Eulerian indicator and rigid velocity field.
pub trait ImmersedSolid3d {
    /// Indicator `χ ∈ [0,1]` (1 inside the solid).
    fn indicator(&self, x: f64, y: f64, z: f64) -> f64;
    /// Solid velocity `(uₛₓ, uₛᵧ, uₛ_z)` (0 for a stationary body).
    fn velocity(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64);
}

/// A sphere. `smooth` is the half-width of a `tanh` interface band (0 = sharp).
#[derive(Clone, Copy, Debug)]
pub struct Sphere {
    pub cx: f64,
    pub cy: f64,
    pub cz: f64,
    pub r: f64,
    pub smooth: f64,
    pub vel: (f64, f64, f64),
}

impl Sphere {
    pub fn new(cx: f64, cy: f64, cz: f64, r: f64) -> Self {
        Self { cx, cy, cz, r, smooth: 0.0, vel: (0.0, 0.0, 0.0) }
    }
}

fn heaviside(d: f64, smooth: f64) -> f64 {
    if smooth <= 0.0 {
        if d < 0.0 {
            1.0
        } else {
            0.0
        }
    } else {
        0.5 * (1.0 - (d / smooth).tanh())
    }
}

impl ImmersedSolid3d for Sphere {
    fn indicator(&self, x: f64, y: f64, z: f64) -> f64 {
        let d = ((x - self.cx).powi(2) + (y - self.cy).powi(2) + (z - self.cz).powi(2)).sqrt() - self.r;
        heaviside(d, self.smooth)
    }
    fn velocity(&self, _x: f64, _y: f64, _z: f64) -> (f64, f64, f64) {
        self.vel
    }
}

/// An axis-aligned ellipsoid with semi-axes `(a, b, c)` centered at `(cx, cy, cz)`.
#[derive(Clone, Copy, Debug)]
pub struct Ellipsoid {
    pub cx: f64,
    pub cy: f64,
    pub cz: f64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub smooth: f64,
}

impl Ellipsoid {
    pub fn new(cx: f64, cy: f64, cz: f64, a: f64, b: f64, c: f64) -> Self {
        Self { cx, cy, cz, a, b, c, smooth: 0.0 }
    }
}

impl ImmersedSolid3d for Ellipsoid {
    fn indicator(&self, x: f64, y: f64, z: f64) -> f64 {
        let q = (((x - self.cx) / self.a).powi(2)
            + ((y - self.cy) / self.b).powi(2)
            + ((z - self.cz) / self.c).powi(2))
        .sqrt();
        let d = (q - 1.0) * self.a.min(self.b).min(self.c); // approx distance, exact sign
        heaviside(d, self.smooth)
    }
    fn velocity(&self, _x: f64, _y: f64, _z: f64) -> (f64, f64, f64) {
        (0.0, 0.0, 0.0)
    }
}

/// Precomputed nodal volume-penalization operator for one rigid solid in 3D.
#[derive(Clone, Debug)]
pub struct VolumePenalization3d {
    pub mask: Vec<f64>,
    pub us_x: Vec<f64>,
    pub us_y: Vec<f64>,
    pub us_z: Vec<f64>,
    pub eta_b: f64,
}

impl VolumePenalization3d {
    /// Sample a solid's indicator and velocity onto the mesh nodes.
    pub fn new(mesh: &Mesh3d, solid: &impl ImmersedSolid3d, eta_b: f64) -> Self {
        let nn = mesh.refh.n_nodes();
        let ndof = mesh.n_elements() * nn;
        let mut mask = vec![0.0; ndof];
        let mut us_x = vec![0.0; ndof];
        let mut us_y = vec![0.0; ndof];
        let mut us_z = vec![0.0; ndof];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let (x, y, z) = (el.geom.x[k], el.geom.y[k], el.geom.z[k]);
                mask[e * nn + k] = solid.indicator(x, y, z);
                let (sx, sy, sz) = solid.velocity(x, y, z);
                us_x[e * nn + k] = sx;
                us_y[e * nn + k] = sy;
                us_z[e * nn + k] = sz;
            }
        }
        Self { mask, us_x, us_y, us_z, eta_b }
    }

    /// Apply the implicit Brinkman relaxation in place over a step `dt`.
    pub fn apply(&self, ux: &mut [f64], uy: &mut [f64], uz: &mut [f64], dt: f64) {
        let r = dt / self.eta_b;
        for i in 0..ux.len() {
            let beta = r * self.mask[i];
            let denom = 1.0 + beta;
            ux[i] = (ux[i] + beta * self.us_x[i]) / denom;
            uy[i] = (uy[i] + beta * self.us_y[i]) / denom;
            uz[i] = (uz[i] + beta * self.us_z[i]) / denom;
        }
    }

    /// Number of nodes flagged solid (`χ > 0.5`).
    pub fn n_solid_nodes(&self) -> usize {
        self.mask.iter().filter(|&&c| c > 0.5).count()
    }

    /// Hydrodynamic force `F = ∫(χ/η_b)(u − u_s) dV`, returned as `(Fx, Fy, Fz)`.
    pub fn force(&self, ux: &[f64], uy: &[f64], uz: &[f64], mesh: &Mesh3d) -> (f64, f64, f64) {
        let nn = mesh.refh.n_nodes();
        let inv = 1.0 / self.eta_b;
        let (mut fx, mut fy, mut fz) = (0.0, 0.0, 0.0);
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let i = e * nn + k;
                let w = el.geom.jw[k] * self.mask[i] * inv;
                fx += w * (ux[i] - self.us_x[i]);
                fy += w * (uy[i] - self.us_y[i]);
                fz += w * (uz[i] - self.us_z[i]);
            }
        }
        (fx, fy, fz)
    }

    /// L2-project the fluid velocity onto rigid motions over the body region,
    /// returning the freely-suspended rigid velocity `(U, V, W, ωx, ωy, ωz)` about
    /// `(cx, cy, cz)`: `U = ∫χu/∫χ`, and `ω = I⁻¹ ∫χ(r×u)` with the symmetric
    /// inertia tensor `I = ∫χ(|r|²δ − r⊗r)`. This is the 3D Jeffery projection.
    pub fn project_rigid(
        &self,
        ux: &[f64],
        uy: &[f64],
        uz: &[f64],
        mesh: &Mesh3d,
        cx: f64,
        cy: f64,
        cz: f64,
    ) -> (f64, f64, f64, f64, f64, f64) {
        let nn = mesh.refh.n_nodes();
        let (mut su, mut sv, mut sw, mut sm) = (0.0, 0.0, 0.0, 0.0);
        let mut l = [0.0; 3]; // ∫χ (r×u)
        // Inertia tensor entries (symmetric): Ixx,Iyy,Izz,Ixy,Ixz,Iyz.
        let (mut ixx, mut iyy, mut izz, mut ixy, mut ixz, mut iyz) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let i = e * nn + k;
                let wgt = el.geom.jw[k] * self.mask[i];
                let (rx, ry, rz) = (el.geom.x[k] - cx, el.geom.y[k] - cy, el.geom.z[k] - cz);
                let (u, v, w) = (ux[i], uy[i], uz[i]);
                su += wgt * u;
                sv += wgt * v;
                sw += wgt * w;
                sm += wgt;
                // r × u.
                l[0] += wgt * (ry * w - rz * v);
                l[1] += wgt * (rz * u - rx * w);
                l[2] += wgt * (rx * v - ry * u);
                let r2 = rx * rx + ry * ry + rz * rz;
                ixx += wgt * (r2 - rx * rx);
                iyy += wgt * (r2 - ry * ry);
                izz += wgt * (r2 - rz * rz);
                ixy += wgt * (-rx * ry);
                ixz += wgt * (-rx * rz);
                iyz += wgt * (-ry * rz);
            }
        }
        if sm <= 0.0 {
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
        // ω = I⁻¹ L (symmetric 3×3 inverse by cofactors).
        let (a, b, c) = (ixx, ixy, ixz);
        let (d, e_, f) = (ixy, iyy, iyz);
        let (g, h, ii) = (ixz, iyz, izz);
        let det = a * (e_ * ii - f * h) - b * (d * ii - f * g) + c * (d * h - e_ * g);
        let (wx, wy, wz) = if det.abs() > 1e-300 {
            let inv = [
                [(e_ * ii - f * h) / det, (c * h - b * ii) / det, (b * f - c * e_) / det],
                [(f * g - d * ii) / det, (a * ii - c * g) / det, (c * d - a * f) / det],
                [(d * h - e_ * g) / det, (b * g - a * h) / det, (a * e_ - b * d) / det],
            ];
            (
                inv[0][0] * l[0] + inv[0][1] * l[1] + inv[0][2] * l[2],
                inv[1][0] * l[0] + inv[1][1] * l[1] + inv[1][2] * l[2],
                inv[2][0] * l[0] + inv[2][1] * l[1] + inv[2][2] * l[2],
            )
        } else {
            (0.0, 0.0, 0.0)
        };
        (su / sm, sv / sm, sw / sm, wx, wy, wz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_indicator_and_penalization_damps_interior() {
        let mesh = Mesh3d::rectangular(3, 4, 4, 4, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
        let sphere = Sphere::new(0.5, 0.5, 0.5, 0.2);
        let (eta_b, dt) = (1e-4, 0.01);
        let p = VolumePenalization3d::new(&mesh, &sphere, eta_b);
        assert!(p.n_solid_nodes() > 0, "sphere captured no nodes");

        let ndof = mesh.n_elements() * mesh.refh.n_nodes();
        let mut ux = vec![1.0; ndof];
        let mut uy = vec![0.0; ndof];
        let mut uz = vec![0.0; ndof];
        // Reference: manual apply on copies.
        let (mut rx, mut ry, mut rz) = (ux.clone(), uy.clone(), uz.clone());
        for i in 0..ndof {
            let beta = (dt / eta_b) * p.mask[i];
            rx[i] = (rx[i] + 0.0) / (1.0 + beta);
            ry[i] = ry[i] / (1.0 + beta);
            rz[i] = rz[i] / (1.0 + beta);
        }
        p.apply(&mut ux, &mut uy, &mut uz, dt);
        assert_eq!(ux, rx);
        assert_eq!(uy, ry);
        assert_eq!(uz, rz);
        // Interior strongly damped; exterior untouched.
        for i in 0..ndof {
            if p.mask[i] > 0.5 {
                assert!(ux[i].abs() < 0.05, "solid node not damped: {}", ux[i]);
            } else {
                assert!((ux[i] - 1.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn drag_is_positive_for_uniform_stream() {
        let mesh = Mesh3d::rectangular(3, 4, 4, 4, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
        let sphere = Sphere::new(0.5, 0.5, 0.5, 0.2);
        let p = VolumePenalization3d::new(&mesh, &sphere, 1e-3);
        let ndof = mesh.n_elements() * mesh.refh.n_nodes();
        let (fx, fy, fz) = p.force(&vec![1.0; ndof], &vec![0.0; ndof], &vec![0.0; ndof], &mesh);
        assert!(fx > 0.0, "Fx = {fx}");
        assert!(fy.abs() < 1e-9 && fz.abs() < 1e-9, "spurious transverse force");
    }

    #[test]
    fn rigid_projection_recovers_exact_rigid_motion() {
        // u = U0 + ω0 × r over the body ⇒ projection returns (U0, ω0) exactly.
        let mesh = Mesh3d::rectangular(3, 4, 4, 4, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
        let sphere = Sphere::new(0.5, 0.5, 0.5, 0.25);
        let p = VolumePenalization3d::new(&mesh, &sphere, 1e-3);
        let nn = mesh.refh.n_nodes();
        let ndof = mesh.n_elements() * nn;
        let (cx, cy, cz) = (0.5, 0.5, 0.5);
        let u0 = [0.3, -0.1, 0.2];
        let w0 = [0.5, -0.2, 0.4];
        let (mut ux, mut uy, mut uz) = (vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof]);
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let i = e * nn + k;
                let (rx, ry, rz) = (el.geom.x[k] - cx, el.geom.y[k] - cy, el.geom.z[k] - cz);
                ux[i] = u0[0] + (w0[1] * rz - w0[2] * ry);
                uy[i] = u0[1] + (w0[2] * rx - w0[0] * rz);
                uz[i] = u0[2] + (w0[0] * ry - w0[1] * rx);
            }
        }
        let (uu, vv, ww, wx, wy, wz) = p.project_rigid(&ux, &uy, &uz, &mesh, cx, cy, cz);
        assert!((uu - u0[0]).abs() < 1e-9 && (vv - u0[1]).abs() < 1e-9 && (ww - u0[2]).abs() < 1e-9);
        assert!((wx - w0[0]).abs() < 1e-8 && (wy - w0[1]).abs() < 1e-8 && (wz - w0[2]).abs() < 1e-8,
            "ω recovered ({wx},{wy},{wz}) vs ({},{},{})", w0[0], w0[1], w0[2]);
    }
}
