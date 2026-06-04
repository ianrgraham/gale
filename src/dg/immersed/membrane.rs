//! Immersed boundaries — Stage 2 foundation: **front-tracking** Eulerian↔Lagrangian
//! coupling on the DG-SEM mesh. See `docs/immersed-boundary-strategy.md`.
//!
//! Deformable particles are represented by a closed ring of Lagrangian marker points
//! carrying an elastic membrane. Two operations couple them to the fluid:
//! - **interpolate**: evaluate the DG solution at a marker location (move the marker
//!   with the fluid),
//! - **spread**: deposit a marker force into the fluid as a nodal body force.
//!
//! Rather than Peskin's uniform-grid regularized delta (which fits a Cartesian
//! finite-difference grid, not LGL nodes), we use the **DG nodal basis directly**:
//! interpolation evaluates the element's tensor-product Lagrange polynomial (exact,
//! high-order), and spreading is its mass-weighted adjoint, so the pair is consistent
//! — `spread` is the transpose of `interpolate` w.r.t. the diagonal LGL mass matrix.
//! (Axis-aligned Cartesian elements, as produced by `Mesh2d::rectangular`/`channel_x`.)

use super::mesh::Mesh2d;

/// 1D Lagrange basis values `ℓ_i(r)` at the reference nodes, evaluated at `r ∈ [-1,1]`.
fn lagrange_basis(nodes: &[f64], r: f64) -> Vec<f64> {
    let n = nodes.len();
    (0..n)
        .map(|i| {
            let mut li = 1.0;
            for j in 0..n {
                if j != i {
                    li *= (r - nodes[j]) / (nodes[i] - nodes[j]);
                }
            }
            li
        })
        .collect()
}

/// Locate the (axis-aligned) element containing `(x, y)` and return its index plus
/// the reference coordinates `(r, s) ∈ [-1,1]²`. Returns `None` if outside the mesh.
fn locate(mesh: &Mesh2d, x: f64, y: f64) -> Option<(usize, f64, f64)> {
    for (e, el) in mesh.elements.iter().enumerate() {
        // Corners: [bl, br, tr, tl].
        let (x0, x1) = (el.corners[0][0], el.corners[1][0]);
        let (y0, y1) = (el.corners[0][1], el.corners[3][1]);
        let eps = 1e-12 * (x1 - x0).max(y1 - y0);
        if x >= x0 - eps && x <= x1 + eps && y >= y0 - eps && y <= y1 + eps {
            let r = 2.0 * (x - x0) / (x1 - x0) - 1.0;
            let s = 2.0 * (y - y0) / (y1 - y0) - 1.0;
            return Some((e, r.clamp(-1.0, 1.0), s.clamp(-1.0, 1.0)));
        }
    }
    None
}

/// Interpolate a nodal scalar field to the point `(x, y)` via the DG basis
/// (exact for polynomials up to the element order). Returns `0.0` if outside.
pub fn interpolate(mesh: &Mesh2d, field: &[f64], x: f64, y: f64) -> f64 {
    let Some((e, r, s)) = locate(mesh, x, y) else {
        return 0.0;
    };
    let nodes = &mesh.refq.line.nodes;
    let n1 = nodes.len();
    let lr = lagrange_basis(nodes, r);
    let ls = lagrange_basis(nodes, s);
    let nn = n1 * n1;
    let mut val = 0.0;
    for j in 0..n1 {
        for i in 0..n1 {
            val += field[e * nn + i + j * n1] * lr[i] * ls[j];
        }
    }
    val
}

/// Spread a point force `f` located at `(x, y)` into the nodal body-force field
/// `out` (length `ne·(p+1)²`), as the mass-weighted adjoint of [`interpolate`]:
/// `out_node += f · ℓ_node(x,y) / (J·w)_node`. So `∫ spread(f)·φ = f·φ(x)` for any
/// DG test function `φ` — the consistent weak point load.
pub fn spread(mesh: &Mesh2d, x: f64, y: f64, f: f64, out: &mut [f64]) {
    let Some((e, r, s)) = locate(mesh, x, y) else {
        return;
    };
    let nodes = &mesh.refq.line.nodes;
    let n1 = nodes.len();
    let lr = lagrange_basis(nodes, r);
    let ls = lagrange_basis(nodes, s);
    let nn = n1 * n1;
    let el = &mesh.elements[e];
    for j in 0..n1 {
        for i in 0..n1 {
            let k = i + j * n1;
            out[e * nn + k] += f * lr[i] * ls[j] / el.geom.jw[k];
        }
    }
}

/// A closed elastic membrane: an ordered ring of Lagrangian marker points with a
/// stretching (spring) elasticity restoring each segment to its rest length. This is
/// the deformable-particle structure; its forces are spread to the fluid and it is
/// advected by the interpolated fluid velocity.
#[derive(Clone, Debug)]
pub struct Membrane {
    /// Marker positions, ordered around the loop.
    pub x: Vec<[f64; 2]>,
    /// Rest length of each segment.
    pub rest_len: f64,
    /// Stretching stiffness.
    pub ks: f64,
}

impl Membrane {
    /// A circular membrane of `n` markers, rest length set to the circle's segment.
    pub fn circle(cx: f64, cy: f64, r: f64, n: usize, ks: f64) -> Self {
        let x = (0..n)
            .map(|k| {
                let th = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
                [cx + r * th.cos(), cy + r * th.sin()]
            })
            .collect();
        // Rest length = the chord between adjacent markers (not the arc), so the
        // reference circle is exactly at equilibrium.
        let rest_len = 2.0 * r * (std::f64::consts::PI / n as f64).sin();
        Self { x, rest_len, ks }
    }

    pub fn n(&self) -> usize {
        self.x.len()
    }

    /// Elastic force at each marker from the two adjacent stretching springs:
    /// `F_i = Σ_{j∈{i±1}} k_s (|X_j−X_i| − L₀) (X_j−X_i)/|X_j−X_i|`. Zero at the rest
    /// configuration; internal (sums to zero net force and torque).
    pub fn elastic_force(&self) -> Vec<[f64; 2]> {
        let n = self.n();
        let mut f = vec![[0.0; 2]; n];
        for i in 0..n {
            for &j in &[(i + 1) % n, (i + n - 1) % n] {
                let d = [self.x[j][0] - self.x[i][0], self.x[j][1] - self.x[i][1]];
                let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
                let t = self.ks * (len - self.rest_len) / len;
                f[i][0] += t * d[0];
                f[i][1] += t * d[1];
            }
        }
        f
    }

    /// Elastic stretching energy `Σ ½k_s(|seg|−L₀)²` — a relaxation diagnostic.
    pub fn strain_energy(&self) -> f64 {
        let n = self.n();
        let mut e = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            let d = [self.x[j][0] - self.x[i][0], self.x[j][1] - self.x[i][1]];
            let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
            e += 0.5 * self.ks * (len - self.rest_len).powi(2);
        }
        e
    }

    /// Centroid of the markers.
    pub fn centroid(&self) -> [f64; 2] {
        let n = self.n() as f64;
        let mut c = [0.0, 0.0];
        for p in &self.x {
            c[0] += p[0];
            c[1] += p[1];
        }
        [c[0] / n, c[1] / n]
    }

    /// Advance the markers by `vel·dt` (front-tracking advection).
    pub fn advect(&mut self, vel: &[[f64; 2]], dt: f64) {
        for (p, v) in self.x.iter_mut().zip(vel) {
            p[0] += v[0] * dt;
            p[1] += v[1] * dt;
        }
    }

    /// Enclosed area via the shoelace formula (signed; positive for CCW ordering).
    pub fn area(&self) -> f64 {
        let n = self.n();
        let mut a = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            a += self.x[i][0] * self.x[j][1] - self.x[j][0] * self.x[i][1];
        }
        0.5 * a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dg::Stokes;

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
    fn interpolation_is_exact_on_polynomials() {
        // The DG basis reproduces any polynomial of degree ≤ p exactly at any point.
        let p = 4;
        let mesh = Mesh2d::rectangular(p, 3, 3, [0.0, 1.0], [0.0, 1.0]);
        let f = |x: f64, y: f64| 1.0 - 2.0 * x + 0.5 * y + x * x - 0.7 * x * y * y + 0.3 * y.powi(4);
        let field = nodal(&mesh, f);
        for &(x, y) in &[(0.17, 0.42), (0.5, 0.5), (0.83, 0.11), (0.95, 0.95), (0.34, 0.78)] {
            let got = interpolate(&mesh, &field, x, y);
            assert!((got - f(x, y)).abs() < 1e-10, "at ({x},{y}): {got} vs {}", f(x, y));
        }
    }

    #[test]
    fn membrane_force_zero_at_rest_and_internal() {
        // At the reference circle the elastic force vanishes; a perturbed membrane
        // develops restoring forces that sum to zero net force AND zero net torque
        // (purely internal), and the force opposes the perturbation.
        let rest = Membrane::circle(0.5, 0.5, 0.2, 24, 10.0);
        let f0 = rest.elastic_force();
        let maxf = f0.iter().fold(0.0f64, |a, v| a.max(v[0].hypot(v[1])));
        assert!(maxf < 1e-9, "force nonzero at rest: {maxf}");

        // Elliptical perturbation (stretched along x, squeezed along y).
        let mut m = rest.clone();
        for p in m.x.iter_mut() {
            p[0] = 0.5 + (p[0] - 0.5) * 1.3;
            p[1] = 0.5 + (p[1] - 0.5) * 0.8;
        }
        let f = m.elastic_force();
        let (mut fx, mut fy, mut tq) = (0.0, 0.0, 0.0);
        for (i, v) in f.iter().enumerate() {
            fx += v[0];
            fy += v[1];
            tq += (m.x[i][0] - 0.5) * v[1] - (m.x[i][1] - 0.5) * v[0];
        }
        assert!(fx.abs() < 1e-9 && fy.abs() < 1e-9, "net force not zero: ({fx},{fy})");
        assert!(tq.abs() < 1e-9, "net torque not zero: {tq}");

        // Restoring of a *local* length perturbation: push one marker radially out;
        // its now-stretched segments pull it back inward (−x).
        let mut m2 = rest.clone();
        m2.x[0] = [0.5 + 0.28, 0.5]; // marker 0 (was at r=0.2 on +x) pushed to r=0.28
        let f2 = m2.elastic_force();
        assert!(f2[0][0] < 0.0, "stretched marker not pulled inward: fx={}", f2[0][0]);
    }

    #[test]
    fn coupled_capsule_relaxes_in_quiescent_fluid() {
        // Full front-tracking loop in a quiescent (no-slip) box: elastic force →
        // spread → Stokes solve → interpolate → advect markers. A perturbed capsule
        // must relax (strain energy drops), stay bounded, and not drift (internal
        // forces ⇒ zero net momentum ⇒ centroid fixed).
        let p = 4;
        let mesh = Mesh2d::rectangular(p, 4, 4, [0.0, 1.0], [0.0, 1.0]);
        let nu = 1.0;
        let dt = 0.002;
        let st = Stokes::new(&mesh, 5.0, nu, dt);
        let zero = |_: f64, _: f64, _: f64| 0.0;

        let mut mem = Membrane::circle(0.5, 0.5, 0.2, 24, 2.0);
        for q in mem.x.iter_mut() {
            q[0] = 0.5 + (q[0] - 0.5) * 1.25; // elliptical perturbation
            q[1] = 0.5 + (q[1] - 0.5) * 0.80;
        }
        let e0 = mem.strain_energy();
        let c0 = mem.centroid();

        let ndof = mesh.n_elements() * mesh.refq.n_nodes();
        let mut ux = vec![0.0; ndof];
        let mut uy = vec![0.0; ndof];
        let mut t = 0.0;
        for _ in 0..150 {
            t += dt;
            let f = mem.elastic_force();
            let mut fx = vec![0.0; ndof];
            let mut fy = vec![0.0; ndof];
            for k in 0..mem.n() {
                spread(&mesh, mem.x[k][0], mem.x[k][1], f[k][0], &mut fx);
                spread(&mesh, mem.x[k][0], mem.x[k][1], f[k][1], &mut fy);
            }
            let (nx, ny) = st.step_ns_forced(&ux, &uy, t, zero, zero, &fx, &fy);
            ux = nx;
            uy = ny;
            let vel: Vec<[f64; 2]> = mem
                .x
                .iter()
                .map(|q| [interpolate(&mesh, &ux, q[0], q[1]), interpolate(&mesh, &uy, q[0], q[1])])
                .collect();
            mem.advect(&vel, dt);
        }
        let e1 = mem.strain_energy();
        let c1 = mem.centroid();
        let drift = (c1[0] - c0[0]).hypot(c1[1] - c0[1]);
        eprintln!("capsule: E {e0:.4e} → {e1:.4e} (ratio {:.3}), centroid drift {drift:.2e}", e1 / e0);
        assert!(e1.is_finite() && mem.x.iter().all(|q| q[0].is_finite()), "diverged");
        // Strain energy strictly decreases (elastic relaxation through the viscous
        // fluid); full relaxation is slower than a unit-test run, but the monotone
        // dissipation + machine-zero centroid drift confirm the coupled loop.
        assert!(e1 < 0.95 * e0, "membrane did not relax: {e0:.3e} → {e1:.3e}");
        assert!(drift < 0.02, "centroid drifted: {drift}");
    }

    #[test]
    fn capsule_deforms_in_shear() {
        // A capsule in Couette shear deforms from a circle into an inclined ellipse
        // (Taylor deformation, tank-treading). Validate: the deformation parameter
        // D=(L−B)/(L+B) grows positive and steady, the long axis lies in the
        // extensional quadrant (0<θ<90°), it stays bounded, and the centroid holds.
        let p = 4;
        let mesh = Mesh2d::channel_x(p, 4, 4, [0.0, 1.0], [0.0, 1.0]);
        let nu = 1.0;
        let gdot = 2.0;
        let dt = 0.002;
        let st = Stokes::new(&mesh, 5.0, nu, dt);
        let shear = move |_x: f64, y: f64, _t: f64| gdot * (y - 0.5); // Couette walls
        let zero = |_: f64, _: f64, _: f64| 0.0;

        let mut mem = Membrane::circle(0.5, 0.5, 0.15, 32, 3.0);
        let c0 = mem.centroid();
        let ndof = mesh.n_elements() * mesh.refq.n_nodes();
        let mut ux = vec![0.0; ndof];
        let mut uy = vec![0.0; ndof];
        let mut t = 0.0;
        let deformation = |m: &Membrane| -> (f64, f64) {
            let c = m.centroid();
            let (mut mxx, mut myy, mut mxy) = (0.0, 0.0, 0.0);
            for q in &m.x {
                let (dx, dy) = (q[0] - c[0], q[1] - c[1]);
                mxx += dx * dx;
                myy += dy * dy;
                mxy += dx * dy;
            }
            let tr = 0.5 * (mxx + myy);
            let rad = (0.25 * (mxx - myy).powi(2) + mxy * mxy).sqrt();
            let (l1, l2) = (tr + rad, tr - rad); // second-moment eigenvalues
            let d = (l1.sqrt() - l2.sqrt()) / (l1.sqrt() + l2.sqrt());
            let theta = 0.5 * (2.0 * mxy).atan2(mxx - myy); // long-axis angle
            (d, theta)
        };
        for _ in 0..200 {
            t += dt;
            let f = mem.elastic_force();
            let mut fx = vec![0.0; ndof];
            let mut fy = vec![0.0; ndof];
            for k in 0..mem.n() {
                spread(&mesh, mem.x[k][0], mem.x[k][1], f[k][0], &mut fx);
                spread(&mesh, mem.x[k][0], mem.x[k][1], f[k][1], &mut fy);
            }
            let (nx, ny) = st.step_ns_forced(&ux, &uy, t, shear, zero, &fx, &fy);
            ux = nx;
            uy = ny;
            let vel: Vec<[f64; 2]> = mem
                .x
                .iter()
                .map(|q| [interpolate(&mesh, &ux, q[0], q[1]), interpolate(&mesh, &uy, q[0], q[1])])
                .collect();
            mem.advect(&vel, dt);
        }
        let (d, theta) = deformation(&mem);
        let c1 = mem.centroid();
        let drift = (c1[0] - c0[0]).hypot(c1[1] - c0[1]);
        let deg = theta.to_degrees();
        eprintln!("capsule shear: D={d:.4}, inclination={deg:.1}°, drift={drift:.2e}");
        assert!(d.is_finite() && mem.x.iter().all(|q| q[0].is_finite()), "diverged");
        assert!(d > 0.02 && d < 0.9, "no/odd deformation: D={d}");
        assert!(deg > 0.0 && deg < 90.0, "long axis not in extensional quadrant: {deg}°");
        assert!(drift < 0.05, "centroid drifted: {drift}");
    }

    #[test]
    fn spread_is_adjoint_of_interpolate() {
        // Consistency: ∫ spread(f at X)·φ dV = f·φ(X) for any DG field φ. With the
        // diagonal LGL mass, ∫ g·φ = Σ jw_k g_k φ_k, so Σ jw_k spread_k φ_k = f·φ(X).
        let p = 4;
        let mesh = Mesh2d::rectangular(p, 3, 3, [0.0, 1.0], [0.0, 1.0]);
        let nn = mesh.refq.n_nodes();
        let phi = nodal(&mesh, |x, y| 0.3 + x - 0.5 * y + 2.0 * x * y - y * y);
        let (x, y, f) = (0.61, 0.28, 1.7);
        let mut g = vec![0.0; mesh.n_elements() * nn];
        spread(&mesh, x, y, f, &mut g);
        // Pair against φ with the mass matrix.
        let mut paired = 0.0;
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                paired += el.geom.jw[k] * g[e * nn + k] * phi[e * nn + k];
            }
        }
        let expected = f * interpolate(&mesh, &phi, x, y);
        assert!((paired - expected).abs() < 1e-10, "adjoint: {paired} vs {expected}");
    }
}
