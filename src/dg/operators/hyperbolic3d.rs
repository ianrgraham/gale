//! Hyperbolic conservation laws on the 3D hex mesh — the 3D analogue of
//! [`Hyperbolic`](super::hyperbolic::Hyperbolic). Weak-form DG-SEM with a Rusanov
//! (local Lax–Friedrichs) interface flux over the 6 hex faces. Conforming meshes
//! only for now (3D octree/mortar AMR is a later step).
//!
//! `∂ₜu = M⁻¹[ Dxᵀ(W Fx) + Dyᵀ(W Fy) + Dzᵀ(W Fz) − ∮ F*·n dS ]`, mass diagonal.

use super::face3d::Face;
use super::mesh3d::{Mesh3d, Neighbor3};

/// A 3D hyperbolic conservation law: physical flux + interface wave speed.
pub trait ConservationLaw3d {
    fn n_vars(&self) -> usize;
    /// Physical flux at a state `u`: writes `fx`, `fy`, `fz` (each length `n_vars`).
    fn flux(&self, u: &[f64], fx: &mut [f64], fy: &mut [f64], fz: &mut [f64]);
    /// Maximum signal speed in direction `(nx, ny, nz)` at state `u` (for Rusanov).
    fn max_wave_speed(&self, u: &[f64], nx: f64, ny: f64, nz: f64) -> f64;
}

/// Constant-coefficient linear advection `∂ₜu + a·∇u = 0` in 3D.
pub struct LinearAdvection3d {
    pub ax: f64,
    pub ay: f64,
    pub az: f64,
}

impl ConservationLaw3d for LinearAdvection3d {
    fn n_vars(&self) -> usize {
        1
    }
    fn flux(&self, u: &[f64], fx: &mut [f64], fy: &mut [f64], fz: &mut [f64]) {
        fx[0] = self.ax * u[0];
        fy[0] = self.ay * u[0];
        fz[0] = self.az * u[0];
    }
    fn max_wave_speed(&self, _u: &[f64], nx: f64, ny: f64, nz: f64) -> f64 {
        (self.ax * nx + self.ay * ny + self.az * nz).abs()
    }
}

/// Weak-form DG-SEM hyperbolic operator on a hex mesh.
pub struct Hyperbolic3d<'m, L: ConservationLaw3d> {
    pub mesh: &'m Mesh3d,
    pub law: L,
    /// Whether the interface flux carries Rusanov dissipation.
    pub dissipation: bool,
}

impl<'m, L: ConservationLaw3d> Hyperbolic3d<'m, L> {
    pub fn new(mesh: &'m Mesh3d, law: L) -> Self {
        Self { mesh, law, dissipation: true }
    }

    pub fn n_vars(&self) -> usize {
        self.law.n_vars()
    }
    pub fn ndof(&self) -> usize {
        self.mesh.n_elements() * self.mesh.refh.n_nodes()
    }

    fn zeros(&self) -> Vec<Vec<f64>> {
        vec![vec![0.0; self.ndof()]; self.n_vars()]
    }

    /// Rusanov (LLF) numerical flux `F*·n` per variable into `out`, outward normal
    /// `(nx, ny, nz)`.
    fn rusanov(&self, um: &[f64], up: &[f64], nx: f64, ny: f64, nz: f64, out: &mut [f64]) {
        let nv = self.n_vars();
        let (mut fxm, mut fym, mut fzm) = (vec![0.0; nv], vec![0.0; nv], vec![0.0; nv]);
        let (mut fxp, mut fyp, mut fzp) = (vec![0.0; nv], vec![0.0; nv], vec![0.0; nv]);
        self.law.flux(um, &mut fxm, &mut fym, &mut fzm);
        self.law.flux(up, &mut fxp, &mut fyp, &mut fzp);
        let lam = self
            .law
            .max_wave_speed(um, nx, ny, nz)
            .max(self.law.max_wave_speed(up, nx, ny, nz));
        let diss = if self.dissipation { lam } else { 0.0 };
        for v in 0..nv {
            let fnm = fxm[v] * nx + fym[v] * ny + fzm[v] * nz;
            let fnp = fxp[v] * nx + fyp[v] * ny + fzp[v] * nz;
            out[v] = 0.5 * (fnm + fnp) - 0.5 * diss * (up[v] - um[v]);
        }
    }

    /// Semi-discrete RHS `∂ₜu = L(u)`. `bc(x, y, z, t, out)` fills the exterior state
    /// at a boundary node.
    pub fn rhs(
        &self,
        state: &[Vec<f64>],
        t: f64,
        bc: &impl Fn(f64, f64, f64, f64, &mut [f64]),
    ) -> Vec<Vec<f64>> {
        let mesh = self.mesh;
        let refh = &mesh.refh;
        let nn = refh.n_nodes();
        let nv = self.n_vars();
        let ndof = self.ndof();

        // Physical fluxes at every node.
        let mut fx = vec![vec![0.0; ndof]; nv];
        let mut fy = vec![vec![0.0; ndof]; nv];
        let mut fz = vec![vec![0.0; ndof]; nv];
        {
            let (mut u, mut a, mut b, mut c) =
                (vec![0.0; nv], vec![0.0; nv], vec![0.0; nv], vec![0.0; nv]);
            for i in 0..ndof {
                for v in 0..nv {
                    u[v] = state[v][i];
                }
                self.law.flux(&u, &mut a, &mut b, &mut c);
                for v in 0..nv {
                    fx[v][i] = a[v];
                    fy[v][i] = b[v];
                    fz[v][i] = c[v];
                }
            }
        }

        // Volume term: Dxᵀ(W Fx) + Dyᵀ(W Fy) + Dzᵀ(W Fz).
        let mut res = self.zeros();
        for v in 0..nv {
            for (e, el) in mesh.elements.iter().enumerate() {
                let wfx: Vec<f64> = (0..nn).map(|k| el.geom.jw[k] * fx[v][e * nn + k]).collect();
                let wfy: Vec<f64> = (0..nn).map(|k| el.geom.jw[k] * fy[v][e * nn + k]).collect();
                let wfz: Vec<f64> = (0..nn).map(|k| el.geom.jw[k] * fz[v][e * nn + k]).collect();
                let a = el.geom.gradx_t(refh, &wfx);
                let b = el.geom.grady_t(refh, &wfy);
                let c = el.geom.gradz_t(refh, &wfz);
                for k in 0..nn {
                    res[v][e * nn + k] += a[k] + b[k] + c[k];
                }
            }
        }

        // Interface term: − ∮ F*·n, Rusanov over the 6 faces (conforming).
        {
            let (mut um, mut up, mut fstar) = (vec![0.0; nv], vec![0.0; nv], vec![0.0; nv]);
            for e in 0..mesh.elements.len() {
                for face in Face::ALL {
                    let fc = &mesh.elements[e].faces[face as usize];
                    match mesh.elements[e].neighbors[face as usize].clone() {
                        Neighbor3::Interior { elem: re, face: rface, perm } => {
                            let rf = &mesh.elements[re].faces[rface as usize];
                            for ai in 0..fc.nodes.len() {
                                let vl = fc.nodes[ai];
                                let (nx, ny, nz, sw) = (fc.nx[ai], fc.ny[ai], fc.nz[ai], fc.sw[ai]);
                                let rnode = rf.nodes[perm[ai]];
                                for v in 0..nv {
                                    um[v] = state[v][e * nn + vl];
                                    up[v] = state[v][re * nn + rnode];
                                }
                                self.rusanov(&um, &up, nx, ny, nz, &mut fstar);
                                for v in 0..nv {
                                    res[v][e * nn + vl] -= sw * fstar[v];
                                }
                            }
                        }
                        Neighbor3::Boundary { .. } => {
                            let g = &mesh.elements[e].geom;
                            for ai in 0..fc.nodes.len() {
                                let vl = fc.nodes[ai];
                                let (nx, ny, nz, sw) = (fc.nx[ai], fc.ny[ai], fc.nz[ai], fc.sw[ai]);
                                for v in 0..nv {
                                    um[v] = state[v][e * nn + vl];
                                }
                                bc(g.x[vl], g.y[vl], g.z[vl], t, &mut up);
                                self.rusanov(&um, &up, nx, ny, nz, &mut fstar);
                                for v in 0..nv {
                                    res[v][e * nn + vl] -= sw * fstar[v];
                                }
                            }
                        }
                    }
                }
            }
        }

        // ∂ₜu = M⁻¹ res.
        let mut dudt = self.zeros();
        for v in 0..nv {
            for (e, el) in mesh.elements.iter().enumerate() {
                for k in 0..nn {
                    dudt[v][e * nn + k] = res[v][e * nn + k] / el.geom.jw[k];
                }
            }
        }
        dudt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::integrate::{ClosureSemi, Integrator, SspRk3};
    use std::f64::consts::PI;

    fn periodic_field(mesh: &Mesh3d, f: impl Fn(f64, f64, f64) -> f64) -> Vec<f64> {
        let nn = mesh.refh.n_nodes();
        let mut u = vec![0.0; mesh.n_elements() * nn];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                u[e * nn + k] = f(el.geom.x[k], el.geom.y[k], el.geom.z[k]);
            }
        }
        u
    }

    const NOBC: fn(f64, f64, f64, f64, &mut [f64]) = |_x, _y, _z, _t, _o| {};

    #[test]
    fn free_stream_preserved() {
        // A uniform state has zero residual to round-off (flux consistency + GCL).
        let mesh = Mesh3d::rectangular_periodic(4, 3, 3, 3, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
        let op = Hyperbolic3d::new(&mesh, LinearAdvection3d { ax: 0.8, ay: -0.5, az: 0.3 });
        let u = vec![3.7; op.ndof()];
        let r = op.rhs(&[u], 0.0, &NOBC);
        let m = r[0].iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        assert!(m < 1e-11, "free-stream residual {m}");
    }

    #[test]
    fn conserves_integral_on_periodic() {
        // ∑ jw·(∂ₜu) = 0: linear advection conserves ∫u on a periodic domain.
        let mesh = Mesh3d::rectangular_periodic(4, 3, 3, 3, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
        let op = Hyperbolic3d::new(&mesh, LinearAdvection3d { ax: 1.0, ay: 0.7, az: -0.4 });
        let nn = mesh.refh.n_nodes();
        let u = periodic_field(&mesh, |x, y, z| {
            (2.0 * PI * x).sin() * (2.0 * PI * y).cos() * (2.0 * PI * z).sin() + 0.3
        });
        let r = op.rhs(&[u], 0.0, &NOBC);
        let mut integral = 0.0;
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                integral += el.geom.jw[k] * r[0][e * nn + k];
            }
        }
        assert!(integral.abs() < 1e-10, "∫∂ₜu = {integral}");
    }

    #[test]
    fn rhs_matches_analytic_advection() {
        // For a smooth periodic field, the DG rhs approximates −a·∇u spectrally.
        let (ax, ay, az) = (1.0, 0.7, -0.4);
        let mesh = Mesh3d::rectangular_periodic(5, 4, 4, 4, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
        let op = Hyperbolic3d::new(&mesh, LinearAdvection3d { ax, ay, az });
        let nn = mesh.refh.n_nodes();
        let u = periodic_field(&mesh, |x, y, z| {
            (2.0 * PI * x).sin() * (2.0 * PI * y).sin() * (2.0 * PI * z).sin()
        });
        let r = op.rhs(&[u], 0.0, &NOBC);
        let mut err = 0.0f64;
        let mut scale = 0.0f64;
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let (x, y, z) = (el.geom.x[k], el.geom.y[k], el.geom.z[k]);
                let (sx, cx) = ((2.0 * PI * x).sin(), (2.0 * PI * x).cos());
                let (sy, cy) = ((2.0 * PI * y).sin(), (2.0 * PI * y).cos());
                let (sz, cz) = ((2.0 * PI * z).sin(), (2.0 * PI * z).cos());
                let ux = 2.0 * PI * cx * sy * sz;
                let uy = 2.0 * PI * sx * cy * sz;
                let uz = 2.0 * PI * sx * sy * cz;
                let exact = -(ax * ux + ay * uy + az * uz);
                err = err.max((r[0][e * nn + k] - exact).abs());
                scale = scale.max(exact.abs());
            }
        }
        assert!(err / scale < 1e-2, "rhs vs analytic rel err {}", err / scale);
    }

    #[test]
    fn advects_smooth_field_via_ssprk3() {
        // Integrate ∂ₜu + a·∇u = 0; exact solution is the translated initial field.
        // Uses the (dimension-agnostic) vector-level SspRk3 from the framework.
        let (ax, ay, az) = (1.0, 0.5, 0.25);
        let mesh = Mesh3d::rectangular_periodic(4, 4, 4, 4, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
        let op = Hyperbolic3d::new(&mesh, LinearAdvection3d { ax, ay, az });
        let nn = mesh.refh.n_nodes();
        let u0 = |x: f64, y: f64, z: f64, t: f64| {
            (2.0 * PI * (x - ax * t)).sin()
                * (2.0 * PI * (y - ay * t)).sin()
                * (2.0 * PI * (z - az * t)).sin()
        };
        let mut u = periodic_field(&mesh, |x, y, z| u0(x, y, z, 0.0));

        let semi = ClosureSemi::new(1, op.ndof(), |s: &[Vec<f64>], t: f64| op.rhs(s, t, &NOBC));
        let integ = SspRk3::new(2e-3);
        let nsteps = 50;
        let mut state = vec![u.clone()];
        let mut t = 0.0;
        for _ in 0..nsteps {
            state = integ.step(&semi, &state, t);
            t += integ.dt();
        }
        u = state.pop().unwrap();

        let tf = integ.dt() * nsteps as f64;
        let mut e2 = 0.0;
        let mut n2 = 0.0;
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let ue = u0(el.geom.x[k], el.geom.y[k], el.geom.z[k], tf);
                let d = u[e * nn + k] - ue;
                e2 += el.geom.jw[k] * d * d;
                n2 += el.geom.jw[k] * ue * ue;
            }
        }
        let rel = (e2 / n2).sqrt();
        assert!(rel < 1e-2, "3D advection translation rel L2 err {rel:e}");
    }
}
