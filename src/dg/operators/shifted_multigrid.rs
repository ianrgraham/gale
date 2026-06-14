//! **p-multigrid preconditioner for the Shifted Boundary Method (SBM) operator**
//! ([`ShiftedPoisson`]). The standard full-mesh [`PMultigrid`](super::multigrid::PMultigrid) is
//! a poor preconditioner for the SBM pressure-Poisson: it ignores the active mask and the
//! surrogate boundary, so it stagnates on the *cylinder-local* modes (`docs/sbm-status.md`,
//! step 2c). This hierarchy uses the **SBM operator itself** at every level, so the smoother and
//! coarse solve see the embedded boundary and damp those local modes.
//!
//! **p-only coarsening** (orders `p, p/2, …, 1` on the same mesh): the active-element mask is
//! element-based and therefore identical across p-levels — only the surrogate shift vectors `d`
//! (at the face nodes) change with order, so a [`ShiftedBoundary`] is rebuilt per level from the
//! level set. Transfers are the same tensor-Lagrange p-restriction/prolongation as `PMultigrid`;
//! smoother is damped Jacobi over a 2-colour-probed diagonal; the coarsest (p=1, full grid)
//! solve uses the loose-tol/cap band-aid (it's only a preconditioner component).

use super::mesh::Mesh2d;
use super::multigrid::{lagrange_matrix, order_levels};
use super::poisson::ShiftedPoisson;
use crate::dg::shifted::{LevelSet, ShiftedBoundary};
use rayon::prelude::*;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// p-multigrid for the SBM operator. Drive it as a preconditioner via [`precondition`]
/// (one V-cycle), e.g. `pres.solve_pcg_from(b, x0, |r| smg.precondition(r), tol, maxit)`, or
/// stand-alone via [`pcg`]/[`pcg_deflated`].
pub struct ShiftedMultigrid {
    orders: Vec<usize>,
    meshes: Vec<Mesh2d>,
    /// Surrogate boundary per level (same geometry, rebuilt at each order's nodes).
    sbs: Vec<ShiftedBoundary>,
    /// 1D Lagrange p-transfer matrix per transition (`fine × coarse`), `len = n_levels − 1`.
    interps: Vec<Vec<f64>>,
    alpha: f64,
    reaction: f64,
    neumann_tags: Vec<u32>,
    taylor: bool,
    surrogate_dirichlet: bool,
    /// Constant-nullspace (pure-Neumann everywhere incl. surrogate) ⇒ deflate the coarse solve.
    singular: bool,
    inv_diag: Vec<Vec<f64>>,
    lam_hi: Vec<f64>,
    n_pre: usize,
    n_post: usize,
    nx: usize,
    ny: usize,
    xr: [f64; 2],
    yr: [f64; 2],
}

impl ShiftedMultigrid {
    /// Build the SBM p-multigrid over the surrogate domain defined by `ls`, matching the
    /// configuration of the finest-level [`ShiftedPoisson`] you intend to solve:
    /// `reaction` (0 ⇒ pressure-Poisson), `neumann_tags` (outer natural BCs), `taylor`
    /// (high-order surrogate Nitsche), and `surrogate_dirichlet` (`false` ⇒ natural-Neumann
    /// surrogate — the pressure projection).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        order: usize, nx: usize, ny: usize, xr: [f64; 2], yr: [f64; 2], alpha: f64, reaction: f64,
        neumann_tags: Vec<u32>, ls: &impl LevelSet, taylor: bool, surrogate_dirichlet: bool,
    ) -> Self {
        Self::build(order, nx, ny, xr, yr, alpha, reaction, neumann_tags, ls, taylor, surrogate_dirichlet, None)
    }

    /// Like [`new`](Self::new) but **reuses a cached smoother** (`inv_diag`, `lam_hi`) instead of
    /// recomputing it (the expensive part of setup: colored-diagonal probing + power iteration).
    /// For a MOVING body whose active-element mask is unchanged step-to-step — the geometry (meshes,
    /// surrogate, shift vectors) is rebuilt fresh so the *operator* is current, while the smoother
    /// (only a preconditioner component) is amortized from the last mask change. Caller passes the
    /// values from [`smoother_data`](Self::smoother_data) of the previous build; they must match the
    /// level structure (same `order`/`nx`/`ny`). Recompute (call `new`) when the mask changes.
    #[allow(clippy::too_many_arguments)]
    pub fn new_reusing_smoother(
        order: usize, nx: usize, ny: usize, xr: [f64; 2], yr: [f64; 2], alpha: f64, reaction: f64,
        neumann_tags: Vec<u32>, ls: &impl LevelSet, taylor: bool, surrogate_dirichlet: bool,
        smoother: (Vec<Vec<f64>>, Vec<f64>),
    ) -> Self {
        Self::build(
            order, nx, ny, xr, yr, alpha, reaction, neumann_tags, ls, taylor, surrogate_dirichlet, Some(smoother),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        order: usize, nx: usize, ny: usize, xr: [f64; 2], yr: [f64; 2], alpha: f64, reaction: f64,
        neumann_tags: Vec<u32>, ls: &impl LevelSet, taylor: bool, surrogate_dirichlet: bool,
        smoother: Option<(Vec<Vec<f64>>, Vec<f64>)>,
    ) -> Self {
        let orders = order_levels(order);
        let meshes: Vec<Mesh2d> = orders.iter().map(|&o| Mesh2d::rectangular(o, nx, ny, xr, yr)).collect();
        let sbs: Vec<ShiftedBoundary> = meshes.iter().map(|m| ShiftedBoundary::new(m, ls)).collect();
        // p-transfer matrices (coarse order → fine order LGL nodes), one per consecutive pair.
        let interps: Vec<Vec<f64>> = (0..orders.len() - 1)
            .map(|l| lagrange_matrix(&meshes[l + 1].refq.line.nodes, &meshes[l].refq.line.nodes))
            .collect();
        // Singular when reaction 0, surrogate natural, AND every outer boundary natural.
        let singular = reaction == 0.0
            && !surrogate_dirichlet
            && meshes[0].boundary_tags().iter().all(|t| neumann_tags.contains(t));
        let mut s = Self {
            orders,
            meshes,
            sbs,
            interps,
            alpha,
            reaction,
            neumann_tags,
            taylor,
            surrogate_dirichlet,
            singular,
            inv_diag: Vec::new(),
            lam_hi: Vec::new(),
            n_pre: 3,
            n_post: 3,
            nx,
            ny,
            xr,
            yr,
        };
        match smoother {
            Some((inv_diag, lam_hi)) => {
                // Amortized: reuse the cached smoother (geometry above is still current).
                s.inv_diag = inv_diag;
                s.lam_hi = lam_hi;
            }
            None => {
                // Full setup: colored-diagonal probing + power iteration (the expensive part).
                s.inv_diag = (0..s.orders.len())
                    .into_par_iter()
                    .map(|l| s.diagonal(l).iter().map(|&v| 1.0 / v).collect())
                    .collect();
                s.lam_hi = (0..s.orders.len()).map(|l| 1.1 * s.power_lambda(l)).collect();
            }
        }
        s
    }

    /// The cached smoother data (`inv_diag` per level, `lam_hi` per level) — pass to
    /// [`new_reusing_smoother`](Self::new_reusing_smoother) to amortize a moving body's per-step setup.
    pub fn smoother_data(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        (self.inv_diag.clone(), self.lam_hi.clone())
    }

    /// Number of multigrid levels (finest … coarsest).
    pub fn n_levels(&self) -> usize {
        self.orders.len()
    }
    fn ndof(&self, l: usize) -> usize {
        self.meshes[l].n_elements() * self.meshes[l].refq.n_nodes()
    }

    // --- accessors for an external (GPU) driver that reuses this setup (mirrors PMultigrid) ---

    /// Pre/post smoother sweep counts.
    pub fn smoothing(&self) -> (usize, usize) {
        (self.n_pre, self.n_post)
    }
    /// Mesh at level `l`.
    pub fn mesh(&self, l: usize) -> &Mesh2d {
        &self.meshes[l]
    }
    /// SIPG penalty scale.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
    /// Helmholtz reaction `λ` baked into the hierarchy.
    pub fn reaction(&self) -> f64 {
        self.reaction
    }
    /// Outer natural (Neumann) boundary tags.
    pub fn neumann_tags(&self) -> &[u32] {
        &self.neumann_tags
    }
    /// Active-element mask (surrogate-fluid domain) — element-based, identical across p-levels.
    pub fn active(&self) -> &[bool] {
        &self.sbs[0].active
    }
    /// Inverse operator diagonal (damped-Jacobi denominator) at level `l`.
    pub fn inv_diagonal(&self, l: usize) -> &[f64] {
        &self.inv_diag[l]
    }
    /// Damped-Jacobi weight `ω = (4/3)/λ_max(D⁻¹A)` at level `l`.
    pub fn jacobi_omega(&self, l: usize) -> f64 {
        (4.0 / 3.0) / self.lam_hi[l]
    }
    /// 1D p-transfer (Lagrange) matrix for the `l → l+1` transition.
    pub fn interp_matrix(&self, l: usize) -> &[f64] {
        &self.interps[l]
    }
    /// Whether the operator is singular (constant nullspace) ⇒ deflate.
    pub fn is_singular(&self) -> bool {
        self.singular
    }
    /// Surrogate boundary (active mask + surrogate faces + shift vectors) at level `l`.
    pub fn shifted_boundary(&self, l: usize) -> &ShiftedBoundary {
        &self.sbs[l]
    }
    /// Surrogate BC type: `true` ⇒ Dirichlet/Nitsche (velocity no-slip), `false` ⇒ natural-Neumann.
    pub fn surrogate_dirichlet(&self) -> bool {
        self.surrogate_dirichlet
    }
    /// Whether the high-order Taylor surrogate correction is enabled.
    pub fn taylor(&self) -> bool {
        self.taylor
    }

    /// Build the SBM operator for level `l` (rebuilt per call, like `PMultigrid::apply_level`).
    fn op(&self, l: usize) -> ShiftedPoisson<'_> {
        let mut o = ShiftedPoisson::with_bc(
            &self.meshes[l], self.alpha, self.reaction, self.neumann_tags.clone(), self.sbs[l].clone(),
        )
        .taylor(self.taylor);
        if !self.surrogate_dirichlet {
            o = o.surrogate_neumann();
        }
        o
    }

    fn apply_level(&self, l: usize, u: &[f64]) -> Vec<f64> {
        self.op(l).apply(u)
    }

    /// Finest-level operator action (for stand-alone PCG).
    pub fn apply(&self, u: &[f64]) -> Vec<f64> {
        self.apply_level(0, u)
    }

    /// Checkerboard parity of element `e` from its centroid (same as `PMultigrid`): SIPG +
    /// the element-local surrogate Nitsche couple only face-neighbours / self, so same-parity
    /// elements never couple ⇒ 2-colour probing recovers the exact diagonal.
    fn elem_color(&self, l: usize, e: usize) -> usize {
        let g = &self.meshes[l].elements[e].geom;
        let nn = self.meshes[l].refq.n_nodes();
        let (mut xc, mut yc) = (0.0, 0.0);
        for k in 0..nn {
            xc += g.x[k];
            yc += g.y[k];
        }
        xc /= nn as f64;
        yc /= nn as f64;
        let cx = (((xc - self.xr[0]) / ((self.xr[1] - self.xr[0]) / self.nx as f64)) as usize).min(self.nx - 1);
        let cy = (((yc - self.yr[0]) / ((self.yr[1] - self.yr[0]) / self.ny as f64)) as usize).min(self.ny - 1);
        (cx + cy) % 2
    }

    /// Operator diagonal via 2-colour probing (O(n·nn)). Inactive dofs read back 1.0 (the SBM
    /// identity block), which is exactly their diagonal.
    fn diagonal(&self, l: usize) -> Vec<f64> {
        let mesh = &self.meshes[l];
        let nn = mesh.refq.n_nodes();
        let ne = mesh.n_elements();
        let n = ne * nn;
        let colors: Vec<usize> = (0..ne).map(|e| self.elem_color(l, e)).collect();
        let mut diag = vec![0.0; n];
        let probes: Vec<(usize, usize)> = (0..2).flat_map(|c| (0..nn).map(move |m| (c, m))).collect();
        let parts: Vec<Vec<(usize, f64)>> = probes
            .par_iter()
            .map(|&(c, m)| {
                let mut e = vec![0.0; n];
                for el in 0..ne {
                    if colors[el] == c {
                        e[el * nn + m] = 1.0;
                    }
                }
                let ae = self.apply_level(l, &e);
                (0..ne).filter(|&el| colors[el] == c).map(|el| (el * nn + m, ae[el * nn + m])).collect()
            })
            .collect();
        for part in parts {
            for (i, v) in part {
                diag[i] = v;
            }
        }
        // Guard against a zero/negative diagonal (shouldn't happen for SPD, but keep Jacobi sane).
        diag.iter_mut().for_each(|d| {
            if !(*d > 0.0) {
                *d = 1.0;
            }
        });
        diag
    }

    fn power_lambda(&self, l: usize) -> f64 {
        let n = self.ndof(l);
        let mut v: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64 * 0.13).collect();
        let nv = norm(&v);
        v.iter_mut().for_each(|x| *x /= nv);
        let mut lam = 1.0;
        for _ in 0..60 {
            let av = self.apply_level(l, &v);
            let mut w: Vec<f64> = (0..n).map(|i| self.inv_diag[l][i] * av[i]).collect();
            lam = norm(&w).max(1e-300);
            w.iter_mut().for_each(|x| *x /= lam);
            v = w;
        }
        lam
    }

    fn smooth(&self, l: usize, x: &mut [f64], b: &[f64], sweeps: usize) {
        let omega = (4.0 / 3.0) / self.lam_hi[l];
        let id = &self.inv_diag[l];
        for _ in 0..sweeps {
            let ax = self.apply_level(l, x);
            for i in 0..x.len() {
                x[i] += omega * id[i] * (b[i] - ax[i]);
            }
        }
    }

    fn prolong(&self, l: usize, coarse: &[f64]) -> Vec<f64> {
        let ncc = self.orders[l + 1] + 1;
        let nff = self.orders[l] + 1;
        let ne = self.meshes[l].n_elements();
        let i = &self.interps[l];
        let (cc, ff) = (ncc * ncc, nff * nff);
        let mut out = vec![0.0; ne * ff];
        let mut tmp = vec![0.0; nff * ncc];
        for e in 0..ne {
            let c = &coarse[e * cc..(e + 1) * cc];
            for jc in 0..ncc {
                for iff in 0..nff {
                    let mut s = 0.0;
                    for ic in 0..ncc {
                        s += i[iff * ncc + ic] * c[ic + jc * ncc];
                    }
                    tmp[iff + jc * nff] = s;
                }
            }
            for jf in 0..nff {
                for iff in 0..nff {
                    let mut s = 0.0;
                    for jc in 0..ncc {
                        s += i[jf * ncc + jc] * tmp[iff + jc * nff];
                    }
                    out[e * ff + iff + jf * nff] = s;
                }
            }
        }
        out
    }

    fn restrict(&self, l: usize, fine: &[f64]) -> Vec<f64> {
        let ncc = self.orders[l + 1] + 1;
        let nff = self.orders[l] + 1;
        let ne = self.meshes[l].n_elements();
        let i = &self.interps[l];
        let (cc, ff) = (ncc * ncc, nff * nff);
        let mut out = vec![0.0; ne * cc];
        let mut tmp = vec![0.0; ncc * nff];
        for e in 0..ne {
            let f = &fine[e * ff..(e + 1) * ff];
            for jf in 0..nff {
                for ic in 0..ncc {
                    let mut s = 0.0;
                    for iff in 0..nff {
                        s += i[iff * ncc + ic] * f[iff + jf * nff];
                    }
                    tmp[ic + jf * ncc] = s;
                }
            }
            for jc in 0..ncc {
                for ic in 0..ncc {
                    let mut s = 0.0;
                    for jf in 0..nff {
                        s += i[jf * ncc + jc] * tmp[ic + jf * ncc];
                    }
                    out[e * cc + ic + jc * ncc] = s;
                }
            }
        }
        out
    }

    fn coarse_solve(&self, l: usize, b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let deflate = |v: &mut [f64]| {
            if self.singular {
                let mean = v.iter().sum::<f64>() / n as f64;
                v.iter_mut().for_each(|x| *x -= mean);
            }
        };
        // p-only coarsening leaves a p=1-on-the-full-grid coarse level — large and (for the
        // pressure) ill-conditioned. It's only a preconditioner component, so use a loose
        // tol/cap for a big coarse grid (mirrors PMultigrid / the GPU band-aid).
        let coarse_small = n <= 1024;
        let (ctol, cap) = if coarse_small { (1e-10, 500) } else { (1e-2, 40) };
        let mut x = vec![0.0; n];
        let mut r = b.to_vec();
        deflate(&mut r);
        let mut p = r.clone();
        let mut rs = dot(&r, &r);
        let bn = norm(b).max(1e-300);
        for _ in 0..cap {
            let ap = self.apply_level(l, &p);
            let pap = dot(&p, &ap);
            if !(pap.abs() > 0.0) {
                break;
            }
            let a = rs / pap;
            for i in 0..n {
                x[i] += a * p[i];
                r[i] -= a * ap[i];
            }
            deflate(&mut r);
            let rsn = dot(&r, &r);
            if rsn.sqrt() / bn < ctol {
                break;
            }
            let be = rsn / rs;
            for i in 0..n {
                p[i] = r[i] + be * p[i];
            }
            rs = rsn;
        }
        x
    }

    fn vcycle(&self, l: usize, b: &[f64]) -> Vec<f64> {
        if l == self.n_levels() - 1 {
            return self.coarse_solve(l, b);
        }
        let mut x = vec![0.0; b.len()];
        self.smooth(l, &mut x, b, self.n_pre);
        let ax = self.apply_level(l, &x);
        let res: Vec<f64> = (0..b.len()).map(|i| b[i] - ax[i]).collect();
        let rc = self.restrict(l, &res);
        let ec = self.vcycle(l + 1, &rc);
        let pe = self.prolong(l, &ec);
        for i in 0..x.len() {
            x[i] += pe[i];
        }
        self.smooth(l, &mut x, b, self.n_post);
        x
    }

    /// One V-cycle as the preconditioner `z = M⁻¹ r`. Pass `|r| smg.precondition(r)` to
    /// [`ShiftedPoisson::solve_pcg_from`](super::poisson::ShiftedPoisson::solve_pcg_from).
    pub fn precondition(&self, r: &[f64]) -> Vec<f64> {
        self.vcycle(0, r)
    }

    /// Stand-alone preconditioned CG (non-singular SBM operator). Returns `(solution, iters)`.
    pub fn pcg(&self, b: &[f64], tol: f64, maxit: usize) -> (Vec<f64>, usize) {
        let n = b.len();
        let bn = norm(b);
        if bn < 1e-300 {
            return (vec![0.0; n], 0); // trivial RHS ⇒ zero solution (avoids the 0/0 in α)
        }
        let mut x = vec![0.0; n];
        let mut r = b.to_vec();
        let mut z = self.precondition(&r);
        let mut p = z.clone();
        let mut rz = dot(&r, &z);
        let mut iters = 0;
        for it in 0..maxit {
            let ap = self.apply(&p);
            let alpha = rz / dot(&p, &ap);
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            iters = it + 1;
            if norm(&r) / bn < tol {
                break;
            }
            z = self.precondition(&r);
            let rz_new = dot(&r, &z);
            let beta = rz_new / rz;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
            rz = rz_new;
        }
        (x, iters)
    }

    /// Preconditioned CG for the **singular** SBM operator (closed-box pressure — pure-Neumann
    /// outer walls + natural-Neumann surrogate ⇒ constant nullspace). Deflates the constant from
    /// the residual and the preconditioned residual each iteration; the coarse solve already
    /// deflates internally. Solution is determined up to an additive constant. `(solution, iters)`.
    pub fn pcg_deflated(&self, b: &[f64], tol: f64, maxit: usize) -> (Vec<f64>, usize) {
        let n = b.len();
        let deflate = |v: &mut [f64]| {
            let mean = v.iter().sum::<f64>() / n as f64;
            v.iter_mut().for_each(|x| *x -= mean);
        };
        let mut x = vec![0.0; n];
        let mut r = b.to_vec();
        deflate(&mut r);
        let bn = norm(&r);
        if bn < 1e-300 {
            return (vec![0.0; n], 0); // trivial (range-projected) RHS ⇒ zero solution
        }
        let mut z = self.precondition(&r);
        deflate(&mut z);
        let mut p = z.clone();
        let mut rz = dot(&r, &z);
        let mut iters = 0;
        for it in 0..maxit {
            let ap = self.apply(&p);
            let alpha = rz / dot(&p, &ap);
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            deflate(&mut r);
            iters = it + 1;
            if norm(&r) / bn < tol {
                break;
            }
            z = self.precondition(&r);
            deflate(&mut z);
            let rz_new = dot(&r, &z);
            let beta = rz_new / rz;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
            rz = rz_new;
        }
        (x, iters)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dg::shifted::CircleLevelSet;
    use crate::dg::{Mesh2d, ShiftedBoundary, ShiftedPoisson};

    /// SBM-MG-PCG must (a) reproduce the direct SBM CG solution and (b) need far fewer
    /// iterations, for both the Helmholtz (reaction>0) and the harder pure-Poisson (reaction 0,
    /// the pressure) Dirichlet-surrogate operators. uex = cos(2x)·sin(3y), circle hole.
    fn check(reaction: f64) -> (usize, usize, f64) {
        let (p, n) = (3usize, 8usize);
        let alpha = 10.0;
        let uex = |x: f64, y: f64| (2.0 * x).cos() * (3.0 * y).sin();
        let ls = CircleLevelSet::new(0.5, 0.5, 0.2);
        let mesh = Mesh2d::rectangular(p, n, n, [0.0, 1.0], [0.0, 1.0]);
        let nn = mesh.refq.n_nodes();
        let sb = ShiftedBoundary::new(&mesh, &ls);
        // Direct operator and the matching MG (same config: all-Dirichlet outer, Taylor surrogate).
        let sp = ShiftedPoisson::with_bc(&mesh, alpha, reaction, vec![], sb).taylor(true);
        let smg = ShiftedMultigrid::new(p, n, n, [0.0, 1.0], [0.0, 1.0], alpha, reaction, vec![], &ls, true, true);
        // f = (reaction − ∇²)u = (reaction + 13) u_exact.
        let mut f = vec![0.0; mesh.n_elements() * nn];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                f[e * nn + k] = (reaction + 13.0) * uex(el.geom.x[k], el.geom.y[k]);
            }
        }
        let b = sp.rhs(&f, uex);
        let (u_direct, it_direct) = sp.solve(&b, 1e-9, 50000);
        let (u_mg, it_mg) = smg.pcg(&b, 1e-9, 2000);
        // Same solution (compare on active dofs — inactive are identically 0 in both).
        let active = sp.active();
        let (mut num, mut den) = (0.0, 0.0);
        for e in 0..mesh.n_elements() {
            if !active[e] {
                continue;
            }
            for k in 0..nn {
                let d = u_mg[e * nn + k] - u_direct[e * nn + k];
                num += d * d;
                den += u_direct[e * nn + k] * u_direct[e * nn + k];
            }
        }
        let rel = (num / den.max(1e-300)).sqrt();
        (it_direct, it_mg, rel)
    }

    #[test]
    fn sbm_mg_pcg_matches_direct_and_cuts_iterations_helmholtz() {
        let (it_direct, it_mg, rel) = check(200.0);
        eprintln!("SBM-MG Helmholtz: direct CG {it_direct} iters, MG-PCG {it_mg} iters, rel {rel:.2e}");
        assert!(rel < 1e-6, "SBM-MG-PCG vs direct rel diff {rel}");
        assert!(it_mg * 2 < it_direct, "SBM-MG-PCG {it_mg} not << direct {it_direct}");
        // The SBM-aware V-cycle converges (no stagnation, unlike the standard full-mesh MG); the
        // iteration count is higher than a clean p-MG (~20) — smoother tuning is a later refinement.
        assert!(it_mg < 200, "SBM-MG-PCG iters not bounded: {it_mg}");
    }

    #[test]
    fn sbm_mg_pcg_matches_direct_and_cuts_iterations_poisson() {
        // Pure Poisson (reaction 0) Dirichlet surrogate — the ill-conditioned case the cylinder
        // pressure needs. MG-PCG must reproduce the direct solve with a large iteration cut.
        let (it_direct, it_mg, rel) = check(0.0);
        eprintln!("SBM-MG Poisson: direct CG {it_direct} iters, MG-PCG {it_mg} iters, rel {rel:.2e}");
        assert!(rel < 1e-6, "SBM-MG-PCG vs direct rel diff {rel}");
        assert!(it_mg * 3 < it_direct, "SBM-MG-PCG {it_mg} not << direct {it_direct}");
    }
}
