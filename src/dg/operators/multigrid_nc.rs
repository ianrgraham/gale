//! **p-multigrid preconditioner for the non-conforming (AMR) SIPG Poisson operator.**
//!
//! The conforming [`PMultigrid`](super::multigrid::PMultigrid) is uniform-rectangle-only
//! (`from_mesh` returns `None` on a refined mesh). Adaptive runs therefore fall back to plain CG on
//! the non-conforming operator, which is unpreconditioned and takes thousands of iterations on the
//! deflated singular pressure solve (`operator_nc` ≈ 80% of step time at scale). This module mirrors
//! the conforming p-multigrid for the NC operator: a hierarchy of polynomial orders `p → p/2 → … → 1`
//! on the SAME refined mesh, V-cycle preconditioning a PCG.
//!
//! Why p-coarsening transfers stay valid on a non-conforming mesh: the inter-level transfer is the
//! element-local tensor-Lagrange nodal interpolation between orders — it never crosses an element
//! face, so the 2:1 mortar non-conformity is entirely the *operator's* business at each level, not
//! the transfer's. The operator at every level is the full mortar-SIPG [`Poisson`] on the refined
//! mesh at that order. The smoother diagonal is obtained by **colored probing** (greedy coloring of
//! the element face-adjacency graph; SIPG couples only face neighbours, so same-colour elements are
//! decoupled and one matvec per (colour, node) reads the exact diagonal).
//!
//! This is the CPU reference (validated to cut the outer iteration count by ~100× vs plain CG); the
//! GPU port mirrors it on `GpuPoissonNc`.

use super::multigrid::{lagrange_matrix, order_levels, PMultigrid};
use crate::dg::mesh::Neighbor;
use crate::dg::reference::Reference1d;
use crate::dg::{Mesh2d, Poisson};
use rayon::prelude::*;
use std::collections::HashSet;

/// Order-1 element id(s) of one base cell in the `cartesian_refined` ordering — `Single` for an
/// unrefined cell, `Quad` (child order `sx+2·sy`) for a 2:1-refined one. The bridge for collapsing the
/// order-1 refined mesh onto the order-1 uniform base mesh (the h-coarsening transition).
#[derive(Clone, Copy)]
enum CellId {
    Single(usize),
    Quad([usize; 4]),
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// p-multigrid hierarchy on a fixed (possibly non-conforming) mesh.
pub struct PMultigridNc {
    orders: Vec<usize>,
    meshes: Vec<Mesh2d>,
    alpha: f64,
    reaction: f64,
    neumann_tags: Vec<u32>,
    /// Per transition `l → l+1`: the 1D Lagrange interp matrix (`fine×coarse`, row-major).
    interp: Vec<Vec<f64>>,
    inv_diag: Vec<Vec<f64>>,
    lam_hi: Vec<f64>,
    colors: Vec<Vec<usize>>,
    n_colors: Vec<usize>,
    singular: bool,
    pre: usize,
    post: usize,
    // ---- h-coarsening tail: collapse order-1-refined → order-1-uniform-base, solve with a conforming
    // p/h-multigrid (the validated `PMultigrid`, which coarsens the uniform base to a tiny grid). This
    // replaces the O(N) order-1-refined coarse solve, which dominated the step at scale. ----
    base_nx: usize,
    base_ny: usize,
    base_mg: PMultigrid,
    /// Per base cell (row-major `cx + cy·nx`): its order-1 element id(s) in the refined mesh.
    cell_ids: Vec<CellId>,
    /// 2:1 geometric transfer matrices `[q*16 + f*4 + a]` (coarse node `a` → fine node `f`, quadrant
    /// `q`), shared with `PMultigrid`'s h-transfer — so the refined↔base collapse uses the SAME P/Pᵀ.
    quad_pq: [f64; 64],
    /// Preconditioned-stationary iterations of the conforming base MG used as the coarse solve.
    coarse_iters: usize,
}

impl PMultigridNc {
    /// Build the p-multigrid for `Poisson::with_bc(mesh, alpha, reaction, neumann_tags)` on a
    /// (refined) `mesh` of order `p`. `refine` is the set of base cells refined to build `mesh`
    /// (so each coarser p-level rebuilds `cartesian_refined` at the lower order). `singular` ⇒ the
    /// all-Neumann pressure operator (deflate the PCG against the constant null space).
    pub fn new(
        p: usize, nx: usize, ny: usize, xr: [f64; 2], yr: [f64; 2], refine: &[(usize, usize)], alpha: f64,
        reaction: f64, neumann_tags: Vec<u32>, singular: bool,
    ) -> Self {
        let orders = order_levels(p);
        let meshes: Vec<Mesh2d> =
            orders.iter().map(|&o| Mesh2d::cartesian_refined(o, nx, ny, xr, yr, refine)).collect();
        // p-transfer matrices between consecutive orders (1D Lagrange, fine←coarse).
        let interp: Vec<Vec<f64>> = (0..orders.len() - 1)
            .map(|l| {
                let cn = Reference1d::new(orders[l + 1]).nodes;
                let fn_ = Reference1d::new(orders[l]).nodes;
                lagrange_matrix(&cn, &fn_)
            })
            .collect();
        let (colors, n_colors): (Vec<Vec<usize>>, Vec<usize>) =
            meshes.iter().map(color_elements).unzip();

        // h-coarsening tail: conforming p/h-multigrid on the order-1 UNIFORM base mesh (coarsens to a
        // tiny grid), plus the refined→base element map and the shared 2:1 transfer matrices.
        let base_mg = PMultigrid::with_bc(1, nx, ny, xr, yr, alpha, reaction, neumann_tags.clone());
        // `base_mg` is order 1 ⇒ its h-transfer matrices are exactly the 64-element (4×4 per
        // quadrant) bilinear set this NC tail uses; copy into the fixed array.
        let quad_pq: [f64; 64] =
            base_mg.quad_prolong().try_into().expect("order-1 base quad_prolong must be 64 elems");
        let cell_ids = compute_cell_ids(nx, ny, refine);

        let mut mg = PMultigridNc {
            orders,
            meshes,
            alpha,
            reaction,
            neumann_tags,
            interp,
            inv_diag: Vec::new(),
            lam_hi: Vec::new(),
            colors,
            n_colors,
            singular,
            pre: 2,
            post: 2,
            base_nx: nx,
            base_ny: ny,
            base_mg,
            cell_ids,
            quad_pq,
            coarse_iters: 2,
        };
        mg.inv_diag = (0..mg.orders.len())
            .map(|l| mg.diagonal(l).iter().map(|d| if d.abs() > 1e-300 { 1.0 / d } else { 0.0 }).collect())
            .collect();
        mg.lam_hi = (0..mg.orders.len()).map(|l| 1.1 * mg.power_lambda(l)).collect();
        mg
    }

    fn ndof(&self, l: usize) -> usize {
        self.meshes[l].n_elements() * self.meshes[l].refq.n_nodes()
    }

    fn apply_level(&self, l: usize, u: &[f64]) -> Vec<f64> {
        Poisson::with_bc(&self.meshes[l], self.alpha, self.reaction, self.neumann_tags.clone()).apply(u)
    }

    /// Exact operator diagonal via colored probing: `n_colors·nn` matvecs. Same-colour elements are
    /// face-decoupled (greedy coloring of the adjacency graph), so probing node `m` of every
    /// colour-`c` element at once and reading back at those nodes yields the diagonal.
    fn diagonal(&self, l: usize) -> Vec<f64> {
        let mesh = &self.meshes[l];
        let nn = mesh.refq.n_nodes();
        let ne = mesh.n_elements();
        let n = ne * nn;
        let colors = &self.colors[l];
        let probes: Vec<(usize, usize)> =
            (0..self.n_colors[l]).flat_map(|c| (0..nn).map(move |m| (c, m))).collect();
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
        let mut diag = vec![0.0; n];
        for part in parts {
            for (i, v) in part {
                diag[i] = v;
            }
        }
        diag
    }

    fn power_lambda(&self, l: usize) -> f64 {
        let n = self.ndof(l);
        let mut v: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64 * 0.13).collect();
        let nv = norm(&v);
        v.iter_mut().for_each(|x| *x /= nv);
        let mut lam = 1.0;
        for _ in 0..50 {
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

    /// p-prolong (coarse level `l+1` → fine level `l`), element-local tensor interpolation.
    fn prolong(&self, l: usize, coarse: &[f64]) -> Vec<f64> {
        let i = &self.interp[l];
        let ncc = self.orders[l + 1] + 1;
        let nff = self.orders[l] + 1;
        let ne = self.meshes[l].n_elements();
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

    /// p-restrict (fine `l` → coarse `l+1`) = transpose of [`prolong`](Self::prolong).
    fn restrict(&self, l: usize, fine: &[f64]) -> Vec<f64> {
        let i = &self.interp[l];
        let ncc = self.orders[l + 1] + 1;
        let nff = self.orders[l] + 1;
        let ne = self.meshes[l].n_elements();
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

    /// Recursive V-cycle for `A x = b` at level `l` (x updated in place; entry x assumed 0 on the
    /// initial outer call but general for recursion).
    fn v_cycle(&self, l: usize, b: &[f64], x: &mut [f64]) {
        if l == self.orders.len() - 1 {
            // Coarsest p-level = order 1 on the REFINED mesh. Instead of an O(N) CG/Jacobi solve here,
            // smooth + H-COARSE-CORRECT: collapse the residual onto the order-1 UNIFORM base mesh and
            // solve that with the conforming p/h-multigrid (which coarsens to a tiny grid ⇒ cheap +
            // mesh-independent), then prolong the correction back. This is the fix for the coarse-solve
            // blowup that made large meshes (256²) take tens of seconds/step.
            self.smooth(l, x, b, self.pre);
            let r: Vec<f64> = {
                let ax = self.apply_level(l, x);
                (0..b.len()).map(|i| b[i] - ax[i]).collect()
            };
            let base_b = self.restrict_to_base(&r);
            let nb = self.base_nx * self.base_ny * 4;
            let mut base_x = vec![0.0; nb];
            // Preconditioned stationary iteration with the conforming base MG (a few V-cycles solve the
            // small base problem well; the base MG is itself mesh-independent).
            for _ in 0..self.coarse_iters {
                let ax = self.base_mg.apply(&base_x);
                let res: Vec<f64> = (0..nb).map(|i| base_b[i] - ax[i]).collect();
                let e = self.base_mg.precondition(&res);
                for i in 0..nb {
                    base_x[i] += e[i];
                }
                if self.singular {
                    let mean = base_x.iter().sum::<f64>() / nb as f64;
                    base_x.iter_mut().for_each(|v| *v -= mean);
                }
            }
            let corr = self.prolong_from_base(&base_x);
            for i in 0..x.len() {
                x[i] += corr[i];
            }
            self.smooth(l, x, b, self.post);
            return;
        }
        self.smooth(l, x, b, self.pre);
        let r: Vec<f64> = {
            let ax = self.apply_level(l, x);
            (0..b.len()).map(|i| b[i] - ax[i]).collect()
        };
        let rc = self.restrict(l, &r);
        let mut ec = vec![0.0; rc.len()];
        self.v_cycle(l + 1, &rc, &mut ec);
        let ef = self.prolong(l, &ec);
        for i in 0..x.len() {
            x[i] += ef[i];
        }
        self.smooth(l, x, b, self.post);
    }

    /// Collapse an order-1 field on the REFINED mesh → order-1 UNIFORM base mesh (the h-coarsening
    /// restriction `Pᵀ`): unrefined cells inject; refined cells `Pᵀ`-gather their 4 children to the
    /// parent via the shared 2:1 matrices. Output indexed by base cell `cx + cy·nx`.
    fn restrict_to_base(&self, refined: &[f64]) -> Vec<f64> {
        let nbc = self.base_nx * self.base_ny;
        let mut base = vec![0.0; nbc * 4];
        for bc in 0..nbc {
            match self.cell_ids[bc] {
                CellId::Single(rid) => base[bc * 4..bc * 4 + 4].copy_from_slice(&refined[rid * 4..rid * 4 + 4]),
                CellId::Quad(c) => {
                    for a in 0..4 {
                        let mut s = 0.0;
                        for q in 0..4 {
                            for f in 0..4 {
                                s += self.quad_pq[q * 16 + f * 4 + a] * refined[c[q] * 4 + f];
                            }
                        }
                        base[bc * 4 + a] = s;
                    }
                }
            }
        }
        base
    }

    /// Prolong an order-1 base field → order-1 refined field (the h-coarsening prolongation `P`,
    /// transpose of [`restrict_to_base`]): unrefined cells inject; refined cells bilinearly evaluate
    /// the parent at each child's nodes.
    fn prolong_from_base(&self, base: &[f64]) -> Vec<f64> {
        let nbc = self.base_nx * self.base_ny;
        let mut refined = vec![0.0; self.ndof(self.orders.len() - 1)];
        for bc in 0..nbc {
            match self.cell_ids[bc] {
                CellId::Single(rid) => refined[rid * 4..rid * 4 + 4].copy_from_slice(&base[bc * 4..bc * 4 + 4]),
                CellId::Quad(c) => {
                    for q in 0..4 {
                        for f in 0..4 {
                            let mut s = 0.0;
                            for a in 0..4 {
                                s += self.quad_pq[q * 16 + f * 4 + a] * base[bc * 4 + a];
                            }
                            refined[c[q] * 4 + f] = s;
                        }
                    }
                }
            }
        }
        refined
    }

    /// One V-cycle as a preconditioner application `z = M⁻¹ r` (finest level, zero initial guess).
    pub fn precondition(&self, r: &[f64]) -> Vec<f64> {
        let mut z = vec![0.0; r.len()];
        self.v_cycle(0, r, &mut z);
        z
    }

    /// Preconditioned CG on the finest NC operator, deflating the constant null space when singular.
    /// Returns `(x, iterations)`. This is the drop-in replacement for the plain-CG NC solve.
    pub fn solve(&self, b: &[f64], tol: f64, maxit: usize) -> (Vec<f64>, usize) {
        let n = b.len();
        let deflate = |v: &mut [f64]| {
            if self.singular {
                let mean = v.iter().sum::<f64>() / n as f64;
                v.iter_mut().for_each(|x| *x -= mean);
            }
        };
        let mut b = b.to_vec();
        deflate(&mut b);
        let bn = norm(&b).max(1e-300);
        let mut x = vec![0.0; n];
        let mut r = b.clone(); // r = b − A·0
        let mut z = self.precondition(&r);
        deflate(&mut z);
        let mut p = z.clone();
        let mut rz = dot(&r, &z);
        let mut iters = maxit;
        for it in 0..maxit {
            if norm(&r) / bn < tol {
                iters = it;
                break;
            }
            let mut ap = self.apply_level(0, &p);
            deflate(&mut ap);
            let pap = dot(&p, &ap);
            if !(pap > 0.0) {
                iters = it;
                break;
            }
            let alpha = rz / pap;
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            let mut zn = self.precondition(&r);
            deflate(&mut zn);
            let rz_new = dot(&r, &zn);
            let beta = rz_new / rz;
            for i in 0..n {
                p[i] = zn[i] + beta * p[i];
            }
            rz = rz_new;
            z = zn;
        }
        let _ = z;
        (x, iters)
    }

    pub fn n_levels(&self) -> usize {
        self.orders.len()
    }

    // ---- accessors for an external (GPU) driver that reuses this CPU setup (mirrors PMultigrid) ----
    /// Polynomial order at level `l` (finest = 0).
    pub fn order(&self, l: usize) -> usize {
        self.orders[l]
    }
    /// Inverse operator diagonal at level `l` (the damped-Jacobi smoother's `D⁻¹`).
    pub fn inv_diag(&self, l: usize) -> &[f64] {
        &self.inv_diag[l]
    }
    /// Damped-Jacobi weight `ω = (4/3)/λ_max(D⁻¹A)` at level `l`.
    pub fn jacobi_omega(&self, l: usize) -> f64 {
        (4.0 / 3.0) / self.lam_hi[l]
    }
    /// 1D Lagrange p-transfer matrix (`fine×coarse`, row-major) for transition `l → l+1`.
    pub fn interp(&self, l: usize) -> &[f64] {
        &self.interp[l]
    }
    /// Pre/post smoothing sweep counts.
    pub fn smoothing(&self) -> (usize, usize) {
        (self.pre, self.post)
    }
    pub fn is_singular(&self) -> bool {
        self.singular
    }

    /// Flattened h-coarsening transfer metadata for a GPU port: `(base_kind, base_ids, quad_pq,
    /// base_nx, base_ny)`. `base_kind[bc]` = 0 (unrefined) / 1 (refined); `base_ids[bc*4+0]` = the
    /// order-1 element id (unrefined) or the 4 child ids (refined); `quad_pq` = the shared 2:1
    /// transfer matrices. The GPU `GpuPMultigridNc` uploads these to drive the identical collapse.
    pub fn base_transfer_data(&self) -> (Vec<u8>, Vec<u32>, [f64; 64], usize, usize) {
        let nbc = self.base_nx * self.base_ny;
        let mut kind = vec![0u8; nbc];
        let mut ids = vec![0u32; nbc * 4];
        for bc in 0..nbc {
            match self.cell_ids[bc] {
                CellId::Single(rid) => {
                    kind[bc] = 0;
                    ids[bc * 4] = rid as u32;
                }
                CellId::Quad(c) => {
                    kind[bc] = 1;
                    for q in 0..4 {
                        ids[bc * 4 + q] = c[q] as u32;
                    }
                }
            }
        }
        (kind, ids, self.quad_pq, self.base_nx, self.base_ny)
    }

    /// dof of the coarsest p-level (order-1 refined mesh) — sizes the GPU h-coarsen transfer output.
    pub fn coarsest_ndof(&self) -> usize {
        self.ndof(self.orders.len() - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: &[f64], b: &[f64]) -> f64 {
        let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
        let n: f64 = b.iter().map(|x| x * x).sum::<f64>().max(1e-300);
        (d / n).sqrt()
    }

    /// Helmholtz (non-singular, Dirichlet) on a 2:1-refined mesh: p-MG-PCG must reach the same answer
    /// as plain CG in DRAMATICALLY fewer iterations.
    #[test]
    fn nc_pmg_helmholtz_cuts_iterations() {
        let (p, nx, ny) = (4usize, 16usize, 16usize);
        let (xr, yr) = ([0.0, 1.0], [0.0, 1.0]);
        let (alpha, reaction) = (5.0, 50.0);
        let refine: Vec<(usize, usize)> =
            (0..nx).flat_map(|cy| (0..nx).map(move |cx| (cx, cy))).filter(|(cx, cy)| (cx + cy) % 4 == 0).collect();
        let mesh = Mesh2d::cartesian_refined(p, nx, ny, xr, yr, &refine);
        let nn = mesh.refq.n_nodes();
        let ndof = mesh.n_elements() * nn;
        let op = Poisson::with_reaction(&mesh, alpha, reaction);

        // Consistent RHS from a manufactured field.
        let mut xt = vec![0.0; ndof];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                xt[e * nn + k] = (3.0 * el.geom.x[k]).sin() * (2.0 * el.geom.y[k]).cos();
            }
        }
        let b = op.apply(&xt);

        let (x_cg, it_cg, _) = op.cg(&b, 1e-10, 20000);
        let mg = PMultigridNc::new(p, nx, ny, xr, yr, &refine, alpha, reaction, vec![], false);
        let (x_mg, it_mg) = mg.solve(&b, 1e-10, 2000);

        println!("  [NC Helmholtz, {} elems, {ndof} dof] plain CG: {it_cg} iters | p-MG-PCG: {it_mg} iters ({:.0}× fewer), levels={}", mesh.n_elements(), it_cg as f64 / it_mg.max(1) as f64, mg.n_levels());
        assert!(rel(&x_mg, &x_cg) < 1e-6, "MG-PCG solution disagrees with CG: rel {}", rel(&x_mg, &x_cg));
        assert!(it_mg * 4 < it_cg, "p-MG-PCG ({it_mg}) should be far fewer iters than CG ({it_cg})");
    }

    /// Singular pure-Neumann pressure operator on a refined mesh (deflated): the hard case that
    /// dominates the NS step. p-MG-PCG vs deflated plain CG.
    #[test]
    fn nc_pmg_singular_pressure_cuts_iterations() {
        let (p, nx, ny) = (4usize, 16usize, 16usize);
        let (xr, yr) = ([0.0, 1.0], [0.0, 1.0]);
        let alpha = 5.0;
        let refine: Vec<(usize, usize)> =
            (0..nx).flat_map(|cy| (0..nx).map(move |cx| (cx, cy))).filter(|(cx, cy)| (cx + cy) % 4 == 0).collect();
        let mesh = Mesh2d::cartesian_refined(p, nx, ny, xr, yr, &refine);
        let nn = mesh.refq.n_nodes();
        let ndof = mesh.n_elements() * nn;
        let tags = mesh.boundary_tags(); // all boundary tags ⇒ pure Neumann ⇒ singular
        let op = Poisson::with_bc(&mesh, alpha, 0.0, tags.clone());

        let mut xt = vec![0.0; ndof];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                xt[e * nn + k] = (std::f64::consts::PI * el.geom.x[k]).cos() * (std::f64::consts::PI * el.geom.y[k]).cos();
            }
        }
        let mut b = op.apply(&xt);
        let mean = b.iter().sum::<f64>() / ndof as f64; // project into range
        b.iter_mut().for_each(|v| *v -= mean);

        let (x_cg, it_cg) = op.cg_deflated(&b, 1e-8, 30000);
        let mg = PMultigridNc::new(p, nx, ny, xr, yr, &refine, alpha, 0.0, tags, true);
        let (x_mg, it_mg) = mg.solve(&b, 1e-8, 2000);

        // Compare up to the constant null space (deflate both).
        let dm = |v: &[f64]| {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            v.iter().map(|x| x - m).collect::<Vec<_>>()
        };
        let r = rel(&dm(&x_mg), &dm(&x_cg));
        println!("  [NC singular pressure, {} elems, {ndof} dof] deflated CG: {it_cg} iters | p-MG-PCG: {it_mg} iters ({:.0}× fewer)", mesh.n_elements(), it_cg as f64 / it_mg.max(1) as f64);
        assert!(r < 1e-5, "MG-PCG pressure disagrees with CG (mod const): rel {r}");
        assert!(it_mg * 4 < it_cg, "p-MG-PCG ({it_mg}) should be far fewer iters than CG ({it_cg})");
    }
}

/// Greedy coloring of the element face-adjacency graph (SIPG couples only face neighbours). Returns
/// the per-element colour and the colour count. Same-colour elements are guaranteed non-adjacent ⇒
/// decoupled, which the colored diagonal probe relies on.
/// Order-1 element id(s) per base cell, replicating `Mesh2d::cartesian_refined`'s element ordering
/// (walk `cy` then `cx`; an unrefined cell pushes one element, a refined cell pushes 4 children in
/// `sx+2·sy` order). Lets the h-coarsening transfer map refined elements ↔ uniform base cells.
fn compute_cell_ids(nx: usize, ny: usize, refine: &[(usize, usize)]) -> Vec<CellId> {
    let refined: HashSet<(usize, usize)> = refine.iter().copied().collect();
    let mut out = vec![CellId::Single(0); nx * ny];
    let mut id = 0usize;
    for cy in 0..ny {
        for cx in 0..nx {
            if refined.contains(&(cx, cy)) {
                out[cx + cy * nx] = CellId::Quad([id, id + 1, id + 2, id + 3]);
                id += 4;
            } else {
                out[cx + cy * nx] = CellId::Single(id);
                id += 1;
            }
        }
    }
    out
}

fn color_elements(mesh: &Mesh2d) -> (Vec<usize>, usize) {
    let ne = mesh.n_elements();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); ne];
    for (e, el) in mesh.elements.iter().enumerate() {
        for nb in el.neighbors.iter() {
            match nb {
                Neighbor::Interior { elem, .. } => adj[e].push(*elem),
                Neighbor::CoarseToFine { fine } => {
                    adj[e].push(fine[0].0);
                    adj[e].push(fine[1].0);
                }
                Neighbor::FineToCoarse { coarse, .. } => adj[e].push(*coarse),
                Neighbor::Boundary { .. } => {}
            }
        }
    }
    let mut color = vec![usize::MAX; ne];
    let mut nc = 0;
    for e in 0..ne {
        let mut used = vec![false; nc + 1];
        for &nb in &adj[e] {
            if color[nb] != usize::MAX && color[nb] <= nc {
                used[color[nb]] = true;
            }
        }
        let c = (0..=nc).find(|&c| !used[c]).unwrap();
        color[e] = c;
        if c == nc {
            nc += 1;
        }
    }
    (color, nc.max(1))
}
