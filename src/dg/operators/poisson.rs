//! Scalar elliptic solve: the **Symmetric Interior Penalty (SIPG)** DG Laplacian
//! for `−∇²u = f` with weak (Nitsche) Dirichlet BCs, applied matrix-free, and a
//! conjugate-gradient solver. This is the CPU scalar-elliptic oracle (build-order
//! step 1 of `docs/implicit-solver-strategy.md`) the GPU port will validate against.
//!
//! Discretization (per `docs/dg-gpu-fluid-simulation.md` §3.4): nodal DG-SEM on
//! quads, LGL collocation (diagonal mass), volume stiffness `Dxᵀ W Dx + Dyᵀ W Dy`,
//! and the standard SIPG face terms — consistency `−∮{∇u·n}[v]`, symmetry
//! `−∮{∇v·n}[u]`, penalty `+∮ τ[u][v]` — with the same form on Dirichlet boundary
//! faces (jump = `u`, single-sided average). Penalty `τ = α (p+1)² / h`.
//!
//! Global DOF layout: element `e`, local node `k` → index `e·nn + k` (DG: no shared
//! nodes between elements).

use super::amr::RefineQuad;
use super::face::Edge;
use super::mesh::{Mesh2d, Neighbor};
use rayon::prelude::*;

/// The SIPG Poisson operator over a mesh, with a penalty coefficient.
pub struct Poisson<'m> {
    pub mesh: &'m Mesh2d,
    /// Penalty scale α in `τ = α (p+1)² / h`.
    pub alpha: f64,
    /// Helmholtz reaction coefficient λ in the operator `λM + A` (0 ⇒ pure Poisson).
    /// Used for the implicit viscous solve `(γ₀/(νΔt) I − ∇²)u` of the Stokes
    /// dual-splitting scheme.
    pub reaction: f64,
    /// Boundary tags treated as **Neumann** (natural BC). Tags not listed are
    /// Dirichlet (weak/Nitsche). Empty ⇒ all-Dirichlet (the pure Poisson default).
    /// The pressure-Poisson of the splitting uses all-Neumann.
    pub neumann_tags: Vec<u32>,
    /// Per-element length scale `h = √area`.
    h: Vec<f64>,
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl<'m> Poisson<'m> {
    pub fn new(mesh: &'m Mesh2d, alpha: f64) -> Self {
        Self::with_reaction(mesh, alpha, 0.0)
    }

    /// Helmholtz operator `λM + A` (reaction `λ ≥ 0`); `λ = 0` is the pure Poisson.
    pub fn with_reaction(mesh: &'m Mesh2d, alpha: f64, reaction: f64) -> Self {
        Self::with_bc(mesh, alpha, reaction, Vec::new())
    }

    /// Full constructor: reaction `λ` and the set of Neumann boundary tags.
    pub fn with_bc(mesh: &'m Mesh2d, alpha: f64, reaction: f64, neumann_tags: Vec<u32>) -> Self {
        let h = mesh
            .elements
            .iter()
            .map(|e| e.geom.jw.iter().sum::<f64>().sqrt())
            .collect();
        Self { mesh, alpha, reaction, neumann_tags, h }
    }

    fn is_neumann(&self, tag: u32) -> bool {
        self.neumann_tags.contains(&tag)
    }

    fn penalty(&self, e: usize, neighbor: Option<usize>) -> f64 {
        let p1 = (self.mesh.order + 1) as f64;
        let h = match neighbor {
            Some(r) => self.h[e].min(self.h[r]),
            None => self.h[e],
        };
        self.alpha * p1 * p1 / h
    }

    /// Total DOF count (`n_elements · (p+1)²`).
    pub fn ndof(&self) -> usize {
        self.mesh.n_elements() * self.mesh.refq.n_nodes()
    }

    /// Matrix-free SIPG operator action `A u`.
    pub fn apply(&self, u: &[f64]) -> Vec<f64> {
        let m = self.mesh;
        let refq = &m.refq;
        let nn = refq.n_nodes();
        let ndof = self.ndof();
        // FUSED gradient + volume-stiffness pass (single parallel region, per-thread scratch,
        // no per-element allocation). Per element it computes the physical gradients into the
        // FLAT `gx`/`gy` buffers (consumed by the face terms below — shared, not recomputed)
        // AND the volume stiffness `r[e] = Drᵀ pr + Dsᵀ ps`, sharing ONE `diff_r`/`diff_s`
        // pair (the old `grad_x`+`grad_y` recomputed both). Bit-for-bit identical to the prior
        // `elem_grads` + `volume_with_grads` (same per-element arithmetic; `+` is commutative).
        let mut r = vec![0.0; ndof];
        let mut gx = vec![0.0; ndof];
        let mut gy = vec![0.0; ndof];
        r.par_chunks_mut(nn)
            .zip(gx.par_chunks_mut(nn))
            .zip(gy.par_chunks_mut(nn))
            .zip(m.elements.par_iter())
            .enumerate()
            .for_each_init(
                || (vec![0.0; nn], vec![0.0; nn], vec![0.0; nn], vec![0.0; nn], vec![0.0; nn]),
                |(fr, fs, pr, ps, tr), (e, (((rc, gxc), gyc), el))| {
                    let ue = &u[e * nn..(e + 1) * nn];
                    refq.diff_r_into(ue, fr);
                    refq.diff_s_into(ue, fs);
                    let g = &el.geom;
                    for k in 0..nn {
                        gxc[k] = g.rx[k] * fr[k] + g.sx[k] * fs[k];
                        gyc[k] = g.ry[k] * fr[k] + g.sy[k] * fs[k];
                        let wx = g.jw[k] * gxc[k];
                        let wy = g.jw[k] * gyc[k];
                        pr[k] = g.rx[k] * wx + g.ry[k] * wy;
                        ps[k] = g.sx[k] * wx + g.sy[k] * wy;
                    }
                    refq.diff_r_t_into(pr, tr);
                    refq.diff_s_t_into(ps, rc);
                    for k in 0..nn {
                        rc[k] += tr[k];
                    }
                },
            );

        // Mortar projections (only used at non-conforming faces).
        let mortar = RefineQuad::new(m.order);
        let sorted = |e: usize, edge: Edge| -> Vec<usize> {
            let f = &m.elements[e].faces[edge as usize];
            let g = &m.elements[e].geom;
            let vert = matches!(edge, Edge::East | Edge::West);
            let mut idx: Vec<usize> = (0..f.nodes.len()).collect();
            idx.sort_by(|&a, &b| {
                let ca = if vert { g.y[f.nodes[a]] } else { g.x[f.nodes[a]] };
                let cb = if vert { g.y[f.nodes[b]] } else { g.x[f.nodes[b]] };
                ca.partial_cmp(&cb).unwrap()
            });
            idx
        };

        // Face terms. Each owning element's contribution is computed in parallel into a
        // record of `(global_index, delta)` pairs — this is where the expensive O(p³) gradᵀ
        // symmetry lifts live — then replayed **serially in element/edge/node order** below.
        // The replay order is identical to the original serial scatter, so accumulation into
        // each node is bit-for-bit unchanged (the symmetry / MMS / non-conforming tests guard
        // this). Writes `r[i] += v` become `push((i, v))`; `r[i] -= v` become `push((i, -v))`.
        let records: Vec<Vec<(usize, f64)>> = m
            .elements
            .par_iter()
            .enumerate()
            .map(|(e, el)| {
                let mut rec: Vec<(usize, f64)> = Vec::new();
                for edge in Edge::ALL {
                    let f = &el.faces[edge as usize];
                    match &el.neighbors[edge as usize] {
                        Neighbor::Interior { elem: re, edge: redge, perm } => {
                            if e >= *re {
                                continue; // process each interior face once (from low side)
                            }
                            let rel = &m.elements[*re];
                            let rf = &rel.faces[*redge as usize];
                            let tau = self.penalty(e, Some(*re));
                            let mut hxl = vec![0.0; nn];
                            let mut hyl = vec![0.0; nn];
                            let mut hxr = vec![0.0; nn];
                            let mut hyr = vec![0.0; nn];
                            for a in 0..f.nodes.len() {
                                let b = perm[a];
                                let (vl, vr) = (f.nodes[a], rf.nodes[b]);
                                let (nx, ny, sw) = (f.nx[a], f.ny[a], f.sw[a]);
                                let dun_l = nx * gx[e * nn + vl] + ny * gy[e * nn + vl];
                                let dun_r = nx * gx[*re * nn + vr] + ny * gy[*re * nn + vr];
                                let avg = 0.5 * (dun_l + dun_r);
                                let jump = u[e * nn + vl] - u[*re * nn + vr];
                                // consistency  −∮{∇u·n}[v]
                                rec.push((e * nn + vl, -sw * avg));
                                rec.push((*re * nn + vr, sw * avg));
                                // penalty  +∮ τ[u][v]
                                rec.push((e * nn + vl, tau * sw * jump));
                                rec.push((*re * nn + vr, -tau * sw * jump));
                                // symmetry  −∮{∇v·n}[u]  (lift, average factor ½, same normal n_L)
                                let g = 0.5 * sw * jump;
                                hxl[vl] += g * nx;
                                hyl[vl] += g * ny;
                                hxr[vr] += g * nx;
                                hyr[vr] += g * ny;
                            }
                            let ll = add(el.geom.gradx_t(refq, &hxl), el.geom.grady_t(refq, &hyl));
                            let lr = add(rel.geom.gradx_t(refq, &hxr), rel.geom.grady_t(refq, &hyr));
                            for k in 0..nn {
                                rec.push((e * nn + k, -ll[k]));
                                rec.push((*re * nn + k, -lr[k]));
                            }
                        }
                        Neighbor::Boundary { tag } => {
                            if self.is_neumann(*tag) {
                                // Natural (Neumann) BC: no operator contribution; the
                                // prescribed flux enters the RHS instead.
                                continue;
                            }
                            let tau = self.penalty(e, None);
                            let mut hx = vec![0.0; nn];
                            let mut hy = vec![0.0; nn];
                            for a in 0..f.nodes.len() {
                                let v = f.nodes[a];
                                let (nx, ny, sw) = (f.nx[a], f.ny[a], f.sw[a]);
                                let dun = nx * gx[e * nn + v] + ny * gy[e * nn + v];
                                let uval = u[e * nn + v];
                                rec.push((e * nn + v, -sw * dun)); // consistency  −∮(∇u·n)v
                                rec.push((e * nn + v, tau * sw * uval)); // penalty  +∮ τ u v
                                hx[v] += sw * uval * nx; // symmetry  −∮(∇v·n)u
                                hy[v] += sw * uval * ny;
                            }
                            let l = add(el.geom.gradx_t(refq, &hx), el.geom.grady_t(refq, &hy));
                            for k in 0..nn {
                                rec.push((e * nn + k, -l[k]));
                            }
                        }
                        Neighbor::FineToCoarse { .. } => { /* handled from the coarse side */ }
                        Neighbor::CoarseToFine { fine } => {
                            // SIPG across a 2:1 interface, integrated on the fine mortar.
                            // Coarse traces (sorted) and their coarse-normal gradient.
                            let ce = sorted(e, edge);
                            let (ncx, ncy) = (f.nx[ce[0]], f.ny[ce[0]]);
                            let uc: Vec<f64> = ce.iter().map(|&i| u[e * nn + f.nodes[i]]).collect();
                            let dnc: Vec<f64> = ce
                                .iter()
                                .map(|&i| {
                                    let v = f.nodes[i];
                                    ncx * gx[e * nn + v] + ncy * gy[e * nn + v]
                                })
                                .collect();
                            let mut hxe = vec![0.0; nn];
                            let mut hye = vec![0.0; nn];
                            for h in 0..2 {
                                let (re, redge) = fine[h];
                                let tau = self.penalty(e, Some(re));
                                let rel = &m.elements[re];
                                let frw = &rel.faces[redge as usize];
                                let rw = sorted(re, redge);
                                let uc_m = mortar.mortar_to_fine(&uc, h);
                                let dnc_m = mortar.mortar_to_fine(&dnc, h);
                                let mut hxr = vec![0.0; nn];
                                let mut hyr = vec![0.0; nn];
                                let mut gc = vec![0.0; rw.len()]; // coarse-test consistency+penalty
                                let mut gl = vec![0.0; rw.len()]; // coarse-test symmetry-lift source
                                for (mi, &i) in rw.iter().enumerate() {
                                    let vf = frw.nodes[i];
                                    let sw = frw.sw[i];
                                    let dnf = ncx * gx[re * nn + vf] + ncy * gy[re * nn + vf];
                                    let jump = uc_m[mi] - u[re * nn + vf];
                                    let avg = 0.5 * (dnc_m[mi] + dnf);
                                    // Fine test (direct): consistency +∮{∇u·n}v_f, penalty −∮τ[u]v_f.
                                    rec.push((re * nn + vf, sw * avg - tau * sw * jump));
                                    // Coarse test (Pᵀ-scattered): −∮{∇u·n}v_c, +∮τ[u]v_c.
                                    gc[mi] = sw * (-avg + tau * jump);
                                    // Symmetry lift g = ½ sw [u] (same coarse normal both sides).
                                    let g = 0.5 * sw * jump;
                                    hxr[vf] += g * ncx;
                                    hyr[vf] += g * ncy;
                                    gl[mi] = g;
                                }
                                // Scatter coarse-test contributions back via Pᵀ.
                                let cc = mortar.mortar_gather(&gc, h);
                                let cl = mortar.mortar_gather(&gl, h);
                                for (mc, &cv) in ce.iter().enumerate() {
                                    let node = f.nodes[cv];
                                    rec.push((e * nn + node, cc[mc]));
                                    hxe[node] += cl[mc] * ncx;
                                    hye[node] += cl[mc] * ncy;
                                }
                                // Fine symmetry-lift → r[re].
                                let lr = add(rel.geom.gradx_t(refq, &hxr), rel.geom.grady_t(refq, &hyr));
                                for k in 0..nn {
                                    rec.push((re * nn + k, -lr[k]));
                                }
                            }
                            // Coarse symmetry-lift → r[e].
                            let le = add(el.geom.gradx_t(refq, &hxe), el.geom.grady_t(refq, &hye));
                            for k in 0..nn {
                                rec.push((e * nn + k, -le[k]));
                            }
                        }
                    }
                }
                rec
            })
            .collect();
        for rec in &records {
            for &(i, d) in rec {
                r[i] += d;
            }
        }
        // Helmholtz reaction term: + λ M u (diagonal mass). Element-local ⇒ parallel.
        if self.reaction != 0.0 {
            r.par_chunks_mut(nn).zip(m.elements.par_iter()).enumerate().for_each(|(e, (out, el))| {
                for k in 0..nn {
                    out[k] += self.reaction * el.geom.jw[k] * u[e * nn + k];
                }
            });
        }
        r
    }

    /// Volume-stiffness action only (no face terms): `Dxᵀ W Dx u + Dyᵀ W Dy u`,
    /// element-local. Written in the fused `pr/ps` form the GPU kernel uses, so it
    /// is the bit-for-bit reference for the GPU volume-operator port.
    pub fn apply_volume(&self, u: &[f64]) -> Vec<f64> {
        self.volume_with_grads(&self.elem_grads(u))
    }

    /// Per-element physical gradients `(∂u/∂x, ∂u/∂y)`, computed in parallel (element-local
    /// ⇒ bit-for-bit identical to a serial precompute). Shared by [`apply`](Self::apply) and
    /// [`apply_volume`](Self::apply_volume) so the gradient contraction is done **once** per
    /// `apply`, not twice (the face term and the volume term both consume it).
    fn elem_grads(&self, u: &[f64]) -> Vec<(Vec<f64>, Vec<f64>)> {
        let refq = &self.mesh.refq;
        let nn = refq.n_nodes();
        self.mesh
            .elements
            .par_iter()
            .enumerate()
            .map(|(e, el)| {
                let ue = &u[e * nn..(e + 1) * nn];
                (el.geom.grad_x(refq, ue), el.geom.grad_y(refq, ue))
            })
            .collect()
    }

    /// Volume-stiffness action from already-computed per-element gradients (the shared core
    /// of [`apply_volume`](Self::apply_volume)). Element-local ⇒ parallel over elements is
    /// bit-for-bit identical to the serial loop (each output chunk is written by exactly one
    /// element's arithmetic).
    fn volume_with_grads(&self, grads: &[(Vec<f64>, Vec<f64>)]) -> Vec<f64> {
        let m = self.mesh;
        let refq = &m.refq;
        let nn = refq.n_nodes();
        let mut r = vec![0.0; self.ndof()];
        r.par_chunks_mut(nn).zip(m.elements.par_iter()).enumerate().for_each(|(e, (out, el))| {
            let (gx, gy) = (&grads[e].0, &grads[e].1);
            let mut pr = vec![0.0; nn];
            let mut ps = vec![0.0; nn];
            for k in 0..nn {
                let wx = el.geom.jw[k] * gx[k];
                let wy = el.geom.jw[k] * gy[k];
                pr[k] = el.geom.rx[k] * wx + el.geom.ry[k] * wy;
                ps[k] = el.geom.sx[k] * wx + el.geom.sy[k] * wy;
            }
            let a = refq.diff_r_t(&pr);
            let b = refq.diff_s_t(&ps);
            for k in 0..nn {
                out[k] = a[k] + b[k];
            }
        });
        r
    }

    /// RHS for `−∇²u = f` with all-Dirichlet data `u = g`.
    pub fn rhs(&self, f: &[f64], g: impl Fn(f64, f64) -> f64) -> Vec<f64> {
        self.rhs_mixed(f, g, |_, _| 0.0)
    }

    /// RHS for mixed boundaries: `g` (value) on Dirichlet faces, `q = ∂u/∂n` (flux)
    /// on Neumann faces (those whose tag is in `neumann_tags`).
    pub fn rhs_mixed(
        &self,
        f: &[f64],
        g: impl Fn(f64, f64) -> f64,
        q: impl Fn(f64, f64) -> f64,
    ) -> Vec<f64> {
        self.rhs_tagged(f, |_, x, y| g(x, y), |_, x, y| q(x, y))
    }

    /// Like [`rhs_mixed`](Self::rhs_mixed) but the Dirichlet value `g(tag, x, y)` and
    /// Neumann flux `q(tag, x, y)` may depend on the boundary tag — the data side of
    /// per-region boundary conditions (operator-side dispatch is via `neumann_tags`).
    pub fn rhs_tagged(
        &self,
        f: &[f64],
        g: impl Fn(u32, f64, f64) -> f64,
        q: impl Fn(u32, f64, f64) -> f64,
    ) -> Vec<f64> {
        let m = self.mesh;
        let refq = &m.refq;
        let nn = refq.n_nodes();
        let mut b = vec![0.0; self.ndof()];

        // Volume load (M f), diagonal mass = Jw.
        for (e, el) in m.elements.iter().enumerate() {
            for k in 0..nn {
                b[e * nn + k] += el.geom.jw[k] * f[e * nn + k];
            }
        }
        for (e, el) in m.elements.iter().enumerate() {
            for edge in Edge::ALL {
                let Neighbor::Boundary { tag } = el.neighbors[edge as usize] else {
                    continue;
                };
                let face = &el.faces[edge as usize];
                if self.is_neumann(tag) {
                    // Neumann flux: + ∮ q v ds (at the face nodes).
                    for a in 0..face.nodes.len() {
                        let v = face.nodes[a];
                        b[e * nn + v] += face.sw[a] * q(tag, el.geom.x[v], el.geom.y[v]);
                    }
                } else {
                    // Dirichlet data: −∮(∇v·n)g + ∮ τ g v.
                    let tau = self.penalty(e, None);
                    let mut hx = vec![0.0; nn];
                    let mut hy = vec![0.0; nn];
                    for a in 0..face.nodes.len() {
                        let v = face.nodes[a];
                        let (nx, ny, sw) = (face.nx[a], face.ny[a], face.sw[a]);
                        let gv = g(tag, el.geom.x[v], el.geom.y[v]);
                        b[e * nn + v] += tau * sw * gv;
                        hx[v] += sw * gv * nx;
                        hy[v] += sw * gv * ny;
                    }
                    let l = add(el.geom.gradx_t(refq, &hx), el.geom.grady_t(refq, &hy));
                    for k in 0..nn {
                        b[e * nn + k] -= l[k];
                    }
                }
            }
        }
        b
    }

    /// CG for the singular (pure-Neumann) system: deflates the constant nullspace
    /// (`A·1 = 0`) by removing the mean from the residual each iteration. Returns
    /// `(u, iterations)`; the solution is determined up to an additive constant.
    pub fn cg_deflated(&self, b: &[f64], tol: f64, maxit: usize) -> (Vec<f64>, usize) {
        let n = b.len();
        let deflate = |v: &mut [f64]| {
            let mean = v.iter().sum::<f64>() / n as f64;
            v.iter_mut().for_each(|x| *x -= mean);
        };
        let mut x = vec![0.0; n];
        let mut r = b.to_vec();
        deflate(&mut r);
        let mut p = r.clone();
        let mut rs = dot(&r, &r);
        let bn = rs.sqrt().max(1e-300);
        for it in 0..maxit {
            let ap = self.apply(&p);
            let alpha = rs / dot(&p, &ap);
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            deflate(&mut r);
            let rs_new = dot(&r, &r);
            if rs_new.sqrt() / bn < tol {
                return (x, it + 1);
            }
            let beta = rs_new / rs;
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
            rs = rs_new;
        }
        (x, maxit)
    }

    /// Conjugate gradient for the SPD operator. Returns `(u, iterations, residual)`.
    pub fn cg(&self, b: &[f64], tol: f64, maxit: usize) -> (Vec<f64>, usize, f64) {
        let n = b.len();
        let mut x = vec![0.0; n];
        let mut r = b.to_vec(); // r = b − A·0
        let mut p = r.clone();
        let mut rs = dot(&r, &r);
        let bnorm = dot(b, b).sqrt().max(1e-300);
        for it in 0..maxit {
            let ap = self.apply(&p);
            let denom = dot(&p, &ap);
            let alpha = rs / denom;
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            let rs_new = dot(&r, &r);
            if rs_new.sqrt() / bnorm < tol {
                return (x, it + 1, rs_new.sqrt());
            }
            let beta = rs_new / rs;
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
            rs = rs_new;
        }
        (x, maxit, rs.sqrt())
    }

    /// L2 norm of a nodal field: `√(Σ Jw·v²)`.
    pub fn l2_norm(&self, v: &[f64]) -> f64 {
        let nn = self.mesh.refq.n_nodes();
        let mut s = 0.0;
        for (e, el) in self.mesh.elements.iter().enumerate() {
            for k in 0..nn {
                s += el.geom.jw[k] * v[e * nn + k] * v[e * nn + k];
            }
        }
        s.sqrt()
    }
}

#[inline]
fn add(mut a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
    a
}

/// **Shifted Boundary Method (SBM)** SIPG Poisson/Helmholtz operator — a *sharp* embedded
/// boundary (`docs/research-sharp-interface.md`, step 2). The SIPG operator is restricted to
/// the ACTIVE (surrogate-fluid) elements, with weak Dirichlet (Nitsche) BCs imposed on the
/// SURROGATE boundary (active-element edges facing inactive elements). Inactive dofs are
/// pinned to zero by an identity block, so the full-length system decouples into the active
/// operator plus a trivial inactive identity (plain SPD CG solves it).
///
/// **First-order SBM:** the boundary value is evaluated at the *shifted* true-boundary point
/// `x̃ + d` (so the surrogate carries the correct data), but the penalty acts on `u(x̃)`
/// directly. The high-order Taylor correction (penalty on `u + ∇u·d`) is the next increment.
pub struct ShiftedPoisson<'m> {
    base: Poisson<'m>,
    sb: crate::dg::shifted::ShiftedBoundary,
    /// `true` ⇒ high-order SBM with the Taylor correction (the BC operator is the shifted
    /// value `S_h u = u + ∇u·d` in the symmetry/penalty terms, eq. 2.14 of arXiv:2006.00872);
    /// `false` ⇒ first-order (penalty acts on `u(x̃)`).
    taylor: bool,
    /// Surrogate-boundary BC type: `true` ⇒ Dirichlet (weak/Nitsche — the no-slip velocity
    /// solve); `false` ⇒ **natural / homogeneous-Neumann** (the surrogate carries no operator
    /// term — the projection pressure-Poisson, where `∂p/∂n = 0` on the embedded surface).
    surrogate_dirichlet: bool,
}

impl<'m> ShiftedPoisson<'m> {
    /// Build the SBM operator for reaction `λ` over the surrogate domain `sb` (all outer mesh
    /// boundaries are Dirichlet here; per-region tags can be added later). Defaults to
    /// first-order Dirichlet surrogate; see [`taylor`](Self::taylor) and
    /// [`surrogate_neumann`](Self::surrogate_neumann).
    pub fn new(mesh: &'m Mesh2d, alpha: f64, reaction: f64, sb: crate::dg::shifted::ShiftedBoundary) -> Self {
        Self { base: Poisson::with_reaction(mesh, alpha, reaction), sb, taylor: false, surrogate_dirichlet: true }
    }

    /// Full constructor with outer Neumann tags (for the pressure: inflow/walls Neumann,
    /// outflow Dirichlet `p=0`).
    pub fn with_bc(mesh: &'m Mesh2d, alpha: f64, reaction: f64, neumann_tags: Vec<u32>, sb: crate::dg::shifted::ShiftedBoundary) -> Self {
        Self { base: Poisson::with_bc(mesh, alpha, reaction, neumann_tags), sb, taylor: false, surrogate_dirichlet: true }
    }

    /// Enable the high-order Taylor correction (the surrogate Nitsche acts on `S_h u = u+∇u·d`).
    pub fn taylor(mut self, taylor: bool) -> Self {
        self.taylor = taylor;
        self
    }

    /// Make the surrogate boundary **natural (homogeneous Neumann)** instead of Dirichlet —
    /// the projection pressure-Poisson (`∂p/∂n = 0` at the embedded surface). Then the
    /// surrogate faces contribute nothing to the operator/RHS (just the active restriction).
    pub fn surrogate_neumann(mut self) -> Self {
        self.surrogate_dirichlet = false;
        self
    }

    /// Is the active block singular (pure-Neumann everywhere ⇒ deflate)? True when all outer
    /// boundaries are Neumann AND the surrogate is natural — the closed-box pressure case.
    pub fn is_singular(&self) -> bool {
        self.base.reaction == 0.0
            && !self.surrogate_dirichlet
            && self.base.mesh.boundary_tags().iter().all(|t| self.base.neumann_tags.contains(t))
    }

    /// Matrix-free SBM action `A u`: real SIPG on the active block, identity on the inactive.
    pub fn apply(&self, u: &[f64]) -> Vec<f64> {
        let m = self.base.mesh;
        let refq = &m.refq;
        let nn = refq.n_nodes();
        let active = &self.sb.active;
        let grads = self.base.elem_grads(u);
        let gx: Vec<&Vec<f64>> = grads.iter().map(|g| &g.0).collect();
        let gy: Vec<&Vec<f64>> = grads.iter().map(|g| &g.1).collect();
        let mut r = self.base.volume_with_grads(&grads);

        // Face + surrogate terms as per-element records computed in PARALLEL (each element's
        // contribution — including its writes to interior neighbours — collected, then replayed
        // serially in element order). Mirrors the parallel Poisson::apply; the MMS test guards
        // correctness. Interior active–active faces use standard SIPG; surrogate faces use the
        // SBM Nitsche (eq. 2.14): consistency −(∂ₙu,v), symmetry −(S_hu,∂ₙv), penalty
        // +(γ/h)(S_hu,S_hv), S_h u = u+∇u·d (taylor) or u; outer walls use plain weak Dirichlet.
        let records: Vec<Vec<(usize, f64)>> = m
            .elements
            .par_iter()
            .enumerate()
            .map(|(e, el)| {
                let mut rec: Vec<(usize, f64)> = Vec::new();
                if !active[e] {
                    return rec;
                }
                for edge in Edge::ALL {
                    let f = &el.faces[edge as usize];
                    match &el.neighbors[edge as usize] {
                        Neighbor::Interior { elem: re, edge: redge, perm } if active[*re] => {
                            if e >= *re {
                                continue;
                            }
                            let rel = &m.elements[*re];
                            let rf = &rel.faces[*redge as usize];
                            let tau = self.base.penalty(e, Some(*re));
                            let (mut hxl, mut hyl, mut hxr, mut hyr) =
                                (vec![0.0; nn], vec![0.0; nn], vec![0.0; nn], vec![0.0; nn]);
                            for a in 0..f.nodes.len() {
                                let b = perm[a];
                                let (vl, vr) = (f.nodes[a], rf.nodes[b]);
                                let (nx, ny, sw) = (f.nx[a], f.ny[a], f.sw[a]);
                                let avg = 0.5 * ((nx * gx[e][vl] + ny * gy[e][vl]) + (nx * gx[*re][vr] + ny * gy[*re][vr]));
                                let jump = u[e * nn + vl] - u[*re * nn + vr];
                                rec.push((e * nn + vl, -sw * avg + tau * sw * jump));
                                rec.push((*re * nn + vr, sw * avg - tau * sw * jump));
                                let g = 0.5 * sw * jump;
                                hxl[vl] += g * nx;
                                hyl[vl] += g * ny;
                                hxr[vr] += g * nx;
                                hyr[vr] += g * ny;
                            }
                            let ll = add(el.geom.gradx_t(refq, &hxl), el.geom.grady_t(refq, &hyl));
                            let lr = add(rel.geom.gradx_t(refq, &hxr), rel.geom.grady_t(refq, &hyr));
                            for k in 0..nn {
                                rec.push((e * nn + k, -ll[k]));
                                rec.push((*re * nn + k, -lr[k]));
                            }
                        }
                        Neighbor::Interior { .. } => {} // surrogate: handled below
                        Neighbor::Boundary { tag } if !self.base.is_neumann(*tag) => {
                            let tau = self.base.penalty(e, None);
                            let (mut hx, mut hy) = (vec![0.0; nn], vec![0.0; nn]);
                            for a in 0..f.nodes.len() {
                                let v = f.nodes[a];
                                let (nx, ny, sw) = (f.nx[a], f.ny[a], f.sw[a]);
                                let dun = nx * gx[e][v] + ny * gy[e][v];
                                let uval = u[e * nn + v];
                                rec.push((e * nn + v, -sw * dun + tau * sw * uval));
                                hx[v] += sw * uval * nx;
                                hy[v] += sw * uval * ny;
                            }
                            let l = add(el.geom.gradx_t(refq, &hx), el.geom.grady_t(refq, &hy));
                            for k in 0..nn {
                                rec.push((e * nn + k, -l[k]));
                            }
                        }
                        _ => {}
                    }
                }
                // Surrogate faces owned by this element (Dirichlet/Nitsche only — for the
                // natural-Neumann pressure they contribute nothing, just the active restriction).
                for &fi in &self.sb.faces_by_elem[e] {
                    if !self.surrogate_dirichlet {
                        break;
                    }
                    let sf = &self.sb.faces[fi];
                    let fd = &el.faces[sf.edge as usize];
                    let tau = self.base.penalty(e, None);
                    let (mut hx, mut hy) = (vec![0.0; nn], vec![0.0; nn]);
                    let (mut px, mut py) = (vec![0.0; nn], vec![0.0; nn]);
                    for (a, sn) in sf.nodes.iter().enumerate() {
                        let v = fd.nodes[a];
                        let (nx, ny, sw) = (fd.nx[a], fd.ny[a], fd.sw[a]);
                        let dun = nx * gx[e][v] + ny * gy[e][v];
                        let su = u[e * nn + v]
                            + if self.taylor { gx[e][v] * sn.dx + gy[e][v] * sn.dy } else { 0.0 };
                        rec.push((e * nn + v, -sw * dun + tau * sw * su));
                        hx[v] += sw * su * nx;
                        hy[v] += sw * su * ny;
                        if self.taylor {
                            px[v] += tau * sw * su * sn.dx;
                            py[v] += tau * sw * su * sn.dy;
                        }
                    }
                    let l = add(el.geom.gradx_t(refq, &hx), el.geom.grady_t(refq, &hy));
                    for k in 0..nn {
                        rec.push((e * nn + k, -l[k]));
                    }
                    if self.taylor {
                        let lp = add(el.geom.gradx_t(refq, &px), el.geom.grady_t(refq, &py));
                        for k in 0..nn {
                            rec.push((e * nn + k, lp[k]));
                        }
                    }
                }
                rec
            })
            .collect();
        for rec in &records {
            for &(i, d) in rec {
                r[i] += d;
            }
        }
        if self.base.reaction != 0.0 {
            for (e, el) in m.elements.iter().enumerate() {
                if active[e] {
                    for k in 0..nn {
                        r[e * nn + k] += self.base.reaction * el.geom.jw[k] * u[e * nn + k];
                    }
                }
            }
        }
        // Identity on inactive dofs ⇒ they solve to 0, decoupled from the active block.
        for e in 0..m.n_elements() {
            if !active[e] {
                for k in 0..nn {
                    r[e * nn + k] = u[e * nn + k];
                }
            }
        }
        r
    }

    /// RHS for SBM with volume load `f` and Dirichlet data `g(x,y)` — applied at the
    /// **shifted true-boundary point** on the surrogate boundary, and at the wall nodes on
    /// outer Dirichlet walls.
    pub fn rhs(&self, f: &[f64], g: impl Fn(f64, f64) -> f64) -> Vec<f64> {
        let m = self.base.mesh;
        let refq = &m.refq;
        let nn = refq.n_nodes();
        let active = &self.sb.active;
        let mut b = vec![0.0; self.base.ndof()];
        for (e, el) in m.elements.iter().enumerate() {
            if active[e] {
                for k in 0..nn {
                    b[e * nn + k] += el.geom.jw[k] * f[e * nn + k];
                }
            }
        }
        let dir_data = |b: &mut [f64], e: usize, f: &crate::dg::FaceData, gvals: &[f64], tau: f64| {
            let el = &m.elements[e];
            let (mut hx, mut hy) = (vec![0.0; nn], vec![0.0; nn]);
            for a in 0..f.nodes.len() {
                let v = f.nodes[a];
                let (nx, ny, sw) = (f.nx[a], f.ny[a], f.sw[a]);
                b[e * nn + v] += tau * sw * gvals[a];
                hx[v] += sw * gvals[a] * nx;
                hy[v] += sw * gvals[a] * ny;
            }
            let l = add(el.geom.gradx_t(refq, &hx), el.geom.grady_t(refq, &hy));
            for k in 0..nn {
                b[e * nn + k] -= l[k];
            }
        };
        // Surrogate boundary: data ḡ = g(true point x̃+d). RHS terms −(ḡ,∂ₙv) (symmetry) and
        // +(γ/h)(ḡ, S_h v) (penalty value + Taylor penalty-gradient). Skipped for natural-Neumann
        // surrogate (homogeneous ∂u/∂n = 0 ⇒ no RHS contribution).
        for sf in self.sb.faces.iter().take(if self.surrogate_dirichlet { usize::MAX } else { 0 }) {
            let e = sf.elem;
            let el = &m.elements[e];
            let fd = &el.faces[sf.edge as usize];
            let tau = self.base.penalty(e, None);
            let (mut hx, mut hy) = (vec![0.0; nn], vec![0.0; nn]); // −(ḡ,∂ₙv)
            let (mut px, mut py) = (vec![0.0; nn], vec![0.0; nn]); // (γ/h)(ḡ)(∇v·d)
            for (a, sn) in sf.nodes.iter().enumerate() {
                let v = fd.nodes[a];
                let (nx, ny, sw) = (fd.nx[a], fd.ny[a], fd.sw[a]);
                let gv = g(sn.x + sn.dx, sn.y + sn.dy);
                b[e * nn + v] += tau * sw * gv; // penalty value
                hx[v] += sw * gv * nx;
                hy[v] += sw * gv * ny;
                if self.taylor {
                    px[v] += tau * sw * gv * sn.dx;
                    py[v] += tau * sw * gv * sn.dy;
                }
            }
            let l = add(el.geom.gradx_t(refq, &hx), el.geom.grady_t(refq, &hy));
            for k in 0..nn {
                b[e * nn + k] -= l[k];
            }
            if self.taylor {
                let lp = add(el.geom.gradx_t(refq, &px), el.geom.grady_t(refq, &py));
                for k in 0..nn {
                    b[e * nn + k] += lp[k];
                }
            }
        }
        // Outer Dirichlet walls of active elements.
        for (e, el) in m.elements.iter().enumerate() {
            if !active[e] {
                continue;
            }
            for edge in Edge::ALL {
                if let Neighbor::Boundary { tag } = el.neighbors[edge as usize] {
                    if self.base.is_neumann(tag) {
                        continue;
                    }
                    let fd = &el.faces[edge as usize];
                    let gvals: Vec<f64> = fd.nodes.iter().map(|&v| g(el.geom.x[v], el.geom.y[v])).collect();
                    dir_data(&mut b, e, fd, &gvals, self.base.penalty(e, None));
                }
            }
        }
        b
    }

    /// Plain CG from a zero initial guess.
    pub fn solve(&self, b: &[f64], tol: f64, maxit: usize) -> (Vec<f64>, usize) {
        self.solve_from(b, vec![0.0; b.len()], tol, maxit)
    }

    /// Plain CG from an initial guess `x0` (the active block is SPD). Warm-starting with the
    /// previous time step's field makes the iteration count collapse as the flow approaches
    /// steady state — essential for time-marching without a multigrid preconditioner.
    pub fn solve_from(&self, b: &[f64], x0: Vec<f64>, tol: f64, maxit: usize) -> (Vec<f64>, usize) {
        let n = b.len();
        let mut x = x0;
        let ax0 = self.apply(&x);
        let mut r: Vec<f64> = b.iter().zip(&ax0).map(|(bi, a)| bi - a).collect();
        let mut p = r.clone();
        let mut rs = dot(&r, &r);
        let bn = dot(b, b).sqrt().max(1e-300);
        for it in 0..maxit {
            let ap = self.apply(&p);
            let alpha = rs / dot(&p, &ap);
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            let rs_new = dot(&r, &r);
            if rs_new.sqrt() / bn < tol {
                return (x, it + 1);
            }
            let beta = rs_new / rs;
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
            rs = rs_new;
        }
        (x, maxit)
    }

    /// **Preconditioned** CG from an initial guess, with an external preconditioner `precond`
    /// (`r ↦ M⁻¹r`). The SBM operator is symmetric and (with an outflow Dirichlet) SPD but
    /// ill-conditioned — unpreconditioned CG needs O(10³) iters. A standard full-mesh
    /// `PMultigrid` V-cycle (same outer Neumann tags, ignoring the active mask/surrogate) is a
    /// good *approximate* preconditioner: it kills the global smooth mode the two operators
    /// share, reaching a loose projection tol in ~tens of mesh-independent iters. Pass e.g.
    /// `|r| mg.precondition(r)`. Returns `(solution, iterations)`.
    pub fn solve_pcg_from(
        &self, b: &[f64], x0: Vec<f64>, precond: impl Fn(&[f64]) -> Vec<f64>, tol: f64, maxit: usize,
    ) -> (Vec<f64>, usize) {
        let n = b.len();
        let mut x = x0;
        let ax0 = self.apply(&x);
        let mut r: Vec<f64> = b.iter().zip(&ax0).map(|(bi, a)| bi - a).collect();
        let mut z = precond(&r);
        let mut p = z.clone();
        let mut rz = dot(&r, &z);
        let bn = dot(b, b).sqrt().max(1e-300);
        for it in 0..maxit {
            let ap = self.apply(&p);
            let alpha = rz / dot(&p, &ap);
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            if dot(&r, &r).sqrt() / bn < tol {
                return (x, it + 1);
            }
            z = precond(&r);
            let rz_new = dot(&r, &z);
            let beta = rz_new / rz;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
            rz = rz_new;
        }
        (x, maxit)
    }

    /// The active-element mask (surrogate fluid domain), for restricting error norms etc.
    pub fn active(&self) -> &[bool] {
        &self.sb.active
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dg::shifted::{CircleLevelSet, ShiftedBoundary};

    /// SBM (shifted-boundary) Poisson, method of manufactured solutions: solve
    /// `−∇²u = f` on the fluid EXTERIOR of a circle embedded in a box, with the weak
    /// Dirichlet BC imposed on the surrogate boundary at the shifted true-circle value.
    /// The manufactured `u = cos(2x)·sin(3y)` gives `−∇²u = 13u`. The error (over the
    /// active surrogate domain) must be small and DECREASE under refinement — the proof
    /// that the embedded BC enforces the right solution. First-order SBM ⇒ ~O(h).
    #[test]
    fn sbm_poisson_manufactured_solution_converges() {
        // First-order SBM (no Taylor correction yet): the surrogate-Nitsche embedded BC must
        // enforce the manufactured solution to a few percent and IMPROVE with refinement.
        // It is only O(h) and geometry-noisy at coarse resolution (the penalty acts on u(x̃)
        // while the data is g(x̃+d) — an O(h) inconsistency), so it plateaus around a couple
        // percent here; clean high-order convergence is the Taylor-correction increment (2b).
        let uex = |x: f64, y: f64| (2.0 * x).cos() * (3.0 * y).sin();
        let ls = CircleLevelSet::new(0.5, 0.5, 0.2);
        let mut errs = Vec::new();
        for &n in &[12usize, 16, 24] {
            let mesh = Mesh2d::rectangular(3, n, n, [0.0, 1.0], [0.0, 1.0]);
            let nn = mesh.refq.n_nodes();
            let sb = ShiftedBoundary::new(&mesh, &ls);
            let sp = ShiftedPoisson::new(&mesh, 10.0, 200.0, sb);
            let active = sp.active().to_vec();
            // f = −∇²u_exact = 13 u_exact.
            let f: Vec<f64> = {
                let mut v = vec![0.0; mesh.n_elements() * nn];
                for (e, el) in mesh.elements.iter().enumerate() {
                    for k in 0..nn {
                        v[e * nn + k] = 213.0 * uex(el.geom.x[k], el.geom.y[k]);
                    }
                }
                v
            };
            let b = sp.rhs(&f, uex);
            let (u, _it) = sp.solve(&b, 1e-9, 30000);
            // L2 error over the ACTIVE (surrogate-fluid) elements only.
            let (mut num, mut den) = (0.0, 0.0);
            for (e, el) in mesh.elements.iter().enumerate() {
                if !active[e] {
                    continue;
                }
                for k in 0..nn {
                    let ue = uex(el.geom.x[k], el.geom.y[k]);
                    let d = u[e * nn + k] - ue;
                    num += el.geom.jw[k] * d * d;
                    den += el.geom.jw[k] * ue * ue;
                }
            }
            let err = (num / den.max(1e-300)).sqrt();
            eprintln!("n={n}: SBM-Poisson rel L2 error = {err:.3e}");
            assert!(err.is_finite() && err < 0.12, "error {err} too large at n={n}");
            errs.push(err);
        }
        // Refinement helps (the embedded BC is consistent) and the finest error is a few %.
        assert!(errs[1] < errs[0], "refinement must reduce the error: {errs:?}");
        assert!(*errs.last().unwrap() < 0.05, "finest first-order SBM error {} should be a few %", errs.last().unwrap());
    }

    /// High-order SBM (step 2b): the Taylor correction (`S_h u = u+∇u·d` in the surrogate
    /// Nitsche) must converge FASTER than first order and reach a far smaller error than the
    /// first-order plateau (~2.7%) — this is what makes SBM beat volume penalization. Checks
    /// the observed convergence rate is super-linear and the finest error is < 1e-3.
    #[test]
    fn sbm_poisson_taylor_correction_is_high_order() {
        let uex = |x: f64, y: f64| (2.0 * x).cos() * (3.0 * y).sin();
        let ls = CircleLevelSet::new(0.5, 0.5, 0.2);
        let ns = [12usize, 16]; // 2 points suffice; the unpreconditioned p=3 CPU solve is slow
        let mut errs = Vec::new();
        for &n in &ns {
            let mesh = Mesh2d::rectangular(3, n, n, [0.0, 1.0], [0.0, 1.0]);
            let nn = mesh.refq.n_nodes();
            let sb = ShiftedBoundary::new(&mesh, &ls);
            let sp = ShiftedPoisson::new(&mesh, 10.0, 200.0, sb).taylor(true);
            let active = sp.active().to_vec();
            let mut f = vec![0.0; mesh.n_elements() * nn];
            for (e, el) in mesh.elements.iter().enumerate() {
                for k in 0..nn {
                    f[e * nn + k] = 213.0 * uex(el.geom.x[k], el.geom.y[k]);
                }
            }
            let (u, _it) = sp.solve(&sp.rhs(&f, uex), 1e-10, 30000);
            let (mut num, mut den) = (0.0, 0.0);
            for (e, el) in mesh.elements.iter().enumerate() {
                if !active[e] {
                    continue;
                }
                for k in 0..nn {
                    let ue = uex(el.geom.x[k], el.geom.y[k]);
                    num += el.geom.jw[k] * (u[e * nn + k] - ue).powi(2);
                    den += el.geom.jw[k] * ue * ue;
                }
            }
            let err = (num / den.max(1e-300)).sqrt();
            eprintln!("n={n}: SBM-Taylor rel L2 error = {err:.3e}");
            errs.push(err);
        }
        // Observed order between the two meshes; high-order (>1.3) vs first-order's ~1.
        let order = (errs[0] / errs[1]).log2() / (ns[1] as f64 / ns[0] as f64).log2();
        eprintln!("observed convergence order ≈ {order:.2}  (errs {errs:?})");
        assert!(order > 1.3, "Taylor SBM should be high-order (vs first-order ~1), got {order:.2}");
        // Finest error must clearly beat both the first-order SBM plateau (~2.7%) and the
        // volume-penalization drag floor (~2%): a few × 1e-3, i.e. ~10× better.
        assert!(*errs.last().unwrap() < 5e-3, "finest Taylor error {} should be a few ×1e-3 (beats penalization ~2%)", errs.last().unwrap());
    }

    /// Sample a closure at every node into the global layout.
    fn nodal(mesh: &Mesh2d, func: impl Fn(f64, f64) -> f64) -> Vec<f64> {
        let nn = mesh.refq.n_nodes();
        let mut v = vec![0.0; mesh.n_elements() * nn];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                v[e * nn + k] = func(el.geom.x[k], el.geom.y[k]);
            }
        }
        v
    }

    #[test]
    fn operator_is_symmetric() {
        let mesh = Mesh2d::rectangular(3, 3, 2, [0.0, 1.5], [0.0, 1.0]);
        let a = Poisson::new(&mesh, 4.0);
        // Deterministic pseudo-random vectors.
        let mk = |seed: u64| -> Vec<f64> {
            let mut s = seed;
            (0..a.ndof())
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((s >> 33) as f64) / (1u64 << 31) as f64 - 1.0
                })
                .collect()
        };
        let (u, v) = (mk(1), mk(2));
        let uav = dot(&u, &a.apply(&v));
        let vau = dot(&v, &a.apply(&u));
        assert!((uav - vau).abs() < 1e-9 * (1.0 + uav.abs()), "uᵀAv={uav} vᵀAu={vau}");
    }

    #[test]
    fn spd_positive_on_nonzero() {
        let mesh = Mesh2d::rectangular(3, 2, 2, [0.0, 1.0], [0.0, 1.0]);
        let a = Poisson::new(&mesh, 5.0);
        let u = nodal(&mesh, |x, y| (x - 0.3) * (y + 0.7) + 0.21);
        let q = dot(&u, &a.apply(&u));
        assert!(q > 0.0, "uᵀAu = {q} not positive");
    }

    #[test]
    fn patch_test_exact_on_quadratic() {
        // u = x² + y² ⇒ −Δu = −4. Degree 2 ≤ p ⇒ the DG solution is exact.
        for p in 2..=4 {
            let mesh = Mesh2d::rectangular(p, 3, 2, [0.0, 1.5], [-1.0, 1.0]);
            let a = Poisson::new(&mesh, 5.0);
            let u_exact = nodal(&mesh, |x, y| x * x + y * y);
            let f = nodal(&mesh, |_, _| -4.0);
            let b = a.rhs(&f, |x, y| x * x + y * y);
            let (uh, _it, _res) = a.cg(&b, 1e-13, 5000);
            let err: Vec<f64> = uh.iter().zip(&u_exact).map(|(a, b)| a - b).collect();
            let e = a.l2_norm(&err);
            assert!(e < 1e-8, "p={p} patch-test L2 error {e}");
        }
    }

    #[test]
    fn patch_test_exact_on_harmonic_cubic() {
        // u = x³ − 3xy² is harmonic ⇒ f = 0. Exact for p ≥ 3.
        for p in 3..=4 {
            let mesh = Mesh2d::rectangular(p, 2, 3, [-0.5, 1.0], [0.0, 1.2]);
            let a = Poisson::new(&mesh, 5.0);
            let exact = |x: f64, y: f64| x * x * x - 3.0 * x * y * y;
            let u_exact = nodal(&mesh, exact);
            let f = vec![0.0; a.ndof()];
            let b = a.rhs(&f, exact);
            let (uh, _it, _res) = a.cg(&b, 1e-13, 5000);
            let err: Vec<f64> = uh.iter().zip(&u_exact).map(|(a, b)| a - b).collect();
            assert!(a.l2_norm(&err) < 1e-7, "p={p} harmonic patch error {}", a.l2_norm(&err));
        }
    }

    /// Mass-weighted mean of a nodal field.
    fn mass_mean(mesh: &Mesh2d, v: &[f64]) -> f64 {
        let nn = mesh.refq.n_nodes();
        let mut s = 0.0;
        let mut w = 0.0;
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                s += el.geom.jw[k] * v[e * nn + k];
                w += el.geom.jw[k];
            }
        }
        s / w
    }

    #[test]
    fn neumann_poisson_converges_up_to_a_constant() {
        // u = cos(πx)cos(πy) on [0,1]² has homogeneous Neumann on all sides;
        // −∇²u = 2π²u = f, ∂u/∂n = 0. Pure-Neumann ⇒ singular; deflated CG + shift.
        use std::f64::consts::PI;
        let p = 4;
        let exact = |x: f64, y: f64| (PI * x).cos() * (PI * y).cos();
        let mut errs = Vec::new();
        for &nx in &[2usize, 4, 8] {
            let mesh = Mesh2d::rectangular(p, nx, nx, [0.0, 1.0], [0.0, 1.0]);
            // All four box boundaries (tags 0..3) are Neumann.
            let a = Poisson::with_bc(&mesh, 5.0, 0.0, vec![0, 1, 2, 3]);
            let f = nodal(&mesh, |x, y| 2.0 * PI * PI * exact(x, y));
            let b = a.rhs_mixed(&f, |_, _| 0.0, |_, _| 0.0); // homogeneous Neumann flux
            let (mut uh, _it) = a.cg_deflated(&b, 1e-11, 20000);
            let u_exact = nodal(&mesh, exact);
            // Fix the constant: match the exact solution's mass-weighted mean.
            let shift = mass_mean(&mesh, &u_exact) - mass_mean(&mesh, &uh);
            uh.iter_mut().for_each(|x| *x += shift);
            let err: Vec<f64> = uh.iter().zip(&u_exact).map(|(a, b)| a - b).collect();
            errs.push(a.l2_norm(&err));
        }
        assert!(errs[1] < errs[0] && errs[2] < errs[1], "Neumann not converging: {errs:?}");
        assert!(errs[2] < 1e-4, "final Neumann error {}", errs[2]);
    }

    #[test]
    fn helmholtz_patch_test_exact_on_quadratic() {
        // (λ − ∇²)u = f with u = x²+y², −∇²u = −4 ⇒ f = λ(x²+y²) − 4. Exact for p≥2.
        let lambda = 12.0;
        for p in 2..=4 {
            let mesh = Mesh2d::rectangular(p, 3, 2, [0.0, 1.5], [-1.0, 1.0]);
            let a = Poisson::with_reaction(&mesh, 5.0, lambda);
            let u_exact = nodal(&mesh, |x, y| x * x + y * y);
            let f = nodal(&mesh, |x, y| lambda * (x * x + y * y) - 4.0);
            let b = a.rhs(&f, |x, y| x * x + y * y);
            let (uh, _it, _res) = a.cg(&b, 1e-13, 5000);
            let err: Vec<f64> = uh.iter().zip(&u_exact).map(|(a, b)| a - b).collect();
            assert!(a.l2_norm(&err) < 1e-8, "p={p} Helmholtz patch error {}", a.l2_norm(&err));
        }
    }

    #[test]
    fn helmholtz_converges_and_is_spd() {
        // (λ − ∇²)u = (λ + 2π²)u for u = sin πx sin πy on [0,1]², homogeneous Dirichlet.
        use std::f64::consts::PI;
        let lambda = 50.0; // ~ 1/(νΔt) scale of the viscous solve
        let p = 4;
        let exact = |x: f64, y: f64| (PI * x).sin() * (PI * y).sin();
        let mut errs = Vec::new();
        for &nx in &[2usize, 4, 8] {
            let mesh = Mesh2d::rectangular(p, nx, nx, [0.0, 1.0], [0.0, 1.0]);
            let a = Poisson::with_reaction(&mesh, 5.0, lambda);
            let f = nodal(&mesh, |x, y| (lambda + 2.0 * PI * PI) * exact(x, y));
            let b = a.rhs(&f, |_, _| 0.0);
            // SPD: a few CG iters must not break down (denominators stay positive).
            let (uh, _it, _res) = a.cg(&b, 1e-12, 20000);
            let u_exact = nodal(&mesh, exact);
            let err: Vec<f64> = uh.iter().zip(&u_exact).map(|(a, b)| a - b).collect();
            errs.push(a.l2_norm(&err));
        }
        assert!(errs[1] < errs[0] && errs[2] < errs[1], "Helmholtz not converging: {errs:?}");
        assert!(errs[2] < 1e-5, "final Helmholtz error {}", errs[2]);
    }

    #[test]
    fn manufactured_solution_converges_at_high_order() {
        // u = sin(πx) sin(πy) on [0,1]² (homogeneous Dirichlet), −Δu = 2π² u.
        use std::f64::consts::PI;
        let p = 4;
        let exact = |x: f64, y: f64| (PI * x).sin() * (PI * y).sin();
        let rhs_f = |x: f64, y: f64| 2.0 * PI * PI * (PI * x).sin() * (PI * y).sin();

        let mut errs = Vec::new();
        for &nx in &[2usize, 4, 8] {
            let mesh = Mesh2d::rectangular(p, nx, nx, [0.0, 1.0], [0.0, 1.0]);
            let a = Poisson::new(&mesh, 5.0);
            let f = nodal(&mesh, rhs_f);
            let b = a.rhs(&f, |_, _| 0.0);
            let (uh, _it, _res) = a.cg(&b, 1e-12, 20000);
            let u_exact = nodal(&mesh, exact);
            let err: Vec<f64> = uh.iter().zip(&u_exact).map(|(a, b)| a - b).collect();
            errs.push(a.l2_norm(&err));
        }
        // Errors decrease, and the observed h-rate is high-order (≥ p, expecting ~p+1).
        assert!(errs[1] < errs[0] && errs[2] < errs[1], "errors not decreasing: {errs:?}");
        let rate = (errs[1] / errs[2]).log2();
        assert!(rate >= p as f64, "observed order {rate} (errs {errs:?})");
        assert!(errs[2] < 1e-5, "final error {} too large", errs[2]);
    }

    #[test]
    fn nonconforming_poisson_is_symmetric() {
        // The decisive guard for the mortar SIPG: A must stay symmetric on a refined
        // mesh (else CG fails). Check ⟨Au,v⟩ = ⟨Av,u⟩ for pseudo-random u,v.
        let p = 4;
        let mesh = Mesh2d::cartesian_refined(p, 3, 3, [0.0, 1.0], [0.0, 1.0], &[(1, 1), (2, 0)]);
        let a = Poisson::new(&mesh, 5.0);
        let ndof = a.ndof();
        let u: Vec<f64> = (0..ndof).map(|i| (((i * 7 + 3) % 13) as f64) * 0.1 - 0.6).collect();
        let v: Vec<f64> = (0..ndof).map(|i| (((i * 5 + 2) % 11) as f64) * 0.1 - 0.5).collect();
        let (au, av) = (a.apply(&u), a.apply(&v));
        let uav: f64 = u.iter().zip(&av).map(|(x, y)| x * y).sum();
        let vau: f64 = v.iter().zip(&au).map(|(x, y)| x * y).sum();
        assert!((uav - vau).abs() < 1e-9 * (uav.abs() + 1.0), "SIPG not symmetric on refined mesh: {uav} vs {vau}");
    }

    #[test]
    fn nonconforming_poisson_mms() {
        // Manufactured solve on a mesh with refined patches: the non-conforming SIPG
        // recovers u = sin(πx)sin(πy) accurately (consistent across hanging nodes).
        use std::f64::consts::PI;
        let p = 4;
        let exact = |x: f64, y: f64| (PI * x).sin() * (PI * y).sin();
        let rhs_f = |x: f64, y: f64| 2.0 * PI * PI * (PI * x).sin() * (PI * y).sin();
        let mesh = Mesh2d::cartesian_refined(p, 4, 4, [0.0, 1.0], [0.0, 1.0], &[(1, 1), (2, 2)]);
        let a = Poisson::new(&mesh, 5.0);
        let f = nodal(&mesh, rhs_f);
        let b = a.rhs(&f, |_, _| 0.0);
        let (uh, _it, _res) = a.cg(&b, 1e-12, 20000);
        let ue = nodal(&mesh, exact);
        let err: Vec<f64> = uh.iter().zip(&ue).map(|(a, b)| a - b).collect();
        let e = a.l2_norm(&err);
        eprintln!("non-conforming Poisson MMS L2 error = {e:.3e}");
        assert!(e < 1e-4, "non-conforming MMS error too large: {e}");
    }

    #[test]
    fn mixed_dirichlet_neumann_by_region() {
        // Per-region BC dispatch via `rhs_tagged`: u = x² + y² (∇²u = 4 ⇒ f = −∇²u = −4)
        // on [0,1]², with Dirichlet data on west/east (tags 3,1) and Neumann flux on
        // south/north (tags 0,2). The degree-2 exact solution is recovered to round-off,
        // proving both the operator's per-tag dispatch and the tagged RHS data.
        let p = 4;
        let mesh = Mesh2d::rectangular(p, 3, 3, [0.0, 1.0], [0.0, 1.0]);
        let exact = |x: f64, y: f64| x * x + y * y;
        let a = Poisson::with_bc(&mesh, 5.0, 0.0, vec![0, 2]); // Neumann: south, north
        let f = nodal(&mesh, |_, _| -4.0);
        // Dirichlet g = exact (used only on tags 1,3). Neumann flux q = ∂u/∂n:
        // south n=(0,−1) ⇒ −∂u/∂y = −2y (= 0 at y=0); north n=(0,1) ⇒ ∂u/∂y = 2y (= 2 at y=1).
        let b = a.rhs_tagged(
            &f,
            |_tag, x, y| exact(x, y),
            |tag, _x, y| if tag == 2 { 2.0 * y } else { -2.0 * y },
        );
        let (uh, _it, _res) = a.cg(&b, 1e-12, 20000);
        let ue = nodal(&mesh, exact);
        let err: Vec<f64> = uh.iter().zip(&ue).map(|(a, b)| a - b).collect();
        let e = a.l2_norm(&err);
        eprintln!("mixed Dirichlet/Neumann-by-region Poisson L2 error = {e:.3e}");
        assert!(e < 1e-10, "per-region BC dispatch inaccurate: {e}");
    }
}
