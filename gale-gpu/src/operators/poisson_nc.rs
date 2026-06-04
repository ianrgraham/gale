//! GPU SIPG-Poisson operator-apply on **2:1 non-conforming (mortar)** quad meshes.
//!
//! This is the AMR companion to the conforming [`super::poisson`] operator: it
//! evaluates the matrix-free SIPG Laplacian `A·u` on a mesh that contains hanging
//! nodes / 2:1 mortar interfaces ([`Neighbor::CoarseToFine`] /
//! [`Neighbor::FineToCoarse`]), with the **same** volume stiffness, symmetry-lift,
//! and consistency/penalty structure as the conforming kernel. Conforming, Dirichlet
//! (`BND`), and Neumann (`NEU`) faces take the identical code path as
//! [`super::poisson`], so a purely-conforming mesh is bit-for-bit unchanged.
//!
//! The device math is a 2-kernel pipeline (`gradient_nc` → `operator_nc`), one block
//! per element, evaluated as a **per-element gather**: block `e` computes `r[e]`,
//! reading its neighbors read-only (each block writes only its own `r`). The mortar
//! coupling is done with the two 1D projection matrices `P0`/`P1`
//! (`RefineQuad::mortar_to_fine`) and their transpose `Pᵀ`
//! (`RefineQuad::mortar_gather`), reconstructed host-side as columns of
//! `mortar_to_fine(eⱼ, half)`.
//!
//! Validated bit-for-bit (max rel < 1e-12) against `gale::dg::Poisson::apply` on a
//! refined mesh by `bin/poisson_nc_check.rs`.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;
use gale::dg::{Edge, Mesh2d, Neighbor, RefineQuad};

const NN_MAX: usize = 81; // (order 8 + 1)²

// Per-edge face-kind tags (`ekind[e*4+t]`).
const K_CONF: u32 = 0; // conforming Interior / Dirichlet(BND) / Neumann(NEU) — `fnbr` path
const K_FINE_TO_COARSE: u32 = 1; // this element is FINE; one coarse neighbor (nbr0), `half0`
const K_COARSE_TO_FINE: u32 = 2; // this element is COARSE; two fine neighbors (nbr0,nbr1)

// Conforming-face neighbor sentinels (same meaning as `super::poisson`).
const BND: u32 = u32::MAX; // Dirichlet boundary face (SIPG consistency+penalty)
const NEU: u32 = u32::MAX - 1; // Neumann boundary face (natural BC ⇒ no contribution)

#[cuda_module]
mod kernels {
    use super::*;

    /// Per-element physical gradient `(gx, gy)` of `u` (identical to the conforming
    /// `gradient`, renamed for the unique crate-wide export requirement).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn gradient_nc(
        d: &[f64], u: &[f64], rx: &[f64], ry: &[f64], sx: &[f64], sy: &[f64], n1: u32,
        mut gx: DisjointSlice<f64>, mut gy: DisjointSlice<f64>,
    ) {
        static mut DS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut US: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        let n1 = n1 as usize;
        let nn = n1 * n1;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let base = e * nn;
        unsafe {
            DS[m] = d[m];
            US[m] = u[base + m];
        }
        thread::sync_threads();
        let i = m % n1;
        let j = m / n1;
        let mut ur = 0.0f64;
        let mut us = 0.0f64;
        let mut k = 0usize;
        while k < n1 {
            unsafe {
                ur += DS[i * n1 + k] * US[k + j * n1];
                us += DS[j * n1 + k] * US[i + k * n1];
            }
            k += 1;
        }
        let gxv = rx[base + m] * ur + sx[base + m] * us;
        let gyv = ry[base + m] * ur + sy[base + m] * us;
        if let Some(o) = gx.get_mut(thread::index_1d()) {
            *o = gxv;
        }
        if let Some(o) = gy.get_mut(thread::index_1d()) {
            *o = gyv;
        }
    }

    /// Non-conforming SIPG operator action `A·u` (+ optional Helmholtz reaction
    /// `λM·u`). One block per element computes `r[e]`. The face loop (single thread,
    /// `m==0`) branches on the per-edge kind tag:
    /// - `K_CONF`: identical to the conforming `operator` (`fnbr` gather, BND/NEU).
    /// - `K_FINE_TO_COARSE`: standard gather with the coarse trace projected to E's
    ///   fine face nodes via `P_half`, using E's own normal.
    /// - `K_COARSE_TO_FINE`: integrate on each fine mortar half and `Pᵀ`-gather back
    ///   to E's coarse face nodes, using E's own normal.
    /// Then: symmetry lift into `pr/ps`, volume sum-fac, reaction.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn operator_nc(
        d: &[f64], u: &[f64], gx: &[f64], gy: &[f64], rx: &[f64], ry: &[f64], sx: &[f64],
        sy: &[f64], jw: &[f64], n1: u32,
        // conforming-face data (natural face order), `(e*4+t)*n1 + a`:
        face_vl: &[u32], face_nx: &[f64], face_ny: &[f64], face_sw: &[f64], face_nbr: &[u32],
        // per-edge metadata:
        ekind: &[u32], enx: &[f64], eny: &[f64], etau: &[f64], half0: &[u32],
        // NC face data (sorted order), `(e*4+t)*n1 + k`:
        self_sorted: &[u32], self_sw: &[f64], nbr0_sorted: &[u32], nbr1_sorted: &[u32],
        swf0: &[f64], swf1: &[f64],
        // mortar projection matrices (n1*n1 each), `P[i*n1+j]`:
        p0: &[f64], p1: &[f64],
        lambda: f64, mut out: DisjointSlice<f64>,
    ) {
        static mut DS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PR: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut RF: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut HX: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut HY: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        let n1 = n1 as usize;
        let nn = n1 * n1;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let b = e * nn + m;
        unsafe {
            DS[m] = d[m];
            RF[m] = 0.0;
            HX[m] = 0.0;
            HY[m] = 0.0;
            let wx = jw[b] * gx[b];
            let wy = jw[b] * gy[b];
            PR[m] = rx[b] * wx + ry[b] * wy;
            PS[m] = sx[b] * wx + sy[b] * wy;
        }
        thread::sync_threads();
        if m == 0 {
            let mut t = 0usize;
            while t < 4 {
                let et = e * 4 + t;
                let kind = ekind[et];
                let tau = etau[et];
                if kind == K_CONF {
                    // ---- conforming Interior / Dirichlet(BND) / Neumann(NEU) ----
                    // Identical to the conforming `operator`: per face node `a` in the
                    // face's natural order, `face_nbr` gives the neighbor global node.
                    let mut a = 0usize;
                    while a < n1 {
                        let idx = et * n1 + a;
                        let vl = face_vl[idx] as usize;
                        let nx = face_nx[idx];
                        let ny = face_ny[idx];
                        let sw = face_sw[idx];
                        let nbr = face_nbr[idx];
                        if nbr == NEU {
                            a += 1;
                            continue;
                        }
                        let dun_e = nx * gx[e * nn + vl] + ny * gy[e * nn + vl];
                        let ug = u[e * nn + vl];
                        let (avg, jump, gfac) = if nbr == BND {
                            (dun_e, ug, 1.0)
                        } else {
                            let ng = nbr as usize;
                            (0.5 * (dun_e + nx * gx[ng] + ny * gy[ng]), ug - u[ng], 0.5)
                        };
                        let g = gfac * sw * jump;
                        unsafe {
                            RF[vl] += -sw * avg + tau * sw * jump;
                            HX[vl] += g * nx;
                            HY[vl] += g * ny;
                        }
                        a += 1;
                    }
                } else if kind == K_FINE_TO_COARSE {
                    // ---- this element E is FINE; one coarse neighbor C ----
                    // Standard gather, but the neighbor trace comes from C projected
                    // to E's fine face nodes via `P_half`. E uses its OWN normal.
                    let fnx = enx[et];
                    let fny = eny[et];
                    let half = half0[et];
                    // Projected coarse trace at each of E's fine sorted nodes i:
                    //   u_nbr_i  = Σ_k P_half[i,k] * uc[k]
                    //   dun_nbr_i= Σ_k P_half[i,k] * dnc[k]   (E's normal on C's grad)
                    let mut i = 0usize;
                    while i < n1 {
                        let si = self_sorted[et * n1 + i] as usize;
                        let sw = self_sw[et * n1 + i];
                        // project coarse trace onto fine node i.
                        let mut u_nbr = 0.0f64;
                        let mut dun_nbr = 0.0f64;
                        let mut k = 0usize;
                        while k < n1 {
                            let ck = nbr0_sorted[et * n1 + k] as usize;
                            let pw = if half == 0 { p0[i * n1 + k] } else { p1[i * n1 + k] };
                            u_nbr += pw * u[ck];
                            dun_nbr += pw * (fnx * gx[ck] + fny * gy[ck]);
                            k += 1;
                        }
                        let u_self = u[e * nn + si];
                        let dun_self = fnx * gx[e * nn + si] + fny * gy[e * nn + si];
                        let avg = 0.5 * (dun_self + dun_nbr);
                        let jump = u_self - u_nbr;
                        let g = 0.5 * sw * jump;
                        unsafe {
                            RF[si] += -sw * avg + tau * sw * jump;
                            HX[si] += g * fnx;
                            HY[si] += g * fny;
                        }
                        i += 1;
                    }
                } else {
                    // ---- this element E is COARSE; two fine neighbors (halves 0,1) ----
                    // Integrate on each fine mortar and `Pᵀ`-gather to E's coarse nodes.
                    // E uses its OWN normal.
                    let fnx = enx[et];
                    let fny = eny[et];
                    let mut h = 0usize;
                    while h < 2 {
                        // For coarse node j we accumulate Σ_i P_h[i,j]·(gc_i, gl_i).
                        let mut j = 0usize;
                        while j < n1 {
                            let cj = self_sorted[et * n1 + j] as usize;
                            let mut rf_acc = 0.0f64;
                            let mut hl_acc = 0.0f64;
                            let mut i = 0usize;
                            while i < n1 {
                                // projected coarse value/flux at fine node i.
                                let mut ucp = 0.0f64;
                                let mut dncp = 0.0f64;
                                let mut k = 0usize;
                                while k < n1 {
                                    let ck = self_sorted[et * n1 + k] as usize;
                                    let pw = if h == 0 { p0[i * n1 + k] } else { p1[i * n1 + k] };
                                    ucp += pw * u[e * nn + ck];
                                    dncp += pw * (fnx * gx[e * nn + ck] + fny * gy[e * nn + ck]);
                                    k += 1;
                                }
                                // fine values.
                                let (fk, swf) = if h == 0 {
                                    (nbr0_sorted[et * n1 + i] as usize, swf0[et * n1 + i])
                                } else {
                                    (nbr1_sorted[et * n1 + i] as usize, swf1[et * n1 + i])
                                };
                                let uf = u[fk];
                                let dunf = fnx * gx[fk] + fny * gy[fk];
                                let jump = ucp - uf;
                                let avg = 0.5 * (dncp + dunf);
                                let gc = -swf * avg + tau * swf * jump;
                                let gl = 0.5 * swf * jump;
                                let pw = if h == 0 { p0[i * n1 + j] } else { p1[i * n1 + j] };
                                rf_acc += pw * gc;
                                hl_acc += pw * gl;
                                i += 1;
                            }
                            unsafe {
                                RF[cj] += rf_acc;
                                HX[cj] += hl_acc * fnx;
                                HY[cj] += hl_acc * fny;
                            }
                            j += 1;
                        }
                        h += 1;
                    }
                }
                t += 1;
            }
        }
        thread::sync_threads();
        let hxm = unsafe { HX[m] };
        let hym = unsafe { HY[m] };
        unsafe {
            PR[m] -= rx[b] * hxm + ry[b] * hym;
            PS[m] -= sx[b] * hxm + sy[b] * hym;
        }
        thread::sync_threads();
        let i = m % n1;
        let j = m / n1;
        let mut acc = 0.0f64;
        let mut k = 0usize;
        while k < n1 {
            unsafe {
                acc += DS[k * n1 + i] * PR[k + j * n1] + DS[k * n1 + j] * PS[i + k * n1];
            }
            k += 1;
        }
        let rfm = unsafe { RF[m] };
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = acc + rfm + lambda * jw[b] * u[b];
        }
    }
}

/// Per-element metrics + flattened (conforming + non-conforming) face metadata,
/// ready to upload. Extends the conforming `MeshArrays` with the NC arrays.
struct NcArrays {
    nn: usize,
    ne: usize,
    ndof: usize,
    n1: u32,
    diff: Vec<f64>,
    rx: Vec<f64>,
    ry: Vec<f64>,
    sx: Vec<f64>,
    sy: Vec<f64>,
    jw: Vec<f64>,
    // conforming face data (natural order), `(e*4+t)*n1 + a`:
    fvl: Vec<u32>,
    fnx: Vec<f64>,
    fny: Vec<f64>,
    fsw: Vec<f64>,
    fnbr: Vec<u32>,
    // per-edge metadata, `e*4+t`:
    ekind: Vec<u32>,
    enx: Vec<f64>,
    eny: Vec<f64>,
    etau: Vec<f64>,
    half0: Vec<u32>,
    // NC face data (sorted order), `(e*4+t)*n1 + k`:
    self_sorted: Vec<u32>,
    self_sw: Vec<f64>,
    nbr0_sorted: Vec<u32>,
    nbr1_sorted: Vec<u32>,
    swf0: Vec<f64>,
    swf1: Vec<f64>,
    // mortar projection matrices (n1*n1 each):
    p0: Vec<f64>,
    p1: Vec<f64>,
}

/// Face-node indices of element `e`'s `edge`, sorted by physical coordinate along the
/// edge (y for East/West, x for North/South) — the SAME ordering as the CPU oracle's
/// `sorted(e,edge)`. Returns positions into `face.nodes` (apply `face.nodes[pos]` for
/// the volume index).
fn sorted_positions(mesh: &Mesh2d, e: usize, edge: Edge) -> Vec<usize> {
    let f = &mesh.elements[e].faces[edge as usize];
    let g = &mesh.elements[e].geom;
    let vert = matches!(edge, Edge::East | Edge::West);
    let mut idx: Vec<usize> = (0..f.nodes.len()).collect();
    idx.sort_by(|&a, &b| {
        let ca = if vert { g.y[f.nodes[a]] } else { g.x[f.nodes[a]] };
        let cb = if vert { g.y[f.nodes[b]] } else { g.x[f.nodes[b]] };
        ca.partial_cmp(&cb).unwrap()
    });
    idx
}

/// Flatten a (possibly non-conforming) mesh: per-element metrics, the conforming
/// SIPG face metadata (matching `super::poisson::flatten_mesh` for `K_CONF` edges),
/// and the mortar metadata for the `CoarseToFine`/`FineToCoarse` edges. Penalty `tau`
/// uses `α(p+1)²/min(h_self, h_nbr)` with `h = √Σjw`; for NC faces the min is taken
/// over the coarse and the relevant fine element(s).
fn flatten_nc(mesh: &Mesh2d, alpha: f64) -> NcArrays {
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let order = mesh.order;
    let n1 = (order + 1) as u32;
    let n1u = n1 as usize;
    assert!(nn <= NN_MAX, "p too large for NN_MAX={NN_MAX}");

    let (mut rx, mut ry, mut sx, mut sy, mut jw) =
        (vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof]);
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            rx[e * nn + k] = el.geom.rx[k];
            ry[e * nn + k] = el.geom.ry[k];
            sx[e * nn + k] = el.geom.sx[k];
            sy[e * nn + k] = el.geom.sy[k];
            jw[e * nn + k] = el.geom.jw[k];
        }
    }

    let p1f = (order + 1) as f64;
    let h: Vec<f64> = mesh.elements.iter().map(|el| el.geom.jw.iter().sum::<f64>().sqrt()).collect();

    // Mortar matrices P0, P1 reconstructed as columns of mortar_to_fine(e_j, half):
    //   P[i*n1+j] = mortar_to_fine(e_j, half)[i].
    let mortar = RefineQuad::new(order);
    let mut p0 = vec![0.0; n1u * n1u];
    let mut p1 = vec![0.0; n1u * n1u];
    for j in 0..n1u {
        let mut ej = vec![0.0; n1u];
        ej[j] = 1.0;
        let col0 = mortar.mortar_to_fine(&ej, 0);
        let col1 = mortar.mortar_to_fine(&ej, 1);
        for i in 0..n1u {
            p0[i * n1u + j] = col0[i];
            p1[i * n1u + j] = col1[i];
        }
    }

    let nfc = ne * 4 * n1u;
    let (mut fvl, mut fnx, mut fny, mut fsw, mut fnbr) =
        (vec![0u32; nfc], vec![0.0; nfc], vec![0.0; nfc], vec![0.0; nfc], vec![BND; nfc]);
    let (mut ekind, mut enx, mut eny, mut etau, mut half0) =
        (vec![K_CONF; ne * 4], vec![0.0; ne * 4], vec![0.0; ne * 4], vec![0.0; ne * 4], vec![0u32; ne * 4]);
    let (mut self_sorted, mut self_sw, mut nbr0_sorted, mut nbr1_sorted, mut swf0, mut swf1) = (
        vec![0u32; nfc], vec![0.0; nfc], vec![0u32; nfc], vec![0u32; nfc], vec![0.0; nfc], vec![0.0; nfc],
    );

    for (e, el) in mesh.elements.iter().enumerate() {
        for (t, edge) in Edge::ALL.iter().enumerate() {
            let et = e * 4 + t;
            let face = &el.faces[*edge as usize];
            let nb = &el.neighbors[*edge as usize];
            match nb {
                Neighbor::Interior { elem: re, edge: redge, perm } => {
                    ekind[et] = K_CONF;
                    etau[et] = alpha * p1f * p1f / h[e].min(h[*re]);
                    let rf = &mesh.elements[*re].faces[*redge as usize];
                    for a in 0..n1u {
                        let idx = et * n1u + a;
                        fvl[idx] = face.nodes[a] as u32;
                        fnx[idx] = face.nx[a];
                        fny[idx] = face.ny[a];
                        fsw[idx] = face.sw[a];
                        fnbr[idx] = (*re * nn + rf.nodes[perm[a]]) as u32;
                    }
                }
                Neighbor::Boundary { .. } => {
                    // All-Dirichlet (BND). (Neumann tagging not needed for the apply
                    // validation; the check uses a default all-Dirichlet operator.)
                    ekind[et] = K_CONF;
                    etau[et] = alpha * p1f * p1f / h[e];
                    for a in 0..n1u {
                        let idx = et * n1u + a;
                        fvl[idx] = face.nodes[a] as u32;
                        fnx[idx] = face.nx[a];
                        fny[idx] = face.ny[a];
                        fsw[idx] = face.sw[a];
                        fnbr[idx] = BND;
                    }
                }
                Neighbor::FineToCoarse { coarse, edge: cedge, half } => {
                    // E (this element) is FINE; one coarse neighbor C.
                    ekind[et] = K_FINE_TO_COARSE;
                    etau[et] = alpha * p1f * p1f / h[e].min(h[*coarse]);
                    half0[et] = *half as u32;
                    // E uses its own (fine) outward normal — constant per Cartesian
                    // face; take the value at the first sorted node.
                    let es = sorted_positions(mesh, e, *edge);
                    enx[et] = face.nx[es[0]];
                    eny[et] = face.ny[es[0]];
                    // E's sorted face nodes (global) + sw.
                    for k in 0..n1u {
                        let pos = es[k];
                        self_sorted[et * n1u + k] = face.nodes[pos] as u32; // LOCAL node idx (0..nn)
                        self_sw[et * n1u + k] = face.sw[pos];
                    }
                    // Coarse C's sorted face nodes (global).
                    let cs = sorted_positions(mesh, *coarse, *cedge);
                    let cf = &mesh.elements[*coarse].faces[*cedge as usize];
                    for k in 0..n1u {
                        nbr0_sorted[et * n1u + k] = (*coarse * nn + cf.nodes[cs[k]]) as u32;
                    }
                }
                Neighbor::CoarseToFine { fine } => {
                    // E (this element) is COARSE; two fine neighbors (halves 0,1).
                    ekind[et] = K_COARSE_TO_FINE;
                    let (re0, _) = fine[0];
                    let (re1, _) = fine[1];
                    etau[et] = alpha * p1f * p1f / h[e].min(h[re0]).min(h[re1]);
                    // E uses its own (coarse) outward normal.
                    let cs = sorted_positions(mesh, e, *edge);
                    enx[et] = face.nx[cs[0]];
                    eny[et] = face.ny[cs[0]];
                    // E's coarse sorted face nodes (global).
                    for k in 0..n1u {
                        let pos = cs[k];
                        self_sorted[et * n1u + k] = face.nodes[pos] as u32; // LOCAL node idx (0..nn)
                    }
                    // Each fine neighbor's sorted face nodes (global) + fine sw.
                    let (rf0e, rf0edge) = fine[0];
                    let (rf1e, rf1edge) = fine[1];
                    let fs0 = sorted_positions(mesh, rf0e, rf0edge);
                    let fs1 = sorted_positions(mesh, rf1e, rf1edge);
                    let ff0 = &mesh.elements[rf0e].faces[rf0edge as usize];
                    let ff1 = &mesh.elements[rf1e].faces[rf1edge as usize];
                    for k in 0..n1u {
                        nbr0_sorted[et * n1u + k] = (rf0e * nn + ff0.nodes[fs0[k]]) as u32;
                        nbr1_sorted[et * n1u + k] = (rf1e * nn + ff1.nodes[fs1[k]]) as u32;
                        swf0[et * n1u + k] = ff0.sw[fs0[k]];
                        swf1[et * n1u + k] = ff1.sw[fs1[k]];
                    }
                }
            }
        }
    }

    NcArrays {
        nn,
        ne,
        ndof,
        n1,
        diff: mesh.refq.line.diff.clone(),
        rx,
        ry,
        sx,
        sy,
        jw,
        fvl,
        fnx,
        fny,
        fsw,
        fnbr,
        ekind,
        enx,
        eny,
        etau,
        half0,
        self_sorted,
        self_sw,
        nbr0_sorted,
        nbr1_sorted,
        swf0,
        swf1,
        p0,
        p1,
    }
}

/// Apply the matrix-free SIPG Poisson (+ optional Helmholtz reaction `λ`) operator
/// `A·u` once on the GPU for a **2:1 non-conforming** mesh, returning the nodal
/// result. Conforming meshes are handled by the identical conforming path, so this
/// is a strict superset of [`super::poisson::poisson_apply`]. `u` must have length
/// `n_elements · n_nodes`. Validated bit-for-bit against `gale::dg::Poisson::apply`.
pub fn poisson_nc_apply(
    mesh: &Mesh2d,
    u: &[f64],
    alpha: f64,
    reaction: f64,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let ma = flatten_nc(mesh, alpha);
    assert_eq!(u.len(), ma.ndof, "state length must be n_elements·n_nodes");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
    let upu = |v: &[u32]| DeviceBuffer::from_host(&stream, v);

    let d_dev = up(&ma.diff)?;
    let u_dev = up(u)?;
    let rx_dev = up(&ma.rx)?;
    let ry_dev = up(&ma.ry)?;
    let sx_dev = up(&ma.sx)?;
    let sy_dev = up(&ma.sy)?;
    let jw_dev = up(&ma.jw)?;
    let fvl_dev = upu(&ma.fvl)?;
    let fnx_dev = up(&ma.fnx)?;
    let fny_dev = up(&ma.fny)?;
    let fsw_dev = up(&ma.fsw)?;
    let fnbr_dev = upu(&ma.fnbr)?;
    let ekind_dev = upu(&ma.ekind)?;
    let enx_dev = up(&ma.enx)?;
    let eny_dev = up(&ma.eny)?;
    let etau_dev = up(&ma.etau)?;
    let half0_dev = upu(&ma.half0)?;
    let self_sorted_dev = upu(&ma.self_sorted)?;
    let self_sw_dev = up(&ma.self_sw)?;
    let nbr0_dev = upu(&ma.nbr0_sorted)?;
    let nbr1_dev = upu(&ma.nbr1_sorted)?;
    let swf0_dev = up(&ma.swf0)?;
    let swf1_dev = up(&ma.swf1)?;
    let p0_dev = up(&ma.p0)?;
    let p1_dev = up(&ma.p1)?;
    let mut gx_dev = DeviceBuffer::<f64>::zeroed(&stream, ma.ndof)?;
    let mut gy_dev = DeviceBuffer::<f64>::zeroed(&stream, ma.ndof)?;
    let mut out_dev = DeviceBuffer::<f64>::zeroed(&stream, ma.ndof)?;

    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig {
        grid_dim: (ma.ne as u32, 1, 1),
        block_dim: (ma.nn as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    module.gradient_nc(
        &stream, cfg, &d_dev, &u_dev, &rx_dev, &ry_dev, &sx_dev, &sy_dev, ma.n1,
        &mut gx_dev, &mut gy_dev,
    )?;
    module.operator_nc(
        &stream, cfg, &d_dev, &u_dev, &gx_dev, &gy_dev, &rx_dev, &ry_dev, &sx_dev, &sy_dev,
        &jw_dev, ma.n1, &fvl_dev, &fnx_dev, &fny_dev, &fsw_dev, &fnbr_dev, &ekind_dev,
        &enx_dev, &eny_dev, &etau_dev, &half0_dev, &self_sorted_dev, &self_sw_dev, &nbr0_dev,
        &nbr1_dev, &swf0_dev, &swf1_dev, &p0_dev, &p1_dev, reaction, &mut out_dev,
    )?;
    Ok(out_dev.to_host_vec(&stream)?)
}
