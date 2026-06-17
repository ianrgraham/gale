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

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;
use gale::dg::{Edge, Mesh2d, Neighbor, RefineQuad};
use std::sync::Arc;

const NN_MAX: usize = 81; // (order 8 + 1)²
const RED: usize = 256; // reduction block size for the CG dot product

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
        sy: &[f64], jw: &[f64], n1: u32, ne: u32,
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
        let _ = ne; // kept for signature stability; this kernel launches one block per element
        static mut DS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PR: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        let n1 = n1 as usize;
        let nn = n1 * n1;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let b = e * nn + m;
        let wx = jw[b] * gx[b];
        let wy = jw[b] * gy[b];
        unsafe {
            DS[m] = d[m];
            PR[m] = rx[b] * wx + ry[b] * wy;
            PS[m] = sx[b] * wx + sy[b] * wy;
        }

        // ---- Face/SIPG/mortar contribution by node-parallel GATHER ----
        // Thread `m` owns local node `m` and accumulates ITS OWN SIPG consistency/penalty `rf` and
        // symmetry-lift `hx,hy` into REGISTERS (no shared accumulators ⇒ no cross-face races, fully
        // parallel — replaces the old single-thread `m==0` face loop, which left 15 of 16 threads idle
        // and ran the matvec at ~18% occupancy). For each of the 4 edges we find node `m`'s position on
        // that edge (a cheap `n1` scan of the edge's node list; `m` is on the edge iff found), then run
        // the SAME per-kind arithmetic as before for that one node — per-node contribution set and
        // accumulation order are identical ⇒ bit-for-bit equal. (Multi-element packing was tried and
        // REVERTED: this gather is register-heavy and latency-bound, so over-subscribing threads/block
        // spills and is measurably slower — same lesson as the conforming operator.)
        let mut rf = 0.0f64;
        let mut hx = 0.0f64;
        let mut hy = 0.0f64;
        let mut t = 0usize;
        while t < 4 {
            let et = e * 4 + t;
            let kind = ekind[et];
            let tau = etau[et];
            if kind == K_CONF {
                // conforming: node list is the face's natural order `face_vl`.
                let mut a = 0usize;
                while a < n1 {
                    if face_vl[et * n1 + a] as usize == m {
                        let idx = et * n1 + a;
                        let nbr = face_nbr[idx];
                        if nbr != NEU {
                            let nx = face_nx[idx];
                            let ny = face_ny[idx];
                            let sw = face_sw[idx];
                            let dun_e = nx * gx[b] + ny * gy[b];
                            let ug = u[b];
                            let (avg, jump, gfac) = if nbr == BND {
                                (dun_e, ug, 1.0)
                            } else {
                                let ng = nbr as usize;
                                (0.5 * (dun_e + nx * gx[ng] + ny * gy[ng]), ug - u[ng], 0.5)
                            };
                            let g = gfac * sw * jump;
                            rf += -sw * avg + tau * sw * jump;
                            hx += g * nx;
                            hy += g * ny;
                        }
                        break; // a node appears once per edge
                    }
                    a += 1;
                }
            } else if kind == K_FINE_TO_COARSE {
                // E is FINE: node list is `self_sorted`; neighbor trace = coarse projected via P_half.
                let fnx = enx[et];
                let fny = eny[et];
                let half = half0[et];
                let mut i = 0usize;
                while i < n1 {
                    if self_sorted[et * n1 + i] as usize == m {
                        let sw = self_sw[et * n1 + i];
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
                        let u_self = u[b];
                        let dun_self = fnx * gx[b] + fny * gy[b];
                        let avg = 0.5 * (dun_self + dun_nbr);
                        let jump = u_self - u_nbr;
                        let g = 0.5 * sw * jump;
                        rf += -sw * avg + tau * sw * jump;
                        hx += g * fnx;
                        hy += g * fny;
                        break;
                    }
                    i += 1;
                }
            } else {
                // E is COARSE: node `m` = coarse node at sorted index `j`; gather Σ_h Σ_i P_h[i,j]·(…).
                let fnx = enx[et];
                let fny = eny[et];
                let mut j = 0usize;
                while j < n1 {
                    if self_sorted[et * n1 + j] as usize == m {
                        let mut h = 0usize;
                        while h < 2 {
                            let mut i = 0usize;
                            while i < n1 {
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
                                rf += pw * gc;
                                hx += pw * gl * fnx;
                                hy += pw * gl * fny;
                                i += 1;
                            }
                            h += 1;
                        }
                        break;
                    }
                    j += 1;
                }
            }
            t += 1;
        }
        unsafe {
            PR[m] -= rx[b] * hx + ry[b] * hy;
            PS[m] -= sx[b] * hx + sy[b] * hy;
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
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = acc + rf + lambda * jw[b] * u[b];
        }
    }

    /// y ← y + a·x  (CG vector op; `_nc` suffix for crate-wide kernel-name uniqueness)
    #[kernel]
    pub fn axpy_nc(mut y: DisjointSlice<f64>, x: &[f64], a: f64) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = y.get_mut(idx) {
            *o += a * x[i];
        }
    }

    /// y ← x + b·y
    #[kernel]
    pub fn xpby_nc(mut y: DisjointSlice<f64>, x: &[f64], b: f64) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = y.get_mut(idx) {
            *o = x[i] + b * *o;
        }
    }

    /// y ← c·y (scale in place).
    #[kernel]
    pub fn scal_nc(mut y: DisjointSlice<f64>, c: f64) {
        let idx = thread::index_1d();
        if let Some(o) = y.get_mut(idx) {
            *o *= c;
        }
    }

    /// out ← a⊙b + c⊙d (Hadamard FMA; the convection primitive u·∂ₓu + v·∂ᵧu).
    #[kernel]
    pub fn fma2_nc(a: &[f64], b: &[f64], c: &[f64], d: &[f64], mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = a[i] * b[i] + c[i] * d[i];
        }
    }

    /// Damped-Jacobi smoother update `x ← x + ω·D⁻¹·(b − A·x)` (one sweep; `ax` is `A·x` precomputed).
    /// The p-multigrid relaxation, pointwise on resident fields.
    #[kernel]
    pub fn jacobi_nc(mut x: DisjointSlice<f64>, inv_diag: &[f64], b: &[f64], ax: &[f64], omega: f64) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = x.get_mut(idx) {
            *o += omega * inv_diag[i] * (b[i] - ax[i]);
        }
    }

    /// **p-prolong** (coarse order → fine order): per element, tensor-Lagrange interpolation
    /// `fine[i,j] = Σ_ab I[i,a] I[j,b] coarse[a,b]` with the 1D `fine×coarse` matrix `interp`
    /// (`nff×ncc`). One block per element, `nff·nff` threads. Element-local ⇒ ignores the mortar
    /// non-conformity (the multigrid transfer never crosses a face). Matches CPU `PMultigridNc::prolong`.
    #[kernel]
    pub fn prolong_p_nc(coarse: &[f64], interp: &[f64], ncc1: u32, nff1: u32, mut out: DisjointSlice<f64>) {
        static mut CF: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        let ncc = ncc1 as usize;
        let nff = nff1 as usize;
        let cc = ncc * ncc;
        let ff = nff * nff;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        if m < cc {
            unsafe { CF[m] = coarse[e * cc + m]; }
        }
        thread::sync_threads();
        if m >= ff {
            return;
        }
        let i = m % nff;
        let j = m / nff;
        let mut s = 0.0f64;
        let mut b = 0usize;
        while b < ncc {
            let mut row = 0.0f64;
            let mut a = 0usize;
            while a < ncc {
                row += interp[i * ncc + a] * unsafe { CF[a + b * ncc] };
                a += 1;
            }
            s += interp[j * ncc + b] * row;
            b += 1;
        }
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = s;
        }
    }

    /// **p-restrict** (fine order → coarse order) = transpose of [`prolong_p_nc`]:
    /// `coarse[a,b] = Σ_ij I[i,a] I[j,b] fine[i,j]`. One block per element, `ncc·ncc` threads.
    #[kernel]
    pub fn restrict_p_nc(fine: &[f64], interp: &[f64], ncc1: u32, nff1: u32, mut out: DisjointSlice<f64>) {
        static mut FF: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        let ncc = ncc1 as usize;
        let nff = nff1 as usize;
        let cc = ncc * ncc;
        let ff = nff * nff;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        if m < ff {
            unsafe { FF[m] = fine[e * ff + m]; }
        }
        thread::sync_threads();
        if m >= cc {
            return;
        }
        let a = m % ncc;
        let b = m / ncc;
        let mut s = 0.0f64;
        let mut j = 0usize;
        while j < nff {
            let mut col = 0.0f64;
            let mut i = 0usize;
            while i < nff {
                col += interp[i * ncc + a] * unsafe { FF[i + j * nff] };
                i += 1;
            }
            s += interp[j * ncc + b] * col;
            j += 1;
        }
        // Output is COARSE-strided (cc per element), but the block is sized to the FINE node count
        // (nff²), so index_1d() = e·nff² + m is NOT the output index — write e·cc + m explicitly.
        // Threads m<cc are unique over (e, coarse-node) ⇒ scatter is race-free.
        let e = thread::blockIdx_x() as usize;
        unsafe {
            *out.get_unchecked_mut(e * cc + m) = s;
        }
    }

    /// **h-coarsen restrict** (order-1 REFINED → order-1 UNIFORM base): per base node, an unrefined
    /// cell injects; a refined cell `Pᵀ`-gathers its 4 children via the shared 2:1 matrices `quad_pq`
    /// (`[q*16+f*4+a]`). One thread per base node. `base_ids[bc*4]` = element id (unrefined) or the 4
    /// child ids (refined); `base_kind[bc]` = 0/1.
    #[kernel]
    pub fn restrict_to_base_nc(
        refined: &[f64], base_kind: &[u8], base_ids: &[u32], quad_pq: &[f64], nbc: u32, mut out: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        let tid = idx.get();
        if tid >= nbc as usize * 4 {
            return;
        }
        let bc = tid / 4;
        let a = tid % 4;
        let val = if base_kind[bc] == 0 {
            refined[base_ids[bc * 4] as usize * 4 + a]
        } else {
            let mut s = 0.0f64;
            let mut q = 0usize;
            while q < 4 {
                let cq = base_ids[bc * 4 + q] as usize;
                let mut f = 0usize;
                while f < 4 {
                    s += quad_pq[q * 16 + f * 4 + a] * refined[cq * 4 + f];
                    f += 1;
                }
                q += 1;
            }
            s
        };
        if let Some(o) = out.get_mut(idx) {
            *o = val;
        }
    }

    /// **h-coarsen prolong** (order-1 UNIFORM base → order-1 REFINED) = transpose of
    /// [`restrict_to_base_nc`]: per base cell, inject (unrefined) or bilinearly evaluate the parent at
    /// each child node (refined). One thread per base cell; scatter writes to disjoint refined elements.
    #[kernel]
    pub fn prolong_from_base_nc(
        base: &[f64], base_kind: &[u8], base_ids: &[u32], quad_pq: &[f64], nbc: u32, mut out: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        let bc = idx.get();
        if bc >= nbc as usize {
            return;
        }
        if base_kind[bc] == 0 {
            let rid = base_ids[bc * 4] as usize;
            let mut a = 0usize;
            while a < 4 {
                unsafe { *out.get_unchecked_mut(rid * 4 + a) = base[bc * 4 + a]; }
                a += 1;
            }
        } else {
            let mut q = 0usize;
            while q < 4 {
                let cq = base_ids[bc * 4 + q] as usize;
                let mut f = 0usize;
                while f < 4 {
                    let mut s = 0.0f64;
                    let mut a = 0usize;
                    while a < 4 {
                        s += quad_pq[q * 16 + f * 4 + a] * base[bc * 4 + a];
                        a += 1;
                    }
                    unsafe { *out.get_unchecked_mut(cq * 4 + f) = s; }
                    f += 1;
                }
                q += 1;
            }
        }
    }

    /// b ← scale·jw⊙f + lift (diagonal-mass RHS assembly: M·(scale·f) + boundary lift).
    #[kernel]
    pub fn rhs_madd_nc(jw: &[f64], f: &[f64], lift: &[f64], scale: f64, mut b: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = b.get_mut(idx) {
            *o = scale * jw[i] * f[i] + lift[i];
        }
    }

    /// Multi-block grid-stride dot product: each block reduces its strided slice into
    /// shared memory and writes one partial to `partial[blockIdx]` (host sums them). With
    /// `gridDim = 1` this is the old single-block reduction; with many blocks it streams
    /// `a`/`b` across all SMs at ~peak bandwidth instead of saturating one SM.
    #[kernel]
    pub fn dot_nc_partial(a: &[f64], b: &[f64], n: u64, mut partial: DisjointSlice<f64>) {
        static mut SH: SharedArray<f64, RED> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let gstride = (thread::gridDim_x() * thread::blockDim_x()) as usize;
        let mut acc = 0.0f64;
        let mut i = (thread::blockIdx_x() * thread::blockDim_x()) as usize + tid;
        while i < n as usize {
            acc += a[i] * b[i];
            i += gstride;
        }
        unsafe { SH[tid] = acc; }
        thread::sync_threads();
        let mut s = thread::blockDim_x() as usize / 2;
        while s > 0 {
            if tid < s {
                unsafe { SH[tid] = SH[tid] + SH[tid + s]; }
            }
            thread::sync_threads();
            s /= 2;
        }
        if tid == 0 {
            unsafe { *partial.get_unchecked_mut(thread::blockIdx_x() as usize) = SH[0]; }
        }
    }
}

/// Number of blocks for the multi-block `dot_nc_partial` reduction (see the 2D
/// `dot_blocks`): enough to stream the vector across all SMs, capped at 1024 so the
/// host-side sum of partials stays trivial. `RED` threads per block.
fn dot_blocks(ndof: usize) -> usize {
    ndof.div_ceil(RED).clamp(1, 1024)
}

/// Launch config for `operator_nc`: one block per element, `nn` threads (the node-parallel face
/// gather makes each node-thread do its own SIPG/mortar work). Multi-element packing was tried and
/// reverted — this register-heavy, latency-bound matvec runs slower when threads/block is raised
/// (register spills), and dynamic-shared tiling defeated the compiler's const-folded indexing.
fn op_launch_cfg(ne: usize, nn: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: (ne as u32, 1, 1), block_dim: (nn as u32, 1, 1), shared_mem_bytes: 0 }
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
fn flatten_nc(mesh: &Mesh2d, alpha: f64, neumann_tags: &[u32]) -> NcArrays {
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
                Neighbor::Boundary { tag } => {
                    // Per-region: tags in `neumann_tags` are natural (NEU), the rest are
                    // Dirichlet SIPG (BND) — same per-tag routing as the conforming
                    // `flatten_mesh`. `&[]` ⇒ all-Dirichlet, all-tags ⇒ all-Neumann.
                    ekind[et] = K_CONF;
                    etau[et] = alpha * p1f * p1f / h[e];
                    let neu = neumann_tags.contains(tag);
                    for a in 0..n1u {
                        let idx = et * n1u + a;
                        fvl[idx] = face.nodes[a] as u32;
                        fnx[idx] = face.nx[a];
                        fny[idx] = face.ny[a];
                        fsw[idx] = face.sw[a];
                        fnbr[idx] = if neu { NEU } else { BND };
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
    let ma = flatten_nc(mesh, alpha, &[]);
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
        &stream, op_launch_cfg(ma.ne, ma.nn), &d_dev, &u_dev, &gx_dev, &gy_dev, &rx_dev, &ry_dev, &sx_dev, &sy_dev,
        &jw_dev, ma.n1, ma.ne as u32, &fvl_dev, &fnx_dev, &fny_dev, &fsw_dev, &fnbr_dev, &ekind_dev,
        &enx_dev, &eny_dev, &etau_dev, &half0_dev, &self_sorted_dev, &self_sw_dev, &nbr0_dev,
        &nbr1_dev, &swf0_dev, &swf1_dev, &p0_dev, &p1_dev, reaction, &mut out_dev,
    )?;
    Ok(out_dev.to_host_vec(&stream)?)
}

/// Solve `(reaction·M + A)·x = b` on a **2:1 non-conforming** mesh by conjugate
/// gradient entirely on the GPU (Dirichlet path). `reaction = 0` is pure Poisson.
/// Mirrors `gale::dg::Poisson::with_reaction(mesh, alpha, reaction).cg`.
pub fn poisson_nc_cg_solve(
    mesh: &Mesh2d, b: &[f64], alpha: f64, reaction: f64, tol: f64, maxit: usize,
) -> Result<(Vec<f64>, usize), Box<dyn std::error::Error>> {
    cg_nc_impl(mesh, b, alpha, reaction, false, &[], tol, maxit)
}

/// Tag-aware NC CG for `(reaction·M + A)·x = b`: boundary tags in `neumann_tags` are
/// natural (Neumann), the rest Dirichlet SIPG — the per-region (inflow/outflow/symmetry)
/// path on a **2:1 non-conforming** mesh. Non-deflated (a Dirichlet boundary makes it
/// non-singular); used for the velocity Helmholtz (`neumann_tags` = outflow + symmetry-
/// tangential) and the outflow-pinned pressure (`reaction = 0`, `neumann_tags` = all
/// except outflow). Mirrors `gale::dg::Poisson::with_bc(mesh, alpha, reaction, neumann_tags).cg`.
pub fn helmholtz_nc_cg_solve_tags(
    mesh: &Mesh2d, b: &[f64], alpha: f64, reaction: f64, neumann_tags: &[u32], tol: f64, maxit: usize,
) -> Result<(Vec<f64>, usize), Box<dyn std::error::Error>> {
    cg_nc_impl(mesh, b, alpha, reaction, false, neumann_tags, tol, maxit)
}

/// Solve the singular pure-Neumann pressure-Poisson `A·x = b` on a non-conforming mesh
/// by **deflated** CG on the GPU (constant nullspace removed each iteration). Mirrors
/// `gale::dg::Poisson::with_bc(mesh, alpha, 0, all-tags).cg_deflated`.
pub fn pressure_nc_cg_solve(
    mesh: &Mesh2d, b: &[f64], alpha: f64, tol: f64, maxit: usize,
) -> Result<(Vec<f64>, usize), Box<dyn std::error::Error>> {
    cg_nc_impl(mesh, b, alpha, 0.0, true, &mesh.boundary_tags(), tol, maxit)
}

/// CG for `(reaction·M + A)·x = b` on a non-conforming mesh; `deflate` removes the
/// constant nullspace each iteration (for the singular pure-Neumann pressure system).
/// `neumann_tags` routes per-region boundaries (NEU vs Dirichlet BND). The matvec is the
/// validated NC `gradient_nc → operator_nc` pipeline; vectors stay device-resident (only
/// the CG scalars transfer host-side).
#[allow(clippy::too_many_arguments)]
fn cg_nc_impl(
    mesh: &Mesh2d, b: &[f64], alpha: f64, reaction: f64, deflate: bool, neumann_tags: &[u32], tol: f64, maxit: usize,
) -> Result<(Vec<f64>, usize), Box<dyn std::error::Error>> {
    let ma = flatten_nc(mesh, alpha, neumann_tags);
    let ndof = ma.ndof;
    assert_eq!(b.len(), ndof, "rhs length must be n_elements·n_nodes");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
    let upu = |v: &[u32]| DeviceBuffer::from_host(&stream, v);

    let d_dev = up(&ma.diff)?;
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
    let ss_dev = upu(&ma.self_sorted)?;
    let ssw_dev = up(&ma.self_sw)?;
    let nbr0_dev = upu(&ma.nbr0_sorted)?;
    let nbr1_dev = upu(&ma.nbr1_sorted)?;
    let swf0_dev = up(&ma.swf0)?;
    let swf1_dev = up(&ma.swf1)?;
    let p0_dev = up(&ma.p0)?;
    let p1_dev = up(&ma.p1)?;

    let mut x = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut r = up(b)?;
    let mut p = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut ap = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut gx = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut gy = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let nb = dot_blocks(ndof);
    let mut partial = DeviceBuffer::<f64>::zeroed(&stream, nb)?;
    let ones = up(&vec![1.0f64; ndof])?;

    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig { grid_dim: (ma.ne as u32, 1, 1), block_dim: (ma.nn as u32, 1, 1), shared_mem_bytes: 0 };
    let red = LaunchConfig { grid_dim: (nb as u32, 1, 1), block_dim: (RED as u32, 1, 1), shared_mem_bytes: 0 };
    let vec_cfg = LaunchConfig::for_num_elems(ndof as u32);
    let n1 = ma.n1;
    let n64 = ndof as u64;
    let ninv = 1.0 / ndof as f64;

    macro_rules! dot {
        ($a:expr, $b:expr) => {{
            module.dot_nc_partial(&stream, red, $a, $b, n64, &mut partial)?;
            partial.to_host_vec(&stream)?.iter().sum::<f64>()
        }};
    }
    macro_rules! deflate {
        ($v:expr) => {{
            if deflate {
                let mean = dot!($v, &ones) * ninv;
                module.axpy_nc(&stream, vec_cfg, $v, &ones, -mean)?;
            }
        }};
    }
    macro_rules! apply {
        ($field:expr, $dst:expr) => {{
            module.gradient_nc(&stream, cfg, &d_dev, $field, &rx_dev, &ry_dev, &sx_dev, &sy_dev, n1, &mut gx, &mut gy)?;
            module.operator_nc(
                &stream, op_launch_cfg(ma.ne, ma.nn), &d_dev, $field, &gx, &gy, &rx_dev, &ry_dev, &sx_dev, &sy_dev, &jw_dev, n1, ma.ne as u32,
                &fvl_dev, &fnx_dev, &fny_dev, &fsw_dev, &fnbr_dev, &ekind_dev, &enx_dev, &eny_dev,
                &etau_dev, &half0_dev, &ss_dev, &ssw_dev, &nbr0_dev, &nbr1_dev, &swf0_dev, &swf1_dev,
                &p0_dev, &p1_dev, reaction, $dst,
            )?;
        }};
    }

    deflate!(&mut r);
    module.xpby_nc(&stream, vec_cfg, &mut p, &r, 0.0)?; // p = r
    let bn = dot!(&r, &r).sqrt().max(1e-300);
    let mut rs = dot!(&r, &r);
    let mut iters = 0;
    for it in 0..maxit {
        apply!(&p, &mut ap);
        let pap = dot!(&p, &ap);
        let alpha_cg = rs / pap;
        module.axpy_nc(&stream, vec_cfg, &mut x, &p, alpha_cg)?;
        module.axpy_nc(&stream, vec_cfg, &mut r, &ap, -alpha_cg)?;
        deflate!(&mut r);
        let rs_new = dot!(&r, &r);
        iters = it + 1;
        if rs_new.sqrt() / bn < tol {
            break;
        }
        let beta = rs_new / rs;
        module.xpby_nc(&stream, vec_cfg, &mut p, &r, beta)?;
        rs = rs_new;
    }
    Ok((x.to_host_vec(&stream)?, iters))
}

// ===== Persistent non-conforming solver handle (P4 for AMR) ======================

/// Persistent GPU **non-conforming** (2:1 mortar) SIPG-Poisson / Helmholtz solver handle
/// — the AMR analogue of [`crate::GpuPoisson`]. Owns the CUDA context, the loaded device
/// module, and the uploaded CONSTANT metrics + conforming/mortar face metadata, so
/// repeated solves on the same adaptive mesh pay the ~0.3 s setup (`CudaContext::new` +
/// `kernels::load` + the large NC upload) **once** instead of per call — the 3 elliptic
/// solves/step of a dual-splitting flow loop on a refined mesh, across every timestep
/// between remeshes, share one handle.
///
/// Only the per-region `neumann_tags`/`reaction`/`deflate` vary between solves;
/// [`solve`](Self::solve) rebuilds just the small boundary `fnbr` entries on the host and
/// uploads them. CG scalars use the multi-block `dot_nc_partial` + host sum (as
/// `cg_nc_impl`). Bit-for-bit equivalent to `helmholtz_nc_cg_solve_tags` /
/// `pressure_nc_cg_solve`.
pub struct GpuPoissonNc {
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
    nn: usize,
    ne: usize,
    ndof: usize,
    n1: u32,
    // constant (neumann-tag-independent) device arrays, uploaded once.
    d_dev: DeviceBuffer<f64>,
    rx_dev: DeviceBuffer<f64>,
    ry_dev: DeviceBuffer<f64>,
    sx_dev: DeviceBuffer<f64>,
    sy_dev: DeviceBuffer<f64>,
    jw_dev: DeviceBuffer<f64>,
    fvl_dev: DeviceBuffer<u32>,
    fnx_dev: DeviceBuffer<f64>,
    fny_dev: DeviceBuffer<f64>,
    fsw_dev: DeviceBuffer<f64>,
    ekind_dev: DeviceBuffer<u32>,
    enx_dev: DeviceBuffer<f64>,
    eny_dev: DeviceBuffer<f64>,
    etau_dev: DeviceBuffer<f64>,
    half0_dev: DeviceBuffer<u32>,
    ss_dev: DeviceBuffer<u32>,
    ssw_dev: DeviceBuffer<f64>,
    nbr0_dev: DeviceBuffer<u32>,
    nbr1_dev: DeviceBuffer<u32>,
    swf0_dev: DeviceBuffer<f64>,
    swf1_dev: DeviceBuffer<f64>,
    p0_dev: DeviceBuffer<f64>,
    p1_dev: DeviceBuffer<f64>,
    // host state to rebuild the per-region `fnbr` cheaply (base = all-Dirichlet; flip the
    // listed boundary-face nodes to NEU when their tag is in `neumann_tags`).
    fnbr_base: Vec<u32>,
    bnodes: Vec<(usize, u32)>,
}

impl GpuPoissonNc {
    /// Build the handle for a (possibly non-conforming) `mesh` with SIPG penalty `alpha`:
    /// create the context, load the module, and upload the constant metrics + mortar data.
    pub fn new(mesh: &Mesh2d, alpha: f64) -> Result<Self, Box<dyn std::error::Error>> {
        let ma = flatten_nc(mesh, alpha, &[]); // fnbr_base: all boundary faces Dirichlet (BND)
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
        let upu = |v: &[u32]| DeviceBuffer::from_host(&stream, v);

        // Boundary-face nodes + tags, in the same flat order flatten_nc uses for the
        // conforming face block, so a per-region solve flips only these entries to NEU.
        let n1u = ma.n1 as usize;
        let mut bnodes = Vec::new();
        for (e, el) in mesh.elements.iter().enumerate() {
            for (t, edge) in Edge::ALL.iter().enumerate() {
                if let Neighbor::Boundary { tag } = el.neighbors[*edge as usize] {
                    for a in 0..n1u {
                        bnodes.push(((e * 4 + t) * n1u + a, tag));
                    }
                }
            }
        }

        Ok(Self {
            module: kernels::load(&ctx)?,
            nn: ma.nn,
            ne: ma.ne,
            ndof: ma.ndof,
            n1: ma.n1,
            d_dev: up(&ma.diff)?,
            rx_dev: up(&ma.rx)?,
            ry_dev: up(&ma.ry)?,
            sx_dev: up(&ma.sx)?,
            sy_dev: up(&ma.sy)?,
            jw_dev: up(&ma.jw)?,
            fvl_dev: upu(&ma.fvl)?,
            fnx_dev: up(&ma.fnx)?,
            fny_dev: up(&ma.fny)?,
            fsw_dev: up(&ma.fsw)?,
            ekind_dev: upu(&ma.ekind)?,
            enx_dev: up(&ma.enx)?,
            eny_dev: up(&ma.eny)?,
            etau_dev: up(&ma.etau)?,
            half0_dev: upu(&ma.half0)?,
            ss_dev: upu(&ma.self_sorted)?,
            ssw_dev: up(&ma.self_sw)?,
            nbr0_dev: upu(&ma.nbr0_sorted)?,
            nbr1_dev: upu(&ma.nbr1_sorted)?,
            swf0_dev: up(&ma.swf0)?,
            swf1_dev: up(&ma.swf1)?,
            p0_dev: up(&ma.p0)?,
            p1_dev: up(&ma.p1)?,
            fnbr_base: ma.fnbr,
            bnodes,
            stream,
        })
    }

    /// Number of degrees of freedom (`n_elements · n_nodes`).
    pub fn ndof(&self) -> usize {
        self.ndof
    }

    /// This handle's CUDA stream (so device-resident callers allocate on the same stream/context).
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }
    /// Upload a host field to a fresh device buffer on this handle's stream.
    pub fn upload(&self, v: &[f64]) -> Result<DeviceBuffer<f64>, Box<dyn std::error::Error>> {
        Ok(DeviceBuffer::from_host(&self.stream, v)?)
    }
    /// Download a device field to host (diagnostics / I/O only).
    pub fn download(&self, v: &DeviceBuffer<f64>) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        Ok(v.to_host_vec(&self.stream)?)
    }
    /// Allocate a zeroed `ndof` device field on this handle's stream.
    pub fn alloc(&self) -> Result<DeviceBuffer<f64>, Box<dyn std::error::Error>> {
        Ok(DeviceBuffer::<f64>::zeroed(&self.stream, self.ndof)?)
    }

    // ===== Device-resident assembly primitives (NC analogues of GpuPoissonMg's) =================
    // Element-local / pointwise ops on resident fields, so a dual-splitting step can run on a
    // non-conforming (AMR) mesh entirely on the GPU. The metrics (`rx_dev`..`jw_dev`) were uploaded
    // once at construction; these reuse them.

    fn vcfg(&self) -> LaunchConfig {
        LaunchConfig::for_num_elems(self.ndof as u32)
    }
    fn ecfg(&self) -> LaunchConfig {
        LaunchConfig { grid_dim: (self.ne as u32, 1, 1), block_dim: (self.nn as u32, 1, 1), shared_mem_bytes: 0 }
    }

    /// Element-local physical gradient `gx=∂src/∂x, gy=∂src/∂y` (same as the matvec's, NC-mesh metrics).
    pub fn gradient_dev(&self, src: &DeviceBuffer<f64>, gx: &mut DeviceBuffer<f64>, gy: &mut DeviceBuffer<f64>) -> Result<(), Box<dyn std::error::Error>> {
        self.module.gradient_nc(&self.stream, self.ecfg(), &self.d_dev, src, &self.rx_dev, &self.ry_dev, &self.sx_dev, &self.sy_dev, self.n1, gx, gy)?;
        Ok(())
    }
    /// `y ← y + a·x`.
    pub fn axpy_dev(&self, y: &mut DeviceBuffer<f64>, x: &DeviceBuffer<f64>, a: f64) -> Result<(), Box<dyn std::error::Error>> {
        self.module.axpy_nc(&self.stream, self.vcfg(), y, x, a)?;
        Ok(())
    }
    /// `y ← c·y`.
    pub fn scal_dev(&self, y: &mut DeviceBuffer<f64>, c: f64) -> Result<(), Box<dyn std::error::Error>> {
        self.module.scal_nc(&self.stream, self.vcfg(), y, c)?;
        Ok(())
    }
    /// `dst ← src` (device-to-device copy).
    pub fn copy_dev(&self, dst: &mut DeviceBuffer<f64>, src: &DeviceBuffer<f64>) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            cuda_core::memory::memcpy_dtod_async(dst.cu_deviceptr(), src.cu_deviceptr(), self.ndof * 8, self.stream.cu_stream())?;
        }
        Ok(())
    }
    /// `out ← a⊙b + c⊙d` (convection primitive).
    pub fn fma2_dev(&self, out: &mut DeviceBuffer<f64>, a: &DeviceBuffer<f64>, b: &DeviceBuffer<f64>, c: &DeviceBuffer<f64>, d: &DeviceBuffer<f64>) -> Result<(), Box<dyn std::error::Error>> {
        self.module.fma2_nc(&self.stream, self.vcfg(), a, b, c, d, out)?;
        Ok(())
    }
    /// `b ← scale·jw⊙f + lift` (diagonal-mass RHS assembly).
    pub fn rhs_madd_dev(&self, b: &mut DeviceBuffer<f64>, jw: &DeviceBuffer<f64>, f: &DeviceBuffer<f64>, lift: &DeviceBuffer<f64>, scale: f64) -> Result<(), Box<dyn std::error::Error>> {
        self.module.rhs_madd_nc(&self.stream, self.vcfg(), jw, f, lift, scale, b)?;
        Ok(())
    }

    // ===== p-multigrid building blocks (device-resident; the V-cycle driver lives in multigrid_nc) =====
    pub fn nn(&self) -> usize {
        self.nn
    }
    pub fn ne(&self) -> usize {
        self.ne
    }
    /// `order + 1` (the 1D node count) — needed to size p-transfers between levels.
    pub fn n1_pub(&self) -> u32 {
        self.n1
    }

    /// Build the device `fnbr` (boundary-face Dirichlet/Neumann marker) for the given `neumann_tags`
    /// once — reused across every V-cycle smooth/matvec at this level (no per-apply host work).
    pub fn build_fnbr_dev(&self, neumann_tags: &[u32]) -> Result<DeviceBuffer<u32>, Box<dyn std::error::Error>> {
        let mut fnbr = self.fnbr_base.clone();
        if !neumann_tags.is_empty() {
            for &(idx, tag) in &self.bnodes {
                if neumann_tags.contains(&tag) {
                    fnbr[idx] = NEU;
                }
            }
        }
        Ok(DeviceBuffer::from_host(&self.stream, &fnbr)?)
    }

    /// Device operator apply `out ← (reaction·M + A)·field` with a prebuilt `fnbr_dev` and caller
    /// scratch `gx,gy`. The matvec used by the smoother and the PCG, no per-call host work.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_dev(
        &self, field: &DeviceBuffer<f64>, reaction: f64, fnbr_dev: &DeviceBuffer<u32>,
        gx: &mut DeviceBuffer<f64>, gy: &mut DeviceBuffer<f64>, out: &mut DeviceBuffer<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = self.ecfg();
        self.module.gradient_nc(&self.stream, cfg, &self.d_dev, field, &self.rx_dev, &self.ry_dev, &self.sx_dev, &self.sy_dev, self.n1, gx, gy)?;
        self.module.operator_nc(
            &self.stream, op_launch_cfg(self.ne, self.nn), &self.d_dev, field, gx, gy, &self.rx_dev, &self.ry_dev, &self.sx_dev, &self.sy_dev, &self.jw_dev, self.n1, self.ne as u32,
            &self.fvl_dev, &self.fnx_dev, &self.fny_dev, &self.fsw_dev, fnbr_dev, &self.ekind_dev, &self.enx_dev, &self.eny_dev,
            &self.etau_dev, &self.half0_dev, &self.ss_dev, &self.ssw_dev, &self.nbr0_dev, &self.nbr1_dev, &self.swf0_dev, &self.swf1_dev,
            &self.p0_dev, &self.p1_dev, reaction, out,
        )?;
        Ok(())
    }

    /// One damped-Jacobi sweep `x ← x + ω·D⁻¹·(b − A·x)` (computes `A·x` into the caller's `ax`).
    #[allow(clippy::too_many_arguments)]
    pub fn jacobi_dev(
        &self, x: &mut DeviceBuffer<f64>, b: &DeviceBuffer<f64>, inv_diag: &DeviceBuffer<f64>, omega: f64, reaction: f64,
        fnbr_dev: &DeviceBuffer<u32>, gx: &mut DeviceBuffer<f64>, gy: &mut DeviceBuffer<f64>, ax: &mut DeviceBuffer<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.apply_dev(x, reaction, fnbr_dev, gx, gy, ax)?;
        self.module.jacobi_nc(&self.stream, self.vcfg(), x, inv_diag, b, ax, omega)?;
        Ok(())
    }

    /// p-prolong `out(this level, order p_f) ← coarse(order p_c)` (element-local tensor interp). `ncc1
    /// = p_c+1`, `nff1 = p_f+1 = this.n1`; `interp` is the 1D `nff×ncc` Lagrange matrix.
    pub fn prolong_p_dev(
        &self, coarse: &DeviceBuffer<f64>, interp: &DeviceBuffer<f64>, ncc1: u32, out: &mut DeviceBuffer<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = LaunchConfig { grid_dim: (self.ne as u32, 1, 1), block_dim: (self.nn as u32, 1, 1), shared_mem_bytes: 0 };
        self.module.prolong_p_nc(&self.stream, cfg, coarse, interp, ncc1, self.n1, out)?;
        Ok(())
    }

    /// p-restrict `out(this level, order p_c) ← fine(order p_f)` = transpose of prolong. Launched with
    /// the FINE block size (`nff1·nff1` threads) so the shared load covers the fine element. `nff1 =
    /// p_f+1`, this level's `n1 = p_c+1`.
    pub fn restrict_p_dev(
        &self, fine: &DeviceBuffer<f64>, interp: &DeviceBuffer<f64>, nff1: u32, out: &mut DeviceBuffer<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = LaunchConfig { grid_dim: (self.ne as u32, 1, 1), block_dim: ((nff1 * nff1) as u32, 1, 1), shared_mem_bytes: 0 };
        self.module.restrict_p_nc(&self.stream, cfg, fine, interp, self.n1, nff1, out)?;
        Ok(())
    }

    /// `Σ a⊙b` over the field (CG scalar; multi-block partial + host sum, like `solve_dev`).
    pub fn dot_dev(&self, a: &DeviceBuffer<f64>, b: &DeviceBuffer<f64>) -> Result<f64, Box<dyn std::error::Error>> {
        let nb = dot_blocks(self.ndof);
        let mut partial = DeviceBuffer::<f64>::zeroed(&self.stream, nb)?;
        let red = LaunchConfig { grid_dim: (nb as u32, 1, 1), block_dim: (RED as u32, 1, 1), shared_mem_bytes: 0 };
        self.module.dot_nc_partial(&self.stream, red, a, b, self.ndof as u64, &mut partial)?;
        Ok(partial.to_host_vec(&self.stream)?.iter().sum::<f64>())
    }

    /// h-coarsen restrict (this order-1 REFINED level → order-1 UNIFORM base, `nbc·4` dof). Device,
    /// no host work — the multigrid_nc handle launches this on the coarsest level.
    #[allow(clippy::too_many_arguments)]
    pub fn restrict_to_base_dev(
        &self, refined: &DeviceBuffer<f64>, base_kind: &DeviceBuffer<u8>, base_ids: &DeviceBuffer<u32>,
        quad_pq: &DeviceBuffer<f64>, nbc: usize, out: &mut DeviceBuffer<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = LaunchConfig::for_num_elems((nbc * 4) as u32);
        self.module.restrict_to_base_nc(&self.stream, cfg, refined, base_kind, base_ids, quad_pq, nbc as u32, out)?;
        Ok(())
    }

    /// h-coarsen prolong (order-1 UNIFORM base → this order-1 REFINED level), transpose of the above.
    #[allow(clippy::too_many_arguments)]
    pub fn prolong_from_base_dev(
        &self, base: &DeviceBuffer<f64>, base_kind: &DeviceBuffer<u8>, base_ids: &DeviceBuffer<u32>,
        quad_pq: &DeviceBuffer<f64>, nbc: usize, out: &mut DeviceBuffer<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = LaunchConfig::for_num_elems(nbc as u32);
        self.module.prolong_from_base_nc(&self.stream, cfg, base, base_kind, base_ids, quad_pq, nbc as u32, out)?;
        Ok(())
    }

    /// Upload a `u8`/`u32` slice to a device buffer on this handle's stream (h-coarsen metadata).
    pub fn upload_u8(&self, v: &[u8]) -> Result<DeviceBuffer<u8>, Box<dyn std::error::Error>> {
        Ok(DeviceBuffer::from_host(&self.stream, v)?)
    }
    pub fn upload_u32(&self, v: &[u32]) -> Result<DeviceBuffer<u32>, Box<dyn std::error::Error>> {
        Ok(DeviceBuffer::from_host(&self.stream, v)?)
    }

    /// Solve `(reaction·M + A)·x = b` on this non-conforming mesh by CG (mortar matvec +
    /// multi-block dot, host-sum scalars). `neumann_tags` are the natural-BC boundary tags;
    /// `deflate` removes the constant nullspace each iteration (pure-Neumann pressure). No
    /// per-call context/module setup or constant upload — only `b`, `fnbr`, and scratch move.
    #[allow(clippy::too_many_arguments)]
    pub fn solve(
        &self,
        b: &[f64],
        reaction: f64,
        neumann_tags: &[u32],
        deflate: bool,
        tol: f64,
        maxit: usize,
    ) -> Result<(Vec<f64>, usize), Box<dyn std::error::Error>> {
        assert_eq!(b.len(), self.ndof, "rhs length must be n_elements·n_nodes");
        let stream = &self.stream;
        let module = &self.module;
        let ndof = self.ndof;

        let mut fnbr = self.fnbr_base.clone();
        if !neumann_tags.is_empty() {
            for &(idx, tag) in &self.bnodes {
                if neumann_tags.contains(&tag) {
                    fnbr[idx] = NEU;
                }
            }
        }
        let fnbr_dev = DeviceBuffer::from_host(stream, &fnbr)?;

        let up = |v: &[f64]| DeviceBuffer::from_host(stream, v);
        let nb = dot_blocks(ndof);
        let mut x = DeviceBuffer::<f64>::zeroed(stream, ndof)?;
        let mut r = up(b)?;
        let mut p = DeviceBuffer::<f64>::zeroed(stream, ndof)?;
        let mut ap = DeviceBuffer::<f64>::zeroed(stream, ndof)?;
        let mut gx = DeviceBuffer::<f64>::zeroed(stream, ndof)?;
        let mut gy = DeviceBuffer::<f64>::zeroed(stream, ndof)?;
        let mut partial = DeviceBuffer::<f64>::zeroed(stream, nb)?;
        let ones = up(&vec![1.0f64; ndof])?;

        let cfg = LaunchConfig { grid_dim: (self.ne as u32, 1, 1), block_dim: (self.nn as u32, 1, 1), shared_mem_bytes: 0 };
        let red = LaunchConfig { grid_dim: (nb as u32, 1, 1), block_dim: (RED as u32, 1, 1), shared_mem_bytes: 0 };
        let vec_cfg = LaunchConfig::for_num_elems(ndof as u32);
        let n1 = self.n1;
        let n64 = ndof as u64;
        let ninv = 1.0 / ndof as f64;

        macro_rules! dot {
            ($a:expr, $b:expr) => {{
                module.dot_nc_partial(stream, red, $a, $b, n64, &mut partial)?;
                partial.to_host_vec(stream)?.iter().sum::<f64>()
            }};
        }
        macro_rules! deflate_v {
            ($v:expr) => {{
                if deflate {
                    let mean = dot!($v, &ones) * ninv;
                    module.axpy_nc(stream, vec_cfg, $v, &ones, -mean)?;
                }
            }};
        }
        macro_rules! apply {
            ($field:expr, $dst:expr) => {{
                module.gradient_nc(stream, cfg, &self.d_dev, $field, &self.rx_dev, &self.ry_dev, &self.sx_dev, &self.sy_dev, n1, &mut gx, &mut gy)?;
                module.operator_nc(
                    stream, op_launch_cfg(self.ne, self.nn), &self.d_dev, $field, &gx, &gy, &self.rx_dev, &self.ry_dev, &self.sx_dev, &self.sy_dev, &self.jw_dev, n1, self.ne as u32,
                    &self.fvl_dev, &self.fnx_dev, &self.fny_dev, &self.fsw_dev, &fnbr_dev, &self.ekind_dev, &self.enx_dev, &self.eny_dev,
                    &self.etau_dev, &self.half0_dev, &self.ss_dev, &self.ssw_dev, &self.nbr0_dev, &self.nbr1_dev, &self.swf0_dev, &self.swf1_dev,
                    &self.p0_dev, &self.p1_dev, reaction, $dst,
                )?;
            }};
        }

        deflate_v!(&mut r);
        module.xpby_nc(stream, vec_cfg, &mut p, &r, 0.0)?; // p = r
        let bn = dot!(&r, &r).sqrt().max(1e-300);
        let mut rs = dot!(&r, &r);
        let mut iters = 0;
        for it in 0..maxit {
            // Convergence check BEFORE forming α — this also catches a zero / near-zero RHS (e.g. the
            // first projection step where div=0 ⇒ rs=0), which would otherwise give the deflated-CG
            // 0/0 = NaN breakdown the conforming path guards against. Then x stays its (zero) iterate.
            if rs.sqrt() / bn < tol {
                iters = it;
                break;
            }
            apply!(&p, &mut ap);
            let pap = dot!(&p, &ap);
            if !(pap > 0.0) {
                iters = it; // operator breakdown (singular direction) ⇒ stop with the current iterate
                break;
            }
            let alpha_cg = rs / pap;
            module.axpy_nc(stream, vec_cfg, &mut x, &p, alpha_cg)?;
            module.axpy_nc(stream, vec_cfg, &mut r, &ap, -alpha_cg)?;
            deflate_v!(&mut r);
            let rs_new = dot!(&r, &r);
            iters = it + 1;
            if rs_new.sqrt() / bn < tol {
                break;
            }
            let beta = rs_new / rs;
            module.xpby_nc(stream, vec_cfg, &mut p, &r, beta)?;
            rs = rs_new;
        }
        Ok((x.to_host_vec(stream)?, iters))
    }

    /// **Device-native** non-conforming solve `(reaction·M + A)·x = b`: `rhs` is already on the
    /// device, the solution is written into `out` (length [`ndof`](Self::ndof)), optionally
    /// warm-started from `x0` — **no field HtoD/DtoH**. The mortar matvec + vector ops run on the
    /// device; only the CG convergence scalar (the dot sum) reads back. The non-conforming/AMR
    /// analogue of [`GpuPoissonMg::solve_dev`], for the device-resident AMR step. Returns iters.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_dev(
        &self,
        rhs: &DeviceBuffer<f64>,
        x0: Option<&DeviceBuffer<f64>>,
        out: &mut DeviceBuffer<f64>,
        reaction: f64,
        neumann_tags: &[u32],
        deflate: bool,
        tol: f64,
        maxit: usize,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let stream = &self.stream;
        let module = &self.module;
        let ndof = self.ndof;

        let mut fnbr = self.fnbr_base.clone();
        if !neumann_tags.is_empty() {
            for &(idx, tag) in &self.bnodes {
                if neumann_tags.contains(&tag) {
                    fnbr[idx] = NEU;
                }
            }
        }
        let fnbr_dev = DeviceBuffer::from_host(stream, &fnbr)?;

        let nb = dot_blocks(ndof);
        let mut r = DeviceBuffer::<f64>::zeroed(stream, ndof)?;
        let mut p = DeviceBuffer::<f64>::zeroed(stream, ndof)?;
        let mut ap = DeviceBuffer::<f64>::zeroed(stream, ndof)?;
        let mut gx = DeviceBuffer::<f64>::zeroed(stream, ndof)?;
        let mut gy = DeviceBuffer::<f64>::zeroed(stream, ndof)?;
        let mut partial = DeviceBuffer::<f64>::zeroed(stream, nb)?;
        let ones = DeviceBuffer::from_host(stream, &vec![1.0f64; ndof])?;

        let cfg = LaunchConfig { grid_dim: (self.ne as u32, 1, 1), block_dim: (self.nn as u32, 1, 1), shared_mem_bytes: 0 };
        let red = LaunchConfig { grid_dim: (nb as u32, 1, 1), block_dim: (RED as u32, 1, 1), shared_mem_bytes: 0 };
        let vec_cfg = LaunchConfig::for_num_elems(ndof as u32);
        let n1 = self.n1;
        let n64 = ndof as u64;
        let ninv = 1.0 / ndof as f64;

        macro_rules! dot {
            ($a:expr, $b:expr) => {{
                module.dot_nc_partial(stream, red, $a, $b, n64, &mut partial)?;
                partial.to_host_vec(stream)?.iter().sum::<f64>()
            }};
        }
        macro_rules! deflate_v {
            ($v:expr) => {{
                if deflate {
                    let mean = dot!($v, &ones) * ninv;
                    module.axpy_nc(stream, vec_cfg, $v, &ones, -mean)?;
                }
            }};
        }
        macro_rules! apply {
            ($field:expr, $dst:expr) => {{
                module.gradient_nc(stream, cfg, &self.d_dev, $field, &self.rx_dev, &self.ry_dev, &self.sx_dev, &self.sy_dev, n1, &mut gx, &mut gy)?;
                module.operator_nc(
                    stream, op_launch_cfg(self.ne, self.nn), &self.d_dev, $field, &gx, &gy, &self.rx_dev, &self.ry_dev, &self.sx_dev, &self.sy_dev, &self.jw_dev, n1, self.ne as u32,
                    &self.fvl_dev, &self.fnx_dev, &self.fny_dev, &self.fsw_dev, &fnbr_dev, &self.ekind_dev, &self.enx_dev, &self.eny_dev,
                    &self.etau_dev, &self.half0_dev, &self.ss_dev, &self.ssw_dev, &self.nbr0_dev, &self.nbr1_dev, &self.swf0_dev, &self.swf1_dev,
                    &self.p0_dev, &self.p1_dev, reaction, $dst,
                )?;
            }};
        }

        // r = b − A·x0 (warm start), x = x0; or r = b, x = 0.
        match x0 {
            Some(x0d) => {
                module.xpby_nc(stream, vec_cfg, out, x0d, 0.0)?; // out = x0
                apply!(&*out, &mut ap);
                module.xpby_nc(stream, vec_cfg, &mut r, rhs, 0.0)?; // r = b
                module.axpy_nc(stream, vec_cfg, &mut r, &ap, -1.0)?; // r -= A·x0
            }
            None => {
                module.xpby_nc(stream, vec_cfg, out, rhs, 0.0)?; // out = b (temp)
                module.axpy_nc(stream, vec_cfg, out, rhs, -1.0)?; // out = 0
                module.xpby_nc(stream, vec_cfg, &mut r, rhs, 0.0)?; // r = b
            }
        }
        deflate_v!(&mut r);
        module.xpby_nc(stream, vec_cfg, &mut p, &r, 0.0)?; // p = r
        let bn = dot!(&r, &r).sqrt().max(1e-300);
        let mut rs = dot!(&r, &r);
        let mut iters = 0;
        for it in 0..maxit {
            // Convergence check BEFORE forming α — this also catches a zero / near-zero RHS (e.g. the
            // first projection step where div=0 ⇒ rs=0), which would otherwise give the deflated-CG
            // 0/0 = NaN breakdown the conforming path guards against. Then x stays its (zero) iterate.
            if rs.sqrt() / bn < tol {
                iters = it;
                break;
            }
            apply!(&p, &mut ap);
            let pap = dot!(&p, &ap);
            if !(pap > 0.0) {
                iters = it; // operator breakdown (singular direction) ⇒ stop with the current iterate
                break;
            }
            let alpha_cg = rs / pap;
            module.axpy_nc(stream, vec_cfg, out, &p, alpha_cg)?;
            module.axpy_nc(stream, vec_cfg, &mut r, &ap, -alpha_cg)?;
            deflate_v!(&mut r);
            let rs_new = dot!(&r, &r);
            iters = it + 1;
            if rs_new.sqrt() / bn < tol {
                break;
            }
            let beta = rs_new / rs;
            module.xpby_nc(stream, vec_cfg, &mut p, &r, beta)?;
            rs = rs_new;
        }
        Ok(iters)
    }
}
