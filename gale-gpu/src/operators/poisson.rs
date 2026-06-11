//! GPU SIPG-Poisson solver kernels — reusable library component.
//!
//! Consolidates the three one-off Poisson GPU ports (operator action, CG, and
//! p-multigrid-preconditioned CG) behind a single `#[cuda_module]` and three host
//! launch wrappers. All kernels are **order-agnostic** (runtime `n1`, shared sized
//! to [`NN_MAX`]), so one set serves every p-multigrid level. The device math is
//! the matrix-free SIPG operator `A·u` = volume stiffness + symmetry-lift + face
//! consistency/penalty, evaluated as a 2-kernel pipeline (`gradient` → `operator`).
//!
//! - [`poisson_apply`] — one application of `A·u` (validated vs `Poisson::apply`).
//! - [`poisson_cg_solve`] — full device-resident CG (validated vs `Poisson::cg`).
//! - [`poisson_pcg_solve`] — full device-resident p-multigrid PCG, V-cycle and all
//!   smoother/transfer steps on the GPU (validated vs `PMultigrid::pcg`).
//!
//! Setup (meshes, transfer matrices, diagonals, smoother weights) is reused from
//! the validated CPU `Poisson` / `PMultigrid`; only the iterations run on-device.
//! Libdevice-free ⇒ the embedded path works on sm_70.

use cuda_core::{CaptureMode, CudaContext, CudaGraphExec, CudaStream, DeviceBuffer, DeviceCopy, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice, DynamicSharedArray, SharedArray};
use std::sync::Arc;
use cuda_host::cuda_module;
use gale::dg::{Edge, Mesh2d, Neighbor, PMultigrid};

const NN_MAX: usize = 81; // (order 8 + 1)²
const RED: usize = 256; // reduction block size
const BND: u32 = u32::MAX; // sentinel: Dirichlet boundary face (SIPG consistency+penalty)
const NEU: u32 = u32::MAX - 1; // sentinel: Neumann boundary face (natural BC ⇒ no contribution)

/// Storage scalar for the matvec / V-cycle, with widen/narrow to the f64 accumulation type. The
/// matvec is **f32-storage / f64-accumulate**: field vectors live in `T` (halving DRAM traffic when
/// `T = f32`) but every contraction accumulates in f64. `T = f64` ⇒ both casts are the identity, so
/// the `::<f64>` instantiation is **bit-for-bit the original FP64 kernel** — the across-the-board
/// FP64 path stays first-class and mixed precision is an opt-in (`T = f32`), never a requirement.
/// (Feasibility + generic-#[kernel] monomorphization proven by probe-mixed-precision / probe-generic-kernel.)
pub trait Scalar: Copy {
    fn to_f64(self) -> f64;
    fn from_f64(x: f64) -> Self;
}
impl Scalar for f64 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline(always)]
    fn from_f64(x: f64) -> Self {
        x
    }
}
impl Scalar for f32 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn from_f64(x: f64) -> Self {
        x as f32
    }
}

/// Number of blocks for the multi-block `dot_partial` reduction: enough to stream the
/// vector across all SMs (one block-partial each), capped so the host-side final sum of
/// the partials stays trivial. `RED` threads per block. Volta has 80 SMs; 1024 blocks ×
/// 256 threads saturates occupancy while summing only 1024 doubles on the host.
fn dot_blocks(ndof: usize) -> usize {
    ndof.div_ceil(RED).clamp(1, 1024)
}

/// Launch configs for the multi-element-per-block matvec (`gradient`, `operator`) over `ne`
/// elements at order `n1-1`. Packs `epb` elements per block targeting ~192 threads (≥1), so the
/// block runs enough warps to hide memory latency (the ncu-identified fix: one element = `nn`
/// threads = 25 at p=4 starved the schedulers). Dynamic shared = the diff matrix `DS[nn]` (once)
/// plus per-element tiles: `gradient` needs `US` (1 tile/elem), `operator` needs `PR`+`PS` (2).
/// Returns `(gradient_cfg, operator_cfg)` — same grid/block, different `shared_mem_bytes`.
fn matvec_cfgs(ne: usize, n1: u32) -> (LaunchConfig, LaunchConfig) {
    let nn = (n1 as usize) * (n1 as usize);
    let epb = 192usize.div_ceil(nn).max(1);
    let grid = ne.div_ceil(epb) as u32;
    let block = (epb * nn) as u32;
    let g = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: ((nn + epb * nn) * 8) as u32 };
    let o = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: ((nn + 2 * epb * nn) * 8) as u32 };
    (g, o)
}

/// Convergence guard for the iterative elliptic solves (CG and MG-PCG alike): emit a
/// one-line stderr warning when a solve exhausts `maxit` without reaching `tol`, so a
/// pathological / under-resolved solve **surfaces** instead of silently feeding an
/// inaccurate velocity or pressure back into the time integrator. `rel` is the achieved
/// relative residual `‖r‖/‖b‖` at the last iteration. Non-fatal (the last iterate is
/// still returned) — parity with the prior behaviour, but now observable.
fn warn_unconverged(label: &str, converged: bool, iters: usize, maxit: usize, rel: f64, tol: f64) {
    if !converged {
        eprintln!(
            "gale-gpu: WARNING: {label} did not converge — {iters}/{maxit} iters, \
             rel residual {rel:.3e} (tol {tol:.1e}); returning the last iterate."
        );
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    // ===== matvec (gradient → operator), MULTI-ELEMENT-PER-BLOCK ============================
    //
    // Each block processes `epb = blockDim_x / nn` elements (≥1), so the block runs ~128–256
    // threads (a multiple-ish of the warp size) instead of one element's `nn` (=25 at p=4) — the
    // ncu-identified fix for the latency/occupancy ceiling (one-element blocks left the SM
    // schedulers starved with too few warps to hide memory stalls). Thread `t` ↦ element-in-block
    // `el = t/nn`, node `m = t%nn`, global element `e = blockIdx·epb + el`. Dynamic shared holds
    // the (element-independent) diff matrix `DS[nn]` once plus one per-element tile per buffer.
    // `ne` guards the partial final block; inactive threads still reach `sync_threads`. Per-element
    // arithmetic is unchanged ⇒ bit-for-bit identical to the one-element kernels.

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn gradient<U: Scalar, G: Scalar>(
        d: &[f64], u: &[U], rx: f64, sy: f64, n1: u32, ne: u32,
        mut gx: DisjointSlice<G>, mut gy: DisjointSlice<G>,
    ) {
        // Two storage scalars, f64 accumulate: `U` = input field, `G` = gradient intermediate
        // written here (read back by `operator`). `U=G=f64` ⇒ identity casts ⇒ bit-exact original;
        // mixed mode uses `G=f32` to halve the gx/gy DRAM round-trip.
        let sm = DynamicSharedArray::<f64>::get(); // [ DS(nn) | US(epb·nn) ]
        let n1 = n1 as usize;
        let nn = n1 * n1;
        let t = thread::threadIdx_x() as usize;
        let epb = thread::blockDim_x() as usize / nn;
        let el = t / nn;
        let m = t % nn;
        let e = thread::blockIdx_x() as usize * epb + el;
        let active = e < ne as usize;
        let us = nn + el * nn; // this element's US tile base
        unsafe {
            if t < nn {
                *sm.add(t) = d[t]; // shared diff matrix, loaded once per block
            }
            if active {
                *sm.add(us + m) = u[e * nn + m].to_f64();
            }
        }
        thread::sync_threads();
        if !active {
            return; // no further barrier in this kernel ⇒ safe to bail
        }
        let i = m % n1;
        let j = m / n1;
        let mut ur = 0.0f64;
        let mut uss = 0.0f64;
        let mut k = 0usize;
        while k < n1 {
            unsafe {
                ur += *sm.add(i * n1 + k) * *sm.add(us + k + j * n1);
                uss += *sm.add(j * n1 + k) * *sm.add(us + i + k * n1);
            }
            k += 1;
        }
        // Affine axis-aligned metrics: gx = rx·u_r (sx=0), gy = sy·u_s (ry=0).
        let gxv = rx * ur;
        let gyv = sy * uss;
        // block_dim = epb·nn ⇒ global 1D thread index == e·nn + m.
        if let Some(o) = gx.get_mut(thread::index_1d()) {
            *o = G::from_f64(gxv);
        }
        if let Some(o) = gy.get_mut(thread::index_1d()) {
            *o = G::from_f64(gyv);
        }
    }

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn operator<U: Scalar, G: Scalar>(
        d: &[f64], u: &[U], gx: &[G], gy: &[G], mass: &[f64], n1: u32, ne: u32, rx: f64,
        sy: f64, jac: f64, _face_vl: &[u32], face_nx: &[f64], face_ny: &[f64],
        face_sw: &[f64], face_nbr: &[u32], face_tau: &[f64], lambda: f64, mut out: DisjointSlice<U>,
    ) {
        // Two storage scalars, f64 accumulate: `U` = field/output, `G` = gradient intermediate
        // (read here, written by `gradient`). Shared PR/PS, metrics, mass stay f64; global field
        // reads widen via to_f64, the result narrows via from_f64. `U=G=f64` ⇒ bit-for-bit FP64.
        let sm = DynamicSharedArray::<f64>::get(); // [ DS(nn) | PR(epb·nn) | PS(epb·nn) ]
        let n1 = n1 as usize;
        let nn = n1 * n1;
        let t = thread::threadIdx_x() as usize;
        let epb = thread::blockDim_x() as usize / nn;
        let el = t / nn;
        let m = t % nn;
        let e = thread::blockIdx_x() as usize * epb + el;
        let active = e < ne as usize;
        let pr = nn + el * nn; // this element's PR tile base
        let ps = nn + epb * nn + el * nn; // this element's PS tile base
        let b = e * nn + m;
        unsafe {
            if t < nn {
                *sm.add(t) = d[t]; // shared diff matrix (DS), loaded once per block
            }
        }
        // affine: per-node jw = constant Jacobian × the (tiny, well-cached) GLL mass diagonal.
        let jw_b = jac * mass[m];
        let mut rf = 0.0f64;
        if active {
            unsafe {
                let wx = jw_b * gx[b].to_f64();
                let wy = jw_b * gy[b].to_f64();
                *sm.add(pr + m) = rx * wx; // ry = 0
                *sm.add(ps + m) = sy * wy; // sx = 0
            }
            // Face contribution by **gather** (race-free, fully parallel): thread `m` owns
            // node `m`, visits the ≤2 faces it lies on (corner ⇒ 2), and accumulates the SIPG
            // consistency/penalty `rf` and the symmetry-lift `hx,hy` into registers — so no
            // shared `RF/HX/HY` and no inter-thread races (each node written by one thread).
            // Node m = (ii, jj)'s face position `a` follows the tensor face-node convention (see
            // `quad_faces`): South/North run along i (a=ii), East/West along j (a=jj). The
            // bit-for-bit operator validator confirms the convention.
            let ii = m % n1;
            let jj = m / n1;
            let mut hx = 0.0f64;
            let mut hy = 0.0f64;
            let mut t4 = 0usize;
            while t4 < 4 {
                let (on, a) = if t4 == 0 {
                    (jj == 0, ii) // South
                } else if t4 == 1 {
                    (ii == n1 - 1, jj) // East
                } else if t4 == 2 {
                    (jj == n1 - 1, ii) // North
                } else {
                    (ii == 0, jj) // West
                };
                if on {
                    let idx = (e * 4 + t4) * n1 + a;
                    let nbr = face_nbr[idx];
                    if nbr != NEU {
                        let tau = face_tau[e * 4 + t4];
                        let nx = face_nx[idx];
                        let ny = face_ny[idx];
                        let sw = face_sw[idx];
                        let dun_e = nx * gx[b].to_f64() + ny * gy[b].to_f64();
                        let ug = u[b].to_f64();
                        let (avg, jump, gfac) = if nbr == BND {
                            (dun_e, ug, 1.0)
                        } else {
                            let ng = nbr as usize;
                            (0.5 * (dun_e + nx * gx[ng].to_f64() + ny * gy[ng].to_f64()), ug - u[ng].to_f64(), 0.5)
                        };
                        let g = gfac * sw * jump;
                        rf += -sw * avg + tau * sw * jump;
                        hx += g * nx;
                        hy += g * ny;
                    }
                }
                t4 += 1;
            }
            unsafe {
                *sm.add(pr + m) -= rx * hx; // ry = 0
                *sm.add(ps + m) -= sy * hy; // sx = 0
            }
        }
        thread::sync_threads(); // ALL threads (active + inactive padding) reach this barrier
        if !active {
            return;
        }
        let i = m % n1;
        let j = m / n1;
        let mut acc = 0.0f64;
        let mut k = 0usize;
        while k < n1 {
            unsafe {
                acc += *sm.add(k * n1 + i) * *sm.add(pr + k + j * n1)
                    + *sm.add(k * n1 + j) * *sm.add(ps + i + k * n1);
            }
            k += 1;
        }
        if let Some(o) = out.get_mut(thread::index_1d()) {
            // SIPG stiffness `A·u` plus the Helmholtz reaction `λ·M·u` (diagonal GLL
            // mass `M = diag(jw)`). `λ = 0` ⇒ pure Poisson, bit-identical to before.
            *o = U::from_f64(acc + rf + lambda * jw_b * u[b].to_f64());
        }
    }

    /// **Fused matvec + damped-Jacobi smoother step.** Computes `Ap = A·u` exactly as
    /// [`operator`], then writes the smoother update `out ← u + ω·invd·(rhs − Ap)` instead of
    /// `Ap` — so the per-node `Ap` never round-trips through DRAM and the separate `jacobi`
    /// launch is eliminated. `out` MUST be a distinct buffer from `u` (the operator reads
    /// neighbour `u`/gradients, so an in-place write would race across blocks); the V-cycle
    /// ping-pongs `xb ↔ sm`. Bit-identical to `operator` then `jacobi` (same arithmetic, same
    /// order). The body is a deliberate copy of [`operator`] up to the final write (cuda-oxide
    /// codegens each `#[kernel]` independently; a shared device helper with shared-mem + a
    /// barrier is unproven, so the duplication buys guaranteed codegen + the bit-exact validator).
    #[kernel]
    pub fn operator_jacobi<U: Scalar, G: Scalar>(
        d: &[f64], u: &[U], gx: &[G], gy: &[G], mass: &[f64], n1: u32, ne: u32, rx: f64,
        sy: f64, jac: f64, _face_vl: &[u32], face_nx: &[f64], face_ny: &[f64],
        face_sw: &[f64], face_nbr: &[u32], face_tau: &[f64], lambda: f64, rhs: &[f64],
        invd: &[f64], omega: f64, mut out: DisjointSlice<U>,
    ) {
        let sm = DynamicSharedArray::<f64>::get(); // [ DS(nn) | PR(epb·nn) | PS(epb·nn) ]
        let n1 = n1 as usize;
        let nn = n1 * n1;
        let t = thread::threadIdx_x() as usize;
        let epb = thread::blockDim_x() as usize / nn;
        let el = t / nn;
        let m = t % nn;
        let e = thread::blockIdx_x() as usize * epb + el;
        let active = e < ne as usize;
        let pr = nn + el * nn;
        let ps = nn + epb * nn + el * nn;
        let b = e * nn + m;
        unsafe {
            if t < nn {
                *sm.add(t) = d[t];
            }
        }
        let jw_b = jac * mass[m];
        let mut rf = 0.0f64;
        if active {
            unsafe {
                let wx = jw_b * gx[b].to_f64();
                let wy = jw_b * gy[b].to_f64();
                *sm.add(pr + m) = rx * wx;
                *sm.add(ps + m) = sy * wy;
            }
            let ii = m % n1;
            let jj = m / n1;
            let mut hx = 0.0f64;
            let mut hy = 0.0f64;
            let mut t4 = 0usize;
            while t4 < 4 {
                let (on, a) = if t4 == 0 {
                    (jj == 0, ii)
                } else if t4 == 1 {
                    (ii == n1 - 1, jj)
                } else if t4 == 2 {
                    (jj == n1 - 1, ii)
                } else {
                    (ii == 0, jj)
                };
                if on {
                    let idx = (e * 4 + t4) * n1 + a;
                    let nbr = face_nbr[idx];
                    if nbr != NEU {
                        let tau = face_tau[e * 4 + t4];
                        let nx = face_nx[idx];
                        let ny = face_ny[idx];
                        let sw = face_sw[idx];
                        let dun_e = nx * gx[b].to_f64() + ny * gy[b].to_f64();
                        let ug = u[b].to_f64();
                        let (avg, jump, gfac) = if nbr == BND {
                            (dun_e, ug, 1.0)
                        } else {
                            let ng = nbr as usize;
                            (0.5 * (dun_e + nx * gx[ng].to_f64() + ny * gy[ng].to_f64()), ug - u[ng].to_f64(), 0.5)
                        };
                        let g = gfac * sw * jump;
                        rf += -sw * avg + tau * sw * jump;
                        hx += g * nx;
                        hy += g * ny;
                    }
                }
                t4 += 1;
            }
            unsafe {
                *sm.add(pr + m) -= rx * hx;
                *sm.add(ps + m) -= sy * hy;
            }
        }
        thread::sync_threads();
        if !active {
            return;
        }
        let i = m % n1;
        let j = m / n1;
        let mut acc = 0.0f64;
        let mut k = 0usize;
        while k < n1 {
            unsafe {
                acc += *sm.add(k * n1 + i) * *sm.add(pr + k + j * n1)
                    + *sm.add(k * n1 + j) * *sm.add(ps + i + k * n1);
            }
            k += 1;
        }
        if let Some(o) = out.get_mut(thread::index_1d()) {
            // Ap = A·u (SIPG + λM), then the damped-Jacobi update folded in: out ← u + ω·invd·(rhs − Ap).
            let ap = acc + rf + lambda * jw_b * u[b].to_f64();
            *o = U::from_f64(u[b].to_f64() + omega * invd[b] * (rhs[b] - ap));
        }
    }

    /// y ← y + ω·invd·(b − ap)  (damped-Jacobi update)
    #[kernel]
    pub fn jacobi(mut y: DisjointSlice<f64>, b: &[f64], ap: &[f64], invd: &[f64], omega: f64) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = y.get_mut(idx) {
            *o += omega * invd[i] * (b[i] - ap[i]);
        }
    }

    /// out ← a − b
    #[kernel]
    pub fn sub(mut out: DisjointSlice<f64>, a: &[f64], b: &[f64]) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = a[i] - b[i];
        }
    }

    /// y ← c·y
    #[kernel]
    pub fn scal(mut y: DisjointSlice<f64>, c: f64) {
        let idx = thread::index_1d();
        if let Some(o) = y.get_mut(idx) {
            *o *= c;
        }
    }

    /// y ← y + a·x
    #[kernel]
    pub fn axpy(mut y: DisjointSlice<f64>, x: &[f64], a: f64) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = y.get_mut(idx) {
            *o += a * x[i];
        }
    }

    /// y ← x + b·y
    #[kernel]
    pub fn xpby(mut y: DisjointSlice<f64>, x: &[f64], b: f64) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = y.get_mut(idx) {
            *o = x[i] + b * *o;
        }
    }

    /// Multi-block grid-stride dot product: each of `gridDim` blocks reduces its
    /// strided slice of `a·b` into shared memory and writes one partial to
    /// `partial[blockIdx]`. The host (or a second reduction) sums the `gridDim`
    /// partials. With `gridDim = 1` this is the old single-block reduction (writes
    /// `partial[0]`); with many blocks it streams `a`/`b` across all SMs at ~peak
    /// bandwidth instead of saturating a single SM.
    #[kernel]
    pub fn dot_partial(a: &[f64], b: &[f64], n: u64, mut partial: DisjointSlice<f64>) {
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

    /// Single-block final reduction of the `n` block-partials from `dot_partial` into the
    /// device scalar `out[0]`. Completes a fully on-device dot (`dot_partial → reduce_scalar`)
    /// so the result never leaves the GPU — the CG scalars stay device-resident and an
    /// iteration issues no host sync. `n ≤ dot_blocks(ndof) ≤ 1024`.
    #[kernel]
    pub fn reduce_scalar(partial: &[f64], n: u64, mut out: DisjointSlice<f64>) {
        static mut SH: SharedArray<f64, RED> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let stride = thread::blockDim_x() as usize;
        let mut acc = 0.0f64;
        let mut i = tid;
        while i < n as usize {
            acc += partial[i];
            i += stride;
        }
        unsafe { SH[tid] = acc; }
        thread::sync_threads();
        let mut s = stride / 2;
        while s > 0 {
            if tid < s {
                unsafe { SH[tid] = SH[tid] + SH[tid + s]; }
            }
            thread::sync_threads();
            s /= 2;
        }
        if tid == 0 {
            unsafe { *out.get_unchecked_mut(0) = SH[0]; }
        }
    }

    /// CG step coefficient `α = rs/pAp`, on-device (1 thread). Writes both `α` and `−α`
    /// (the latter for the residual update `r −= α·Ap` via [`axpy_s`]).
    #[kernel]
    pub fn cg_alpha(rs: &[f64], pap: &[f64], mut alpha: DisjointSlice<f64>, mut nalpha: DisjointSlice<f64>) {
        if thread::threadIdx_x() == 0 {
            let a = rs[0] / pap[0];
            unsafe {
                *alpha.get_unchecked_mut(0) = a;
                *nalpha.get_unchecked_mut(0) = -a;
            }
        }
    }

    /// CG step coefficient `β = rs_new/rs`, on-device (1 thread), and advance the residual
    /// scalar `rs ← rs_new` for the next iteration.
    #[kernel]
    pub fn cg_beta(rs_new: &[f64], mut rs: DisjointSlice<f64>, mut beta: DisjointSlice<f64>) {
        if thread::threadIdx_x() == 0 {
            let rn = rs_new[0];
            unsafe {
                let ro = *rs.get_unchecked_mut(0);
                *beta.get_unchecked_mut(0) = rn / ro;
                *rs.get_unchecked_mut(0) = rn;
            }
        }
    }

    /// In-place `s[0] ← −s[0]·ninv`, on-device (1 thread): turns a reduced `1ᵀr` sum into
    /// the negative arithmetic mean, for the deflation `r ← r − mean(r)·1` via [`axpy_s`].
    #[kernel]
    pub fn cg_negmean(mut s: DisjointSlice<f64>, ninv: f64) {
        if thread::threadIdx_x() == 0 {
            unsafe {
                let v = *s.get_unchecked_mut(0);
                *s.get_unchecked_mut(0) = -v * ninv;
            }
        }
    }

    /// `y ← y + a[0]·x` with the scalar read from a device buffer (the on-device-scalar
    /// companion to [`axpy`], so CG coefficients never round-trip to the host).
    #[kernel]
    pub fn axpy_s(mut y: DisjointSlice<f64>, x: &[f64], a: &[f64]) {
        let av = a[0];
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = y.get_mut(idx) {
            *o += av * x[i];
        }
    }

    /// `y ← x + b[0]·y` with the scalar read from a device buffer (on-device-scalar
    /// companion to [`xpby`]).
    #[kernel]
    pub fn xpby_s(mut y: DisjointSlice<f64>, x: &[f64], b: &[f64]) {
        let bv = b[0];
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = y.get_mut(idx) {
            *o = x[i] + bv * *o;
        }
    }

    /// Prolong coarse→fine, per element (block = fine nodes). `out[e·nf + m]`.
    #[kernel]
    pub fn prolong(interp: &[f64], coarse: &[f64], n1f: u32, n1c: u32, mut out: DisjointSlice<f64>) {
        static mut CS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut IM: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        let n1f = n1f as usize;
        let n1c = n1c as usize;
        let nc = n1c * n1c;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        unsafe {
            if m < nc {
                CS[m] = coarse[e * nc + m];
            }
            if m < n1f * n1c {
                IM[m] = interp[m];
            }
        }
        thread::sync_threads();
        let iff = m % n1f;
        let jf = m / n1f;
        let mut sacc = 0.0f64;
        let mut jc = 0usize;
        while jc < n1c {
            let mut ic = 0usize;
            while ic < n1c {
                sacc += unsafe { IM[iff * n1c + ic] * IM[jf * n1c + jc] * CS[ic + jc * n1c] };
                ic += 1;
            }
            jc += 1;
        }
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = sacc;
        }
    }

    /// Restrict fine→coarse, per element (block = fine nodes, write coarse for m<nc).
    #[kernel]
    pub fn restrict(interp: &[f64], fine: &[f64], n1f: u32, n1c: u32, mut out: DisjointSlice<f64>) {
        static mut FS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut IM: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        let n1f = n1f as usize;
        let n1c = n1c as usize;
        let nf = n1f * n1f;
        let nc = n1c * n1c;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        unsafe {
            FS[m] = fine[e * nf + m];
            if m < n1f * n1c {
                IM[m] = interp[m];
            }
        }
        thread::sync_threads();
        if m < nc {
            let ic = m % n1c;
            let jc = m / n1c;
            let mut sacc = 0.0f64;
            let mut jf = 0usize;
            while jf < n1f {
                let mut iff = 0usize;
                while iff < n1f {
                    sacc += unsafe { IM[iff * n1c + ic] * IM[jf * n1c + jc] * FS[iff + jf * n1f] };
                    iff += 1;
                }
                jf += 1;
            }
            unsafe {
                *out.get_unchecked_mut(e * nc + m) = sacc;
            }
        }
    }

    /// **h-prolong** (2:1 geometric, order-1 both levels): one block per FINE element, 4
    /// threads (the 4 fine nodes). The parent coarse element `ec` and child quadrant `q` are
    /// derived from the fine element index and the two grid widths `nxf`/`nxc`; `pq` holds the
    /// four 4×4 per-quadrant matrices `[q*16 + f*4 + a]`. `out[ef*4 + f]`. Matches the host
    /// `PMultigrid::prolong_h`.
    #[kernel]
    pub fn h_prolong(pq: &[f64], coarse: &[f64], nxf: u32, nxc: u32, mut out: DisjointSlice<f64>) {
        let ef = thread::blockIdx_x() as usize;
        let f = thread::threadIdx_x() as usize;
        if f >= 4 {
            return;
        }
        let (nxf, nxc) = (nxf as usize, nxc as usize);
        let (fx, fy) = (ef % nxf, ef / nxf);
        let ec = (fx / 2) + (fy / 2) * nxc;
        let q = (fx % 2) + 2 * (fy % 2);
        let mut s = 0.0f64;
        let mut a = 0usize;
        while a < 4 {
            s += pq[q * 16 + f * 4 + a] * coarse[ec * 4 + a];
            a += 1;
        }
        // block_dim is 4 (order-1) ⇒ global 1D thread index == ef*4 + f.
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = s;
        }
    }

    /// **h-restrict** (the transpose `Rᵀ` of [`h_prolong`]): one block per COARSE element, 4
    /// threads (the 4 coarse nodes `a`). Each coarse element gathers its 4 fine children
    /// (quadrants), accumulating `pqᵀ`-weighted child contributions. `out[ec*4 + a]`. Matches
    /// the host `PMultigrid::restrict_h`.
    #[kernel]
    pub fn h_restrict(pq: &[f64], fine: &[f64], nxf: u32, nxc: u32, mut out: DisjointSlice<f64>) {
        let ec = thread::blockIdx_x() as usize;
        let a = thread::threadIdx_x() as usize;
        if a >= 4 {
            return;
        }
        let (nxf, nxc) = (nxf as usize, nxc as usize);
        let (cx, cy) = (ec % nxc, ec / nxc);
        let mut s = 0.0f64;
        let mut q = 0usize;
        while q < 4 {
            let (qx, qy) = (q % 2, q / 2);
            let ef = (2 * cx + qx) + (2 * cy + qy) * nxf;
            let mut f = 0usize;
            while f < 4 {
                s += pq[q * 16 + f * 4 + a] * fine[ef * 4 + f];
                f += 1;
            }
            q += 1;
        }
        // block_dim is 4 (order-1) ⇒ global 1D thread index == ec*4 + a.
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = s;
        }
    }
}

/// Per-element metrics + flattened face metadata for one mesh, ready to upload.
/// Built host-side exactly as the original Poisson bins did, then shared by the
/// `gradient`/`operator` launches.
struct MeshArrays {
    nn: usize,
    ne: usize,
    ndof: usize,
    n1: u32,
    diff: Vec<f64>,
    /// 2D GLL mass diagonal `w_i·w_j` (length `nn`), shared by all elements — `jw = jac·mass`.
    mass: Vec<f64>,
    /// **Affine metrics** for a uniform axis-aligned rectangular mesh: `rx = 2/dx`, `sy = 2/dy`
    /// are constant across the whole mesh and `ry = sx = 0`, and the Jacobian `jac` is constant.
    /// So the 5 per-node metric arrays collapse to these 3 scalars + the shared `mass` — the
    /// matvec stops streaming `rx/ry/sx/sy/jw` from DRAM entirely (the big memory-bound win).
    rx: f64,
    sy: f64,
    jac: f64,
    fvl: Vec<u32>,
    fnx: Vec<f64>,
    fny: Vec<f64>,
    fsw: Vec<f64>,
    fnbr: Vec<u32>,
    ftau: Vec<f64>,
}

/// Flatten a mesh's metrics and SIPG face metadata (penalty `tau` from `alpha`),
/// matching the host-side assembly in the original `gpu_poisson_*` bins. A boundary
/// face whose tag is in `neumann_tags` is marked Neumann (`NEU`, natural BC ⇒ no
/// operator contribution); all other boundary faces stay Dirichlet SIPG (`BND`). So
/// `&[]` is all-Dirichlet (velocity Helmholtz) and `&mesh.boundary_tags()` is the
/// pure-Neumann pressure-Poisson; a partial set is the per-region (inflow/outflow)
/// case where outflow tags are Dirichlet (`p=0`) and the rest Neumann.
fn flatten_mesh(mesh: &Mesh2d, alpha: f64, neumann_tags: &[u32]) -> MeshArrays {
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let order = mesh.order;
    let n1 = (order + 1) as u32;
    assert!(nn <= NN_MAX, "p too large for NN_MAX={NN_MAX}");

    // Affine-metric extraction: a uniform axis-aligned rectangular mesh has rx, sy, jac constant
    // across every node of every element and ry = sx = 0. Take them from element 0 and ASSERT the
    // whole mesh matches — so a non-axis-aligned / non-uniform mesh (which would need the per-node
    // metrics) fails loudly here rather than silently producing the wrong operator.
    let (rx, sy, jac) = (mesh.elements[0].geom.rx[0], mesh.elements[0].geom.sy[0], mesh.elements[0].geom.jac[0]);
    let tol = 1e-9 * (1.0 + rx.abs() + sy.abs() + jac.abs());
    for el in &mesh.elements {
        for k in 0..nn {
            assert!(
                (el.geom.rx[k] - rx).abs() < tol
                    && (el.geom.sy[k] - sy).abs() < tol
                    && (el.geom.jac[k] - jac).abs() < tol
                    && el.geom.ry[k].abs() < tol
                    && el.geom.sx[k].abs() < tol,
                "gale-gpu matvec requires a uniform axis-aligned rectangular mesh (affine metrics)"
            );
        }
    }
    let mass = mesh.refq.mass.clone(); // nn, the 2D GLL mass diagonal w_i·w_j (jw = jac·mass)

    let p1 = (order + 1) as f64;
    let h: Vec<f64> = mesh.elements.iter().map(|el| el.geom.jw.iter().sum::<f64>().sqrt()).collect();
    let n1u = n1 as usize;
    let nfc = ne * 4 * n1u;
    let (mut fvl, mut fnx, mut fny, mut fsw, mut fnbr, mut ftau) =
        (vec![0u32; nfc], vec![0.0; nfc], vec![0.0; nfc], vec![0.0; nfc], vec![BND; nfc], vec![0.0; ne * 4]);
    for (e, el) in mesh.elements.iter().enumerate() {
        for (t, edge) in Edge::ALL.iter().enumerate() {
            let face = &el.faces[*edge as usize];
            let nb = &el.neighbors[*edge as usize];
            ftau[e * 4 + t] = match nb {
                Neighbor::Interior { elem: re, .. } => alpha * p1 * p1 / h[e].min(h[*re]),
                Neighbor::Boundary { .. } => alpha * p1 * p1 / h[e],
                Neighbor::CoarseToFine { .. } | Neighbor::FineToCoarse { .. } => unreachable!(),
            };
            let neumann_boundary = matches!(nb, Neighbor::Boundary { tag } if neumann_tags.contains(tag));
            for a in 0..n1u {
                let idx = (e * 4 + t) * n1u + a;
                fvl[idx] = face.nodes[a] as u32;
                fnx[idx] = face.nx[a];
                fny[idx] = face.ny[a];
                fsw[idx] = face.sw[a];
                if let Neighbor::Interior { elem: re, edge: redge, perm } = nb {
                    let rf = &mesh.elements[*re].faces[*redge as usize];
                    fnbr[idx] = (*re * nn + rf.nodes[perm[a]]) as u32;
                } else if neumann_boundary {
                    fnbr[idx] = NEU; // natural BC: kernel skips this face (else stays BND)
                }
            }
        }
    }

    MeshArrays {
        nn,
        ne,
        ndof,
        n1,
        diff: mesh.refq.line.diff.clone(),
        mass,
        rx,
        sy,
        jac,
        fvl,
        fnx,
        fny,
        fsw,
        fnbr,
        ftau,
    }
}

/// Apply the matrix-free SIPG Poisson operator `A·u` once on the GPU, returning the
/// nodal result. Reusable host wrapper around the [`kernels::gradient`] →
/// [`kernels::operator`] 2-kernel pipeline: flattens the mesh metrics + SIPG face
/// metadata (penalty from `alpha`), uploads, launches one block per element, and
/// gathers the result. Bit-for-bit equal to `gale::dg::Poisson::apply`.
///
/// `u` must have length `n_elements · n_nodes`.
pub fn poisson_apply(
    mesh: &Mesh2d,
    u: &[f64],
    alpha: f64,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let ma = flatten_mesh(mesh, alpha, &[]);
    assert_eq!(u.len(), ma.ndof, "state length must be n_elements·n_nodes");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
    let upu = |v: &[u32]| DeviceBuffer::from_host(&stream, v);

    let d_dev = up(&ma.diff)?;
    let u_dev = up(u)?;
    let mass_dev = up(&ma.mass)?;
    let fvl_dev = upu(&ma.fvl)?;
    let fnx_dev = up(&ma.fnx)?;
    let fny_dev = up(&ma.fny)?;
    let fsw_dev = up(&ma.fsw)?;
    let fnbr_dev = upu(&ma.fnbr)?;
    let ftau_dev = up(&ma.ftau)?;
    let mut gx_dev = DeviceBuffer::<f64>::zeroed(&stream, ma.ndof)?;
    let mut gy_dev = DeviceBuffer::<f64>::zeroed(&stream, ma.ndof)?;
    let mut out_dev = DeviceBuffer::<f64>::zeroed(&stream, ma.ndof)?;

    let module = kernels::load(&ctx)?;
    let nev = ma.ne as u32;
    let (gcfg, ocfg) = matvec_cfgs(ma.ne, ma.n1);
    module.gradient::<f64, f64>(&stream, gcfg, &d_dev, &u_dev, ma.rx, ma.sy, ma.n1, nev, &mut gx_dev, &mut gy_dev)?;
    module.operator::<f64, f64>(
        &stream, ocfg, &d_dev, &u_dev, &gx_dev, &gy_dev, &mass_dev, ma.n1, nev, ma.rx, ma.sy, ma.jac,
        &fvl_dev, &fnx_dev, &fny_dev, &fsw_dev, &fnbr_dev, &ftau_dev, 0.0, &mut out_dev,
    )?;
    Ok(out_dev.to_host_vec(&stream)?)
}

/// Solve the SIPG Poisson system `A·u = b` by **conjugate gradient entirely on the
/// GPU**, returning the solution and the iteration count. Vectors stay resident on
/// the device; only the CG scalars (`α`, `β`, residual) transfer host-side. The
/// convergence criterion (`‖r‖ / ‖b‖ < tol`) and reductions mirror
/// `gale::dg::Poisson::cg` exactly. `b` is the already-assembled right-hand side
/// (e.g. from `Poisson::rhs`), length `n_elements · n_nodes`.
pub fn poisson_cg_solve(
    mesh: &Mesh2d,
    b: &[f64],
    alpha: f64,
    tol: f64,
    maxit: usize,
) -> Result<(Vec<f64>, usize), Box<dyn std::error::Error>> {
    cg_solve_impl(mesh, b, alpha, 0.0, &[], tol, maxit)
}

/// Solve the SIPG **Helmholtz** system `(λM + A)·u = b` by conjugate gradient on the
/// GPU, returning the solution and iteration count. Identical to [`poisson_cg_solve`]
/// but the operator carries the diagonal-mass reaction term `λM` (`reaction = λ`),
/// matching `gale::dg::Poisson::with_reaction(mesh, alpha, λ).cg(b, …)`. This is the
/// viscous-velocity solve of the dual-splitting Stokes/NS scheme.
pub fn helmholtz_cg_solve(
    mesh: &Mesh2d,
    b: &[f64],
    alpha: f64,
    reaction: f64,
    tol: f64,
    maxit: usize,
) -> Result<(Vec<f64>, usize), Box<dyn std::error::Error>> {
    cg_solve_impl(mesh, b, alpha, reaction, &[], tol, maxit)
}

/// Tag-aware CG for `(reaction·M + A)·x = b` on the GPU: boundary tags in
/// `neumann_tags` are natural (Neumann), the rest Dirichlet SIPG — the per-region
/// (inflow/outflow/wall) boundary path. The operator is non-singular as long as at
/// least one boundary is Dirichlet, so this is the *non-deflated* solve used for both
/// the viscous-velocity Helmholtz (`neumann_tags` = outflow) and the pressure-Poisson
/// when an outflow pins it (`reaction = 0`, `neumann_tags` = everything except outflow).
/// Mirrors `gale::dg::Poisson::with_bc(mesh, alpha, reaction, neumann_tags).cg`.
pub fn helmholtz_cg_solve_tags(
    mesh: &Mesh2d,
    b: &[f64],
    alpha: f64,
    reaction: f64,
    neumann_tags: &[u32],
    tol: f64,
    maxit: usize,
) -> Result<(Vec<f64>, usize), Box<dyn std::error::Error>> {
    cg_solve_impl(mesh, b, alpha, reaction, neumann_tags, tol, maxit)
}

/// Unpreconditioned CG for `(reaction·M + A)·x = b` on the GPU. `reaction = 0` is the
/// pure SIPG Poisson; `reaction = λ > 0` is the viscous Helmholtz operator.
fn cg_solve_impl(
    mesh: &Mesh2d,
    b: &[f64],
    alpha: f64,
    reaction: f64,
    neumann_tags: &[u32],
    tol: f64,
    maxit: usize,
) -> Result<(Vec<f64>, usize), Box<dyn std::error::Error>> {
    let ma = flatten_mesh(mesh, alpha, neumann_tags);
    let ndof = ma.ndof;
    assert_eq!(b.len(), ndof, "rhs length must be n_elements·n_nodes");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
    let upu = |v: &[u32]| DeviceBuffer::from_host(&stream, v);

    let d_dev = up(&ma.diff)?;
    let mass_dev = up(&ma.mass)?;
    let fvl_dev = upu(&ma.fvl)?;
    let fnx_dev = up(&ma.fnx)?;
    let fny_dev = up(&ma.fny)?;
    let fsw_dev = up(&ma.fsw)?;
    let fnbr_dev = upu(&ma.fnbr)?;
    let ftau_dev = up(&ma.ftau)?;

    // CG vectors (resident on device). r = b − A·0 = b, p = r.
    let mut x = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut r = up(b)?;
    let mut p = up(b)?;
    let mut ap = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut gx = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut gy = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let nb = dot_blocks(ndof);
    let mut partial = DeviceBuffer::<f64>::zeroed(&stream, nb)?;
    // CG scalars kept **on-device** (length-1 buffers) so an iteration issues no host sync.
    let mut d_rs = DeviceBuffer::<f64>::zeroed(&stream, 1)?;
    let mut d_rsnew = DeviceBuffer::<f64>::zeroed(&stream, 1)?;
    let mut d_pap = DeviceBuffer::<f64>::zeroed(&stream, 1)?;
    let mut d_alpha = DeviceBuffer::<f64>::zeroed(&stream, 1)?;
    let mut d_nalpha = DeviceBuffer::<f64>::zeroed(&stream, 1)?;
    let mut d_beta = DeviceBuffer::<f64>::zeroed(&stream, 1)?;

    let module = kernels::load(&ctx)?;
    let nev = ma.ne as u32;
    let (gcfg, ocfg) = matvec_cfgs(ma.ne, ma.n1);
    let red = LaunchConfig { grid_dim: (nb as u32, 1, 1), block_dim: (RED as u32, 1, 1), shared_mem_bytes: 0 };
    let red1 = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (RED as u32, 1, 1), shared_mem_bytes: 0 };
    let one = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
    let vec_cfg = LaunchConfig::for_num_elems(ndof as u32);
    let n1 = ma.n1;
    let n64 = ndof as u64;
    let nb64 = nb as u64;

    // Fully on-device dot a·b → device scalar `$out` (dot_partial → reduce_scalar), no sync.
    macro_rules! dot_to {
        ($a:expr, $b:expr, $out:expr) => {{
            module.dot_partial(&stream, red, $a, $b, n64, &mut partial)?;
            module.reduce_scalar(&stream, red1, &partial, nb64, $out)?;
        }};
    }
    macro_rules! apply {
        ($field:expr, $dst:expr) => {{
            module.gradient::<f64, f64>(&stream, gcfg, &d_dev, $field, ma.rx, ma.sy, n1, nev, &mut gx, &mut gy)?;
            module.operator::<f64, f64>(
                &stream, ocfg, &d_dev, $field, &gx, &gy, &mass_dev, n1, nev, ma.rx, ma.sy, ma.jac,
                &fvl_dev, &fnx_dev, &fny_dev, &fsw_dev, &fnbr_dev, &ftau_dev, reaction, $dst,
            )?;
        }};
    }

    // Standard CG with device-resident scalars; the residual is polled to the host only
    // every CHECK iterations (instead of twice per iteration). Same arithmetic as the host
    // CG — at most CHECK−1 extra iterations past convergence, far cheaper than the syncs.
    const CHECK: usize = 25;
    dot_to!(&r, &r, &mut d_rs);
    let bnorm = d_rs.to_host_vec(&stream)?[0].sqrt().max(1e-300);
    let mut iters = 0;
    let mut converged = false;
    for it in 0..maxit {
        apply!(&p, &mut ap);
        dot_to!(&p, &ap, &mut d_pap);
        module.cg_alpha(&stream, one, &d_rs, &d_pap, &mut d_alpha, &mut d_nalpha)?;
        module.axpy_s(&stream, vec_cfg, &mut x, &p, &d_alpha)?; // x += α p
        module.axpy_s(&stream, vec_cfg, &mut r, &ap, &d_nalpha)?; // r −= α ap
        dot_to!(&r, &r, &mut d_rsnew);
        iters = it + 1;
        if (it + 1) % CHECK == 0 || it + 1 == maxit {
            if d_rsnew.to_host_vec(&stream)?[0].sqrt() / bnorm < tol {
                converged = true;
                break;
            }
        }
        module.cg_beta(&stream, one, &d_rsnew, &mut d_rs, &mut d_beta)?; // β = rsnew/rs; rs ← rsnew
        module.xpby_s(&stream, vec_cfg, &mut p, &r, &d_beta)?; // p = r + β p
    }
    let rel = d_rsnew.to_host_vec(&stream)?[0].sqrt() / bnorm;
    warn_unconverged("Helmholtz/Poisson CG", converged, iters, maxit, rel, tol);

    Ok((x.to_host_vec(&stream)?, iters))
}

/// Solve the **singular pure-Neumann** SIPG pressure-Poisson system `A·u = b` by
/// **deflated conjugate gradient on the GPU**, returning the solution (determined up
/// to an additive constant) and the iteration count. The constant nullspace `A·1 = 0`
/// is deflated by removing the arithmetic mean from the residual each iteration.
/// Boundary faces use the natural (Neumann) BC. Mirrors
/// `gale::dg::Poisson::with_bc(mesh, alpha, 0, all-tags).cg_deflated` exactly — this
/// is the pressure-projection solve of the dual-splitting Stokes/NS scheme.
pub fn pressure_cg_solve(
    mesh: &Mesh2d,
    b: &[f64],
    alpha: f64,
    tol: f64,
    maxit: usize,
) -> Result<(Vec<f64>, usize), Box<dyn std::error::Error>> {
    let ma = flatten_mesh(mesh, alpha, &mesh.boundary_tags()); // pure-Neumann boundaries
    let ndof = ma.ndof;
    assert_eq!(b.len(), ndof, "rhs length must be n_elements·n_nodes");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
    let upu = |v: &[u32]| DeviceBuffer::from_host(&stream, v);

    let d_dev = up(&ma.diff)?;
    let mass_dev = up(&ma.mass)?;
    let fvl_dev = upu(&ma.fvl)?;
    let fnx_dev = up(&ma.fnx)?;
    let fny_dev = up(&ma.fny)?;
    let fsw_dev = up(&ma.fsw)?;
    let fnbr_dev = upu(&ma.fnbr)?;
    let ftau_dev = up(&ma.ftau)?;

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
    let nev = ma.ne as u32;
    let (gcfg, ocfg) = matvec_cfgs(ma.ne, ma.n1);
    let red = LaunchConfig { grid_dim: (nb as u32, 1, 1), block_dim: (RED as u32, 1, 1), shared_mem_bytes: 0 };
    let vec_cfg = LaunchConfig::for_num_elems(ndof as u32);
    let n1 = ma.n1;
    let n64 = ndof as u64;
    let ninv = 1.0 / ndof as f64;

    macro_rules! dot {
        ($a:expr, $b:expr) => {{
            module.dot_partial(&stream, red, $a, $b, n64, &mut partial)?;
            partial.to_host_vec(&stream)?.iter().sum::<f64>()
        }};
    }
    // Remove the constant nullspace component: v ← v − mean(v) (arithmetic mean,
    // matching the CPU `deflate`). mean = (1ᵀv)/n via dot with the ones vector.
    macro_rules! deflate {
        ($v:expr) => {{
            let mean = dot!($v, &ones) * ninv;
            module.axpy(&stream, vec_cfg, $v, &ones, -mean)?;
        }};
    }
    macro_rules! apply {
        ($field:expr, $dst:expr) => {{
            module.gradient::<f64, f64>(&stream, gcfg, &d_dev, $field, ma.rx, ma.sy, n1, nev, &mut gx, &mut gy)?;
            module.operator::<f64, f64>(
                &stream, ocfg, &d_dev, $field, &gx, &gy, &mass_dev, n1, nev, ma.rx, ma.sy, ma.jac,
                &fvl_dev, &fnx_dev, &fny_dev, &fsw_dev, &fnbr_dev, &ftau_dev, 0.0, $dst,
            )?;
        }};
    }

    deflate!(&mut r); // r = deflate(b)
    // p = r (copy via xpby with β=0: p ← r + 0·p).
    module.xpby(&stream, vec_cfg, &mut p, &r, 0.0)?;
    let bn = dot!(&r, &r).sqrt().max(1e-300);
    let mut rs = dot!(&r, &r);
    let mut iters = 0;
    let mut converged = false;
    let mut rel = (rs.sqrt() / bn).min(1.0);
    for it in 0..maxit {
        apply!(&p, &mut ap);
        let pap = dot!(&p, &ap);
        let alpha_cg = rs / pap;
        module.axpy(&stream, vec_cfg, &mut x, &p, alpha_cg)?; // x += α p
        module.axpy(&stream, vec_cfg, &mut r, &ap, -alpha_cg)?; // r −= α ap
        deflate!(&mut r);
        let rs_new = dot!(&r, &r);
        iters = it + 1;
        rel = rs_new.sqrt() / bn;
        if rel < tol {
            converged = true;
            break;
        }
        let beta = rs_new / rs;
        module.xpby(&stream, vec_cfg, &mut p, &r, beta)?; // p = r + β p
        rs = rs_new;
    }
    warn_unconverged("deflated pressure CG", converged, iters, maxit, rel, tol);
    Ok((x.to_host_vec(&stream)?, iters))
}

/// All **constant** per-level device state for one fixed p-multigrid hierarchy: the
/// uploaded mesh metrics + SIPG face metadata, inverse diagonals, transfer matrices,
/// launch configs, and the deflation constants for a singular pure-Neumann operator.
/// Built once (`MgConst::build`) and reused across every [`pcg_solve_with`] call, so
/// repeated solves on a fixed mesh skip the host `flatten_mesh` + H2D uploads (~ms per
/// level). Only `rhs` and the per-solve scratch move thereafter.
struct MgConst {
    nlev: usize,
    n_pre: usize,
    n_post: usize,
    n0: usize,
    n1v: Vec<u32>,
    nev: Vec<u32>,
    ndofv: Vec<usize>,
    dl: Vec<DeviceBuffer<f64>>,
    /// 2D GLL mass diagonal per level (affine matvec: `jw = jac·mass`).
    massl: Vec<DeviceBuffer<f64>>,
    /// Affine metric scalars per level (uniform axis-aligned rect mesh): `rx=2/dx`, `sy=2/dy`, Jacobian.
    rxs: Vec<f64>,
    sys: Vec<f64>,
    jacs: Vec<f64>,
    invd: Vec<DeviceBuffer<f64>>,
    fvl: Vec<DeviceBuffer<u32>>,
    fnbr: Vec<DeviceBuffer<u32>>,
    fnx: Vec<DeviceBuffer<f64>>,
    fny: Vec<DeviceBuffer<f64>>,
    fsw: Vec<DeviceBuffer<f64>>,
    ftau: Vec<DeviceBuffer<f64>>,
    omega: Vec<f64>,
    interp: Vec<DeviceBuffer<f64>>,
    /// Per-transition (`len = nlev−1`): true ⇒ h-coarsening (2:1 geometric, the `h_prolong`/
    /// `h_restrict` kernels + `pq`); false ⇒ p-coarsening (the tensor `interp` + restrict/prolong).
    is_h: Vec<bool>,
    /// Element-grid width `nx` per level (for the h-transfer kernels' index arithmetic).
    nxv: Vec<u32>,
    /// The four 2:1 geometric prolongation matrices (shared across all h-transfers), uploaded once.
    pq: DeviceBuffer<f64>,
    cfg: Vec<LaunchConfig>,
    vcfg: Vec<LaunchConfig>,
    reaction: f64,
    deflate: bool,
    ones0: DeviceBuffer<f64>,
    ones_c: DeviceBuffer<f64>,
    ninv0: f64,
    ninv_c: f64,
    clast: usize,
}

impl MgConst {
    /// Flatten + upload every level of `mg` once. `stream` owns the CUDA context the
    /// buffers live in; it must be the same stream later passed to [`pcg_solve_with`].
    fn build(stream: &CudaStream, mg: &PMultigrid) -> Result<Self, Box<dyn std::error::Error>> {
        let nlev = mg.n_levels();
        let (n_pre, n_post) = mg.smoothing();
        let n0 = mg.mesh(0).n_elements() * mg.mesh(0).refq.n_nodes();
        let up = |v: &[f64]| DeviceBuffer::from_host(stream, v);
        let upu = |v: &[u32]| DeviceBuffer::from_host(stream, v);

        let mut n1v = Vec::new();
        let mut nev = Vec::new();
        let mut ndofv = Vec::new();
        let (mut dl, mut massl, mut invd) = (vec![], vec![], vec![]);
        let (mut rxs, mut sys, mut jacs): (Vec<f64>, Vec<f64>, Vec<f64>) = (vec![], vec![], vec![]);
        let (mut fvl, mut fnbr) = (vec![], vec![]);
        let (mut fnx, mut fny, mut fsw, mut ftau) = (vec![], vec![], vec![], vec![]);
        let mut omega = Vec::new();

        // Per-region Neumann tags (rediscretized at every level) — all-tags + reaction 0 is
        // the singular pressure operator (auto-deflated below); empty is all-Dirichlet.
        let neumann_tags = mg.neumann_tags();
        for l in 0..nlev {
            let m = mg.mesh(l);
            let ma = flatten_mesh(m, mg.alpha, neumann_tags);
            n1v.push(ma.n1);
            nev.push(ma.ne as u32);
            ndofv.push(ma.ndof);
            dl.push(up(&ma.diff)?);
            massl.push(up(&ma.mass)?);
            rxs.push(ma.rx);
            sys.push(ma.sy);
            jacs.push(ma.jac);
            invd.push(up(mg.inv_diagonal(l))?);
            fvl.push(upu(&ma.fvl)?);
            fnbr.push(upu(&ma.fnbr)?);
            fnx.push(up(&ma.fnx)?);
            fny.push(up(&ma.fny)?);
            fsw.push(up(&ma.fsw)?);
            ftau.push(up(&ma.ftau)?);
            omega.push(mg.jacobi_omega(l));
        }
        // transfer operators (coarse l+1 → fine l): p-transfers carry a tensor interp matrix;
        // h-transfers carry none (they use the shared `pq` + the kernels' index arithmetic),
        // so a 1-elem dummy keeps the per-transition vec index-aligned.
        let mut interp = Vec::new();
        let mut is_h = Vec::new();
        for l in 0..nlev - 1 {
            let h = mg.is_h_transfer(l);
            is_h.push(h);
            let m = mg.interp_matrix(l);
            interp.push(up(if h { &[1.0f64][..] } else { m })?);
        }
        let nxv: Vec<u32> = (0..nlev).map(|l| mg.level_dims(l).0 as u32).collect();
        let pq = up(&mg.quad_prolong()[..])?;

        // per-level launch configs
        let cfg: Vec<LaunchConfig> = (0..nlev)
            .map(|l| LaunchConfig { grid_dim: (nev[l], 1, 1), block_dim: (n1v[l] * n1v[l], 1, 1), shared_mem_bytes: 0 })
            .collect();
        let vcfg: Vec<LaunchConfig> = ndofv.iter().map(|&n| LaunchConfig::for_num_elems(n as u32)).collect();

        // Singular pure-Neumann pressure operator (constant nullspace) ⇒ deflate: project the
        // residual onto the range (subtract its mean) in the OUTER PCG and in the COARSEST
        // solve; the SPD V-cycle preconditioner itself is unmodified (see the research note).
        let clast = nlev - 1;
        let deflate = mg.is_singular(&mg.mesh(0).boundary_tags());
        let ones0 = up(&vec![1.0f64; n0])?;
        let ones_c = up(&vec![1.0f64; ndofv[clast]])?;
        let ninv_c = 1.0 / ndofv[clast] as f64;
        Ok(Self {
            nlev,
            n_pre,
            n_post,
            n0,
            n1v,
            nev,
            ndofv,
            dl,
            massl,
            rxs,
            sys,
            jacs,
            invd,
            fvl,
            fnbr,
            fnx,
            fny,
            fsw,
            ftau,
            omega,
            interp,
            is_h,
            nxv,
            pq,
            cfg,
            vcfg,
            reaction: mg.reaction(),
            deflate,
            ones0,
            ones_c,
            ninv0: 1.0 / n0 as f64,
            ninv_c,
            clast,
        })
    }
}

/// Solve the SIPG Poisson system `A·u = rhs` by **p-multigrid-preconditioned CG,
/// fully on-device**, returning the finest-level solution and the outer PCG
/// iteration count. The preconditioner is one p-multigrid V-cycle (down-sweep
/// damped-Jacobi smoothing → coarsest-level CG → up-sweep prolong + smoothing),
/// with every smoother/transfer/operator step launched on the GPU. Setup (per-level
/// meshes, inter-level interpolation matrices, inverse diagonals, Jacobi `omega`,
/// pre/post smoothing counts) comes from the validated CPU `mg`. Convergence
/// (`‖r‖ / ‖rhs‖ < tol`), the inner coarse-CG tolerance/cap, and the V-cycle
/// structure mirror `gale::dg::PMultigrid::pcg` exactly.
///
/// `rhs` is the already-assembled finest-level right-hand side (e.g. from
/// `Poisson::new(mg.mesh(0), mg.alpha).rhs(...)`).
pub fn poisson_pcg_solve(
    mg: &PMultigrid,
    rhs: &[f64],
    tol: f64,
    maxit: usize,
) -> Result<(Vec<f64>, usize), Box<dyn std::error::Error>> {
    // One-shot: create the context, load the module, upload, run. For repeated solves on a
    // fixed hierarchy use the persistent [`GpuPoissonMg`] handle (amortizes the ~0.3 s load
    // AND the per-level uploads).
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let dev = MgConst::build(&stream, mg)?;
    pcg_solve_with::<f64>(&stream, &module, &dev, rhs, tol, maxit, false)
}

/// The p-MG-PCG loop given an already-created stream + loaded module + uploaded
/// hierarchy (`MgConst`) — shared by the one-shot [`poisson_pcg_solve`] and the
/// persistent [`GpuPoissonMg`] handle. The constant per-level device arrays live in
/// `dev`; only the per-solve scratch (V-cycle work vectors + PCG vectors) is allocated
/// here, and only `rhs` is uploaded.
fn pcg_solve_with<G: Scalar + DeviceCopy>(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
    dev: &MgConst,
    rhs: &[f64],
    tol: f64,
    maxit: usize,
    use_graph: bool,
) -> Result<(Vec<f64>, usize), Box<dyn std::error::Error>> {
    // `use_graph`: capture the V-cycle launch sequence into a CUDA graph once and replay it each
    // outer PCG iteration, instead of re-issuing its ~500 tiny kernel launches. The solve is
    // launch-bound (cuLaunchKernel ≈ 83% of host time), so this is the dominant lever. Requires
    // a NON-legacy stream (the caller supplies one) and a host-branch-free V-cycle — hence the
    // coarse CG runs a FIXED iteration count (no residual readback) under this flag.
    // `G` = the gradient-intermediate (gxb/gyb) storage precision. `G=f64` is the across-the-board
    // FP64 path (bit-exact); `G=f32` halves the gx/gy DRAM traffic (the matvec's biggest field
    // round-trip, incl. the uncoalesced neighbor reads). All other vectors + reductions stay f64.
    // Bind the constant per-level state to bare locals (disjoint immutable borrows of
    // `dev`) so the matvec/V-cycle/deflation macros below read exactly as the original
    // single-function loop did.
    let MgConst {
        nlev,
        n_pre,
        n_post,
        n0,
        ref n1v,
        ref nev,
        ref ndofv,
        ref dl,
        ref massl,
        ref rxs,
        ref sys,
        ref jacs,
        ref invd,
        ref fvl,
        ref fnbr,
        ref fnx,
        ref fny,
        ref fsw,
        ref ftau,
        ref omega,
        ref interp,
        ref is_h,
        ref nxv,
        ref pq,
        ref cfg,
        ref vcfg,
        reaction,
        deflate,
        ref ones0,
        ref ones_c,
        ninv0,
        ninv_c,
        clast,
    } = *dev;
    // Per-level packed matvec launch configs (gradient_cfg, operator_cfg) — derived from the
    // element count + order at each level; the transfer/vector kernels keep `cfg`/`vcfg`.
    let mvcfg: Vec<(LaunchConfig, LaunchConfig)> =
        (0..nlev).map(|l| matvec_cfgs(nev[l] as usize, n1v[l])).collect();
    assert_eq!(rhs.len(), n0, "rhs length must match the finest level");

    // Per-solve scratch: V-cycle work vectors (struct-of-arrays per level) — allocated
    // fresh each solve (cheap device zeroed-malloc; the expensive uploads live in `dev`).
    let (mut xb, mut bb, mut rb, mut apb, mut gxb, mut gyb, mut tmpb, mut sm) =
        (vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    for l in 0..nlev {
        let nd = ndofv[l];
        xb.push(DeviceBuffer::<f64>::zeroed(stream, nd)?);
        bb.push(DeviceBuffer::<f64>::zeroed(stream, nd)?);
        rb.push(DeviceBuffer::<f64>::zeroed(stream, nd)?);
        apb.push(DeviceBuffer::<f64>::zeroed(stream, nd)?);
        gxb.push(DeviceBuffer::<G>::zeroed(stream, nd)?);
        gyb.push(DeviceBuffer::<G>::zeroed(stream, nd)?);
        tmpb.push(DeviceBuffer::<f64>::zeroed(stream, nd)?);
        sm.push(DeviceBuffer::<f64>::zeroed(stream, nd)?); // damped-Jacobi smoother pong buffer
    }

    // PCG vectors (finest level, separate from V-cycle scratch).
    let mut psol = DeviceBuffer::<f64>::zeroed(stream, n0)?;
    let mut pres = DeviceBuffer::from_host(stream, rhs)?;
    let mut pp = DeviceBuffer::<f64>::zeroed(stream, n0)?;
    let pz = DeviceBuffer::<f64>::zeroed(stream, n0)?;
    let mut pap = DeviceBuffer::<f64>::zeroed(stream, n0)?;
    let rhs_dev = DeviceBuffer::from_host(stream, rhs)?;
    // Multi-block reduction partials, sized for the finest level (the largest dot).
    let mut partial = DeviceBuffer::<f64>::zeroed(stream, dot_blocks(n0))?;
    // CG/PCG scalars kept **on device** (length-1 buffers) so neither the outer PCG nor the
    // coarse-grid CG issues a host sync per iteration — the dominant cost of the previous
    // host-readback `dot` (a V-cycle's coarse solve runs hundreds of iters, each formerly a
    // `to_host_vec`). The OUTER pool must survive a V-cycle (rz spans the preconditioner
    // call), so it is disjoint from the COARSE pool the V-cycle clobbers.
    let mut d_rz = DeviceBuffer::<f64>::zeroed(stream, 1)?; // outer: r·z (PCG)
    let mut d_rznew = DeviceBuffer::<f64>::zeroed(stream, 1)?;
    let mut d_pap_o = DeviceBuffer::<f64>::zeroed(stream, 1)?;
    let mut d_alpha_o = DeviceBuffer::<f64>::zeroed(stream, 1)?;
    let mut d_nalpha_o = DeviceBuffer::<f64>::zeroed(stream, 1)?;
    let mut d_beta_o = DeviceBuffer::<f64>::zeroed(stream, 1)?;
    let mut d_rr = DeviceBuffer::<f64>::zeroed(stream, 1)?; // outer residual ‖r‖²
    let mut d_nmean_o = DeviceBuffer::<f64>::zeroed(stream, 1)?;
    let mut d_rs = DeviceBuffer::<f64>::zeroed(stream, 1)?; // coarse: r·r (CG)
    let mut d_rsnew = DeviceBuffer::<f64>::zeroed(stream, 1)?;
    let mut d_pap_c = DeviceBuffer::<f64>::zeroed(stream, 1)?;
    let mut d_alpha_c = DeviceBuffer::<f64>::zeroed(stream, 1)?;
    let mut d_nalpha_c = DeviceBuffer::<f64>::zeroed(stream, 1)?;
    let mut d_beta_c = DeviceBuffer::<f64>::zeroed(stream, 1)?;
    let mut d_nmean_c = DeviceBuffer::<f64>::zeroed(stream, 1)?;
    let one = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };

    // Fully on-device dot a·b → device scalar `$out` (dot_partial → reduce_scalar), no sync.
    macro_rules! dot_to {
        ($a:expr, $b:expr, $n:expr, $out:expr) => {{
            let nbl = dot_blocks($n);
            let redcfg = LaunchConfig { grid_dim: (nbl as u32, 1, 1), block_dim: (RED as u32, 1, 1), shared_mem_bytes: 0 };
            let red1 = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (RED as u32, 1, 1), shared_mem_bytes: 0 };
            module.dot_partial(&stream, redcfg, $a, $b, $n as u64, &mut partial)?;
            module.reduce_scalar(&stream, red1, &partial, nbl as u64, $out)?;
        }};
    }
    macro_rules! dcopy {
        ($dst:expr, $src:expr, $n:expr) => {{
            unsafe {
                cuda_core::memory::memcpy_dtod_async($dst.cu_deviceptr(), $src.cu_deviceptr(), $n * 8, stream.cu_stream())?;
            }
        }};
    }
    // Singular pure-Neumann pressure operator (constant nullspace) ⇒ deflate: project the
    // residual onto the range (subtract its mean) in the OUTER PCG and in the COARSEST solve;
    // the SPD V-cycle preconditioner itself is unmodified (see docs/research-pressure-multigrid.md).
    // `deflate`, `ones0`, `ones_c`, `ninv0`, `ninv_c`, `clast` all come from `dev` (above).
    // The mean is removed on-device (dot → cg_negmean(−mean) → axpy_s), no host round-trip.
    macro_rules! deflate0 {
        ($v:expr) => {{
            if deflate {
                dot_to!($v, &ones0, n0, &mut d_nmean_o);
                module.cg_negmean(&stream, one, &mut d_nmean_o, ninv0)?;
                module.axpy_s(&stream, vcfg[0], $v, &ones0, &d_nmean_o)?;
            }
        }};
    }
    // Coarsest-level range projection (nullspace-consistent coarse solve, Kaasschieter).
    macro_rules! deflate_c {
        ($v:expr) => {{
            if deflate {
                dot_to!($v, &ones_c, ndofv[clast], &mut d_nmean_c);
                module.cg_negmean(&stream, one, &mut d_nmean_c, ninv_c)?;
                module.axpy_s(&stream, vcfg[clast], $v, &ones_c, &d_nmean_c)?;
            }
        }};
    }
    // matvec: dst = A·src at level $l (src/dst external; gx/gy scratch from Vecs).
    // Helmholtz reaction λ (0 ⇒ pure Poisson) — `reaction` comes from `dev`; the per-level
    // diagonal/omega were computed for it at build time, so the on-device V-cycle matches
    // the CPU setup.
    macro_rules! matvec {
        ($l:expr, $src:expr, $dst:expr) => {{
            let l = $l;
            module.gradient::<f64, G>(&stream, mvcfg[l].0, &dl[l], $src, rxs[l], sys[l], n1v[l], nev[l], &mut gxb[l], &mut gyb[l])?;
            module.operator::<f64, G>(
                &stream, mvcfg[l].1, &dl[l], $src, &gxb[l], &gyb[l], &massl[l], n1v[l], nev[l], rxs[l], sys[l], jacs[l],
                &fvl[l], &fnx[l], &fny[l], &fsw[l], &fnbr[l], &ftau[l], reaction, $dst,
            )?;
        }};
    }
    // One fused damped-Jacobi smoother sweep: gradient + `operator_jacobi` (the matvec with the
    // Jacobi update `x + ω·invd·(b − Ap)` folded into the operator's output write — no `Ap` DRAM
    // round-trip, no separate `jacobi` launch), writing to the pong buffer `sm[l]` then swapping
    // it into `xb[l]` (the operator reads neighbour `u`, so the update cannot be in place). After
    // the swap `xb[l]` holds the freshest iterate. Bit-identical to `matvec! + jacobi`.
    macro_rules! smooth_sweep {
        ($l:expr) => {{
            let l = $l;
            module.gradient::<f64, G>(&stream, mvcfg[l].0, &dl[l], &xb[l], rxs[l], sys[l], n1v[l], nev[l], &mut gxb[l], &mut gyb[l])?;
            module.operator_jacobi::<f64, G>(
                &stream, mvcfg[l].1, &dl[l], &xb[l], &gxb[l], &gyb[l], &massl[l], n1v[l], nev[l], rxs[l], sys[l], jacs[l],
                &fvl[l], &fnx[l], &fny[l], &fsw[l], &fnbr[l], &ftau[l], reaction, &bb[l], &invd[l], omega[l], &mut sm[l],
            )?;
            std::mem::swap(&mut xb[l], &mut sm[l]);
        }};
    }
    // Two operator applications MUST stay FP64 even in mixed mode, with their own f64 gx/gy
    // intermediates (NOT the V-cycle's `G`-typed gxb/gyb):
    //  • the OUTER PCG's `A·p` (`matvec0!`, finest level) — CG breaks with an inexact/varying
    //    operator; the matvec/residual precision sets the attainable accuracy.
    //  • the COARSE-grid CG (`matvec_c!`, coarsest level) — the deflated *singular* coarse solve is
    //    a CG too and likewise breaks (→ NaN) on an FP32-perturbed, non-symmetric operator. The
    //    coarse grid is tiny (h-coarsened), so FP64 there is free.
    // Only the V-cycle *smoother* matvecs (a robust stationary Jacobi) carry the FP32 intermediates.
    let mut gx0 = DeviceBuffer::<f64>::zeroed(stream, n0)?;
    let mut gy0 = DeviceBuffer::<f64>::zeroed(stream, n0)?;
    let mut gxc = DeviceBuffer::<f64>::zeroed(stream, ndofv[clast])?;
    let mut gyc = DeviceBuffer::<f64>::zeroed(stream, ndofv[clast])?;
    macro_rules! matvec0 {
        ($src:expr, $dst:expr) => {{
            module.gradient::<f64, f64>(&stream, mvcfg[0].0, &dl[0], $src, rxs[0], sys[0], n1v[0], nev[0], &mut gx0, &mut gy0)?;
            module.operator::<f64, f64>(
                &stream, mvcfg[0].1, &dl[0], $src, &gx0, &gy0, &massl[0], n1v[0], nev[0], rxs[0], sys[0], jacs[0],
                &fvl[0], &fnx[0], &fny[0], &fsw[0], &fnbr[0], &ftau[0], reaction, $dst,
            )?;
        }};
    }
    macro_rules! matvec_c {
        ($l:expr, $src:expr, $dst:expr) => {{
            let l = $l;
            module.gradient::<f64, f64>(&stream, mvcfg[l].0, &dl[l], $src, rxs[l], sys[l], n1v[l], nev[l], &mut gxc, &mut gyc)?;
            module.operator::<f64, f64>(
                &stream, mvcfg[l].1, &dl[l], $src, &gxc, &gyc, &massl[l], n1v[l], nev[l], rxs[l], sys[l], jacs[l],
                &fvl[l], &fnx[l], &fny[l], &fsw[l], &fnbr[l], &ftau[l], reaction, $dst,
            )?;
        }};
    }
    // Coarsest-level solve tolerance/cap. With h-coarsening the coarsest grid is TINY (a few
    // elements), so a tight, near-exact solve is cheap (~handful of iters) and minimizes outer
    // PCG iterations. Without it (odd / non-power-of-2 grids that can't h-coarsen, where the
    // coarsest is still order-1 on the full grid) we keep the loose-tol band-aid: the coarse
    // solve is just a preconditioner component, so a loose tol + low cap keeps the V-cycle cheap
    // and the outer PCG absorbs the inexactness.
    let coarse_small = ndofv[clast] <= 1024;
    let coarse_tol = if coarse_small { 1e-9 } else { 1e-2 };
    // Adaptive (non-graph) iteration cap. Under `use_graph` the coarse CG instead runs a small
    // FIXED count: the h-coarsened coarsest grid is tiny (CG converges in ≤ ndof iters), so a
    // handful suffices, and a long fixed chain would bloat the captured graph (cuGraphInstantiate
    // cost scales with node count — a 100-iter coarse chain is ~1300 extra nodes).
    let coarse_cap = if use_graph {
        (ndofv[clast] + 4).min(if coarse_small { 100 } else { 40 })
    } else if coarse_small {
        100
    } else {
        40
    };
    // Coarsest-level CG, on-device scalars; the residual is polled to the host only every
    // COARSE_CHECK iterations (no per-iter sync).
    macro_rules! coarse_cg {
        ($l:expr) => {{
            let l = $l;
            let n = ndofv[l];
            module.scal(&stream, vcfg[l], &mut xb[l], 0.0)?;
            dcopy!(&rb[l], &bb[l], n);
            deflate_c!(&mut rb[l]); // project the coarse RHS onto the range (singular op)
            dcopy!(&tmpb[l], &rb[l], n);
            dot_to!(&rb[l], &rb[l], n, &mut d_rs);
            // `bn` (and the relative-residual break below) require a host readback, which is
            // illegal mid graph-capture. Under `use_graph` the coarse CG instead runs a FIXED
            // `coarse_cap` iterations with no readback (a tiny h-coarsened grid converges well
            // within the cap; the extra iters are cheap and, captured into the graph, add ~zero
            // launch cost). Otherwise it polls every COARSE_CHECK iters and breaks at `coarse_tol`.
            let bn = if use_graph { 1.0 } else { d_rs.to_host_vec(&stream)?[0].sqrt().max(1e-300) };
            const COARSE_CHECK: usize = 10;
            for it in 0..coarse_cap {
                matvec_c!(l, &tmpb[l], &mut apb[l]); // coarse A·p — FP64 (deflated CG needs exactness)
                dot_to!(&tmpb[l], &apb[l], n, &mut d_pap_c);
                module.cg_alpha(&stream, one, &d_rs, &d_pap_c, &mut d_alpha_c, &mut d_nalpha_c)?;
                module.axpy_s(&stream, vcfg[l], &mut xb[l], &tmpb[l], &d_alpha_c)?; // x += α p
                module.axpy_s(&stream, vcfg[l], &mut rb[l], &apb[l], &d_nalpha_c)?; // r −= α ap
                deflate_c!(&mut rb[l]); // keep the coarse residual in the range each iter
                dot_to!(&rb[l], &rb[l], n, &mut d_rsnew);
                if !use_graph && ((it + 1) % COARSE_CHECK == 0 || it + 1 == coarse_cap) {
                    if d_rsnew.to_host_vec(&stream)?[0].sqrt() / bn < coarse_tol {
                        break;
                    }
                }
                module.cg_beta(&stream, one, &d_rsnew, &mut d_rs, &mut d_beta_c)?; // β=rsnew/rs; rs←rsnew
                module.xpby_s(&stream, vcfg[l], &mut tmpb[l], &rb[l], &d_beta_c)?; // p = r + β p
            }
        }};
    }
    // One V-cycle: input in bb[0], solution in xb[0].
    macro_rules! vcycle {
        () => {{
            let last = nlev - 1;
            for l in 0..last {
                module.scal(&stream, vcfg[l], &mut xb[l], 0.0)?;
                for _ in 0..n_pre {
                    smooth_sweep!(l);
                }
                matvec!(l, &xb[l], &mut apb[l]);
                module.sub(&stream, vcfg[l], &mut rb[l], &bb[l], &apb[l])?;
                if is_h[l] {
                    // h-restrict: one block per COARSE element (cfg[l+1]); grids nxv[l]→nxv[l+1].
                    module.h_restrict(&stream, cfg[l + 1], pq, &rb[l], nxv[l], nxv[l + 1], &mut bb[l + 1])?;
                } else {
                    module.restrict(&stream, cfg[l], &interp[l], &rb[l], n1v[l], n1v[l + 1], &mut bb[l + 1])?;
                }
            }
            coarse_cg!(last);
            for l in (0..last).rev() {
                if is_h[l] {
                    // h-prolong: one block per FINE element (cfg[l]); coarse l+1 → fine l.
                    module.h_prolong(&stream, cfg[l], pq, &xb[l + 1], nxv[l], nxv[l + 1], &mut tmpb[l])?;
                } else {
                    module.prolong(&stream, cfg[l], &interp[l], &xb[l + 1], n1v[l], n1v[l + 1], &mut tmpb[l])?;
                }
                module.axpy(&stream, vcfg[l], &mut xb[l], &tmpb[l], 1.0)?;
                for _ in 0..n_post {
                    smooth_sweep!(l);
                }
            }
        }};
    }

    // Launch-bound remediation: capture the V-cycle's ~500-launch sequence into a CUDA graph
    // ONCE (recorded, not executed — `vcycle!` re-zeroes its working state each call, so the
    // graph is a pure bb[0] → xb[0] map reusable every outer iteration), then replay it with a
    // single `cuGraphLaunch`. Requires the host-branch-free coarse CG (above, under `use_graph`)
    // and a non-legacy stream (the handle supplies one in graph mode). `run_vcycle!` dispatches
    // to the replay or the direct launches.
    let vcycle_graph: Option<CudaGraphExec> = if use_graph {
        Some(stream.capture(CaptureMode::ThreadLocal, || {
            vcycle!();
            Ok(())
        })?)
    } else {
        None
    };
    macro_rules! run_vcycle {
        () => {{
            if let Some(ref g) = vcycle_graph {
                g.launch(stream)?;
            } else {
                vcycle!();
            }
        }};
    }

    // Preconditioned CG on the device. α/β and the deflation mean stay on-device; only the
    // outer residual norm is polled — once per outer iter, which is cheap (PCG converges in
    // ~tens of iters), unlike the coarse solve's hundreds.
    module.scal(&stream, vcfg[0], &mut psol, 0.0)?;
    deflate0!(&mut pres); // project the initial residual (=RHS) onto the range
    dcopy!(&bb[0], &pres, n0);
    run_vcycle!();
    dcopy!(&pz, &xb[0], n0);
    dcopy!(&pp, &pz, n0);
    dot_to!(&rhs_dev, &rhs_dev, n0, &mut d_rr);
    let bn = d_rr.to_host_vec(&stream)?[0].sqrt().max(1e-300);
    dot_to!(&pres, &pz, n0, &mut d_rz);
    let mut iters = 0;
    let mut converged = false;
    let mut rel = 1.0;
    for it in 0..maxit {
        matvec0!(&pp, &mut pap); // outer A·p — FP64 (CG needs an exact operator)
        dot_to!(&pp, &pap, n0, &mut d_pap_o);
        module.cg_alpha(&stream, one, &d_rz, &d_pap_o, &mut d_alpha_o, &mut d_nalpha_o)?;
        module.axpy_s(&stream, vcfg[0], &mut psol, &pp, &d_alpha_o)?; // x += α p
        module.axpy_s(&stream, vcfg[0], &mut pres, &pap, &d_nalpha_o)?; // r −= α Ap
        deflate0!(&mut pres); // keep the residual in the range each iteration
        iters = it + 1;
        dot_to!(&pres, &pres, n0, &mut d_rr);
        rel = d_rr.to_host_vec(&stream)?[0].sqrt() / bn;
        if rel < tol {
            converged = true;
            break;
        }
        dcopy!(&bb[0], &pres, n0);
        run_vcycle!();
        dcopy!(&pz, &xb[0], n0);
        dot_to!(&pres, &pz, n0, &mut d_rznew);
        module.cg_beta(&stream, one, &d_rznew, &mut d_rz, &mut d_beta_o)?; // β=rznew/rz; rz←rznew
        module.xpby_s(&stream, vcfg[0], &mut pp, &pz, &d_beta_o)?; // p = z + β p
    }
    let kind = if deflate { "singular-Neumann pressure" } else { "Helmholtz" };
    warn_unconverged(&format!("p-MG-PCG ({kind})"), converged, iters, maxit, rel, tol);

    Ok((psol.to_host_vec(&stream)?, iters))
}

/// Persistent p-multigrid-PCG handle: owns the CUDA context, the loaded device module, and
/// the **uploaded** p-multigrid hierarchy ([`MgConst`]), so repeated solves on a fixed mesh
/// skip both the ~0.3 s context/module setup AND the per-level `flatten_mesh` + H2D uploads.
/// Each solve only uploads `rhs`, allocates the V-cycle/PCG scratch, and downloads the
/// solution. Drives the deflated singular-Neumann pressure (and Helmholtz velocity) elliptic
/// solves through one V-cycle-preconditioned CG. The flow integrators hold these across
/// timesteps (one per distinct operator: pressure + per-component velocity Helmholtz).
pub struct GpuPoissonMg {
    // `stream` and `module` keep an `Arc<CudaContext>`, so the context outlives the handle;
    // `dev`'s device buffers live in that same context.
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
    dev: MgConst,
    /// Mixed-precision V-cycle: store the gradient intermediates (gx/gy) in FP32 to halve their
    /// DRAM traffic. The outer CG, residual, reductions, and deflation stay FP64, so the final
    /// solution keeps full accuracy. `false` (default) ⇒ the across-the-board FP64 path (bit-exact).
    mixed: bool,
    /// Capture the V-cycle into a CUDA graph and replay it (launch-bound remediation). Opt-in;
    /// `false` (default) ⇒ the launch-per-iteration path. When enabled, `stream` is a non-legacy
    /// stream (capture is illegal on the default/legacy stream) and the coarse CG runs a fixed
    /// iteration count (the captured V-cycle must be host-branch-free). The replayed launches are
    /// identical to the direct ones, so the result matches the non-graph path to solver tolerance.
    graph: bool,
}

impl GpuPoissonMg {
    /// Build the handle for an already-set-up p-multigrid hierarchy (pressure: all-Neumann +
    /// reaction 0 ⇒ deflated; velocity: reaction λ + the per-region neumann tags). Uploads
    /// every level once; `mg` is consumed (its device-side image lives in `dev` thereafter).
    pub fn new(mg: PMultigrid) -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let module = kernels::load(&ctx)?;
        let dev = MgConst::build(&stream, &mg)?;
        Ok(Self { stream, module, dev, mixed: false, graph: false })
    }

    /// Enable the mixed-precision V-cycle (FP32 gradient intermediates). Opt-in; default is the
    /// bit-exact FP64 path. The final solution stays FP64-accurate (outer CG + reductions are FP64).
    pub fn with_mixed_precision(mut self, mixed: bool) -> Self {
        self.mixed = mixed;
        self
    }

    /// Enable CUDA-graph capture/replay of the V-cycle (launch-bound remediation; the solve is
    /// ~83% `cuLaunchKernel`). Opt-in; default is the launch-per-iteration path. Capture is illegal
    /// on the default/legacy stream, so enabling this swaps in a freshly created stream. The coarse
    /// CG then runs a fixed iteration count (host-branch-free), so the result matches the non-graph
    /// path to solver tolerance rather than bit-for-bit. Fallible because it may create a stream.
    pub fn with_cuda_graph(mut self, graph: bool) -> Result<Self, Box<dyn std::error::Error>> {
        self.graph = graph;
        if graph && self.stream.cu_stream().is_null() {
            // Stream capture cannot be initiated on CU_STREAM_LEGACY (the null default stream).
            self.stream = self.stream.context().new_stream()?;
        }
        Ok(self)
    }

    /// Degrees of freedom on the finest level (`n_elements · n_nodes`).
    pub fn ndof(&self) -> usize {
        self.dev.n0
    }

    /// The Helmholtz reaction `λ` baked into this hierarchy (0 for the pure-Poisson
    /// pressure operator). Lets a caller detect a `Δt` change (⇒ rebuild) for velocity.
    pub fn reaction(&self) -> f64 {
        self.dev.reaction
    }

    /// Solve `A·x = rhs` (the hierarchy's operator) by p-MG-PCG; singular pure-Neumann
    /// systems are auto-deflated. `rhs` is the finest-level RHS.
    pub fn solve(&self, rhs: &[f64], tol: f64, maxit: usize) -> Result<(Vec<f64>, usize), Box<dyn std::error::Error>> {
        if self.mixed {
            pcg_solve_with::<f32>(&self.stream, &self.module, &self.dev, rhs, tol, maxit, self.graph)
        } else {
            pcg_solve_with::<f64>(&self.stream, &self.module, &self.dev, rhs, tol, maxit, self.graph)
        }
    }
}

// ===== Roofline microbenchmark ====================================================

/// One kernel's measured average per-launch time (device-side, via CUDA events).
#[derive(Clone, Debug)]
pub struct KernelTime {
    pub name: &'static str,
    pub ms: f64,
}

/// Device-side timings of the 2D SIPG-Poisson kernels on a mesh, for roofline analysis.
/// `gradient`/`operator` are the matvec pipeline (one block per element, `nn` threads);
/// `axpy`/`xpby`/`dot` are the CG vector ops (the same launches the solve issues). Mesh
/// dimensions are returned so the caller can do the byte/FLOP accounting.
#[derive(Clone, Debug)]
pub struct PoissonBench {
    pub ne: usize,
    pub nn: usize,
    pub n1: u32,
    pub ndof: usize,
    /// Host wall-clock per `axpy` launch (single final sync) — the host-side launch cost.
    pub axpy_wall_us: f64,
    pub kernels: Vec<KernelTime>,
}

/// Microbenchmark each 2D Poisson kernel in isolation: launch it `reps` times between
/// CUDA timing events (after a 3-launch warmup) and report the per-launch average. The
/// mesh metrics/face data are uploaded once. This isolates per-kernel device time from
/// the per-solve host overhead (context/module setup, per-iteration sync) that
/// [`poisson_cg_solve`] also pays — quantified by comparing to a full-solve wall clock.
pub fn bench_poisson_kernels(
    mesh: &Mesh2d,
    alpha: f64,
    reps: u32,
) -> Result<PoissonBench, Box<dyn std::error::Error>> {
    let ma = flatten_mesh(mesh, alpha, &[]);
    let ndof = ma.ndof;
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
    let upu = |v: &[u32]| DeviceBuffer::from_host(&stream, v);

    let d_dev = up(&ma.diff)?;
    let mass_dev = up(&ma.mass)?;
    let fvl_dev = upu(&ma.fvl)?;
    let fnx_dev = up(&ma.fnx)?;
    let fny_dev = up(&ma.fny)?;
    let fsw_dev = up(&ma.fsw)?;
    let fnbr_dev = upu(&ma.fnbr)?;
    let ftau_dev = up(&ma.ftau)?;

    let u_dev = up(&vec![1.0f64; ndof])?;
    let x_dev0 = up(&vec![0.5f64; ndof])?;
    let mut y_dev = up(&vec![0.25f64; ndof])?;
    let mut gx = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut gy = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut out = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let nb = dot_blocks(ndof);
    let mut partial = DeviceBuffer::<f64>::zeroed(&stream, nb)?;

    let module = kernels::load(&ctx)?;
    let nev = ma.ne as u32;
    let (gcfg, ocfg) = matvec_cfgs(ma.ne, ma.n1);
    let red = LaunchConfig { grid_dim: (nb as u32, 1, 1), block_dim: (RED as u32, 1, 1), shared_mem_bytes: 0 };
    let vec_cfg = LaunchConfig::for_num_elems(ndof as u32);
    let n1 = ma.n1;
    let n64 = ndof as u64;

    // Time a kernel launch (expression `$body`) averaged over `reps`, after warmup.
    let ev = || ctx.new_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT));
    macro_rules! timed {
        ($body:block) => {{
            for _ in 0..3 {
                $body
            }
            stream.synchronize()?;
            let s = ev()?;
            let e = ev()?;
            s.record(&stream)?;
            for _ in 0..reps {
                $body
            }
            e.record(&stream)?;
            e.synchronize()?;
            (s.elapsed_ms(&e)? as f64) / reps as f64
        }};
    }

    let grad_ms = timed!({
        module.gradient::<f64, f64>(&stream, gcfg, &d_dev, &u_dev, ma.rx, ma.sy, n1, nev, &mut gx, &mut gy)?;
    });
    let op_ms = timed!({
        module.operator::<f64, f64>(
            &stream, ocfg, &d_dev, &u_dev, &gx, &gy, &mass_dev, n1, nev, ma.rx, ma.sy, ma.jac,
            &fvl_dev, &fnx_dev, &fny_dev, &fsw_dev, &fnbr_dev, &ftau_dev, 0.0, &mut out,
        )?;
    });
    let axpy_ms = timed!({
        module.axpy(&stream, vec_cfg, &mut y_dev, &x_dev0, 0.7)?;
    });
    let xpby_ms = timed!({
        module.xpby(&stream, vec_cfg, &mut y_dev, &x_dev0, 0.7)?;
    });
    let dot_ms = timed!({
        module.dot_partial(&stream, red, &x_dev0, &u_dev, n64, &mut partial)?;
    });

    // Host-side per-launch overhead: wall clock (host Instant) of `reps` axpy launches
    // with a SINGLE final sync, vs the device-event time of the same. If wall ≫ event the
    // host is the bottleneck (cuda-oxide's sync launch path issues cuLaunchKernel but the
    // per-call arg-marshalling / FFI dominates), which caps how fast a launch-heavy CG
    // iteration can go regardless of removing readback syncs.
    stream.synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        module.axpy(&stream, vec_cfg, &mut y_dev, &x_dev0, 0.7)?;
    }
    stream.synchronize()?;
    let axpy_wall_us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;

    Ok(PoissonBench {
        ne: ma.ne,
        nn: ma.nn,
        n1: ma.n1,
        ndof,
        axpy_wall_us,
        kernels: vec![
            KernelTime { name: "gradient", ms: grad_ms },
            KernelTime { name: "operator", ms: op_ms },
            KernelTime { name: "axpy", ms: axpy_ms },
            KernelTime { name: "xpby", ms: xpby_ms },
            KernelTime { name: "dot_partial", ms: dot_ms },
        ],
    })
}

// ===== Persistent solver handle (P4) =============================================

/// Persistent GPU SIPG-Poisson / Helmholtz solver handle. Owns the CUDA context, the
/// loaded device module, and the uploaded **constant** mesh metrics + face metadata, so
/// repeated solves on the same mesh pay the heavy setup (`CudaContext::new` +
/// `kernels::load` cubin/JIT + mesh upload, ~0.3 s) **once** instead of per call. The
/// dual-splitting flow loop's 3 solves/step (and every timestep) share one handle.
///
/// Only the per-region `neumann_tags` (which boundary faces are natural) and the
/// `reaction`/`deflate` flags vary between solves; [`solve`](Self::solve) rebuilds just the
/// small `fnbr` array on the host and uploads it, then runs the on-device-scalar CG on
/// freshly-allocated (cheap) scratch. Bit-for-bit equivalent to [`cg_solve_impl`] /
/// [`pressure_cg_solve`].
pub struct GpuPoisson {
    // `stream` and `module` each hold an `Arc<CudaContext>`, keeping the context alive.
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
    ne: usize,
    ndof: usize,
    n1: u32,
    // constant (neumann-tag-independent) device arrays, uploaded once.
    d_dev: DeviceBuffer<f64>,
    mass_dev: DeviceBuffer<f64>, // 2D GLL mass diagonal (affine matvec: jw = jac·mass)
    rx: f64,                     // affine metric scalars (uniform axis-aligned rect mesh)
    sy: f64,
    jac: f64,
    fvl_dev: DeviceBuffer<u32>,
    fnx_dev: DeviceBuffer<f64>,
    fny_dev: DeviceBuffer<f64>,
    fsw_dev: DeviceBuffer<f64>,
    ftau_dev: DeviceBuffer<f64>,
    // host state to rebuild the per-region `fnbr` cheaply (base = all-Dirichlet; flip the
    // listed boundary-face nodes to NEU when their tag is in `neumann_tags`).
    fnbr_base: Vec<u32>,
    bnodes: Vec<(usize, u32)>,
}

impl GpuPoisson {
    /// Build the handle for `mesh` with SIPG penalty factor `alpha`: create the context,
    /// load the module, and upload the constant metrics/face data. Conforming meshes only
    /// (the non-conforming mortar path keeps its own solver for now).
    pub fn new(mesh: &Mesh2d, alpha: f64) -> Result<Self, Box<dyn std::error::Error>> {
        let ma = flatten_mesh(mesh, alpha, &[]); // fnbr_base: all boundary faces Dirichlet (BND)
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
        let upu = |v: &[u32]| DeviceBuffer::from_host(&stream, v);

        // Boundary-face nodes and their tags, in the same flat order flatten_mesh uses, so
        // a per-region solve only flips these few entries to NEU (no mesh walk per solve).
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
            ne: ma.ne,
            ndof: ma.ndof,
            n1: ma.n1,
            d_dev: up(&ma.diff)?,
            mass_dev: up(&ma.mass)?,
            rx: ma.rx,
            sy: ma.sy,
            jac: ma.jac,
            fvl_dev: upu(&ma.fvl)?,
            fnx_dev: up(&ma.fnx)?,
            fny_dev: up(&ma.fny)?,
            fsw_dev: up(&ma.fsw)?,
            ftau_dev: up(&ma.ftau)?,
            fnbr_base: ma.fnbr,
            bnodes,
            stream,
        })
    }

    /// Number of degrees of freedom (`n_elements · n_nodes`).
    pub fn ndof(&self) -> usize {
        self.ndof
    }

    /// Solve `(reaction·M + A)·x = b` on this mesh by on-device CG. `neumann_tags` are the
    /// natural-BC boundary tags (`&[]` = all-Dirichlet velocity Helmholtz;
    /// `&mesh.boundary_tags()` = pure-Neumann pressure). `deflate` removes the constant
    /// nullspace each iteration (the singular pure-Neumann pressure system). No per-call
    /// context/module setup or constant-mesh upload — only `b`, `fnbr`, and scratch move.
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

        // Per-region fnbr: start from the all-Dirichlet base, flip listed boundary nodes to
        // NEU where their tag is natural. Cheap host rebuild + one small upload per solve.
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
        let mut p = up(b)?;
        let mut ap = DeviceBuffer::<f64>::zeroed(stream, ndof)?;
        let mut gx = DeviceBuffer::<f64>::zeroed(stream, ndof)?;
        let mut gy = DeviceBuffer::<f64>::zeroed(stream, ndof)?;
        let mut partial = DeviceBuffer::<f64>::zeroed(stream, nb)?;
        let mut d_rs = DeviceBuffer::<f64>::zeroed(stream, 1)?;
        let mut d_rsnew = DeviceBuffer::<f64>::zeroed(stream, 1)?;
        let mut d_pap = DeviceBuffer::<f64>::zeroed(stream, 1)?;
        let mut d_alpha = DeviceBuffer::<f64>::zeroed(stream, 1)?;
        let mut d_nalpha = DeviceBuffer::<f64>::zeroed(stream, 1)?;
        let mut d_beta = DeviceBuffer::<f64>::zeroed(stream, 1)?;
        let mut d_nmean = DeviceBuffer::<f64>::zeroed(stream, 1)?;
        let ones = up(&vec![1.0f64; ndof])?;

        let nev = self.ne as u32;
        let (gcfg, ocfg) = matvec_cfgs(self.ne, self.n1);
        let red = LaunchConfig { grid_dim: (nb as u32, 1, 1), block_dim: (RED as u32, 1, 1), shared_mem_bytes: 0 };
        let red1 = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (RED as u32, 1, 1), shared_mem_bytes: 0 };
        let one = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
        let vec_cfg = LaunchConfig::for_num_elems(ndof as u32);
        let n1 = self.n1;
        let n64 = ndof as u64;
        let nb64 = nb as u64;
        let ninv = 1.0 / ndof as f64;

        macro_rules! dot_to {
            ($a:expr, $b:expr, $out:expr) => {{
                module.dot_partial(stream, red, $a, $b, n64, &mut partial)?;
                module.reduce_scalar(stream, red1, &partial, nb64, $out)?;
            }};
        }
        // Deflate r ← r − mean(r) on-device (constant-nullspace removal for pure-Neumann).
        macro_rules! deflate_r {
            () => {{
                if deflate {
                    dot_to!(&r, &ones, &mut d_nmean);
                    module.cg_negmean(stream, one, &mut d_nmean, ninv)?;
                    module.axpy_s(stream, vec_cfg, &mut r, &ones, &d_nmean)?;
                }
            }};
        }
        macro_rules! apply {
            ($field:expr, $dst:expr) => {{
                module.gradient::<f64, f64>(stream, gcfg, &self.d_dev, $field, self.rx, self.sy, n1, nev, &mut gx, &mut gy)?;
                module.operator::<f64, f64>(
                    stream, ocfg, &self.d_dev, $field, &gx, &gy, &self.mass_dev, n1, nev, self.rx, self.sy, self.jac,
                    &self.fvl_dev, &self.fnx_dev, &self.fny_dev, &self.fsw_dev, &fnbr_dev, &self.ftau_dev, reaction, $dst,
                )?;
            }};
        }

        const CHECK: usize = 25;
        deflate_r!(); // r = deflate(b)
        module.xpby(stream, vec_cfg, &mut p, &r, 0.0)?; // p = r (β=0)
        dot_to!(&r, &r, &mut d_rs);
        let bnorm = d_rs.to_host_vec(stream)?[0].sqrt().max(1e-300);
        let mut iters = 0;
        let mut converged = false;
        for it in 0..maxit {
            apply!(&p, &mut ap);
            dot_to!(&p, &ap, &mut d_pap);
            module.cg_alpha(stream, one, &d_rs, &d_pap, &mut d_alpha, &mut d_nalpha)?;
            module.axpy_s(stream, vec_cfg, &mut x, &p, &d_alpha)?; // x += α p
            module.axpy_s(stream, vec_cfg, &mut r, &ap, &d_nalpha)?; // r −= α ap
            deflate_r!();
            dot_to!(&r, &r, &mut d_rsnew);
            iters = it + 1;
            if (it + 1) % CHECK == 0 || it + 1 == maxit {
                if d_rsnew.to_host_vec(stream)?[0].sqrt() / bnorm < tol {
                    converged = true;
                    break;
                }
            }
            module.cg_beta(stream, one, &d_rsnew, &mut d_rs, &mut d_beta)?; // β = rsnew/rs; rs ← rsnew
            module.xpby_s(stream, vec_cfg, &mut p, &r, &d_beta)?; // p = r + β p
        }
        let rel = d_rsnew.to_host_vec(stream)?[0].sqrt() / bnorm;
        let label = if deflate { "GpuPoisson deflated-pressure CG" } else { "GpuPoisson CG" };
        warn_unconverged(label, converged, iters, maxit, rel, tol);
        Ok((x.to_host_vec(stream)?, iters))
    }
}
