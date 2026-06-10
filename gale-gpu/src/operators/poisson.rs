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

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use std::sync::Arc;
use cuda_host::cuda_module;
use gale::dg::{Edge, Mesh2d, Neighbor, PMultigrid};

const NN_MAX: usize = 81; // (order 8 + 1)²
const RED: usize = 256; // reduction block size
const BND: u32 = u32::MAX; // sentinel: Dirichlet boundary face (SIPG consistency+penalty)
const NEU: u32 = u32::MAX - 1; // sentinel: Neumann boundary face (natural BC ⇒ no contribution)

/// Number of blocks for the multi-block `dot_partial` reduction: enough to stream the
/// vector across all SMs (one block-partial each), capped so the host-side final sum of
/// the partials stays trivial. `RED` threads per block. Volta has 80 SMs; 1024 blocks ×
/// 256 threads saturates occupancy while summing only 1024 doubles on the host.
fn dot_blocks(ndof: usize) -> usize {
    ndof.div_ceil(RED).clamp(1, 1024)
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn gradient(
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

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn operator(
        d: &[f64], u: &[f64], gx: &[f64], gy: &[f64], rx: &[f64], ry: &[f64], sx: &[f64],
        sy: &[f64], jw: &[f64], n1: u32, _face_vl: &[u32], face_nx: &[f64], face_ny: &[f64],
        face_sw: &[f64], face_nbr: &[u32], face_tau: &[f64], lambda: f64, mut out: DisjointSlice<f64>,
    ) {
        static mut DS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PR: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        let n1 = n1 as usize;
        let nn = n1 * n1;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let b = e * nn + m;
        unsafe {
            DS[m] = d[m];
            let wx = jw[b] * gx[b];
            let wy = jw[b] * gy[b];
            PR[m] = rx[b] * wx + ry[b] * wy;
            PS[m] = sx[b] * wx + sy[b] * wy;
        }
        // Face contribution by **gather** (race-free, fully parallel): thread `m` owns
        // node `m`, scans the element's 4·n1 face entries for the ones at this node
        // (`face_vl == m`; corners match on two faces), and accumulates the SIPG
        // consistency/penalty `rf` and the symmetry-lift `hx,hy` into registers — so no
        // shared `RF/HX/HY` and no inter-thread races (each node written by one thread).
        // Since `face_vl == m`, the interior trace is `gx[b]`/`gy[b]`/`u[b]`.
        // Node m = (ii, jj) lies on at most 2 of the 4 faces (it's an element corner at
        // most). Its face position `a` follows the tensor face-node convention (see
        // `quad_faces`): South/North run along i (a=ii), East/West along j (a=jj). So we
        // visit only the ≤2 faces this node is on — no scan over all 4·n1 entries. The
        // bit-for-bit operator validator confirms the convention. `face_vl[idx] == m`.
        let ii = m % n1;
        let jj = m / n1;
        let mut rf = 0.0f64;
        let mut hx = 0.0f64;
        let mut hy = 0.0f64;
        let mut t = 0usize;
        while t < 4 {
            let (on, a) = if t == 0 {
                (jj == 0, ii) // South
            } else if t == 1 {
                (ii == n1 - 1, jj) // East
            } else if t == 2 {
                (jj == n1 - 1, ii) // North
            } else {
                (ii == 0, jj) // West
            };
            if on {
                let idx = (e * 4 + t) * n1 + a;
                let nbr = face_nbr[idx];
                if nbr != NEU {
                    let tau = face_tau[e * 4 + t];
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
            // SIPG stiffness `A·u` plus the Helmholtz reaction `λ·M·u` (diagonal GLL
            // mass `M = diag(jw)`). `λ = 0` ⇒ pure Poisson, bit-identical to before.
            *o = acc + rf + lambda * jw[b] * u[b];
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
    rx: Vec<f64>,
    ry: Vec<f64>,
    sx: Vec<f64>,
    sy: Vec<f64>,
    jw: Vec<f64>,
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
    let ftau_dev = up(&ma.ftau)?;
    let mut gx_dev = DeviceBuffer::<f64>::zeroed(&stream, ma.ndof)?;
    let mut gy_dev = DeviceBuffer::<f64>::zeroed(&stream, ma.ndof)?;
    let mut out_dev = DeviceBuffer::<f64>::zeroed(&stream, ma.ndof)?;

    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig {
        grid_dim: (ma.ne as u32, 1, 1),
        block_dim: (ma.nn as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    module.gradient(
        &stream, cfg, &d_dev, &u_dev, &rx_dev, &ry_dev, &sx_dev, &sy_dev, ma.n1,
        &mut gx_dev, &mut gy_dev,
    )?;
    module.operator(
        &stream, cfg, &d_dev, &u_dev, &gx_dev, &gy_dev, &rx_dev, &ry_dev, &sx_dev, &sy_dev,
        &jw_dev, ma.n1, &fvl_dev, &fnx_dev, &fny_dev, &fsw_dev, &fnbr_dev, &ftau_dev,
        0.0, &mut out_dev,
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
    let cfg = LaunchConfig {
        grid_dim: (ma.ne as u32, 1, 1),
        block_dim: (ma.nn as u32, 1, 1),
        shared_mem_bytes: 0,
    };
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
            module.gradient(&stream, cfg, &d_dev, $field, &rx_dev, &ry_dev, &sx_dev, &sy_dev, n1, &mut gx, &mut gy)?;
            module.operator(
                &stream, cfg, &d_dev, $field, &gx, &gy, &rx_dev, &ry_dev, &sx_dev, &sy_dev, &jw_dev, n1,
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
                break;
            }
        }
        module.cg_beta(&stream, one, &d_rsnew, &mut d_rs, &mut d_beta)?; // β = rsnew/rs; rs ← rsnew
        module.xpby_s(&stream, vec_cfg, &mut p, &r, &d_beta)?; // p = r + β p
    }

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
    let cfg = LaunchConfig {
        grid_dim: (ma.ne as u32, 1, 1),
        block_dim: (ma.nn as u32, 1, 1),
        shared_mem_bytes: 0,
    };
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
            module.gradient(&stream, cfg, &d_dev, $field, &rx_dev, &ry_dev, &sx_dev, &sy_dev, n1, &mut gx, &mut gy)?;
            module.operator(
                &stream, cfg, &d_dev, $field, &gx, &gy, &rx_dev, &ry_dev, &sx_dev, &sy_dev, &jw_dev, n1,
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
    for it in 0..maxit {
        apply!(&p, &mut ap);
        let pap = dot!(&p, &ap);
        let alpha_cg = rs / pap;
        module.axpy(&stream, vec_cfg, &mut x, &p, alpha_cg)?; // x += α p
        module.axpy(&stream, vec_cfg, &mut r, &ap, -alpha_cg)?; // r −= α ap
        deflate!(&mut r);
        let rs_new = dot!(&r, &r);
        iters = it + 1;
        if rs_new.sqrt() / bn < tol {
            break;
        }
        let beta = rs_new / rs;
        module.xpby(&stream, vec_cfg, &mut p, &r, beta)?; // p = r + β p
        rs = rs_new;
    }
    Ok((x.to_host_vec(&stream)?, iters))
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
    let nlev = mg.n_levels();
    let (n_pre, n_post) = mg.smoothing();
    let nn0 = mg.mesh(0).refq.n_nodes();
    let n0 = mg.mesh(0).n_elements() * nn0;
    assert_eq!(rhs.len(), n0, "rhs length must match the finest level");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
    let upu = |v: &[u32]| DeviceBuffer::from_host(&stream, v);

    // device buffers (struct-of-arrays per level)
    let mut n1v = Vec::new();
    let mut nev = Vec::new();
    let mut ndofv = Vec::new();
    let (mut dl, mut rxl, mut ryl, mut sxl, mut syl, mut jwl, mut invd) =
        (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    let (mut fvl, mut fnbr) = (vec![], vec![]);
    let (mut fnx, mut fny, mut fsw, mut ftau) = (vec![], vec![], vec![], vec![]);
    let mut omega = Vec::new();
    let (mut xb, mut bb, mut rb, mut apb, mut gxb, mut gyb, mut tmpb) =
        (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);

    for l in 0..nlev {
        let m = mg.mesh(l);
        let ma = flatten_mesh(m, mg.alpha, &[]);
        n1v.push(ma.n1);
        nev.push(ma.ne as u32);
        ndofv.push(ma.ndof);
        dl.push(up(&ma.diff)?);
        rxl.push(up(&ma.rx)?);
        ryl.push(up(&ma.ry)?);
        sxl.push(up(&ma.sx)?);
        syl.push(up(&ma.sy)?);
        jwl.push(up(&ma.jw)?);
        invd.push(up(mg.inv_diagonal(l))?);
        fvl.push(upu(&ma.fvl)?);
        fnbr.push(upu(&ma.fnbr)?);
        fnx.push(up(&ma.fnx)?);
        fny.push(up(&ma.fny)?);
        fsw.push(up(&ma.fsw)?);
        ftau.push(up(&ma.ftau)?);
        omega.push(mg.jacobi_omega(l));
        xb.push(DeviceBuffer::<f64>::zeroed(&stream, ma.ndof)?);
        bb.push(DeviceBuffer::<f64>::zeroed(&stream, ma.ndof)?);
        rb.push(DeviceBuffer::<f64>::zeroed(&stream, ma.ndof)?);
        apb.push(DeviceBuffer::<f64>::zeroed(&stream, ma.ndof)?);
        gxb.push(DeviceBuffer::<f64>::zeroed(&stream, ma.ndof)?);
        gyb.push(DeviceBuffer::<f64>::zeroed(&stream, ma.ndof)?);
        tmpb.push(DeviceBuffer::<f64>::zeroed(&stream, ma.ndof)?);
    }
    // transfer matrices (coarse l+1 → fine l)
    let mut interp = Vec::new();
    for l in 0..nlev - 1 {
        interp.push(up(mg.interp_matrix(l))?);
    }

    // per-level launch configs
    let cfg: Vec<LaunchConfig> = (0..nlev)
        .map(|l| LaunchConfig { grid_dim: (nev[l], 1, 1), block_dim: (n1v[l] * n1v[l], 1, 1), shared_mem_bytes: 0 })
        .collect();
    let vcfg: Vec<LaunchConfig> = ndofv.iter().map(|&n| LaunchConfig::for_num_elems(n as u32)).collect();

    // PCG vectors (finest level, separate from V-cycle scratch).
    let mut psol = DeviceBuffer::<f64>::zeroed(&stream, n0)?;
    let mut pres = up(rhs)?;
    let mut pp = DeviceBuffer::<f64>::zeroed(&stream, n0)?;
    let pz = DeviceBuffer::<f64>::zeroed(&stream, n0)?;
    let mut pap = DeviceBuffer::<f64>::zeroed(&stream, n0)?;
    let rhs_dev = up(rhs)?;
    // Multi-block reduction: partials sized for the finest level (the largest dot); each
    // dot launches dot_blocks($n) blocks and the host sums that many partials.
    let mut partial = DeviceBuffer::<f64>::zeroed(&stream, dot_blocks(n0))?;

    let module = kernels::load(&ctx)?;

    macro_rules! dot {
        ($a:expr, $b:expr, $n:expr) => {{
            let nbl = dot_blocks($n);
            let redcfg = LaunchConfig { grid_dim: (nbl as u32, 1, 1), block_dim: (RED as u32, 1, 1), shared_mem_bytes: 0 };
            module.dot_partial(&stream, redcfg, $a, $b, $n as u64, &mut partial)?;
            partial.to_host_vec(&stream)?[..nbl].iter().sum::<f64>()
        }};
    }
    macro_rules! dcopy {
        ($dst:expr, $src:expr, $n:expr) => {{
            unsafe {
                cuda_core::memory::memcpy_dtod_async($dst.cu_deviceptr(), $src.cu_deviceptr(), $n * 8, stream.cu_stream())?;
            }
        }};
    }
    // matvec: dst = A·src at level $l (src/dst external; gx/gy scratch from Vecs).
    // Helmholtz reaction λ (0 ⇒ pure Poisson) — the per-level diagonal/omega in `mg` are
    // already computed for this reaction, so the on-device V-cycle matches the CPU setup.
    let reaction = mg.reaction();
    macro_rules! matvec {
        ($l:expr, $src:expr, $dst:expr) => {{
            let l = $l;
            module.gradient(&stream, cfg[l], &dl[l], $src, &rxl[l], &ryl[l], &sxl[l], &syl[l], n1v[l], &mut gxb[l], &mut gyb[l])?;
            module.operator(
                &stream, cfg[l], &dl[l], $src, &gxb[l], &gyb[l], &rxl[l], &ryl[l], &sxl[l], &syl[l], &jwl[l], n1v[l],
                &fvl[l], &fnx[l], &fny[l], &fsw[l], &fnbr[l], &ftau[l], reaction, $dst,
            )?;
        }};
    }
    macro_rules! coarse_cg {
        ($l:expr) => {{
            let l = $l;
            let n = ndofv[l];
            module.scal(&stream, vcfg[l], &mut xb[l], 0.0)?;
            dcopy!(&rb[l], &bb[l], n);
            dcopy!(&tmpb[l], &bb[l], n);
            let bn = dot!(&bb[l], &bb[l], n).sqrt().max(1e-300);
            let mut rs = dot!(&rb[l], &rb[l], n);
            for _ in 0..300 {
                matvec!(l, &tmpb[l], &mut apb[l]);
                let pap_d = dot!(&tmpb[l], &apb[l], n);
                let alpha = rs / pap_d;
                module.axpy(&stream, vcfg[l], &mut xb[l], &tmpb[l], alpha)?;
                module.axpy(&stream, vcfg[l], &mut rb[l], &apb[l], -alpha)?;
                let rsn = dot!(&rb[l], &rb[l], n);
                if rsn.sqrt() / bn < 1e-10 {
                    break;
                }
                let beta = rsn / rs;
                module.xpby(&stream, vcfg[l], &mut tmpb[l], &rb[l], beta)?;
                rs = rsn;
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
                    matvec!(l, &xb[l], &mut apb[l]);
                    module.jacobi(&stream, vcfg[l], &mut xb[l], &bb[l], &apb[l], &invd[l], omega[l])?;
                }
                matvec!(l, &xb[l], &mut apb[l]);
                module.sub(&stream, vcfg[l], &mut rb[l], &bb[l], &apb[l])?;
                module.restrict(&stream, cfg[l], &interp[l], &rb[l], n1v[l], n1v[l + 1], &mut bb[l + 1])?;
            }
            coarse_cg!(last);
            for l in (0..last).rev() {
                module.prolong(&stream, cfg[l], &interp[l], &xb[l + 1], n1v[l], n1v[l + 1], &mut tmpb[l])?;
                module.axpy(&stream, vcfg[l], &mut xb[l], &tmpb[l], 1.0)?;
                for _ in 0..n_post {
                    matvec!(l, &xb[l], &mut apb[l]);
                    module.jacobi(&stream, vcfg[l], &mut xb[l], &bb[l], &apb[l], &invd[l], omega[l])?;
                }
            }
        }};
    }

    // preconditioned CG on the device
    module.scal(&stream, vcfg[0], &mut psol, 0.0)?;
    dcopy!(&bb[0], &pres, n0);
    vcycle!();
    dcopy!(&pz, &xb[0], n0);
    dcopy!(&pp, &pz, n0);
    let bn = dot!(&rhs_dev, &rhs_dev, n0).sqrt().max(1e-300);
    let mut rz = dot!(&pres, &pz, n0);
    let mut iters = 0;
    for it in 0..maxit {
        matvec!(0, &pp, &mut pap);
        let pap_d = dot!(&pp, &pap, n0);
        let alpha = rz / pap_d;
        module.axpy(&stream, vcfg[0], &mut psol, &pp, alpha)?;
        module.axpy(&stream, vcfg[0], &mut pres, &pap, -alpha)?;
        iters = it + 1;
        if dot!(&pres, &pres, n0).sqrt() / bn < tol {
            break;
        }
        dcopy!(&bb[0], &pres, n0);
        vcycle!();
        dcopy!(&pz, &xb[0], n0);
        let rz_new = dot!(&pres, &pz, n0);
        let beta = rz_new / rz;
        module.xpby(&stream, vcfg[0], &mut pp, &pz, beta)?;
        rz = rz_new;
    }

    Ok((psol.to_host_vec(&stream)?, iters))
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
    let cfg = LaunchConfig { grid_dim: (ma.ne as u32, 1, 1), block_dim: (ma.nn as u32, 1, 1), shared_mem_bytes: 0 };
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
        module.gradient(&stream, cfg, &d_dev, &u_dev, &rx_dev, &ry_dev, &sx_dev, &sy_dev, n1, &mut gx, &mut gy)?;
    });
    let op_ms = timed!({
        module.operator(
            &stream, cfg, &d_dev, &u_dev, &gx, &gy, &rx_dev, &ry_dev, &sx_dev, &sy_dev, &jw_dev, n1,
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

        let cfg = LaunchConfig { grid_dim: (self.ne as u32, 1, 1), block_dim: (self.nn as u32, 1, 1), shared_mem_bytes: 0 };
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
                module.gradient(stream, cfg, &self.d_dev, $field, &self.rx_dev, &self.ry_dev, &self.sx_dev, &self.sy_dev, n1, &mut gx, &mut gy)?;
                module.operator(
                    stream, cfg, &self.d_dev, $field, &gx, &gy, &self.rx_dev, &self.ry_dev, &self.sx_dev, &self.sy_dev, &self.jw_dev, n1,
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
                    break;
                }
            }
            module.cg_beta(stream, one, &d_rsnew, &mut d_rs, &mut d_beta)?; // β = rsnew/rs; rs ← rsnew
            module.xpby_s(stream, vec_cfg, &mut p, &r, &d_beta)?; // p = r + β p
        }
        Ok((x.to_host_vec(stream)?, iters))
    }
}
