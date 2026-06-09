//! GPU symmetric interior-penalty (SIPG) Poisson / Helmholtz on the **3D hex mesh** —
//! the GPU analogue of `gale::dg::Poisson3d`, and the 3D counterpart of
//! [`crate::operators::poisson`]. Matrix-free operator `λM + A` with conjugate-gradient
//! solves entirely on the GPU; `M = diag(jw)` is the diagonal GLL mass.
//!
//! `A u` = volume stiffness `Drᵀ W Dr + Dsᵀ W Ds + Dtᵀ W Dt` (in physical coords via
//! the chain-rule metrics) plus the SIPG face terms (consistency `−∮{∇u·n}[v]`,
//! penalty `+∮τ[u][v]`, symmetry lift `−∮{∇v·n}[u]`) over the 6 hex faces, in the
//! per-element **gather** form (each block computes its own element's output, reading
//! neighbor values/gradients read-only) — so no cross-block races. Mirrors
//! `gale::dg::Poisson3d::apply` exactly; validated bit-for-bit against it.
//!
//! Two-kernel pipeline `gradient3d → operator3d`. Metrics are packed node-major
//! (`met[b*9 + {rx,ry,rz,sx,sy,sz,tx,ty,tz}]`), gradients packed (`g[b*3+{x,y,z}]`),
//! face floats packed (`fmet[idx*4+{nx,ny,nz,sw}]`) to keep kernel signatures narrow
//! (the NVVM-text backend mis-parses very wide signatures).

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice, DynamicSharedArray, SharedArray};
use cuda_host::cuda_module;
use gale::dg::{Face, Mesh3d, Neighbor3};
use std::sync::Arc;

const NN_MAX: usize = 125; // (p+1)³ up to p=4
const RED: usize = 256; // reduction block size
const BND: u32 = u32::MAX; // Dirichlet boundary face (SIPG consistency+penalty)
const NEU: u32 = u32::MAX - 1; // Neumann boundary face (natural BC ⇒ no contribution)

#[cuda_module]
mod kernels {
    use super::*;

    /// Physical gradient `∇u` on each hex, packed `g[b*3 + {x,y,z}]`. `e = blockIdx.x`,
    /// node `m = threadIdx.x`. Reference derivatives by sum-factorization, then the
    /// chain rule `gx = rx·ur + sx·us + tx·ut` (metric *columns*).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn gradient3d(
        d: &[f64], u: &[f64], met: &[f64], n1: u32,
        mut gx: DisjointSlice<f64>, mut gy: DisjointSlice<f64>, mut gz: DisjointSlice<f64>,
    ) {
        // Dynamic shared sized to the runtime element (`shared_mem_bytes = 2·nn·8` at
        // launch), not the compile-time `NN_MAX` worst case — so a low-order solve doesn't
        // over-reserve shared and throttle SM occupancy. Layout: DS=[0,nn), US=[nn,2nn).
        let n1 = n1 as usize;
        let n2 = n1 * n1;
        let nn = n1 * n2;
        let sm = DynamicSharedArray::<f64>::get();
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let b = e * nn + m;
        unsafe {
            *sm.add(m) = d[m];
            *sm.add(nn + m) = u[b];
        }
        thread::sync_threads();
        let i = m % n1;
        let j = (m / n1) % n1;
        let k = m / n2;
        let mut ur = 0.0f64;
        let mut us = 0.0f64;
        let mut ut = 0.0f64;
        let mut a = 0usize;
        while a < n1 {
            unsafe {
                ur += *sm.add(i * n1 + a) * *sm.add(nn + a + j * n1 + k * n2);
                us += *sm.add(j * n1 + a) * *sm.add(nn + i + a * n1 + k * n2);
                ut += *sm.add(k * n1 + a) * *sm.add(nn + i + j * n1 + a * n2);
            }
            a += 1;
        }
        let mo = b * 9;
        let gxv = met[mo] * ur + met[mo + 3] * us + met[mo + 6] * ut;
        let gyv = met[mo + 1] * ur + met[mo + 4] * us + met[mo + 7] * ut;
        let gzv = met[mo + 2] * ur + met[mo + 5] * us + met[mo + 8] * ut;
        if let Some(o) = gx.get_mut(thread::index_1d()) {
            *o = gxv;
        }
        if let Some(o) = gy.get_mut(thread::index_1d()) {
            *o = gyv;
        }
        if let Some(o) = gz.get_mut(thread::index_1d()) {
            *o = gzv;
        }
    }

    /// SIPG operator action `(λM + A)·u`, packed-gradient gather form. `g` is the
    /// global packed gradient from [`gradient3d`]; `fnbr` indexes global nodes
    /// (`BND`/`NEU` sentinels for boundary faces). One block per element.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn operator3d(
        d: &[f64], u: &[f64], gx: &[f64], gy: &[f64], gz: &[f64], met: &[f64], jw: &[f64], n1: u32,
        _face_vl: &[u32], fmet: &[f64], face_nbr: &[u32], face_tau: &[f64], lambda: f64,
        mut out: DisjointSlice<f64>,
    ) {
        // Dynamic shared sized to the runtime element (`shared_mem_bytes = 4·nn·8` at
        // launch), not the compile-time `NN_MAX`/`P3MAX` worst case — right-sizing this is a
        // ~1.5× win on the 3D solve at the orders we run (occupancy is shared-bound here).
        // Layout: DS=[0,nn); packed P pr/ps/pt = [nn,2nn)/[2nn,3nn)/[3nn,4nn).
        let n1 = n1 as usize;
        let n2 = n1 * n1;
        let nn = n1 * n2;
        let sm = DynamicSharedArray::<f64>::get();
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let b = e * nn + m;
        let mo = b * 9;
        unsafe {
            *sm.add(m) = d[m];
            // Volume flux W·∇u, projected to (r,s,t) contravariant directions (rows).
            let wx = jw[b] * gx[b];
            let wy = jw[b] * gy[b];
            let wz = jw[b] * gz[b];
            *sm.add(nn + m) = met[mo] * wx + met[mo + 1] * wy + met[mo + 2] * wz;
            *sm.add(2 * nn + m) = met[mo + 3] * wx + met[mo + 4] * wy + met[mo + 5] * wz;
            *sm.add(3 * nn + m) = met[mo + 6] * wx + met[mo + 7] * wy + met[mo + 8] * wz;
        }
        // Face contribution by **direct membership** (race-free; the 3D analogue of the 2D
        // operator). A hex node m=(ii,jj,kk) lies on at most 3 of the 6 faces (an element
        // corner). Its in-face position `a` follows the tensor face-node convention (see
        // `hex_faces`): Bottom/Top run over (i,j) ⇒ a=ii+jj·n1; South/North over (i,k) ⇒
        // a=ii+kk·n1; West/East over (j,k) ⇒ a=jj+kk·n1. So we visit only the ≤3 faces this
        // node is on — no scan over all 6·n2 entries. The bit-for-bit operator validator
        // confirms the convention. `face_vl[idx] == m` ⇒ interior trace `g{x,y,z}[b]`.
        let ii = m % n1;
        let jj = (m / n1) % n1;
        let kk = m / n2;
        let mut rf = 0.0f64;
        let mut hx = 0.0f64;
        let mut hy = 0.0f64;
        let mut hz = 0.0f64;
        let mut t = 0usize;
        while t < 6 {
            let (on, a) = if t == 0 {
                (kk == 0, ii + jj * n1) // Bottom
            } else if t == 1 {
                (kk == n1 - 1, ii + jj * n1) // Top
            } else if t == 2 {
                (jj == 0, ii + kk * n1) // South
            } else if t == 3 {
                (jj == n1 - 1, ii + kk * n1) // North
            } else if t == 4 {
                (ii == 0, jj + kk * n1) // West
            } else {
                (ii == n1 - 1, jj + kk * n1) // East
            };
            if on {
                let idx = (e * 6 + t) * n2 + a;
                let nbr = face_nbr[idx];
                if nbr != NEU {
                    let tau = face_tau[e * 6 + t];
                    let fo = idx * 4;
                    let nx = fmet[fo];
                    let ny = fmet[fo + 1];
                    let nz = fmet[fo + 2];
                    let sw = fmet[fo + 3];
                    let dun_e = nx * gx[b] + ny * gy[b] + nz * gz[b];
                    let ug = u[b];
                    let (avg, jump, gfac) = if nbr == BND {
                        (dun_e, ug, 1.0) // Dirichlet
                    } else {
                        let ng = nbr as usize;
                        (0.5 * (dun_e + nx * gx[ng] + ny * gy[ng] + nz * gz[ng]), ug - u[ng], 0.5)
                    };
                    let gg = gfac * sw * jump;
                    rf += -sw * avg + tau * sw * jump;
                    hx += gg * nx;
                    hy += gg * ny;
                    hz += gg * nz;
                }
            }
            t += 1;
        }
        // Subtract the symmetry-lift sources (project H to (r,s,t) like the volume).
        unsafe {
            *sm.add(nn + m) -= met[mo] * hx + met[mo + 1] * hy + met[mo + 2] * hz;
            *sm.add(2 * nn + m) -= met[mo + 3] * hx + met[mo + 4] * hy + met[mo + 5] * hz;
            *sm.add(3 * nn + m) -= met[mo + 6] * hx + met[mo + 7] * hy + met[mo + 8] * hz;
        }
        thread::sync_threads();
        let i = m % n1;
        let j = (m / n1) % n1;
        let k = m / n2;
        let mut acc = 0.0f64;
        let mut a = 0usize;
        while a < n1 {
            unsafe {
                acc += *sm.add(a * n1 + i) * *sm.add(nn + a + j * n1 + k * n2);
                acc += *sm.add(a * n1 + j) * *sm.add(2 * nn + i + a * n1 + k * n2);
                acc += *sm.add(a * n1 + k) * *sm.add(3 * nn + i + j * n1 + a * n2);
            }
            a += 1;
        }
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = acc + rf + lambda * jw[b] * u[b];
        }
    }

    /// y ← y + a·x
    #[kernel]
    pub fn axpy3v(mut y: DisjointSlice<f64>, x: &[f64], a: f64) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = y.get_mut(idx) {
            *o += a * x[i];
        }
    }

    /// y ← x + b·y
    #[kernel]
    pub fn xpby3v(mut y: DisjointSlice<f64>, x: &[f64], b: f64) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = y.get_mut(idx) {
            *o = x[i] + b * *o;
        }
    }

    /// Multi-block grid-stride dot product: each block reduces its strided slice into
    /// shared memory and writes one partial to `partial[blockIdx]` (host sums them). With
    /// `gridDim = 1` this is the old single-block reduction; with many blocks it streams
    /// `a`/`b` across all SMs at ~peak bandwidth instead of saturating one SM.
    #[kernel]
    pub fn dot3v_partial(a: &[f64], b: &[f64], n: u64, mut partial: DisjointSlice<f64>) {
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

/// Number of blocks for the multi-block `dot3v_partial` reduction (see the 2D
/// `dot_blocks`): enough to stream the vector across all SMs, capped at 1024 so the
/// host-side sum of partials stays trivial. `RED` threads per block.
fn dot_blocks(ndof: usize) -> usize {
    ndof.div_ceil(RED).clamp(1, 1024)
}

/// Per-element metrics + flattened face metadata for one hex mesh, ready to upload.
struct MeshArrays3d {
    nn: usize,
    ne: usize,
    ndof: usize,
    n1: u32,
    diff_pad: Vec<f64>, // n1² diff matrix padded to n³
    met: Vec<f64>,      // [ndof*9] node-major rx,ry,rz,sx,sy,sz,tx,ty,tz
    jw: Vec<f64>,
    fvl: Vec<u32>,
    fmet: Vec<f64>, // [nfc*4] nx,ny,nz,sw
    fnbr: Vec<u32>,
    ftau: Vec<f64>, // [ne*6]
}

fn flatten3d(mesh: &Mesh3d, alpha: f64, neumann_tags: &[u32]) -> MeshArrays3d {
    let nn = mesh.refh.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let order = mesh.order;
    let n1 = (order + 1) as u32;
    assert!(nn <= NN_MAX, "p too large for NN_MAX={NN_MAX}");
    let n2 = (n1 * n1) as usize;

    let mut met = vec![0.0; ndof * 9];
    let mut jw = vec![0.0; ndof];
    for (e, el) in mesh.elements.iter().enumerate() {
        let d = &el.geom;
        for k in 0..nn {
            let b = e * nn + k;
            met[b * 9] = d.rx[k];
            met[b * 9 + 1] = d.ry[k];
            met[b * 9 + 2] = d.rz[k];
            met[b * 9 + 3] = d.sx[k];
            met[b * 9 + 4] = d.sy[k];
            met[b * 9 + 5] = d.sz[k];
            met[b * 9 + 6] = d.tx[k];
            met[b * 9 + 7] = d.ty[k];
            met[b * 9 + 8] = d.tz[k];
            jw[b] = d.jw[k];
        }
    }

    let p1 = (order + 1) as f64;
    let h: Vec<f64> = mesh.elements.iter().map(|el| el.geom.jw.iter().sum::<f64>().cbrt()).collect();
    let nfc = ne * 6 * n2;
    let mut fvl = vec![0u32; nfc];
    let mut fmet = vec![0.0; nfc * 4];
    let mut fnbr = vec![BND; nfc];
    let mut ftau = vec![0.0; ne * 6];
    for (e, el) in mesh.elements.iter().enumerate() {
        for (t, face) in Face::ALL.iter().enumerate() {
            let fc = &el.faces[*face as usize];
            let nb = &el.neighbors[*face as usize];
            ftau[e * 6 + t] = match nb {
                Neighbor3::Interior { elem: re, .. } => alpha * p1 * p1 / h[e].min(h[*re]),
                Neighbor3::Boundary { .. } => alpha * p1 * p1 / h[e],
            };
            let neumann_boundary = matches!(nb, Neighbor3::Boundary { tag } if neumann_tags.contains(tag));
            for a in 0..n2 {
                let idx = (e * 6 + t) * n2 + a;
                fvl[idx] = fc.nodes[a] as u32;
                fmet[idx * 4] = fc.nx[a];
                fmet[idx * 4 + 1] = fc.ny[a];
                fmet[idx * 4 + 2] = fc.nz[a];
                fmet[idx * 4 + 3] = fc.sw[a];
                if let Neighbor3::Interior { elem: re, face: rface, perm } = nb {
                    let rf = &mesh.elements[*re].faces[*rface as usize];
                    fnbr[idx] = (*re * nn + rf.nodes[perm[a]]) as u32;
                } else if neumann_boundary {
                    fnbr[idx] = NEU;
                }
            }
        }
    }

    let mut diff_pad = vec![0.0; nn];
    diff_pad[..mesh.refh.line.diff.len()].copy_from_slice(&mesh.refh.line.diff);

    MeshArrays3d { nn, ne, ndof, n1, diff_pad, met, jw, fvl, fmet, fnbr, ftau }
}

/// Apply the matrix-free SIPG Poisson/Helmholtz operator `(λM + A)·u` once on the GPU
/// (Dirichlet boundaries). Bit-for-bit equal to `gale::dg::Poisson3d::apply`.
pub fn poisson3d_apply(mesh: &Mesh3d, u: &[f64], alpha: f64, reaction: f64) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let ma = flatten3d(mesh, alpha, &[]);
    assert_eq!(u.len(), ma.ndof, "state length must be n_elements·n_nodes");
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
    let upu = |v: &[u32]| DeviceBuffer::from_host(&stream, v);
    let d_dev = up(&ma.diff_pad)?;
    let u_dev = up(u)?;
    let met_dev = up(&ma.met)?;
    let jw_dev = up(&ma.jw)?;
    let fvl_dev = upu(&ma.fvl)?;
    let fmet_dev = up(&ma.fmet)?;
    let fnbr_dev = upu(&ma.fnbr)?;
    let ftau_dev = up(&ma.ftau)?;
    let mut gx_dev = DeviceBuffer::<f64>::zeroed(&stream, ma.ndof)?;
    let mut gy_dev = DeviceBuffer::<f64>::zeroed(&stream, ma.ndof)?;
    let mut gz_dev = DeviceBuffer::<f64>::zeroed(&stream, ma.ndof)?;
    let mut out_dev = DeviceBuffer::<f64>::zeroed(&stream, ma.ndof)?;
    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig { grid_dim: (ma.ne as u32, 1, 1), block_dim: (ma.nn as u32, 1, 1), shared_mem_bytes: (4 * ma.nn * 8) as u32 };
    module.gradient3d(&stream, cfg, &d_dev, &u_dev, &met_dev, ma.n1, &mut gx_dev, &mut gy_dev, &mut gz_dev)?;
    module.operator3d(
        &stream, cfg, &d_dev, &u_dev, &gx_dev, &gy_dev, &gz_dev, &met_dev, &jw_dev, ma.n1,
        &fvl_dev, &fmet_dev, &fnbr_dev, &ftau_dev, reaction, &mut out_dev,
    )?;
    Ok(out_dev.to_host_vec(&stream)?)
}

/// Solve the 3D SIPG **Helmholtz** system `(λM + A)·u = b` (`reaction = λ ≥ 0`;
/// `λ = 0` is pure Poisson) by conjugate gradient on the GPU (Dirichlet boundaries).
/// Mirrors `gale::dg::Poisson3d::with_reaction(mesh, alpha, λ).cg`.
pub fn helmholtz3d_cg_solve(mesh: &Mesh3d, b: &[f64], alpha: f64, reaction: f64, tol: f64, maxit: usize) -> Result<(Vec<f64>, usize), Box<dyn std::error::Error>> {
    cg3d_impl(mesh, b, alpha, reaction, &[], false, tol, maxit)
}

/// Tag-aware 3D CG for `(reaction·M + A)·x = b`: boundary tags in `neumann_tags` are
/// natural (Neumann), the rest Dirichlet SIPG — the per-region (inflow/outflow/symmetry)
/// path. Non-deflated (a Dirichlet boundary makes it non-singular); used for the velocity
/// Helmholtz (`neumann_tags` = outflow + symmetry-tangential) and the outflow-pinned
/// pressure (`reaction = 0`, `neumann_tags` = all except outflow). Mirrors
/// `gale::dg::Poisson3d::with_bc(mesh, alpha, reaction, neumann_tags).cg`.
pub fn helmholtz3d_cg_solve_tags(
    mesh: &Mesh3d,
    b: &[f64],
    alpha: f64,
    reaction: f64,
    neumann_tags: &[u32],
    tol: f64,
    maxit: usize,
) -> Result<(Vec<f64>, usize), Box<dyn std::error::Error>> {
    cg3d_impl(mesh, b, alpha, reaction, neumann_tags, false, tol, maxit)
}

/// Solve the singular pure-Neumann 3D pressure-Poisson `A·u = b` by **deflated** CG on
/// the GPU (constant nullspace removed each iteration). Mirrors
/// `gale::dg::Poisson3d::with_bc(mesh, alpha, 0, all-tags).cg_deflated`.
pub fn pressure3d_cg_solve(mesh: &Mesh3d, b: &[f64], alpha: f64, tol: f64, maxit: usize) -> Result<(Vec<f64>, usize), Box<dyn std::error::Error>> {
    cg3d_impl(mesh, b, alpha, 0.0, &mesh.boundary_tags(), true, tol, maxit)
}

/// CG for `(reaction·M + A)·x = b`; `deflate` removes the constant nullspace each
/// iteration (for the singular pure-Neumann pressure system).
#[allow(clippy::too_many_arguments)]
fn cg3d_impl(mesh: &Mesh3d, b: &[f64], alpha: f64, reaction: f64, neumann_tags: &[u32], deflate: bool, tol: f64, maxit: usize) -> Result<(Vec<f64>, usize), Box<dyn std::error::Error>> {
    let ma = flatten3d(mesh, alpha, neumann_tags);
    let ndof = ma.ndof;
    assert_eq!(b.len(), ndof, "rhs length must be n_elements·n_nodes");
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
    let upu = |v: &[u32]| DeviceBuffer::from_host(&stream, v);
    let d_dev = up(&ma.diff_pad)?;
    let met_dev = up(&ma.met)?;
    let jw_dev = up(&ma.jw)?;
    let fvl_dev = upu(&ma.fvl)?;
    let fmet_dev = up(&ma.fmet)?;
    let fnbr_dev = upu(&ma.fnbr)?;
    let ftau_dev = up(&ma.ftau)?;

    let mut x = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut r = up(b)?;
    let mut p = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut ap = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut gx = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut gy = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut gz = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let nb = dot_blocks(ndof);
    let mut partial = DeviceBuffer::<f64>::zeroed(&stream, nb)?;
    let ones = up(&vec![1.0f64; ndof])?;

    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig { grid_dim: (ma.ne as u32, 1, 1), block_dim: (ma.nn as u32, 1, 1), shared_mem_bytes: (4 * ma.nn * 8) as u32 };
    let red = LaunchConfig { grid_dim: (nb as u32, 1, 1), block_dim: (RED as u32, 1, 1), shared_mem_bytes: 0 };
    let vec_cfg = LaunchConfig::for_num_elems(ndof as u32);
    let n1 = ma.n1;
    let n64 = ndof as u64;
    let ninv = 1.0 / ndof as f64;

    macro_rules! dot {
        ($a:expr, $b:expr) => {{
            module.dot3v_partial(&stream, red, $a, $b, n64, &mut partial)?;
            partial.to_host_vec(&stream)?.iter().sum::<f64>()
        }};
    }
    macro_rules! deflate {
        ($v:expr) => {{
            if deflate {
                let mean = dot!($v, &ones) * ninv;
                module.axpy3v(&stream, vec_cfg, $v, &ones, -mean)?;
            }
        }};
    }
    macro_rules! apply {
        ($field:expr, $dst:expr) => {{
            module.gradient3d(&stream, cfg, &d_dev, $field, &met_dev, n1, &mut gx, &mut gy, &mut gz)?;
            module.operator3d(
                &stream, cfg, &d_dev, $field, &gx, &gy, &gz, &met_dev, &jw_dev, n1,
                &fvl_dev, &fmet_dev, &fnbr_dev, &ftau_dev, reaction, $dst,
            )?;
        }};
    }

    deflate!(&mut r);
    module.xpby3v(&stream, vec_cfg, &mut p, &r, 0.0)?; // p = r
    let bn = dot!(&r, &r).sqrt().max(1e-300);
    let mut rs = dot!(&r, &r);
    let mut iters = 0;
    for it in 0..maxit {
        apply!(&p, &mut ap);
        let pap = dot!(&p, &ap);
        let alpha_cg = rs / pap;
        module.axpy3v(&stream, vec_cfg, &mut x, &p, alpha_cg)?;
        module.axpy3v(&stream, vec_cfg, &mut r, &ap, -alpha_cg)?;
        deflate!(&mut r);
        let rs_new = dot!(&r, &r);
        iters = it + 1;
        if rs_new.sqrt() / bn < tol {
            break;
        }
        let beta = rs_new / rs;
        module.xpby3v(&stream, vec_cfg, &mut p, &r, beta)?;
        rs = rs_new;
    }
    Ok((x.to_host_vec(&stream)?, iters))
}

// ===== Persistent 3D solver handle (P4) ==========================================

/// Persistent GPU 3D SIPG-Poisson / Helmholtz solver handle — the hex-mesh analogue of
/// [`crate::GpuPoisson`]. Owns the CUDA context, the loaded device module, and the
/// uploaded **constant** mesh arrays (metrics + face geometry + tau), so repeated solves
/// on the same mesh pay the heavy setup (`CudaContext::new` + `kernels::load` cubin/JIT +
/// mesh upload, ~0.3 s) **once** instead of per call — the 3 elliptic solves/step of the
/// 3D dual-splitting flow loop, across every timestep, share one handle.
///
/// Only the per-region `neumann_tags`/`reaction`/`deflate` vary between solves;
/// [`solve`](Self::solve) rebuilds just the small `fnbr` array on the host and uploads it.
/// CG scalars are read back per dot (the multi-block `dot3v_partial` + host sum — the same
/// as `cg3d_impl`; the on-device-scalar refinement is 2D-only for now). Bit-for-bit
/// equivalent to `cg3d_impl` (i.e. `helmholtz3d_cg_solve_tags` / `pressure3d_cg_solve`).
pub struct GpuPoisson3d {
    // `stream` and `module` each hold an `Arc<CudaContext>`, keeping the context alive.
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
    nn: usize,
    ne: usize,
    ndof: usize,
    n1: u32,
    // constant (neumann-tag-independent) device arrays, uploaded once.
    d_dev: DeviceBuffer<f64>,
    met_dev: DeviceBuffer<f64>,
    jw_dev: DeviceBuffer<f64>,
    fvl_dev: DeviceBuffer<u32>,
    fmet_dev: DeviceBuffer<f64>,
    ftau_dev: DeviceBuffer<f64>,
    // host state to rebuild the per-region `fnbr` cheaply (base = all-Dirichlet; flip the
    // listed boundary-face nodes to NEU when their tag is in `neumann_tags`).
    fnbr_base: Vec<u32>,
    bnodes: Vec<(usize, u32)>,
}

impl GpuPoisson3d {
    /// Build the handle for `mesh` with SIPG penalty factor `alpha`: create the context,
    /// load the module, and upload the constant metrics/face data.
    pub fn new(mesh: &Mesh3d, alpha: f64) -> Result<Self, Box<dyn std::error::Error>> {
        let ma = flatten3d(mesh, alpha, &[]); // fnbr_base: all boundary faces Dirichlet (BND)
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
        let upu = |v: &[u32]| DeviceBuffer::from_host(&stream, v);

        // Boundary-face nodes + tags, in the same flat order flatten3d uses (6 faces, n2
        // nodes each), so a per-region solve flips only these entries to NEU.
        let n2 = (ma.n1 * ma.n1) as usize;
        let mut bnodes = Vec::new();
        for (e, el) in mesh.elements.iter().enumerate() {
            for (t, face) in Face::ALL.iter().enumerate() {
                if let Neighbor3::Boundary { tag } = el.neighbors[*face as usize] {
                    for a in 0..n2 {
                        bnodes.push(((e * 6 + t) * n2 + a, tag));
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
            d_dev: up(&ma.diff_pad)?,
            met_dev: up(&ma.met)?,
            jw_dev: up(&ma.jw)?,
            fvl_dev: upu(&ma.fvl)?,
            fmet_dev: up(&ma.fmet)?,
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

    /// Solve `(reaction·M + A)·x = b` on this hex mesh by CG (multi-block dot + host-sum
    /// scalars). `neumann_tags` are the natural-BC boundary tags (`&[]` = all-Dirichlet
    /// velocity Helmholtz; `&mesh.boundary_tags()` = pure-Neumann pressure). `deflate`
    /// removes the constant nullspace each iteration. No per-call context/module setup or
    /// constant-mesh upload — only `b`, the small `fnbr`, and scratch move.
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
        let mut gz = DeviceBuffer::<f64>::zeroed(stream, ndof)?;
        let mut partial = DeviceBuffer::<f64>::zeroed(stream, nb)?;
        let ones = up(&vec![1.0f64; ndof])?;

        let cfg = LaunchConfig { grid_dim: (self.ne as u32, 1, 1), block_dim: (self.nn as u32, 1, 1), shared_mem_bytes: (4 * self.nn * 8) as u32 };
        let red = LaunchConfig { grid_dim: (nb as u32, 1, 1), block_dim: (RED as u32, 1, 1), shared_mem_bytes: 0 };
        let vec_cfg = LaunchConfig::for_num_elems(ndof as u32);
        let n1 = self.n1;
        let n64 = ndof as u64;
        let ninv = 1.0 / ndof as f64;

        macro_rules! dot {
            ($a:expr, $b:expr) => {{
                module.dot3v_partial(stream, red, $a, $b, n64, &mut partial)?;
                partial.to_host_vec(stream)?.iter().sum::<f64>()
            }};
        }
        macro_rules! deflate_v {
            ($v:expr) => {{
                if deflate {
                    let mean = dot!($v, &ones) * ninv;
                    module.axpy3v(stream, vec_cfg, $v, &ones, -mean)?;
                }
            }};
        }
        macro_rules! apply {
            ($field:expr, $dst:expr) => {{
                module.gradient3d(stream, cfg, &self.d_dev, $field, &self.met_dev, n1, &mut gx, &mut gy, &mut gz)?;
                module.operator3d(
                    stream, cfg, &self.d_dev, $field, &gx, &gy, &gz, &self.met_dev, &self.jw_dev, n1,
                    &self.fvl_dev, &self.fmet_dev, &fnbr_dev, &self.ftau_dev, reaction, $dst,
                )?;
            }};
        }

        deflate_v!(&mut r);
        module.xpby3v(stream, vec_cfg, &mut p, &r, 0.0)?; // p = r
        let bn = dot!(&r, &r).sqrt().max(1e-300);
        let mut rs = dot!(&r, &r);
        let mut iters = 0;
        for it in 0..maxit {
            apply!(&p, &mut ap);
            let pap = dot!(&p, &ap);
            let alpha_cg = rs / pap;
            module.axpy3v(stream, vec_cfg, &mut x, &p, alpha_cg)?;
            module.axpy3v(stream, vec_cfg, &mut r, &ap, -alpha_cg)?;
            deflate_v!(&mut r);
            let rs_new = dot!(&r, &r);
            iters = it + 1;
            if rs_new.sqrt() / bn < tol {
                break;
            }
            let beta = rs_new / rs;
            module.xpby3v(stream, vec_cfg, &mut p, &r, beta)?;
            rs = rs_new;
        }
        Ok((x.to_host_vec(stream)?, iters))
    }
}
