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

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;
use gale::dg::{Face, Mesh3d, Neighbor3};

const NN_MAX: usize = 125; // (p+1)³ up to p=4
const P3MAX: usize = 3 * NN_MAX; // packed 3-vector shared arrays
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
        static mut DS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut US: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        let n1 = n1 as usize;
        let n2 = n1 * n1;
        let nn = n1 * n2;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let b = e * nn + m;
        unsafe {
            DS[m] = d[m];
            US[m] = u[b];
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
                ur += DS[i * n1 + a] * US[a + j * n1 + k * n2];
                us += DS[j * n1 + a] * US[i + a * n1 + k * n2];
                ut += DS[k * n1 + a] * US[i + j * n1 + a * n2];
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
        face_vl: &[u32], fmet: &[f64], face_nbr: &[u32], face_tau: &[f64], lambda: f64,
        mut out: DisjointSlice<f64>,
    ) {
        static mut DS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut P: SharedArray<f64, P3MAX> = SharedArray::UNINIT; // packed pr/ps/pt
        let n1 = n1 as usize;
        let n2 = n1 * n1;
        let nn = n1 * n2;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let b = e * nn + m;
        let mo = b * 9;
        unsafe {
            DS[m] = d[m];
            // Volume flux W·∇u, projected to (r,s,t) contravariant directions (rows).
            let wx = jw[b] * gx[b];
            let wy = jw[b] * gy[b];
            let wz = jw[b] * gz[b];
            P[m] = met[mo] * wx + met[mo + 1] * wy + met[mo + 2] * wz;
            P[NN_MAX + m] = met[mo + 3] * wx + met[mo + 4] * wy + met[mo + 5] * wz;
            P[2 * NN_MAX + m] = met[mo + 6] * wx + met[mo + 7] * wy + met[mo + 8] * wz;
        }
        // Face contribution by **gather** (race-free, fully parallel; see the 2D operator):
        // thread `m` scans the element's 6·n2 face entries for the ones at its node
        // (`face_vl == m`; hex corners match on three faces), accumulating the SIPG
        // consistency/penalty `rf` and symmetry-lift `hx,hy,hz` into registers — no shared
        // `RF/H`, no races. `face_vl == m` ⇒ the interior trace is `gx[b]`/`gy[b]`/`gz[b]`.
        let mut rf = 0.0f64;
        let mut hx = 0.0f64;
        let mut hy = 0.0f64;
        let mut hz = 0.0f64;
        let mut t = 0usize;
        while t < 6 {
            let tau = face_tau[e * 6 + t];
            let mut a = 0usize;
            while a < n2 {
                let idx = (e * 6 + t) * n2 + a;
                if face_vl[idx] as usize == m {
                    let nbr = face_nbr[idx];
                    if nbr != NEU {
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
                a += 1;
            }
            t += 1;
        }
        // Subtract the symmetry-lift sources (project H to (r,s,t) like the volume).
        unsafe {
            P[m] -= met[mo] * hx + met[mo + 1] * hy + met[mo + 2] * hz;
            P[NN_MAX + m] -= met[mo + 3] * hx + met[mo + 4] * hy + met[mo + 5] * hz;
            P[2 * NN_MAX + m] -= met[mo + 6] * hx + met[mo + 7] * hy + met[mo + 8] * hz;
        }
        thread::sync_threads();
        let i = m % n1;
        let j = (m / n1) % n1;
        let k = m / n2;
        let mut acc = 0.0f64;
        let mut a = 0usize;
        while a < n1 {
            unsafe {
                acc += DS[a * n1 + i] * P[a + j * n1 + k * n2];
                acc += DS[a * n1 + j] * P[NN_MAX + i + a * n1 + k * n2];
                acc += DS[a * n1 + k] * P[2 * NN_MAX + i + j * n1 + a * n2];
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
    let cfg = LaunchConfig { grid_dim: (ma.ne as u32, 1, 1), block_dim: (ma.nn as u32, 1, 1), shared_mem_bytes: 0 };
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
    let cfg = LaunchConfig { grid_dim: (ma.ne as u32, 1, 1), block_dim: (ma.nn as u32, 1, 1), shared_mem_bytes: 0 };
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
