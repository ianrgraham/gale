//! Distributed **multi-GPU** advection — reusable library components.
//!
//! These host wrappers run scalar linear advection split across *multiple* devices
//! (the 2× Titan V), driven by gale's framework [`Device`](gale::sim::Device) /
//! [`DomainDecomposition`](gale::sim::DomainDecomposition) abstraction. The
//! element→GPU partition comes from the decomposition (the same one validated
//! CPU-side against the monolithic operator); each GPU owns a partition whose state
//! lives in a combined `[local | halo]` buffer. Cross-partition neighbour traces are
//! filled into the halo region by P2P `cuMemcpyPeerAsync` (peer access enabled), so
//! the *same* advection kernel — which reads `u[face_nbr]` — runs unchanged:
//! `face_nbr` simply points into local state or the halo.
//!
//! Two flavours are provided: [`multigpu_advection_2d`] (quad mesh) and
//! [`multigpu_advection_3d`] (hex mesh). Each computes the weak-form RHS across the
//! devices and gathers it back into global element order, bit-for-bit equal to the
//! monolithic CPU operator (`gale::dg::Hyperbolic` / `Hyperbolic3d`).
//!
//! Device *module* code needs the cuda-oxide backend, but these wrappers live in the
//! normally-built lib because the embedded artifact is anchored across the rlib
//! boundary (see `docs/cuda-oxide-codegen-notes.md` §3). The two `#[cuda_module]`s
//! here ([`kernels2d`], [`kernels3d`]) carry distinct module names and distinct
//! kernel export names (`advect2d_mg_rhs`, `advect3d_mg_rhs`) so they coexist in the
//! single crate-wide device bundle alongside the single-GPU advection kernels.

use cuda_core::peer::{can_access_peer, enable_peer_access};
use cuda_core::{memory, CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;
use gale::dg::{Edge, Face, Mesh2d, Mesh3d, Neighbor, Neighbor3};
use gale::sim::{Device, DomainDecomposition, Partition};

const NN_MAX_2D: usize = 81; // (p+1)² up to p=8
const NN_MAX_3D: usize = 125; // (p+1)³, p=4
const P3MAX_3D: usize = 3 * NN_MAX_3D;

#[cuda_module]
mod kernels2d {
    use super::*;

    /// Weak-form linear advection RHS (2D quad). `u` is the combined `[local | halo]`
    /// buffer; `face_nbr` indexes into it (local node or halo slot). The device body
    /// is identical to the single-GPU `advect2d_rhs`; only the export name differs.
    ///
    /// Named `advect2d_mg_rhs` (not `advect_rhs`) because kernel export names share a
    /// single crate-wide device bundle in cuda-oxide, so they must be unique across
    /// gale-gpu (cf. `advect2d_rhs`, `advect3d_rhs`, and the 3D multi-GPU kernel).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn advect2d_mg_rhs(
        d: &[f64], u: &[f64], rx: &[f64], ry: &[f64], sx: &[f64], sy: &[f64], jw: &[f64],
        ax: f64, ay: f64, n1: u32,
        face_vl: &[u32], face_nx: &[f64], face_ny: &[f64], face_sw: &[f64], face_nbr: &[u32],
        mut out: DisjointSlice<f64>,
    ) {
        static mut DS: SharedArray<f64, NN_MAX_2D> = SharedArray::UNINIT;
        static mut PR: SharedArray<f64, NN_MAX_2D> = SharedArray::UNINIT;
        static mut PS: SharedArray<f64, NN_MAX_2D> = SharedArray::UNINIT;
        static mut RFACE: SharedArray<f64, NN_MAX_2D> = SharedArray::UNINIT;
        let n1 = n1 as usize;
        let nn = n1 * n1;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let b = e * nn + m;
        unsafe {
            DS[m] = d[m];
            RFACE[m] = 0.0;
            let um = u[b];
            let wfx = jw[b] * ax * um;
            let wfy = jw[b] * ay * um;
            PR[m] = rx[b] * wfx + ry[b] * wfy;
            PS[m] = sx[b] * wfx + sy[b] * wfy;
        }
        thread::sync_threads();
        if m == 0 {
            let mut t = 0usize;
            while t < 4 {
                let mut a = 0usize;
                while a < n1 {
                    let idx = (e * 4 + t) * n1 + a;
                    let vl = face_vl[idx] as usize;
                    let nx = face_nx[idx];
                    let ny = face_ny[idx];
                    let sw = face_sw[idx];
                    let um = u[e * nn + vl];
                    let up = u[face_nbr[idx] as usize];
                    let an = ax * nx + ay * ny;
                    let aan = if an < 0.0 { -an } else { an };
                    let fstar = 0.5 * an * (um + up) - 0.5 * aan * (up - um);
                    unsafe {
                        RFACE[vl] += sw * fstar;
                    }
                    a += 1;
                }
                t += 1;
            }
        }
        thread::sync_threads();
        let i = m % n1;
        let j = m / n1;
        let mut vol = 0.0f64;
        let mut k = 0usize;
        while k < n1 {
            unsafe {
                vol += DS[k * n1 + i] * PR[k + j * n1] + DS[k * n1 + j] * PS[i + k * n1];
            }
            k += 1;
        }
        let rf = unsafe { RFACE[m] };
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = (vol - rf) / jw[b];
        }
    }
}

#[cuda_module]
mod kernels3d {
    use super::*;

    /// 3D weak-form linear-advection RHS over a hex. `u` is the combined
    /// `[local | halo]` buffer; `face_nbr` indexes into it. The device body is
    /// identical to the single-GPU `advect3d_rhs`; only the export name differs.
    /// Metrics packed node-major in `met` (`met[b*9 + c]`); face floats in `fmet`
    /// (`fmet[idx*4 + c]`).
    ///
    /// Named `advect3d_mg_rhs` (not `advect_rhs`) because kernel export names share a
    /// single crate-wide device bundle in cuda-oxide, so they must be unique across
    /// gale-gpu (cf. `advect3d_rhs` and the 2D multi-GPU kernel).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn advect3d_mg_rhs(
        d: &[f64], u: &[f64], met: &[f64], jw: &[f64], ax: f64, ay: f64, az: f64, n1: u32,
        face_vl: &[u32], fmet: &[f64], face_nbr: &[u32],
        mut out: DisjointSlice<f64>,
    ) {
        static mut DS: SharedArray<f64, NN_MAX_3D> = SharedArray::UNINIT;
        static mut P: SharedArray<f64, P3MAX_3D> = SharedArray::UNINIT;
        static mut RFACE: SharedArray<f64, NN_MAX_3D> = SharedArray::UNINIT;
        let n1 = n1 as usize;
        let n2 = n1 * n1;
        let nn = n1 * n2;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let b = e * nn + m;
        unsafe {
            DS[m] = d[m];
            RFACE[m] = 0.0;
            let um = u[b];
            let wfx = jw[b] * ax * um;
            let wfy = jw[b] * ay * um;
            let wfz = jw[b] * az * um;
            let mo = b * 9;
            P[m] = met[mo] * wfx + met[mo + 1] * wfy + met[mo + 2] * wfz;
            P[NN_MAX_3D + m] = met[mo + 3] * wfx + met[mo + 4] * wfy + met[mo + 5] * wfz;
            P[2 * NN_MAX_3D + m] = met[mo + 6] * wfx + met[mo + 7] * wfy + met[mo + 8] * wfz;
        }
        thread::sync_threads();
        if m == 0 {
            let mut t = 0usize;
            while t < 6 {
                let mut a = 0usize;
                while a < n2 {
                    let idx = (e * 6 + t) * n2 + a;
                    let vl = face_vl[idx] as usize;
                    let fo = idx * 4;
                    let nx = fmet[fo];
                    let ny = fmet[fo + 1];
                    let nz = fmet[fo + 2];
                    let sw = fmet[fo + 3];
                    let um = u[e * nn + vl];
                    let up = u[face_nbr[idx] as usize];
                    let an = ax * nx + ay * ny + az * nz;
                    let fstar = 0.5 * an * (um + up) - 0.5 * an.abs() * (up - um);
                    unsafe {
                        RFACE[vl] += sw * fstar;
                    }
                    a += 1;
                }
                t += 1;
            }
        }
        thread::sync_threads();
        let i = m % n1;
        let j = (m / n1) % n1;
        let k = m / n2;
        let mut vol = 0.0f64;
        let mut a = 0usize;
        while a < n1 {
            unsafe {
                vol += DS[a * n1 + i] * P[a + j * n1 + k * n2];
                vol += DS[a * n1 + j] * P[NN_MAX_3D + i + a * n1 + k * n2];
                vol += DS[a * n1 + k] * P[2 * NN_MAX_3D + i + j * n1 + a * n2];
            }
            a += 1;
        }
        let rf = unsafe { RFACE[m] };
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = (vol - rf) / jw[b];
        }
    }
}

/// Per-GPU 2D partition data (host side, ready to upload).
struct Part2d {
    global: Vec<usize>, // local elem → global elem
    state: Vec<f64>,    // local_state, length n_local*nn
    rx: Vec<f64>,
    ry: Vec<f64>,
    sx: Vec<f64>,
    sy: Vec<f64>,
    jw: Vec<f64>,
    fvl: Vec<u32>,
    fnx: Vec<f64>,
    fny: Vec<f64>,
    fsw: Vec<f64>,
    fnbr: Vec<u32>,                // index into combined [local | halo]
    halo_src: Vec<(usize, usize)>, // per halo slot: (source GPU, index in its local_state)
}

/// Per-GPU 3D partition data (host side, ready to upload).
struct Part3d {
    global: Vec<usize>,
    state: Vec<f64>,
    met: Vec<f64>,
    jw: Vec<f64>,
    fvl: Vec<u32>,
    fmet: Vec<f64>,
    fnbr: Vec<u32>,                // index into combined [local | halo]
    halo_src: Vec<(usize, usize)>, // per halo slot: (source GPU, index in its local_state)
}

/// Compute the scalar linear-advection weak-form RHS for a **periodic** quad mesh
/// distributed across **2 GPUs** with P2P halo exchange, returning the gathered
/// nodal time-derivative in global element order.
///
/// The element→GPU ownership comes from gale's framework
/// [`DomainDecomposition`](gale::sim::DomainDecomposition); each GPU holds its
/// partition's state in a combined `[local | halo]` buffer, cross-partition
/// neighbour traces are copied into the halo via `cuMemcpyPeerAsync` (peer access
/// enabled bidirectionally), and the [`kernels2d::advect2d_mg_rhs`] kernel runs per
/// device. The gathered result is bit-for-bit equal to the monolithic CPU operator
/// `gale::dg::Hyperbolic`.
///
/// Requires a 2× GPU machine with mutual P2P access; panics on a non-periodic mesh.
pub fn multigpu_advection_2d(
    mesh: &Mesh2d,
    gstate: &[f64],
    ax: f64,
    ay: f64,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let nn = mesh.refq.n_nodes();
    let n1 = mesh.order + 1;
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    assert_eq!(gstate.len(), ndof, "state length must be n_elements·n_nodes");
    assert!(nn <= NN_MAX_2D, "p too large for NN_MAX_2D={NN_MAX_2D}");

    // Partition via the framework Device abstraction. The DomainDecomposition is the
    // single source of element→GPU ownership (validated CPU-side against monolithic).
    let device = Device::MultiGpu { ordinals: vec![0, 1], partition: Partition::Blocks };
    let n_gpu = device.n_devices();
    assert_eq!(n_gpu, 2, "this routine targets the 2× Titan V; device declares {n_gpu}");
    let dd = DomainDecomposition::new(ne, &device);
    println!(
        "partition (framework DomainDecomposition): {} elements → {:?} per GPU",
        ne,
        dd.counts()
    );
    let parts: Vec<usize> = dd.parts.clone();
    let mut local_of = vec![0usize; ne];
    let mut global_of: [Vec<usize>; 2] = [vec![], vec![]];
    for e in 0..ne {
        local_of[e] = global_of[parts[e]].len();
        global_of[parts[e]].push(e);
    }

    let build = |g: usize| -> Part2d {
        let locals = &global_of[g];
        let nl = locals.len();
        let nldof = nl * nn;
        let mut part = Part2d {
            global: locals.clone(),
            state: vec![0.0; nldof],
            rx: vec![0.0; nldof], ry: vec![0.0; nldof], sx: vec![0.0; nldof], sy: vec![0.0; nldof], jw: vec![0.0; nldof],
            fvl: vec![0; nl * 4 * n1], fnx: vec![0.0; nl * 4 * n1], fny: vec![0.0; nl * 4 * n1],
            fsw: vec![0.0; nl * 4 * n1], fnbr: vec![0; nl * 4 * n1],
            halo_src: vec![],
        };
        for (le, &e) in locals.iter().enumerate() {
            let el = &mesh.elements[e];
            for k in 0..nn {
                part.state[le * nn + k] = gstate[e * nn + k];
                part.rx[le * nn + k] = el.geom.rx[k];
                part.ry[le * nn + k] = el.geom.ry[k];
                part.sx[le * nn + k] = el.geom.sx[k];
                part.sy[le * nn + k] = el.geom.sy[k];
                part.jw[le * nn + k] = el.geom.jw[k];
            }
            for (t, edge) in Edge::ALL.iter().enumerate() {
                let face = &el.faces[*edge as usize];
                let Neighbor::Interior { elem: re, edge: redge, perm } = &el.neighbors[*edge as usize] else {
                    panic!("periodic mesh: no boundary");
                };
                let rf = &mesh.elements[*re].faces[*redge as usize];
                for a in 0..n1 {
                    let idx = (le * 4 + t) * n1 + a;
                    part.fvl[idx] = face.nodes[a] as u32;
                    part.fnx[idx] = face.nx[a];
                    part.fny[idx] = face.ny[a];
                    part.fsw[idx] = face.sw[a];
                    let rnode = rf.nodes[perm[a]];
                    if parts[*re] == g {
                        part.fnbr[idx] = (local_of[*re] * nn + rnode) as u32; // local
                    } else {
                        let slot = part.halo_src.len();
                        part.halo_src.push((parts[*re], local_of[*re] * nn + rnode));
                        part.fnbr[idx] = (nldof + slot) as u32; // halo region
                    }
                }
            }
        }
        part
    };
    let parts_data = [build(0), build(1)];

    // Two device contexts + bidirectional peer access.
    let ctx0 = CudaContext::new(0)?;
    let ctx1 = CudaContext::new(1)?;
    if !can_access_peer(&ctx0, &ctx1)? || !can_access_peer(&ctx1, &ctx0)? {
        return Err("devices cannot access each other via P2P".into());
    }
    // cuCtxEnablePeerAccess operates on the *current* context, so bind `from` first.
    ctx0.bind_to_thread()?;
    enable_peer_access(&ctx0, &ctx1)?;
    ctx1.bind_to_thread()?;
    enable_peer_access(&ctx1, &ctx0)?;
    let ctxs = [&ctx0, &ctx1];

    // Per-GPU device buffers (combined u = [local | halo]).
    let mut combined = Vec::new();
    let mut bufs = Vec::new();
    let mut outs = Vec::new();
    let mut modules = Vec::new();
    for g in 0..2 {
        let ctx = ctxs[g];
        ctx.bind_to_thread()?;
        let stream = ctx.default_stream();
        let pd = &parts_data[g];
        let nl = pd.global.len();
        let nldof = nl * nn;
        let mut u = vec![0.0; nldof + pd.halo_src.len()];
        u[..nldof].copy_from_slice(&pd.state);
        let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
        let upu = |v: &[u32]| DeviceBuffer::from_host(&stream, v);
        let b = (
            up(&mesh.refq.line.diff)?, up(&pd.rx)?, up(&pd.ry)?, up(&pd.sx)?, up(&pd.sy)?, up(&pd.jw)?,
            upu(&pd.fvl)?, up(&pd.fnx)?, up(&pd.fny)?, up(&pd.fsw)?, upu(&pd.fnbr)?,
        );
        combined.push(DeviceBuffer::from_host(&stream, &u)?);
        outs.push(DeviceBuffer::<f64>::zeroed(&stream, nldof)?);
        modules.push(kernels2d::load(ctx)?);
        bufs.push(b);
    }

    // Halo exchange via P2P memcpy_dtod (peer access enabled). For each GPU, fill its
    // halo region from the source GPU's combined buffer.
    for g in 0..2 {
        let ctx = ctxs[g];
        ctx.bind_to_thread()?;
        let stream = ctx.default_stream();
        let nldof = parts_data[g].global.len() * nn;
        let sz = std::mem::size_of::<f64>();
        for (slot, &(src_g, src_idx)) in parts_data[g].halo_src.iter().enumerate() {
            let dst = combined[g].cu_deviceptr() + ((nldof + slot) * sz) as u64;
            let src = combined[src_g].cu_deviceptr() + (src_idx * sz) as u64;
            // Cross-device P2P copy (NVLink/PCIe peer path).
            unsafe {
                memory::memcpy_peer_async(dst, ctx.cu_ctx(), src, ctxs[src_g].cu_ctx(), sz, stream.cu_stream())?
            };
        }
        stream.synchronize()?;
    }

    // Launch the kernel on each GPU, gather results back to global order.
    let mut got = vec![0.0; ndof];
    for g in 0..2 {
        let ctx = ctxs[g];
        ctx.bind_to_thread()?;
        let stream = ctx.default_stream();
        let pd = &parts_data[g];
        let nl = pd.global.len();
        let b = &bufs[g];
        let cfg = LaunchConfig { grid_dim: (nl as u32, 1, 1), block_dim: (nn as u32, 1, 1), shared_mem_bytes: 0 };
        modules[g].advect2d_mg_rhs(
            &stream, cfg, &b.0, &combined[g], &b.1, &b.2, &b.3, &b.4, &b.5, ax, ay, n1 as u32,
            &b.6, &b.7, &b.8, &b.9, &b.10, &mut outs[g],
        )?;
        let local_out = outs[g].to_host_vec(&stream)?;
        for (le, &e) in pd.global.iter().enumerate() {
            got[e * nn..(e + 1) * nn].copy_from_slice(&local_out[le * nn..(le + 1) * nn]);
        }
    }
    Ok(got)
}

/// Compute the scalar linear-advection weak-form RHS for a **periodic** hex mesh
/// distributed across **2 GPUs** with P2P halo exchange, returning the gathered
/// nodal time-derivative in global element order — the 3D analogue of
/// [`multigpu_advection_2d`].
///
/// Element→GPU ownership comes from gale's framework
/// [`DomainDecomposition`](gale::sim::DomainDecomposition); each GPU holds its
/// partition's state in a combined `[local | halo]` buffer, cross-partition hex-face
/// traces (`n_1d²` values per face) are copied into the halo via `cuMemcpyPeerAsync`
/// (peer access enabled bidirectionally), and the [`kernels3d::advect3d_mg_rhs`]
/// kernel runs per device. The gathered result is bit-for-bit equal to the
/// monolithic CPU operator `gale::dg::Hyperbolic3d`.
///
/// Requires a 2× GPU machine with mutual P2P access; panics on a non-periodic mesh.
pub fn multigpu_advection_3d(
    mesh: &Mesh3d,
    gstate: &[f64],
    ax: f64,
    ay: f64,
    az: f64,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let nn = mesh.refh.n_nodes();
    let n1 = mesh.order + 1;
    let n2 = n1 * n1;
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    assert_eq!(gstate.len(), ndof, "state length must be n_elements·n_nodes");
    assert!(nn <= NN_MAX_3D, "p too large for NN_MAX_3D={NN_MAX_3D}");

    // Partition via the framework Device abstraction.
    let device = Device::MultiGpu { ordinals: vec![0, 1], partition: Partition::Blocks };
    assert_eq!(device.n_devices(), 2, "this routine targets the 2× Titan V");
    let dd = DomainDecomposition::new(ne, &device);
    let parts = dd.parts.clone();
    println!("partition (framework DomainDecomposition): {} hexes → {:?} per GPU", ne, dd.counts());

    let mut local_of = vec![0usize; ne];
    let mut global_of: [Vec<usize>; 2] = [vec![], vec![]];
    for e in 0..ne {
        local_of[e] = global_of[parts[e]].len();
        global_of[parts[e]].push(e);
    }

    let build = |g: usize| -> Part3d {
        let locals = &global_of[g];
        let nl = locals.len();
        let nldof = nl * nn;
        let mut part = Part3d {
            global: locals.clone(),
            state: vec![0.0; nldof],
            met: vec![0.0; nldof * 9],
            jw: vec![0.0; nldof],
            fvl: vec![0; nl * 6 * n2],
            fmet: vec![0.0; nl * 6 * n2 * 4],
            fnbr: vec![0; nl * 6 * n2],
            halo_src: vec![],
        };
        for (le, &e) in locals.iter().enumerate() {
            let el = &mesh.elements[e];
            for k in 0..nn {
                let b = le * nn + k;
                part.state[b] = gstate[e * nn + k];
                let d = &el.geom;
                part.met[b * 9] = d.rx[k];
                part.met[b * 9 + 1] = d.ry[k];
                part.met[b * 9 + 2] = d.rz[k];
                part.met[b * 9 + 3] = d.sx[k];
                part.met[b * 9 + 4] = d.sy[k];
                part.met[b * 9 + 5] = d.sz[k];
                part.met[b * 9 + 6] = d.tx[k];
                part.met[b * 9 + 7] = d.ty[k];
                part.met[b * 9 + 8] = d.tz[k];
                part.jw[b] = d.jw[k];
            }
            for (t, face) in Face::ALL.iter().enumerate() {
                let fc = &el.faces[*face as usize];
                let Neighbor3::Interior { elem: re, face: rface, perm } = &el.neighbors[*face as usize] else {
                    panic!("periodic mesh: no boundary");
                };
                let rf = &mesh.elements[*re].faces[*rface as usize];
                for a in 0..n2 {
                    let idx = (le * 6 + t) * n2 + a;
                    part.fvl[idx] = fc.nodes[a] as u32;
                    part.fmet[idx * 4] = fc.nx[a];
                    part.fmet[idx * 4 + 1] = fc.ny[a];
                    part.fmet[idx * 4 + 2] = fc.nz[a];
                    part.fmet[idx * 4 + 3] = fc.sw[a];
                    let rnode = rf.nodes[perm[a]];
                    if parts[*re] == g {
                        part.fnbr[idx] = (local_of[*re] * nn + rnode) as u32; // local
                    } else {
                        let slot = part.halo_src.len();
                        part.halo_src.push((parts[*re], local_of[*re] * nn + rnode));
                        part.fnbr[idx] = (nldof + slot) as u32; // halo region
                    }
                }
            }
        }
        part
    };
    let parts_data = [build(0), build(1)];

    // Two device contexts + bidirectional peer access.
    let ctx0 = CudaContext::new(0)?;
    let ctx1 = CudaContext::new(1)?;
    if !can_access_peer(&ctx0, &ctx1)? || !can_access_peer(&ctx1, &ctx0)? {
        return Err("devices cannot access each other via P2P".into());
    }
    ctx0.bind_to_thread()?;
    enable_peer_access(&ctx0, &ctx1)?;
    ctx1.bind_to_thread()?;
    enable_peer_access(&ctx1, &ctx0)?;
    let ctxs = [&ctx0, &ctx1];

    // Per-GPU device buffers (combined u = [local | halo]).
    let mut combined = Vec::new();
    let mut bufs = Vec::new();
    let mut outs = Vec::new();
    let mut modules = Vec::new();
    for g in 0..2 {
        let ctx = ctxs[g];
        ctx.bind_to_thread()?;
        let stream = ctx.default_stream();
        let pd = &parts_data[g];
        let nldof = pd.global.len() * nn;
        let mut u = vec![0.0; nldof + pd.halo_src.len()];
        u[..nldof].copy_from_slice(&pd.state);
        let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
        let upu = |v: &[u32]| DeviceBuffer::from_host(&stream, v);
        let b = (up(&mesh.refh.line.diff)?, up(&pd.met)?, up(&pd.jw)?, upu(&pd.fvl)?, up(&pd.fmet)?, upu(&pd.fnbr)?);
        combined.push(DeviceBuffer::from_host(&stream, &u)?);
        outs.push(DeviceBuffer::<f64>::zeroed(&stream, nldof)?);
        modules.push(kernels3d::load(ctx)?);
        bufs.push(b);
    }

    // P2P halo exchange: fill each GPU's halo region from the source GPU's buffer.
    for g in 0..2 {
        let ctx = ctxs[g];
        ctx.bind_to_thread()?;
        let stream = ctx.default_stream();
        let nldof = parts_data[g].global.len() * nn;
        let sz = std::mem::size_of::<f64>();
        for (slot, &(src_g, src_idx)) in parts_data[g].halo_src.iter().enumerate() {
            let dst = combined[g].cu_deviceptr() + ((nldof + slot) * sz) as u64;
            let src = combined[src_g].cu_deviceptr() + (src_idx * sz) as u64;
            unsafe {
                memory::memcpy_peer_async(dst, ctx.cu_ctx(), src, ctxs[src_g].cu_ctx(), sz, stream.cu_stream())?
            };
        }
        stream.synchronize()?;
    }

    // Launch on each GPU, gather to global order.
    let mut got = vec![0.0; ndof];
    for g in 0..2 {
        let ctx = ctxs[g];
        ctx.bind_to_thread()?;
        let stream = ctx.default_stream();
        let pd = &parts_data[g];
        let nl = pd.global.len();
        let b = &bufs[g];
        let cfg = LaunchConfig { grid_dim: (nl as u32, 1, 1), block_dim: (nn as u32, 1, 1), shared_mem_bytes: 0 };
        modules[g].advect3d_mg_rhs(
            &stream, cfg, &b.0, &combined[g], &b.1, &b.2, ax, ay, az, n1 as u32,
            &b.3, &b.4, &b.5, &mut outs[g],
        )?;
        let local_out = outs[g].to_host_vec(&stream)?;
        for (le, &e) in pd.global.iter().enumerate() {
            got[e * nn..(e + 1) * nn].copy_from_slice(&local_out[le * nn..(le + 1) * nn]);
        }
    }
    Ok(got)
}
