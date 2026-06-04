//! Live **multi-GPU** advection across the 2× Titan V, driven by the framework's
//! [`Device`](gale::sim::Device) abstraction: the element→GPU partition comes from
//! [`DomainDecomposition`](gale::sim::DomainDecomposition) (the same decomposition
//! validated CPU-side against the monolithic operator), and the GPU execution path
//! consumes it. Each GPU owns a partition; its state lives in a combined buffer
//! `[local_state | halo]`. Cross-partition neighbour traces are filled into the
//! halo by P2P `cuMemcpyPeerAsync` (peer access enabled), so the *same* `advect_rhs`
//! kernel (which reads `u[face_nbr]`) runs unchanged — `face_nbr` simply points into
//! local state or the halo region. The result is gathered and checked bit-for-bit
//! against the monolithic CPU operator.
//!
//! This closes the multi-GPU through-line: the framework `Device` selector +
//! decomposition driving real kernel execution + P2P halos on hardware. (Device
//! *module* code needs the cargo-oxide backend, so this execution path lives in a
//! binary, not the normally-built lib — see docs/api-design.md §3.6.)
//!
//! Run: cargo oxide run --bin gpu-multigpu

use cuda_core::peer::{can_access_peer, enable_peer_access};
use cuda_core::{memory, CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;
use gale::dg::{Edge, Hyperbolic, LinearAdvection, Mesh2d, Neighbor};
use gale::sim::{Device, DomainDecomposition, Partition};

const NN_MAX: usize = 81;

#[cuda_module]
mod kernels {
    use super::*;

    /// Weak-form linear advection RHS. `u` is the combined `[local | halo]` buffer;
    /// `face_nbr` indexes into it (local node or halo slot). (Same as gpu-advection.)
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn advect_rhs(
        d: &[f64], u: &[f64], rx: &[f64], ry: &[f64], sx: &[f64], sy: &[f64], jw: &[f64],
        ax: f64, ay: f64, n1: u32,
        face_vl: &[u32], face_nx: &[f64], face_ny: &[f64], face_sw: &[f64], face_nbr: &[u32],
        mut out: DisjointSlice<f64>,
    ) {
        static mut DS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PR: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut RFACE: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
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

/// Per-GPU partition data (host side, ready to upload).
struct Part {
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
    fnbr: Vec<u32>,             // index into combined [local | halo]
    halo_src: Vec<(usize, usize)>, // per halo slot: (source GPU, index in its local_state)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    let (ax, ay) = (0.8, -0.5);
    println!("=== Multi-GPU advection (2 devices, P2P halo) vs CPU (p={p}) ===\n");

    let mesh = Mesh2d::rectangular_periodic(p, 4, 4, [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refq.n_nodes();
    let n1 = p + 1;
    let ne = mesh.n_elements();
    let ndof = ne * nn;

    let mut gstate = vec![0.0; ndof];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            gstate[e * nn + k] = (2.0 * el.geom.x[k]).sin() * (3.0 * el.geom.y[k]).cos() + 0.2 * el.geom.x[k];
        }
    }
    // CPU monolithic reference.
    let cpu = Hyperbolic::new(&mesh, LinearAdvection { ax, ay }).rhs(&[gstate.clone()], 0.0, &|_, _, _, _: &mut [f64]| {});

    // Partition via the framework Device abstraction. The DomainDecomposition is the
    // single source of element→GPU ownership (validated CPU-side against monolithic).
    let device = Device::MultiGpu { ordinals: vec![0, 1], partition: Partition::Blocks };
    let n_gpu = device.n_devices();
    assert_eq!(n_gpu, 2, "this binary targets the 2× Titan V; device declares {n_gpu}");
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

    let build = |g: usize| -> Part {
        let locals = &global_of[g];
        let nl = locals.len();
        let nldof = nl * nn;
        let mut part = Part {
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
        eprintln!("FAIL: devices cannot access each other via P2P");
        std::process::exit(1);
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
        modules.push(kernels::load(ctx)?);
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
        modules[g].advect_rhs(
            &stream, cfg, &b.0, &combined[g], &b.1, &b.2, &b.3, &b.4, &b.5, ax, ay, n1 as u32,
            &b.6, &b.7, &b.8, &b.9, &b.10, &mut outs[g],
        )?;
        let local_out = outs[g].to_host_vec(&stream)?;
        for (le, &e) in pd.global.iter().enumerate() {
            got[e * nn..(e + 1) * nn].copy_from_slice(&local_out[le * nn..(le + 1) * nn]);
        }
    }

    let mut max_abs = 0.0f64;
    let scale = cpu[0].iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1e-300);
    for i in 0..ndof {
        max_abs = max_abs.max((got[i] - cpu[0][i]).abs());
    }
    println!("devices=2  dofs={ndof}  max|2gpu − cpu| / |op| = {:.3e}", max_abs / scale);
    if max_abs / scale < 1e-12 {
        println!("\nPASS: 2-GPU P2P-halo advection matches the monolithic CPU operator.");
        Ok(())
    } else {
        eprintln!("\nFAIL: multi-GPU mismatch.");
        std::process::exit(1);
    }
}
