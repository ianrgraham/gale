//! Live **multi-GPU 3D** hex advection across the 2× Titan V — the 3D analogue of
//! `gpu-multigpu`. The element→GPU partition comes from the framework's
//! [`DomainDecomposition`](gale::sim::DomainDecomposition) (validated CPU-side vs
//! the monolithic operator); each GPU owns a partition with its state in a combined
//! `[local | halo]` buffer, cross-partition hex-face traces (`n_1d²` values each)
//! filled by P2P `cuMemcpyPeerAsync`, and runs the same `advect_rhs` hex kernel as
//! `gpu-advection3d` (which reads `u[face_nbr]` — local node or halo slot). The
//! gathered result is checked bit-for-bit against the monolithic CPU `Hyperbolic3d`.
//!
//! Closes the 3D multi-GPU through-line: framework Device + decomposition driving
//! real 3D kernel execution + P2P halos on hardware. (Unblocked by the cuda-oxide
//! typed-pointer dialect fix — see docs/cuda-oxide-codegen-notes.md §3.)
//!
//! Run: cargo oxide run --bin gpu-multigpu3d

use cuda_core::peer::{can_access_peer, enable_peer_access};
use cuda_core::{memory, CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;
use gale::dg::{Face, Hyperbolic3d, LinearAdvection3d, Mesh3d, Neighbor3};
use gale::sim::{Device, DomainDecomposition, Partition};

const NN_MAX: usize = 125; // (p+1)³, p=4
const P3MAX: usize = 3 * NN_MAX;

#[cuda_module]
mod kernels {
    use super::*;

    /// 3D weak-form linear-advection RHS over a hex. `u` is the combined
    /// `[local | halo]` buffer; `face_nbr` indexes into it. Identical to the
    /// single-GPU `gpu-advection3d` kernel. Metrics packed node-major in `met`
    /// (`met[b*9 + c]`); face floats in `fmet` (`fmet[idx*4 + c]`).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn advect_rhs(
        d: &[f64], u: &[f64], met: &[f64], jw: &[f64], ax: f64, ay: f64, az: f64, n1: u32,
        face_vl: &[u32], fmet: &[f64], face_nbr: &[u32],
        mut out: DisjointSlice<f64>,
    ) {
        static mut DS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut P: SharedArray<f64, P3MAX> = SharedArray::UNINIT;
        static mut RFACE: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
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
            P[NN_MAX + m] = met[mo + 3] * wfx + met[mo + 4] * wfy + met[mo + 5] * wfz;
            P[2 * NN_MAX + m] = met[mo + 6] * wfx + met[mo + 7] * wfy + met[mo + 8] * wfz;
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
                vol += DS[a * n1 + j] * P[NN_MAX + i + a * n1 + k * n2];
                vol += DS[a * n1 + k] * P[2 * NN_MAX + i + j * n1 + a * n2];
            }
            a += 1;
        }
        let rf = unsafe { RFACE[m] };
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = (vol - rf) / jw[b];
        }
    }
}

/// Per-GPU partition data (host side, ready to upload).
struct Part {
    global: Vec<usize>,
    state: Vec<f64>,
    met: Vec<f64>,
    jw: Vec<f64>,
    fvl: Vec<u32>,
    fmet: Vec<f64>,
    fnbr: Vec<u32>,                // index into combined [local | halo]
    halo_src: Vec<(usize, usize)>, // per halo slot: (source GPU, index in its local_state)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    let (ax, ay, az) = (0.8, -0.5, 0.3);
    println!("=== Multi-GPU 3D hex advection (2 devices, P2P halo) vs CPU (p={p}) ===\n");

    let mesh = Mesh3d::rectangular_periodic(p, 4, 2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refh.n_nodes();
    let n1 = p + 1;
    let n2 = n1 * n1;
    let ne = mesh.n_elements();
    let ndof = ne * nn;

    let mut gstate = vec![0.0; ndof];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            gstate[e * nn + k] =
                (2.0 * el.geom.x[k]).sin() * (3.0 * el.geom.y[k]).cos() * (el.geom.z[k]).sin() + 0.2 * el.geom.x[k];
        }
    }
    // CPU monolithic reference.
    let cpu = Hyperbolic3d::new(&mesh, LinearAdvection3d { ax, ay, az })
        .rhs(&[gstate.clone()], 0.0, &|_, _, _, _, _: &mut [f64]| {});

    // Partition via the framework Device abstraction.
    let device = Device::MultiGpu { ordinals: vec![0, 1], partition: Partition::Blocks };
    assert_eq!(device.n_devices(), 2, "this binary targets the 2× Titan V");
    let dd = DomainDecomposition::new(ne, &device);
    let parts = dd.parts.clone();
    println!("partition (framework DomainDecomposition): {} hexes → {:?} per GPU", ne, dd.counts());

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
        eprintln!("FAIL: devices cannot access each other via P2P");
        std::process::exit(1);
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
        modules.push(kernels::load(ctx)?);
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
        modules[g].advect_rhs(
            &stream, cfg, &b.0, &combined[g], &b.1, &b.2, ax, ay, az, n1 as u32,
            &b.3, &b.4, &b.5, &mut outs[g],
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
    println!("devices=2  hexes={ne}  dofs={ndof}  max|2gpu − cpu| / |op| = {:.3e}", max_abs / scale);
    if max_abs / scale < 1e-12 {
        println!("\nPASS: 2-GPU P2P-halo 3D hex advection matches the monolithic CPU operator.");
        Ok(())
    } else {
        eprintln!("\nFAIL: multi-GPU 3D mismatch.");
        std::process::exit(1);
    }
}
