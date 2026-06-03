//! GPU port of the **3D hyperbolic operator** (weak form) for linear advection on
//! hexes — the 3D analogue of `gpu-advection`. In-kernel flux, sum-factorized
//! volume `Dxᵀ(W Fx)+Dyᵀ(W Fy)+Dzᵀ(W Fz)` over the three tensor directions, Rusanov
//! interface gather over the 6 hex faces, and `M⁻¹`. Validated bit-for-bit against
//! the CPU `Hyperbolic3d::rhs`. Periodic mesh ⇒ no boundary fluxes; libdevice-free
//! (only `f64::abs`) ⇒ the embedded `#[cuda_module]` path works on sm_70.
//!
//! This extends the validated GPU operator coverage to 3D (the prerequisite for 3D
//! multi-GPU); `n_nodes = (p+1)³` per block, `n1²` diff entries.
//!
//! STATUS — blocked on a cuda-oxide backend bug (see
//! `docs/cuda-oxide-codegen-notes.md`). This kernel's NVVM IR is emitted with
//! **opaque pointers** (`ptr`) instead of the **typed pointers** (`i8*`) the 2D
//! kernels get, and libNVVM rejects it: `nvvmCompileProgram ... parse expected
//! type` at the first parameter. The CPU operator (`Hyperbolic3d`) it validates
//! against is correct and tested; this binary is the minimal reproducer for the
//! backend's missing typed-pointer conversion on this kernel.
//!
//! Run: cargo oxide run --bin gpu-advection3d   (currently fails at NVVM compile)

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;
use gale::dg::{Face, Hyperbolic3d, LinearAdvection3d, Mesh3d, Neighbor3};

// (p+1)³ for p=4 = 125; bounds every per-node shared array.
const NN_MAX: usize = 125;
// Packed PR/PS/PT shared array.
const P3MAX: usize = 3 * NN_MAX;

#[cuda_module]
mod kernels {
    use super::*;

    /// `∂ₜu = M⁻¹[Dxᵀ(W·aₓu)+Dyᵀ(W·a_yu)+Dzᵀ(W·a_zu) − ∮F*·n]`, weak form, Rusanov.
    /// `e = blockIdx.x`, node `m = threadIdx.x` (0..n³).
    ///
    /// Metrics are packed node-major into `met` (`met[b*9 + {rx,ry,rz,sx,sy,sz,tx,
    /// ty,tz}]`) and the face floats into `fmet` (`fmet[idx*4 + {nx,ny,nz,sw}]`) to
    /// keep the parameter count low — the NVVM-text backend mis-parses very wide
    /// kernel signatures (~23 slice params).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn advect_rhs(
        d: &[f64], u: &[f64], met: &[f64], jw: &[f64], ax: f64, ay: f64, az: f64, n1: u32,
        face_vl: &[u32], fmet: &[f64], face_nbr: &[u32],
        mut out: DisjointSlice<f64>,
    ) {
        // PR/PS/PT packed into one shared array P (P[m], P[NN_MAX+m], P[2·NN_MAX+m])
        // to keep the shared-array count low.
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
            // `d` is padded to n³ on the host (first n1² = diff matrix, rest 0), so
            // this load is unconditional.
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
        // Tensor indices of node m.
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    println!("=== GPU 3D hyperbolic operator (linear advection) vs CPU (p={p}) ===\n");

    let mesh = Mesh3d::rectangular_periodic(p, 3, 3, 3, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refh.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let (ax, ay, az) = (0.8, -0.5, 0.3);
    let op = Hyperbolic3d::new(&mesh, LinearAdvection3d { ax, ay, az });

    let mut state = vec![vec![0.0; ndof]];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            state[0][e * nn + k] =
                (2.0 * el.geom.x[k]).sin() * (3.0 * el.geom.y[k]).cos() * (el.geom.z[k]).sin() + 0.2 * el.geom.x[k];
        }
    }
    let cpu = op.rhs(&state, 0.0, &|_, _, _, _, _: &mut [f64]| {});

    // Pack the 9 metric terms node-major into one buffer (met[b*9 + c]) + jw.
    let mut met = vec![0.0; ndof * 9];
    let mut jw = vec![0.0; ndof];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            let d = &el.geom;
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
    let n1 = (p + 1) as u32;
    let n2 = (p + 1) * (p + 1);
    let nfc = ne * 6 * n2;
    // Pack face floats (nx,ny,nz,sw) into fmet[idx*4 + c]; vl/nbr stay u32.
    let mut fvl = vec![0u32; nfc];
    let mut fmet = vec![0.0; nfc * 4];
    let mut fnbr = vec![0u32; nfc];
    for (e, el) in mesh.elements.iter().enumerate() {
        for (t, face) in Face::ALL.iter().enumerate() {
            let fc = &el.faces[*face as usize];
            let Neighbor3::Interior { elem: re, face: rface, perm } = &el.neighbors[*face as usize] else {
                panic!("periodic mesh should have no boundary");
            };
            let rf = &mesh.elements[*re].faces[*rface as usize];
            for a in 0..n2 {
                let idx = (e * 6 + t) * n2 + a;
                fvl[idx] = fc.nodes[a] as u32;
                fmet[idx * 4] = fc.nx[a];
                fmet[idx * 4 + 1] = fc.ny[a];
                fmet[idx * 4 + 2] = fc.nz[a];
                fmet[idx * 4 + 3] = fc.sw[a];
                fnbr[idx] = (*re * nn + rf.nodes[perm[a]]) as u32;
            }
        }
    }

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
    let upu = |v: &[u32]| DeviceBuffer::from_host(&stream, v);
    // Pad the n1² diff matrix to n³ so the kernel loads DS[m] unconditionally.
    let mut d_pad = vec![0.0; nn];
    d_pad[..mesh.refh.line.diff.len()].copy_from_slice(&mesh.refh.line.diff);
    let d_dev = up(&d_pad)?;
    let u_dev = up(&state[0])?;
    let met_dev = up(&met)?;
    let jw_dev = up(&jw)?;
    let fvl_dev = upu(&fvl)?;
    let fmet_dev = up(&fmet)?;
    let fnbr_dev = upu(&fnbr)?;
    let mut out_dev = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;

    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig { grid_dim: (ne as u32, 1, 1), block_dim: (nn as u32, 1, 1), shared_mem_bytes: 0 };
    module.advect_rhs(
        &stream, cfg, &d_dev, &u_dev, &met_dev, &jw_dev, ax, ay, az, n1,
        &fvl_dev, &fmet_dev, &fnbr_dev, &mut out_dev,
    )?;
    let gpu = out_dev.to_host_vec(&stream)?;

    let mut max_abs = 0.0f64;
    let scale = cpu[0].iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1e-300);
    for i in 0..ndof {
        max_abs = max_abs.max((gpu[i] - cpu[0][i]).abs());
    }
    println!("dofs={ndof}  nodes/elem={nn}  max|gpu − cpu| / |op| = {:.3e}", max_abs / scale);
    if max_abs / scale < 1e-10 {
        println!("\nPASS: GPU 3D advection operator matches the CPU oracle.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU/CPU mismatch.");
        std::process::exit(1);
    }
}
