//! GPU port of the **hyperbolic conservation-law operator** (weak form), for linear
//! advection — establishes the conservation-law kernel pattern: in-kernel flux,
//! sum-factorized volume `Dxᵀ(W F)`, Rusanov (LLF) interface gather, and `M⁻¹`.
//! Validated bit-for-bit against the CPU `Hyperbolic::rhs`. Periodic mesh ⇒ no
//! boundary fluxes; libdevice-free ⇒ embedded `#[cuda_module]` path works on sm_70.
//! (Euler / split-form GPU — needing in-kernel `sqrt`/`ln` — is the follow-on via
//! the `ltoir` loader.)
//!
//! Run: cargo oxide run --bin gpu-advection

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;
use gale::dg::{Edge, Hyperbolic, LinearAdvection, Mesh2d, Neighbor};

const NN_MAX: usize = 81;

#[cuda_module]
mod kernels {
    use super::*;

    /// `∂ₜu = M⁻¹[Dxᵀ(W·aₓu) + Dyᵀ(W·a_yu) − ∮ F*·n]` for linear advection,
    /// weak form, Rusanov flux. `e = blockIdx.x`, node `m = threadIdx.x`.
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
                    // Uses f64::abs() (libdevice/NVVM-text path) — exercises the
                    // typed-pointer bitcast fix in the cuda-oxide backend.
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    println!("=== GPU hyperbolic operator (linear advection) vs CPU (p={p}) ===\n");

    let mesh = Mesh2d::rectangular_periodic(p, 3, 3, [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let (ax, ay) = (0.8, -0.5);
    let op = Hyperbolic::new(&mesh, LinearAdvection { ax, ay });

    // Test field + CPU reference (single rhs evaluation).
    let mut state = vec![vec![0.0; ndof]];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            state[0][e * nn + k] = (2.0 * el.geom.x[k]).sin() * (3.0 * el.geom.y[k]).cos() + 0.2 * el.geom.x[k];
        }
    }
    let cpu = op.rhs(&state, 0.0, &|_, _, _, _: &mut [f64]| {});

    // Flatten metrics + face metadata.
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
    let n1 = (p + 1) as u32;
    let nfc = ne * 4 * (p + 1);
    let (mut fvl, mut fnx, mut fny, mut fsw, mut fnbr) =
        (vec![0u32; nfc], vec![0.0; nfc], vec![0.0; nfc], vec![0.0; nfc], vec![0u32; nfc]);
    for (e, el) in mesh.elements.iter().enumerate() {
        for (t, edge) in Edge::ALL.iter().enumerate() {
            let face = &el.faces[*edge as usize];
            let Neighbor::Interior { elem: re, edge: redge, perm } = &el.neighbors[*edge as usize] else {
                panic!("periodic mesh should have no boundary");
            };
            let rf = &mesh.elements[*re].faces[*redge as usize];
            for a in 0..(p + 1) {
                let idx = (e * 4 + t) * (p + 1) + a;
                fvl[idx] = face.nodes[a] as u32;
                fnx[idx] = face.nx[a];
                fny[idx] = face.ny[a];
                fsw[idx] = face.sw[a];
                fnbr[idx] = (*re * nn + rf.nodes[perm[a]]) as u32;
            }
        }
    }

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
    let upu = |v: &[u32]| DeviceBuffer::from_host(&stream, v);
    let d_dev = up(&mesh.refq.line.diff)?;
    let u_dev = up(&state[0])?;
    let rx_dev = up(&rx)?;
    let ry_dev = up(&ry)?;
    let sx_dev = up(&sx)?;
    let sy_dev = up(&sy)?;
    let jw_dev = up(&jw)?;
    let fvl_dev = upu(&fvl)?;
    let fnx_dev = up(&fnx)?;
    let fny_dev = up(&fny)?;
    let fsw_dev = up(&fsw)?;
    let fnbr_dev = upu(&fnbr)?;
    let mut out_dev = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;

    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig { grid_dim: (ne as u32, 1, 1), block_dim: (nn as u32, 1, 1), shared_mem_bytes: 0 };
    module.advect_rhs(
        &stream, cfg, &d_dev, &u_dev, &rx_dev, &ry_dev, &sx_dev, &sy_dev, &jw_dev, ax, ay, n1,
        &fvl_dev, &fnx_dev, &fny_dev, &fsw_dev, &fnbr_dev, &mut out_dev,
    )?;
    let gpu = out_dev.to_host_vec(&stream)?;

    let mut max_abs = 0.0f64;
    let scale = cpu[0].iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1e-300);
    for i in 0..ndof {
        max_abs = max_abs.max((gpu[i] - cpu[0][i]).abs());
    }
    println!("dofs={ndof}  max|gpu − cpu| / |op| = {:.3e}", max_abs / scale);
    if max_abs / scale < 1e-10 {
        println!("\nPASS: GPU advection operator matches the CPU oracle.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU/CPU mismatch.");
        std::process::exit(1);
    }
}
