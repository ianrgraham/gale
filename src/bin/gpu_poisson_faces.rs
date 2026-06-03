//! GPU port (step 7b): the **SIPG face/flux terms** on the Titan Vs via cuda-oxide,
//! validated against the CPU oracle (`apply − apply_volume`).
//!
//! Element-centric **gather** design (one block per element): each element computes
//! its *own* residual by reading neighbor traces read-only — no scatter, no atomics.
//! The key identity: an interior face's contribution has the *same form* in an
//! element's own outward-normal frame whether it is the "−" or "+" side, so every
//! element applies one formula per edge:
//!   r_e += −sw·{∇u·n} + τ·sw·[u]   (at face nodes; boundary: single-sided, [u]=u)
//!   r_e −= liftᵀ(½·sw·[u]·n)        (symmetry term, spread over the element)
//! Gradients are precomputed on the host here to isolate the face logic; step 7c
//! chains a gradient kernel. Libdevice-free ⇒ embedded path works on sm_70.
//!
//! Run: cargo oxide run --bin gpu-poisson-faces

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;
use gale::dg::{Edge, Mesh2d, Neighbor, Poisson};

const P: usize = 4;
const N1: usize = P + 1; // 5
const NN: usize = N1 * N1; // 25
const BND: u32 = u32::MAX; // boundary sentinel for the neighbor-node index

#[cuda_module]
mod kernels {
    use super::*;

    /// Per-element SIPG face contribution. `e = blockIdx.x`, local node `m = threadIdx.x`.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn face_terms(
        d: &[f64],  // 1D D matrix, N1×N1
        u: &[f64],  // field, ne·NN
        gx: &[f64], // ∂u/∂x, ne·NN (precomputed)
        gy: &[f64], // ∂u/∂y, ne·NN
        rx: &[f64], // metrics, ne·NN (for the lift)
        ry: &[f64],
        sx: &[f64],
        sy: &[f64],
        face_vl: &[u32],  // local node index of face node, ne·4·N1
        face_nx: &[f64],  // outward normal & surface weight, ne·4·N1
        face_ny: &[f64],
        face_sw: &[f64],
        face_nbr: &[u32], // matched neighbor GLOBAL node index (or BND), ne·4·N1
        face_tau: &[f64], // penalty per (element,edge), ne·4
        mut out: DisjointSlice<f64>,
    ) {
        static mut DS: SharedArray<f64, NN> = SharedArray::UNINIT;
        static mut RF: SharedArray<f64, NN> = SharedArray::UNINIT; // face-local point terms
        static mut HX: SharedArray<f64, NN> = SharedArray::UNINIT; // lift source (·n_x)
        static mut HY: SharedArray<f64, NN> = SharedArray::UNINIT;
        static mut PRF: SharedArray<f64, NN> = SharedArray::UNINIT;
        static mut PSF: SharedArray<f64, NN> = SharedArray::UNINIT;

        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;

        unsafe {
            DS[m] = d[m];
            RF[m] = 0.0;
            HX[m] = 0.0;
            HY[m] = 0.0;
        }
        thread::sync_threads();

        // Thread 0 accumulates the face-node point terms and lift sources.
        // (Corner nodes belong to two edges; serial accumulation avoids the race.)
        if m == 0 {
            let mut t = 0usize;
            while t < 4 {
                let tau = face_tau[e * 4 + t];
                let mut a = 0usize;
                while a < N1 {
                    let idx = (e * 4 + t) * N1 + a;
                    let vl = face_vl[idx] as usize;
                    let nx = face_nx[idx];
                    let ny = face_ny[idx];
                    let sw = face_sw[idx];
                    let nbr = face_nbr[idx];
                    let ug = u[e * NN + vl];
                    let dun_e = nx * gx[e * NN + vl] + ny * gy[e * NN + vl];
                    let (avg, jump, gfac) = if nbr == BND {
                        (dun_e, ug, 1.0) // boundary: single-sided, [u]=u, lift factor 1
                    } else {
                        let ng = nbr as usize;
                        let dun_n = nx * gx[ng] + ny * gy[ng];
                        (0.5 * (dun_e + dun_n), ug - u[ng], 0.5)
                    };
                    let g = gfac * sw * jump;
                    unsafe {
                        RF[vl] += -sw * avg + tau * sw * jump;
                        HX[vl] += g * nx;
                        HY[vl] += g * ny;
                    }
                    a += 1;
                }
                t += 1;
            }
        }
        thread::sync_threads();

        // Fuse metrics into the transpose-derivative sources.
        let hxm = unsafe { HX[m] };
        let hym = unsafe { HY[m] };
        unsafe {
            PRF[m] = rx[e * NN + m] * hxm + ry[e * NN + m] * hym;
            PSF[m] = sx[e * NN + m] * hxm + sy[e * NN + m] * hym;
        }
        thread::sync_threads();

        // Lift = Drᵀ(pr) + Dsᵀ(ps); residual subtracts it.
        let i = m % N1;
        let j = m / N1;
        let mut lift = 0.0f64;
        let mut k = 0usize;
        while k < N1 {
            unsafe {
                lift += DS[k * N1 + i] * PRF[k + j * N1] + DS[k * N1 + j] * PSF[i + k * N1];
            }
            k += 1;
        }
        let rfm = unsafe { RF[m] };
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = rfm - lift;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU SIPG face terms vs CPU oracle (p={P}) ===\n");

    let mesh = Mesh2d::rectangular(P, 3, 3, [0.0, 1.0], [0.0, 1.0]);
    let refq = &mesh.refq;
    let nn = refq.n_nodes();
    let ne = mesh.n_elements();
    assert_eq!(nn, NN);
    let poisson = Poisson::new(&mesh, 5.0);

    // Test field + CPU face reference (= full operator − volume).
    let mut u = vec![0.0; ne * nn];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            let (x, y) = (el.geom.x[k], el.geom.y[k]);
            u[e * nn + k] = (1.3 * x + 0.7 * y).sin() + 0.5 * x * x - 0.4 * y;
        }
    }
    let full = poisson.apply(&u);
    let vol = poisson.apply_volume(&u);
    let cpu: Vec<f64> = full.iter().zip(&vol).map(|(a, b)| a - b).collect();

    // Host-side gradients and flattened geometry.
    let mut gx = vec![0.0; ne * nn];
    let mut gy = vec![0.0; ne * nn];
    let (mut rx, mut ry, mut sx, mut sy) =
        (vec![0.0; ne * nn], vec![0.0; ne * nn], vec![0.0; ne * nn], vec![0.0; ne * nn]);
    for (e, el) in mesh.elements.iter().enumerate() {
        let ue = &u[e * nn..(e + 1) * nn];
        let g_x = el.geom.grad_x(refq, ue);
        let g_y = el.geom.grad_y(refq, ue);
        for k in 0..nn {
            gx[e * nn + k] = g_x[k];
            gy[e * nn + k] = g_y[k];
            rx[e * nn + k] = el.geom.rx[k];
            ry[e * nn + k] = el.geom.ry[k];
            sx[e * nn + k] = el.geom.sx[k];
            sy[e * nn + k] = el.geom.sy[k];
        }
    }

    // Flattened face metadata (ne·4·N1) + penalty (ne·4).
    let alpha = 5.0;
    let p1 = (P + 1) as f64;
    let h: Vec<f64> = mesh
        .elements
        .iter()
        .map(|el| el.geom.jw.iter().sum::<f64>().sqrt())
        .collect();
    let nf = ne * 4 * N1;
    let mut face_vl = vec![0u32; nf];
    let mut face_nx = vec![0.0; nf];
    let mut face_ny = vec![0.0; nf];
    let mut face_sw = vec![0.0; nf];
    let mut face_nbr = vec![BND; nf];
    let mut face_tau = vec![0.0; ne * 4];
    for (e, el) in mesh.elements.iter().enumerate() {
        for (t, edge) in Edge::ALL.iter().enumerate() {
            let face = &el.faces[*edge as usize];
            let nb = &el.neighbors[*edge as usize];
            face_tau[e * 4 + t] = match nb {
                Neighbor::Interior { elem: re, .. } => alpha * p1 * p1 / h[e].min(h[*re]),
                Neighbor::Boundary { .. } => alpha * p1 * p1 / h[e],
            };
            for a in 0..N1 {
                let idx = (e * 4 + t) * N1 + a;
                face_vl[idx] = face.nodes[a] as u32;
                face_nx[idx] = face.nx[a];
                face_ny[idx] = face.ny[a];
                face_sw[idx] = face.sw[a];
                if let Neighbor::Interior { elem: re, edge: redge, perm } = nb {
                    let rf = &mesh.elements[*re].faces[*redge as usize];
                    face_nbr[idx] = (*re * NN + rf.nodes[perm[a]]) as u32;
                }
            }
        }
    }

    // Launch.
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let dbuf = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
    let d_dev = dbuf(&refq.line.diff)?;
    let u_dev = dbuf(&u)?;
    let gx_dev = dbuf(&gx)?;
    let gy_dev = dbuf(&gy)?;
    let rx_dev = dbuf(&rx)?;
    let ry_dev = dbuf(&ry)?;
    let sx_dev = dbuf(&sx)?;
    let sy_dev = dbuf(&sy)?;
    let fvl_dev = DeviceBuffer::from_host(&stream, &face_vl)?;
    let fnx_dev = dbuf(&face_nx)?;
    let fny_dev = dbuf(&face_ny)?;
    let fsw_dev = dbuf(&face_sw)?;
    let fnbr_dev = DeviceBuffer::from_host(&stream, &face_nbr)?;
    let ftau_dev = dbuf(&face_tau)?;
    let mut out_dev = DeviceBuffer::<f64>::zeroed(&stream, ne * nn)?;

    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig {
        grid_dim: (ne as u32, 1, 1),
        block_dim: (nn as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    module.face_terms(
        &stream, cfg, &d_dev, &u_dev, &gx_dev, &gy_dev, &rx_dev, &ry_dev, &sx_dev, &sy_dev,
        &fvl_dev, &fnx_dev, &fny_dev, &fsw_dev, &fnbr_dev, &ftau_dev, &mut out_dev,
    )?;
    let gpu = out_dev.to_host_vec(&stream)?;

    let mut max_abs = 0.0f64;
    let scale = cpu.iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1e-300);
    for i in 0..ne * nn {
        max_abs = max_abs.max((gpu[i] - cpu[i]).abs());
    }
    println!("elements={ne}  nodes/elem={nn}  dofs={}", ne * nn);
    println!("max|gpu - cpu|        = {max_abs:.3e}");
    println!("max|gpu - cpu| / |op| = {:.3e}", max_abs / scale);
    if max_abs / scale < 1e-10 {
        println!("\nPASS: GPU SIPG face terms match the CPU oracle.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU/CPU mismatch.");
        std::process::exit(1);
    }
}
