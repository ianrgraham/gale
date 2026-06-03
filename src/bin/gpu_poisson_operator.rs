//! GPU port (step 7c): the **full matrix-free SIPG operator** `A u` = volume + faces,
//! as a 2-kernel pipeline, validated against the CPU oracle `Poisson::apply`.
//!
//! Kernel 1 (`gradient`): per element, `∇u` at every node → stored globally so a
//! neighbor's gradient is readable in kernel 2.
//! Kernel 2 (`operator`): per element, the fused residual
//!   `A u = Drᵀ(pr_tot) + Dsᵀ(ps_tot) + r_face_pt`,
//! where `pr_tot = pr_vol − pr_face` combines the volume stiffness and the SIPG
//! symmetry-lift into one transpose, and `r_face_pt` is the face-node
//! consistency+penalty. Element-centric gather (no atomics). Libdevice-free ⇒ the
//! embedded path works on sm_70.
//!
//! Run: cargo oxide run --bin gpu-poisson-operator

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;
use gale::dg::{Edge, Mesh2d, Neighbor, Poisson};

const P: usize = 4;
const N1: usize = P + 1;
const NN: usize = N1 * N1;
const BND: u32 = u32::MAX;

#[cuda_module]
mod kernels {
    use super::*;

    /// `∇u` per element by sum factorization. `e = blockIdx.x`, node `m = threadIdx.x`.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn gradient(
        d: &[f64],
        u: &[f64],
        rx: &[f64],
        ry: &[f64],
        sx: &[f64],
        sy: &[f64],
        mut gx: DisjointSlice<f64>,
        mut gy: DisjointSlice<f64>,
    ) {
        static mut DS: SharedArray<f64, NN> = SharedArray::UNINIT;
        static mut US: SharedArray<f64, NN> = SharedArray::UNINIT;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        unsafe {
            DS[m] = d[m];
            US[m] = u[e * NN + m];
        }
        thread::sync_threads();
        let i = m % N1;
        let j = m / N1;
        let mut ur = 0.0f64;
        let mut us = 0.0f64;
        let mut k = 0usize;
        while k < N1 {
            unsafe {
                ur += DS[i * N1 + k] * US[k + j * N1];
                us += DS[j * N1 + k] * US[i + k * N1];
            }
            k += 1;
        }
        let b = e * NN + m;
        let gxv = rx[b] * ur + sx[b] * us;
        let gyv = ry[b] * ur + sy[b] * us;
        if let Some(o) = gx.get_mut(thread::index_1d()) {
            *o = gxv;
        }
        if let Some(o) = gy.get_mut(thread::index_1d()) {
            *o = gyv;
        }
    }

    /// Full SIPG operator action `A u` (volume + faces) per element.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn operator(
        d: &[f64],
        u: &[f64],
        gx: &[f64],
        gy: &[f64],
        rx: &[f64],
        ry: &[f64],
        sx: &[f64],
        sy: &[f64],
        jw: &[f64],
        face_vl: &[u32],
        face_nx: &[f64],
        face_ny: &[f64],
        face_sw: &[f64],
        face_nbr: &[u32],
        face_tau: &[f64],
        mut out: DisjointSlice<f64>,
    ) {
        static mut DS: SharedArray<f64, NN> = SharedArray::UNINIT;
        static mut PR: SharedArray<f64, NN> = SharedArray::UNINIT;
        static mut PS: SharedArray<f64, NN> = SharedArray::UNINIT;
        static mut RF: SharedArray<f64, NN> = SharedArray::UNINIT;
        static mut HX: SharedArray<f64, NN> = SharedArray::UNINIT;
        static mut HY: SharedArray<f64, NN> = SharedArray::UNINIT;

        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let b = e * NN + m;

        // Volume sources pr_vol/ps_vol from the (global) gradient; init face accumulators.
        unsafe {
            DS[m] = d[m];
            RF[m] = 0.0;
            HX[m] = 0.0;
            HY[m] = 0.0;
            let wx = jw[b] * gx[b];
            let wy = jw[b] * gy[b];
            PR[m] = rx[b] * wx + ry[b] * wy;
            PS[m] = sx[b] * wx + sy[b] * wy;
        }
        thread::sync_threads();

        // Face-node consistency/penalty + lift sources (serial on thread 0).
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
                    let dun_e = nx * gx[e * NN + vl] + ny * gy[e * NN + vl];
                    let ug = u[e * NN + vl];
                    let (avg, jump, gfac) = if nbr == BND {
                        (dun_e, ug, 1.0)
                    } else {
                        let ng = nbr as usize;
                        (0.5 * (dun_e + nx * gx[ng] + ny * gy[ng]), ug - u[ng], 0.5)
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

        // Subtract the lift sources: pr_tot = pr_vol − pr_face.
        let hxm = unsafe { HX[m] };
        let hym = unsafe { HY[m] };
        unsafe {
            PR[m] -= rx[b] * hxm + ry[b] * hym;
            PS[m] -= sx[b] * hxm + sy[b] * hym;
        }
        thread::sync_threads();

        // A u = Drᵀ(pr_tot) + Dsᵀ(ps_tot) + face-node terms.
        let i = m % N1;
        let j = m / N1;
        let mut acc = 0.0f64;
        let mut k = 0usize;
        while k < N1 {
            unsafe {
                acc += DS[k * N1 + i] * PR[k + j * N1] + DS[k * N1 + j] * PS[i + k * N1];
            }
            k += 1;
        }
        let rfm = unsafe { RF[m] };
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = acc + rfm;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU full SIPG operator A·u vs CPU oracle (p={P}) ===\n");

    let mesh = Mesh2d::rectangular(P, 3, 3, [0.0, 1.0], [0.0, 1.0]);
    let refq = &mesh.refq;
    let nn = refq.n_nodes();
    let ne = mesh.n_elements();
    assert_eq!(nn, NN);
    let poisson = Poisson::new(&mesh, 5.0);

    let mut u = vec![0.0; ne * nn];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            let (x, y) = (el.geom.x[k], el.geom.y[k]);
            u[e * nn + k] = (1.3 * x + 0.7 * y).sin() + 0.5 * x * x - 0.4 * y;
        }
    }
    let cpu = poisson.apply(&u);

    // Flatten metrics.
    let (mut rx, mut ry, mut sx, mut sy, mut jw) = (
        vec![0.0; ne * nn],
        vec![0.0; ne * nn],
        vec![0.0; ne * nn],
        vec![0.0; ne * nn],
        vec![0.0; ne * nn],
    );
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            rx[e * nn + k] = el.geom.rx[k];
            ry[e * nn + k] = el.geom.ry[k];
            sx[e * nn + k] = el.geom.sx[k];
            sy[e * nn + k] = el.geom.sy[k];
            jw[e * nn + k] = el.geom.jw[k];
        }
    }

    // Flatten face metadata.
    let alpha = 5.0;
    let p1 = (P + 1) as f64;
    let h: Vec<f64> = mesh.elements.iter().map(|el| el.geom.jw.iter().sum::<f64>().sqrt()).collect();
    let nf = ne * 4 * N1;
    let (mut fvl, mut fnx, mut fny, mut fsw, mut fnbr, mut ftau) = (
        vec![0u32; nf],
        vec![0.0; nf],
        vec![0.0; nf],
        vec![0.0; nf],
        vec![BND; nf],
        vec![0.0; ne * 4],
    );
    for (e, el) in mesh.elements.iter().enumerate() {
        for (t, edge) in Edge::ALL.iter().enumerate() {
            let face = &el.faces[*edge as usize];
            let nb = &el.neighbors[*edge as usize];
            ftau[e * 4 + t] = match nb {
                Neighbor::Interior { elem: re, .. } => alpha * p1 * p1 / h[e].min(h[*re]),
                Neighbor::Boundary { .. } => alpha * p1 * p1 / h[e],
            };
            for a in 0..N1 {
                let idx = (e * 4 + t) * N1 + a;
                fvl[idx] = face.nodes[a] as u32;
                fnx[idx] = face.nx[a];
                fny[idx] = face.ny[a];
                fsw[idx] = face.sw[a];
                if let Neighbor::Interior { elem: re, edge: redge, perm } = nb {
                    let rf = &mesh.elements[*re].faces[*redge as usize];
                    fnbr[idx] = (*re * NN + rf.nodes[perm[a]]) as u32;
                }
            }
        }
    }

    // Launch the 2-kernel pipeline.
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let f = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
    let d_dev = f(&refq.line.diff)?;
    let u_dev = f(&u)?;
    let rx_dev = f(&rx)?;
    let ry_dev = f(&ry)?;
    let sx_dev = f(&sx)?;
    let sy_dev = f(&sy)?;
    let jw_dev = f(&jw)?;
    let mut gx_dev = DeviceBuffer::<f64>::zeroed(&stream, ne * nn)?;
    let mut gy_dev = DeviceBuffer::<f64>::zeroed(&stream, ne * nn)?;
    let fvl_dev = DeviceBuffer::from_host(&stream, &fvl)?;
    let fnx_dev = f(&fnx)?;
    let fny_dev = f(&fny)?;
    let fsw_dev = f(&fsw)?;
    let fnbr_dev = DeviceBuffer::from_host(&stream, &fnbr)?;
    let ftau_dev = f(&ftau)?;
    let mut out_dev = DeviceBuffer::<f64>::zeroed(&stream, ne * nn)?;

    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig {
        grid_dim: (ne as u32, 1, 1),
        block_dim: (nn as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    module.gradient(&stream, cfg, &d_dev, &u_dev, &rx_dev, &ry_dev, &sx_dev, &sy_dev, &mut gx_dev, &mut gy_dev)?;
    module.operator(
        &stream, cfg, &d_dev, &u_dev, &gx_dev, &gy_dev, &rx_dev, &ry_dev, &sx_dev, &sy_dev, &jw_dev,
        &fvl_dev, &fnx_dev, &fny_dev, &fsw_dev, &fnbr_dev, &ftau_dev, &mut out_dev,
    )?;
    let gpu = out_dev.to_host_vec(&stream)?;

    let mut max_abs = 0.0f64;
    let scale = cpu.iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1e-300);
    for i in 0..ne * nn {
        max_abs = max_abs.max((gpu[i] - cpu[i]).abs());
    }
    println!("elements={ne}  dofs={}", ne * nn);
    println!("max|gpu - cpu| / |op| = {:.3e}", max_abs / scale);
    if max_abs / scale < 1e-10 {
        println!("\nPASS: full GPU SIPG operator matches the CPU oracle.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU/CPU mismatch.");
        std::process::exit(1);
    }
}
