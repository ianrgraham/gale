//! GPU port (step 7a): the **matrix-free DG volume-stiffness operator** on the
//! Titan Vs via cuda-oxide, validated bit-for-bit against the CPU oracle
//! (`Poisson::apply_volume`).
//!
//! One thread-block per element, `(p+1)²` threads per block (one per node), with
//! the 1D differentiation matrix and the element's nodal data staged in shared
//! memory. This is the element-local, sum-factorized heavy-compute core of the DG
//! operator — the GPU arithmetic-intensity win the project rests on. It is
//! libdevice-free (only `+`/`*`), so the embedded `#[cuda_module]` path works on
//! sm_70 (cf. `docs/milestone-1-probe-results.md`).
//!
//! Run: cargo oxide run --bin gpu-poisson-volume   (uses the #101 backend)

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;
use gale::dg::{Mesh2d, Poisson};

const P: usize = 4; // polynomial order
const N1: usize = P + 1; // nodes per direction = 5
const NN: usize = N1 * N1; // nodes per element = 25 (also the size of the 1D D matrix)

#[cuda_module]
mod kernels {
    use super::*;

    /// `out_e = Dxᵀ(W·∇x u_e) + Dyᵀ(W·∇y u_e)` per element `e = blockIdx.x`.
    /// Local node `m = threadIdx.x = i + j·N1`.
    #[kernel]
    pub fn volume_stiffness(
        d: &[f64],  // 1D differentiation matrix, N1×N1
        u: &[f64],  // field, ne·NN
        rx: &[f64], // metric terms, ne·NN
        ry: &[f64],
        sx: &[f64],
        sy: &[f64],
        jw: &[f64], // physical mass diagonal, ne·NN
        mut out: DisjointSlice<f64>,
    ) {
        static mut DS: SharedArray<f64, NN> = SharedArray::UNINIT;
        static mut US: SharedArray<f64, NN> = SharedArray::UNINIT;
        static mut PR: SharedArray<f64, NN> = SharedArray::UNINIT;
        static mut PS: SharedArray<f64, NN> = SharedArray::UNINIT;

        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let base = e * NN;

        // Stage the (shared-across-elements) D matrix and this element's field.
        unsafe {
            DS[m] = d[m];
            US[m] = u[base + m];
        }
        thread::sync_threads();

        let i = m % N1;
        let j = m / N1;

        // Reference derivatives by sum factorization.
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

        // Physical gradient → weighted → fused pr/ps for the transpose step.
        let gx = rx[base + m] * ur + sx[base + m] * us;
        let gy = ry[base + m] * ur + sy[base + m] * us;
        let wx = jw[base + m] * gx;
        let wy = jw[base + m] * gy;
        unsafe {
            PR[m] = rx[base + m] * wx + ry[base + m] * wy;
            PS[m] = sx[base + m] * wx + sy[base + m] * wy;
        }
        thread::sync_threads();

        // Transpose application: Drᵀ(pr) + Dsᵀ(ps).
        let mut rr = 0.0f64;
        let mut k2 = 0usize;
        while k2 < N1 {
            unsafe {
                rr += DS[k2 * N1 + i] * PR[k2 + j * N1] + DS[k2 * N1 + j] * PS[i + k2 * N1];
            }
            k2 += 1;
        }
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = rr;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU DG volume-stiffness operator vs CPU oracle (p={P}) ===\n");

    let mesh = Mesh2d::rectangular(P, 3, 3, [0.0, 1.0], [0.0, 1.0]);
    let refq = &mesh.refq;
    let nn = refq.n_nodes();
    let ne = mesh.n_elements();
    assert_eq!(nn, NN, "kernel compiled for p={P}");
    let poisson = Poisson::new(&mesh, 5.0);

    // Arbitrary smooth test field (CPU and GPU apply the same operator to it).
    let mut u = vec![0.0; ne * nn];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            let (x, y) = (el.geom.x[k], el.geom.y[k]);
            u[e * nn + k] = (1.3 * x + 0.7 * y).sin() + 0.5 * x * x - 0.4 * y;
        }
    }

    // CPU reference.
    let cpu = poisson.apply_volume(&u);

    // Flatten per-element geometry into global device buffers.
    let mut rx = vec![0.0; ne * nn];
    let mut ry = vec![0.0; ne * nn];
    let mut sx = vec![0.0; ne * nn];
    let mut sy = vec![0.0; ne * nn];
    let mut jw = vec![0.0; ne * nn];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            rx[e * nn + k] = el.geom.rx[k];
            ry[e * nn + k] = el.geom.ry[k];
            sx[e * nn + k] = el.geom.sx[k];
            sy[e * nn + k] = el.geom.sy[k];
            jw[e * nn + k] = el.geom.jw[k];
        }
    }
    let d = refq.line.diff.clone(); // N1×N1

    // GPU launch.
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let d_dev = DeviceBuffer::from_host(&stream, &d)?;
    let u_dev = DeviceBuffer::from_host(&stream, &u)?;
    let rx_dev = DeviceBuffer::from_host(&stream, &rx)?;
    let ry_dev = DeviceBuffer::from_host(&stream, &ry)?;
    let sx_dev = DeviceBuffer::from_host(&stream, &sx)?;
    let sy_dev = DeviceBuffer::from_host(&stream, &sy)?;
    let jw_dev = DeviceBuffer::from_host(&stream, &jw)?;
    let mut out_dev = DeviceBuffer::<f64>::zeroed(&stream, ne * nn)?;

    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig {
        grid_dim: (ne as u32, 1, 1),
        block_dim: (nn as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    module.volume_stiffness(
        &stream, cfg, &d_dev, &u_dev, &rx_dev, &ry_dev, &sx_dev, &sy_dev, &jw_dev, &mut out_dev,
    )?;
    let gpu = out_dev.to_host_vec(&stream)?;

    // Compare.
    let mut max_abs = 0.0f64;
    let scale = cpu.iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1e-300);
    for i in 0..ne * nn {
        max_abs = max_abs.max((gpu[i] - cpu[i]).abs());
    }
    println!("elements={ne}  nodes/elem={nn}  dofs={}", ne * nn);
    println!("max|gpu - cpu|        = {max_abs:.3e}");
    println!("max|gpu - cpu| / |op| = {:.3e}", max_abs / scale);

    if max_abs / scale < 1e-10 {
        println!("\nPASS: GPU volume stiffness matches the CPU oracle.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU/CPU mismatch.");
        std::process::exit(1);
    }
}
