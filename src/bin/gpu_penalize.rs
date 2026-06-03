//! GPU port of the **volume-penalization** apply (immersed rigid bodies) — the
//! implicit Brinkman relaxation `u ← (u + β·u_s)/(1+β)`, `β = χ·dt/η_b`, applied
//! per node. Embarrassingly parallel, libdevice-free. Validated bit-for-bit against
//! `VolumePenalization::apply`.
//!
//! Run: cargo oxide run --bin gpu-penalize

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice};
use cuda_host::cuda_module;
use gale::dg::{Disk, Mesh2d, VolumePenalization};

#[cuda_module]
mod kernels {
    use super::*;

    /// In-place implicit penalization of both velocity components. One thread per dof.
    #[kernel]
    pub fn penalize(
        mut ux: DisjointSlice<f64>,
        mut uy: DisjointSlice<f64>,
        mask: &[f64],
        usx: &[f64],
        usy: &[f64],
        r: f64,
        nn: u32,
    ) {
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let g = e * (nn as usize) + m;
        let beta = r * mask[g];
        let denom = 1.0 + beta;
        let (bx, by) = (beta * usx[g], beta * usy[g]);
        if let Some(o) = ux.get_mut(thread::index_1d()) {
            *o = (*o + bx) / denom;
        }
        if let Some(o) = uy.get_mut(thread::index_1d()) {
            *o = (*o + by) / denom;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    println!("=== GPU volume-penalization apply vs CPU (p={p}) ===\n");

    let mesh = Mesh2d::rectangular(p, 4, 4, [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let dt = 0.02;
    let eta_b = 1e-3;
    let disk = Disk::new(0.5, 0.5, 0.2);
    let pen = VolumePenalization::new(&mesh, &disk, eta_b);

    // A test velocity field; CPU reference applies the relaxation.
    let mut ux: Vec<f64> = (0..ndof).map(|i| 1.0 + 0.3 * (i as f64 * 0.01).sin()).collect();
    let mut uy: Vec<f64> = (0..ndof).map(|i| -0.5 + 0.2 * (i as f64 * 0.02).cos()).collect();
    let (mut cx, mut cy) = (ux.clone(), uy.clone());
    pen.apply(&mut cx, &mut cy, dt);

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
    let mut ux_dev = up(&ux)?;
    let mut uy_dev = up(&uy)?;
    let mask_dev = up(&pen.mask)?;
    let usx_dev = up(&pen.us_x)?;
    let usy_dev = up(&pen.us_y)?;

    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig { grid_dim: (ne as u32, 1, 1), block_dim: (nn as u32, 1, 1), shared_mem_bytes: 0 };
    module.penalize(&stream, cfg, &mut ux_dev, &mut uy_dev, &mask_dev, &usx_dev, &usy_dev, dt / eta_b, nn as u32)?;
    ux = ux_dev.to_host_vec(&stream)?;
    uy = uy_dev.to_host_vec(&stream)?;

    let mut max_abs = 0.0f64;
    for i in 0..ndof {
        max_abs = max_abs.max((ux[i] - cx[i]).abs()).max((uy[i] - cy[i]).abs());
    }
    println!("dofs={ndof}  max|gpu − cpu| = {max_abs:.3e}");
    if max_abs < 1e-14 {
        println!("\nPASS: GPU penalization matches the CPU oracle.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU/CPU mismatch.");
        std::process::exit(1);
    }
}
