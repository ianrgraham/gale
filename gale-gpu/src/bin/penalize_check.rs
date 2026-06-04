//! Validation harness for the reusable [`gale_gpu::penalize_apply`] library
//! component: applies GPU volume penalization and checks it bit-for-bit against
//! `gale::dg::VolumePenalization::apply`. Also exercises a second `#[cuda_module]`
//! (immersed) coexisting with the advection one in the same crate bundle.
//!
//! Run: cargo oxide run --bin penalize-check

use gale::dg::{Disk, Mesh2d, VolumePenalization};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    println!("=== gale_gpu::penalize_apply (library) vs CPU oracle (p={p}) ===\n");

    let mesh = Mesh2d::rectangular(p, 4, 4, [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let dt = 0.02;
    let eta_b = 1e-3;
    let disk = Disk::new(0.5, 0.5, 0.2);
    let pen = VolumePenalization::new(&mesh, &disk, eta_b);

    // A test velocity field; CPU reference applies the relaxation.
    let ux: Vec<f64> = (0..ndof).map(|i| 1.0 + 0.3 * (i as f64 * 0.01).sin()).collect();
    let uy: Vec<f64> = (0..ndof).map(|i| -0.5 + 0.2 * (i as f64 * 0.02).cos()).collect();
    let (mut cx, mut cy) = (ux.clone(), uy.clone());
    pen.apply(&mut cx, &mut cy, dt);

    // GPU path through the reusable library wrapper.
    let (mut gx, mut gy) = (ux.clone(), uy.clone());
    gale_gpu::penalize_apply(&mesh, &mut gx, &mut gy, &pen, dt)?;

    let mut max_abs = 0.0f64;
    for i in 0..ndof {
        max_abs = max_abs.max((gx[i] - cx[i]).abs()).max((gy[i] - cy[i]).abs());
    }
    println!("dofs={ndof}  max|gpu − cpu| = {max_abs:.3e}");
    if max_abs < 1e-14 {
        println!("\nPASS: GPU penalization (library) matches the CPU oracle.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU/CPU mismatch.");
        std::process::exit(1);
    }
}
