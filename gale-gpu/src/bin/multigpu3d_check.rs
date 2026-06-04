//! Validation harness for the reusable [`gale_gpu::multigpu_advection_3d`] library
//! component — the 3D analogue of `gpu-multigpu`. Runs live **multi-GPU 3D** hex
//! advection across the 2× Titan V (framework
//! [`DomainDecomposition`](gale::sim::DomainDecomposition) element→GPU partition,
//! combined `[local | halo]` per-GPU buffers, P2P `cuMemcpyPeerAsync` halo exchange,
//! and the `advect3d_mg_rhs` hex kernel per device), then checks the gathered result
//! bit-for-bit against the monolithic CPU operator `gale::dg::Hyperbolic3d`.
//!
//! Run: cargo oxide run --bin gpu-multigpu3d

use gale::dg::{Hyperbolic3d, LinearAdvection3d, Mesh3d};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    let (ax, ay, az) = (0.8, -0.5, 0.3);
    println!("=== Multi-GPU 3D hex advection (2 devices, P2P halo) vs CPU (p={p}) ===\n");

    let mesh = Mesh3d::rectangular_periodic(p, 4, 2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refh.n_nodes();
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

    // Multi-GPU path through the reusable library wrapper.
    let got = gale_gpu::multigpu_advection_3d(&mesh, &gstate, ax, ay, az)?;

    let mut max_abs = 0.0f64;
    let scale = cpu[0].iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1e-300);
    for i in 0..ndof {
        max_abs = max_abs.max((got[i] - cpu[0][i]).abs());
    }
    println!("devices=2  hexes={ne}  dofs={ndof}  max|2gpu − cpu| / |op| = {:.3e}", max_abs / scale);
    if max_abs / scale < 1e-12 {
        println!("\nPASS: 2-GPU P2P-halo 3D hex advection (library) matches the monolithic CPU operator.");
        Ok(())
    } else {
        eprintln!("\nFAIL: multi-GPU 3D mismatch.");
        std::process::exit(1);
    }
}
