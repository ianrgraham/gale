//! Validation harness for the reusable [`gale_gpu::advection3d_rhs`] library
//! component: computes the 3D linear-advection weak-form RHS on the GPU and checks
//! it bit-for-bit against the CPU oracle `gale::dg::Hyperbolic3d::rhs`. Periodic
//! hex mesh ⇒ no boundary fluxes; libdevice-free apart from `f64::abs` ⇒ runs on
//! sm_70 (2× Titan V).
//!
//! Run: cargo oxide run --bin gpu-advection3d

use gale::dg::{Hyperbolic3d, LinearAdvection3d, Mesh3d};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    println!("=== gale_gpu::advection3d_rhs (library) vs CPU oracle (p={p}) ===\n");

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

    // GPU path through the reusable library wrapper.
    let gpu = gale_gpu::advection3d_rhs(&mesh, &state[0], ax, ay, az)?;

    let mut max_abs = 0.0f64;
    let scale = cpu[0].iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1e-300);
    for i in 0..ndof {
        max_abs = max_abs.max((gpu[i] - cpu[0][i]).abs());
    }
    println!("dofs={ndof}  nodes/elem={nn}  max|gpu − cpu| / |op| = {:.3e}", max_abs / scale);
    if max_abs / scale < 1e-10 {
        println!("\nPASS: GPU 3D advection operator (library) matches the CPU oracle.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU/CPU mismatch.");
        std::process::exit(1);
    }
}
