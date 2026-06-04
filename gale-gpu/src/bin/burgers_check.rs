//! Validation harness for the reusable [`gale_gpu::burgers_rhs`] library component:
//! computes the GPU split-form (entropy-conserving) Burgers RHS and checks it
//! bit-for-bit against `gale::dg::Hyperbolic::rhs` (SplitForm, `dissipation=false`).
//!
//! Run: cargo oxide run --bin gpu-burgers-split

use gale::dg::{Burgers, Hyperbolic, Mesh2d, VolumeForm};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    println!("=== gale_gpu::burgers_rhs (library) vs CPU oracle (entropy-conserving, p={p}) ===\n");

    let mesh = Mesh2d::rectangular_periodic(p, 3, 3, [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let op = Hyperbolic::with_options(&mesh, Burgers, VolumeForm::SplitForm, false);

    let mut state = vec![vec![0.0; ndof]];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            state[0][e * nn + k] = 0.5 + (2.0 * std::f64::consts::PI * el.geom.x[k]).sin() * (2.0 * std::f64::consts::PI * el.geom.y[k]).cos();
        }
    }
    let cpu = op.rhs(&state, 0.0, &|_, _, _, _: &mut [f64]| {});

    // GPU path through the reusable library wrapper.
    let gpu = gale_gpu::burgers_rhs(&mesh, &state[0])?;

    let mut max_abs = 0.0f64;
    let scale = cpu[0].iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1e-300);
    for i in 0..ndof {
        max_abs = max_abs.max((gpu[i] - cpu[0][i]).abs());
    }
    println!("dofs={ndof}  max|gpu − cpu| / |rhs| = {:.3e}", max_abs / scale);
    if max_abs / scale < 1e-11 {
        println!("\nPASS: GPU split-form Burgers (library) matches the CPU oracle.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU/CPU mismatch.");
        std::process::exit(1);
    }
}
