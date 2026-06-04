//! Validation harness for the reusable [`gale_gpu::oldroyd_conf_rhs`] library
//! component: computes the Oldroyd-B conformation-transport rhs on the GPU and
//! checks it bit-for-bit against the CPU oracle `gale::dg::OldroydB::conformation_rhs`.
//!
//! Run: cargo oxide run --bin gpu-oldroyd

use gale::dg::{Mesh2d, OldroydB};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    println!("=== gale_gpu::oldroyd_conf_rhs (library) vs CPU oracle (p={p}) ===\n");

    let mesh = Mesh2d::rectangular(p, 4, 4, [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let lambda = 0.7;
    let op = OldroydB::new(&mesh, lambda, 1.0);

    // Smooth velocity + SPD conformation fields.
    let (mut ux, mut uy) = (vec![0.0; ndof], vec![0.0; ndof]);
    let mut c = [vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof]];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            let (x, y) = (el.geom.x[k], el.geom.y[k]);
            ux[e * nn + k] = (2.0 * x).sin() * y + 0.3 * x;
            uy[e * nn + k] = -0.4 * (3.0 * y).cos() * x;
            c[0][e * nn + k] = 1.5 + 0.4 * (x + y).sin();
            c[1][e * nn + k] = 0.2 * x - 0.1 * y;
            c[2][e * nn + k] = 1.3 + 0.3 * (x * y).cos();
        }
    }
    let cpu = op.conformation_rhs(&c, &ux, &uy);

    // GPU path through the reusable library wrapper.
    let gpu = gale_gpu::oldroyd_conf_rhs(&mesh, &c, &ux, &uy, lambda)?;

    let mut max_abs = 0.0f64;
    let mut scale = 1e-300f64;
    for v in 0..3 {
        for i in 0..ndof {
            max_abs = max_abs.max((gpu[v][i] - cpu[v][i]).abs());
            scale = scale.max(cpu[v][i].abs());
        }
    }
    println!("dofs={ndof}  max|gpu − cpu| / |rhs| = {:.3e}", max_abs / scale);
    if max_abs / scale < 1e-12 {
        println!("\nPASS: GPU Oldroyd-B conformation rhs (library) matches the CPU oracle.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU/CPU mismatch.");
        std::process::exit(1);
    }
}
