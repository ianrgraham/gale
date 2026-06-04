//! Validation harness for the reusable [`gale_gpu::operators::poisson::poisson_apply`]
//! library component: applies the full matrix-free SIPG operator `A·u` on the GPU and
//! checks it against the CPU oracle `gale::dg::Poisson::apply`.
//!
//! Run: cargo oxide run --bin gpu-poisson-operator

use gale::dg::{Mesh2d, Poisson};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const P: usize = 4;
    let alpha = 5.0;
    println!("=== gale_gpu::poisson_apply (library) vs CPU oracle (p={P}) ===\n");

    let mesh = Mesh2d::rectangular(P, 3, 3, [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let poisson = Poisson::new(&mesh, alpha);

    let mut u = vec![0.0; ne * nn];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            let (x, y) = (el.geom.x[k], el.geom.y[k]);
            u[e * nn + k] = (1.3 * x + 0.7 * y).sin() + 0.5 * x * x - 0.4 * y;
        }
    }
    let cpu = poisson.apply(&u);

    // GPU path through the reusable library wrapper.
    let gpu = gale_gpu::operators::poisson::poisson_apply(&mesh, &u, alpha)?;

    let mut max_abs = 0.0f64;
    let scale = cpu.iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1e-300);
    for i in 0..ne * nn {
        max_abs = max_abs.max((gpu[i] - cpu[i]).abs());
    }
    println!("elements={ne}  dofs={}", ne * nn);
    println!("max|gpu - cpu| / |op| = {:.3e}", max_abs / scale);
    if max_abs / scale < 1e-10 {
        println!("\nPASS: full GPU SIPG operator (library) matches the CPU oracle.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU/CPU mismatch.");
        std::process::exit(1);
    }
}
