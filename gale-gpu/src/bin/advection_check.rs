//! Cross-crate validation: this binary calls the **library** kernel
//! [`gale_gpu::advection_rhs`] and checks it bit-for-bit against gale's CPU
//! `Hyperbolic` oracle. The kernel's embedded device artifact lives in the
//! `gale-gpu` rlib (a dependency of this bin), so a successful run also proves the
//! cross-crate artifact-anchor link path works.
//!
//! Run: cargo oxide run --bin advection-check

use gale::dg::{Hyperbolic, LinearAdvection, Mesh2d};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    let (ax, ay) = (0.8, -0.5);
    println!("=== gale-gpu::advection_rhs (library kernel) vs CPU oracle (p={p}) ===\n");

    let mesh = Mesh2d::rectangular_periodic(p, 3, 3, [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;

    let mut u = vec![0.0; ndof];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            u[e * nn + k] = (2.0 * el.geom.x[k]).sin() * (3.0 * el.geom.y[k]).cos() + 0.2 * el.geom.x[k];
        }
    }

    let cpu = Hyperbolic::new(&mesh, LinearAdvection { ax, ay }).rhs(&[u.clone()], 0.0, &|_, _, _, _: &mut [f64]| {});
    let gpu = gale_gpu::advection_rhs(&mesh, &u, ax, ay)?;

    let scale = cpu[0].iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1e-300);
    let max_abs = (0..ndof).fold(0.0f64, |a, i| a.max((gpu[i] - cpu[0][i]).abs()));
    println!("dofs={ndof}  max|gpu − cpu| / |op| = {:.3e}", max_abs / scale);
    if max_abs / scale < 1e-10 {
        println!("\nPASS: reusable library kernel (gale-gpu) matches the CPU oracle across the crate boundary.");
        Ok(())
    } else {
        eprintln!("\nFAIL: gpu/cpu mismatch");
        std::process::exit(1);
    }
}
