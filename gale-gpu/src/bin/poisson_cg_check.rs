//! Validation harness for the reusable [`gale_gpu::operators::poisson::poisson_cg_solve`]
//! library component: solves the SIPG Poisson system by conjugate gradient entirely on
//! the GPU and checks the solution against the CPU oracle `gale::dg::Poisson::cg`.
//!
//! Run: cargo oxide run --bin gpu-poisson-cg

use gale::dg::{Mesh2d, Poisson};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::f64::consts::PI;
    const P: usize = 4;
    let alpha = 5.0;
    println!("=== gale_gpu::poisson_cg_solve (library) vs CPU oracle (p={P}) ===\n");

    let mesh = Mesh2d::rectangular(P, 3, 3, [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let poisson = Poisson::new(&mesh, alpha);

    // MMS: u = sin(πx)sin(πy), −Δu = 2π²u, homogeneous Dirichlet.
    let exact = |x: f64, y: f64| (PI * x).sin() * (PI * y).sin();
    let mut f = vec![0.0; ndof];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            f[e * nn + k] = 2.0 * PI * PI * exact(el.geom.x[k], el.geom.y[k]);
        }
    }
    let b = poisson.rhs(&f, |_, _| 0.0);

    // CPU reference solve.
    let (u_cpu, it_cpu, _res) = poisson.cg(&b, 1e-10, 20000);

    // GPU path through the reusable library wrapper (same tol / max-iters).
    let (u_gpu, iters) = gale_gpu::operators::poisson::poisson_cg_solve(&mesh, &b, alpha, 1e-10, 20000)?;

    // Compare GPU vs CPU solve.
    let mut diff = 0.0f64;
    let mut cpun = 0.0f64;
    let mut err_gpu = 0.0f64;
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            let i = e * nn + k;
            diff += el.geom.jw[k] * (u_gpu[i] - u_cpu[i]).powi(2);
            cpun += el.geom.jw[k] * u_cpu[i].powi(2);
            err_gpu += el.geom.jw[k] * (u_gpu[i] - exact(el.geom.x[k], el.geom.y[k])).powi(2);
        }
    }
    let rel = (diff / cpun.max(1e-300)).sqrt();
    println!("dofs={ndof}   CG iters: cpu={it_cpu}  gpu={iters}");
    println!("‖u_gpu − u_cpu‖ / ‖u_cpu‖ = {rel:.3e}");
    println!("‖u_gpu − u_exact‖ (MMS)   = {:.3e}", err_gpu.sqrt());
    if rel < 1e-8 {
        println!("\nPASS: GPU CG (library) matches the CPU solve.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU/CPU solve mismatch.");
        std::process::exit(1);
    }
}
