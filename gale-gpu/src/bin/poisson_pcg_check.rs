//! Validation harness for the reusable [`gale_gpu::operators::poisson::poisson_pcg_solve`]
//! library component: solves the SIPG Poisson system by p-multigrid-preconditioned CG
//! entirely on the GPU and checks the solution against the CPU oracle
//! `gale::dg::PMultigrid::pcg`.
//!
//! Run: cargo oxide run --bin gpu-poisson-pcg

use gale::dg::{PMultigrid, Poisson};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::f64::consts::PI;
    let order = 4;
    println!("=== gale_gpu::poisson_pcg_solve (library) vs CPU oracle (p={order}) ===\n");

    // CPU setup (validated) + reference solve.
    let mg = PMultigrid::new(order, 3, 3, [0.0, 1.0], [0.0, 1.0], 5.0);
    let nlev = mg.n_levels();
    let fine = mg.mesh(0);
    let nn0 = fine.refq.n_nodes();
    let n0 = fine.n_elements() * nn0;
    let exact = |x: f64, y: f64| (PI * x).sin() * (PI * y).sin();
    let mut frc = vec![0.0; n0];
    for (e, el) in fine.elements.iter().enumerate() {
        for k in 0..nn0 {
            frc[e * nn0 + k] = 2.0 * PI * PI * exact(el.geom.x[k], el.geom.y[k]);
        }
    }
    let rhs = Poisson::new(fine, mg.alpha).rhs(&frc, |_, _| 0.0);
    let (u_cpu, it_cpu) = mg.pcg(&rhs, 1e-10, 2000);

    // GPU path through the reusable library wrapper (same tol / max-iters).
    let (u_gpu, iters) = gale_gpu::operators::poisson::poisson_pcg_solve(&mg, &rhs, 1e-10, 2000)?;

    // compare
    let mut diff = 0.0f64;
    let mut nrm = 0.0f64;
    for i in 0..n0 {
        diff += (u_gpu[i] - u_cpu[i]).powi(2);
        nrm += u_cpu[i].powi(2);
    }
    let rel = (diff / nrm.max(1e-300)).sqrt();
    println!("dofs={n0}  levels={nlev}  PCG iters: cpu={it_cpu}  gpu={iters}");
    println!("‖u_gpu − u_cpu‖ / ‖u_cpu‖ = {rel:.3e}");
    if rel < 1e-7 {
        println!("\nPASS: GPU p-multigrid PCG (library) matches the CPU oracle.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU/CPU mismatch.");
        std::process::exit(1);
    }
}
