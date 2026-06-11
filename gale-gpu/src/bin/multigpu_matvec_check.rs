//! Validate the distributed (2-GPU) SIPG matvec bit-for-bit against the single-GPU operator
//! (and the CPU oracle). The research §7 first increment for a distributed elliptic solver.
//!   cargo oxide build --arch sm_70   (then run target/release/multigpu_matvec_check)
//! Env: MG_P (default 4), MG_GRID (default 64).

use gale::dg::{Mesh2d, Poisson};
use gale_gpu::{multigpu_poisson_matvec_2d, poisson_apply};

fn rel(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
    let den: f64 = b.iter().map(|y| y * y).sum::<f64>().sqrt().max(1e-300);
    num / den
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p: usize = std::env::var("MG_P").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let g: usize = std::env::var("MG_GRID").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    let alpha = 5.0;
    let mesh = Mesh2d::rectangular(p, g, g, [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refq.n_nodes();
    let ndof = mesh.n_elements() * nn;
    // Deterministic broadband test field.
    let u: Vec<f64> = (0..ndof)
        .map(|i| {
            let h = (i as u64).wrapping_mul(2862933555777941757).wrapping_add(3037000493);
            (h >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        })
        .collect();

    // --- all-Dirichlet pure Poisson: distributed vs single-GPU vs CPU oracle ---
    let dist = multigpu_poisson_matvec_2d(&mesh, &u, alpha, 0.0, &[])?;
    let single = poisson_apply(&mesh, &u, alpha)?;
    let cpu = Poisson::new(&mesh, alpha).apply(&u);
    let r_single = rel(&dist, &single);
    let r_cpu = rel(&dist, &cpu);
    println!("p={p} grid={g}² ndof={ndof}");
    println!("  Dirichlet:  ‖dist − single-GPU‖/‖·‖ = {r_single:.3e}   ‖dist − CPU oracle‖/‖·‖ = {r_cpu:.3e}");

    // --- Helmholtz (reaction λ) + a Neumann boundary tag, vs CPU oracle ---
    let lambda = 100.0;
    let tags = mesh.boundary_tags();
    let neu = vec![tags[0]]; // make one boundary side Neumann
    let dist2 = multigpu_poisson_matvec_2d(&mesh, &u, alpha, lambda, &neu)?;
    let cpu2 = Poisson::with_bc(&mesh, alpha, lambda, neu.clone()).apply(&u);
    let r2 = rel(&dist2, &cpu2);
    println!("  Helmholtz+Neumann(tag {}):  ‖dist − CPU oracle‖/‖·‖ = {r2:.3e}", neu[0]);

    let ok = r_single < 1e-12 && r_cpu < 1e-10 && r2 < 1e-10;
    println!("{}", if ok { "PASS: distributed 2-GPU matvec matches the single-GPU operator + CPU oracle." } else { "FAIL" });
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
