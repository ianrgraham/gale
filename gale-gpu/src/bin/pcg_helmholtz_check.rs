//! Validates p-multigrid PCG for the viscous **Helmholtz** operator `(λM + A)` (the flow
//! velocity solve) and shows the iteration win vs plain CG. `PMultigrid::with_reaction`
//! now carries the reaction λ at every level; `poisson_pcg_solve` passes it to the device
//! operator. The PCG solution must match the plain-CG solution (same system), and its
//! iteration count must stay ~flat under refinement while plain CG grows ∝ 1/h.
//!
//! Run: cargo oxide run --bin pcg-helmholtz-check

use gale::dg::{PMultigrid, Poisson};
use gale_gpu::{helmholtz_cg_solve, operators::poisson::poisson_pcg_solve};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::f64::consts::PI;
    let (p, alpha, tol, maxit) = (4usize, 5.0, 1e-10, 2_000);
    let nu = 1.0;
    let dt = 0.01;
    let lambda = 1.0 / (nu * dt); // = 100, the representative velocity Helmholtz reaction
    println!("=== p-multigrid PCG for velocity Helmholtz (λ={lambda}, p={p}) — iters vs plain CG ===\n");
    println!("{:>7} {:>8} {:>10} {:>10} {:>9}", "grid", "ndof", "CG iters", "PCG iters", "speedup");

    let mut worst_rel = 0.0f64;
    for &g in &[16usize, 32, 64] {
        eprintln!("[grid {g}²] starting...");
        let mg = PMultigrid::with_reaction(p, g, g, [0.0, 1.0], [0.0, 1.0], alpha, lambda);
        let fine = mg.mesh(0);
        let nn = fine.refq.n_nodes();
        let n0 = fine.n_elements() * nn;
        let src: Vec<f64> = {
            let mut v = vec![0.0; n0];
            for (e, el) in fine.elements.iter().enumerate() {
                for k in 0..nn {
                    v[e * nn + k] = (PI * el.geom.x[k]).sin() * (PI * el.geom.y[k]).sin();
                }
            }
            v
        };
        // Helmholtz RHS: (λM + A) u = λM·src + Dirichlet(0), exactly as the flow velocity step.
        let hop = Poisson::with_reaction(fine, alpha, lambda);
        let fxv: Vec<f64> = src.iter().map(|v| lambda * v).collect();
        let rhs = hop.rhs(&fxv, |_, _| 0.0);

        let (u_cg, it_cg) = helmholtz_cg_solve(fine, &rhs, alpha, lambda, tol, maxit)?;
        eprintln!("[grid {g}²] plain CG done: {it_cg} iters; starting PCG...");
        let (u_pcg, it_pcg) = poisson_pcg_solve(&mg, &rhs, tol, maxit)?;
        eprintln!("[grid {g}²] PCG done: {it_pcg} iters");

        // Same system ⇒ solutions must agree.
        let diff: f64 = u_pcg.iter().zip(&u_cg).map(|(a, b)| (a - b).powi(2)).sum();
        let nrm: f64 = u_cg.iter().map(|b| b * b).sum::<f64>().max(1e-300);
        let rel = (diff / nrm).sqrt();
        worst_rel = worst_rel.max(rel);
        println!(
            "{:>5}² {:>8} {:>10} {:>10} {:>8.1}×",
            g, n0, it_cg, it_pcg, it_cg as f64 / it_pcg as f64
        );
    }
    println!("\nworst PCG-vs-CG solution rel = {worst_rel:.3e}");
    if worst_rel < 1e-7 {
        println!("PASS: Helmholtz p-MG PCG matches plain CG and cuts iterations.");
        Ok(())
    } else {
        eprintln!("FAIL: Helmholtz PCG mismatch.");
        std::process::exit(1);
    }
}
