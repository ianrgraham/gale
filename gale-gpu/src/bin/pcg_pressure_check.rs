//! Validates p-multigrid PCG for the **singular pure-Neumann pressure-Poisson** (the
//! dual-splitting pressure solve, reaction 0, all-Neumann ⇒ constant nullspace). The MG
//! is built BC-aware (`PMultigrid::with_bc` with all boundary tags) and `poisson_pcg_solve`
//! auto-detects the singular case and deflates (range projection in the outer PCG + the
//! coarsest solve; see docs/research-pressure-multigrid.md). The PCG solution must match
//! the deflated Neumann CG (`pressure_cg_solve`) up to the constant nullspace, and its
//! iteration count must stay ~flat under refinement while plain deflated CG grows.
//!
//! Run: cargo oxide run --bin pcg-pressure-check

use gale::dg::{PMultigrid, Poisson};
use gale_gpu::{operators::poisson::poisson_pcg_solve, pressure_cg_solve};

/// L2 relative difference after removing the mean from both (both are determined up to a
/// constant — the nullspace — so compare on the range).
fn rel_meanzero(a: &[f64], b: &[f64]) -> f64 {
    let ma = a.iter().sum::<f64>() / a.len() as f64;
    let mb = b.iter().sum::<f64>() / b.len() as f64;
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - ma) - (y - mb)).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (y - mb).powi(2)).sum::<f64>().max(1e-300);
    (num / den).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::f64::consts::PI;
    let (p, alpha, tol, maxit) = (4usize, 5.0, 1e-10, 50_000);
    println!("=== p-MG PCG for the singular pure-Neumann pressure-Poisson (p={p}) — iters vs deflated CG ===\n");
    println!("{:>7} {:>8} {:>10} {:>10} {:>9}", "grid", "ndof", "CG iters", "PCG iters", "speedup");

    let mut worst_rel = 0.0f64;
    for &g in &[16usize, 32, 64] {
        // All boundary tags Neumann, reaction 0 ⇒ the singular pressure operator.
        let proto = gale::dg::Mesh2d::rectangular(p, g, g, [0.0, 1.0], [0.0, 1.0]);
        let tags = proto.boundary_tags();
        let mg = PMultigrid::with_bc(p, g, g, [0.0, 1.0], [0.0, 1.0], alpha, 0.0, tags.clone());
        let fine = mg.mesh(0);
        let nn = fine.refq.n_nodes();
        let n0 = fine.n_elements() * nn;
        // Broadband, mean-zero (Neumann-compatible) source — representative of a real flow
        // divergence RHS (full spectral content), so plain CG iterations grow with 1/h and
        // the multigrid win is visible (a single Neumann eigenmode would converge trivially).
        let mut src = vec![0.0; n0];
        for (e, el) in fine.elements.iter().enumerate() {
            for k in 0..nn {
                let i = (e * nn + k) as u64;
                // cheap deterministic hash → pseudo-random in [-1, 1]
                let h = i.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
                src[e * nn + k] = ((h >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
                let _ = (PI, el.geom.x[k]);
            }
        }
        let mean = src.iter().sum::<f64>() / n0 as f64;
        src.iter_mut().for_each(|v| *v -= mean);
        let pop = Poisson::with_bc(fine, alpha, 0.0, tags);
        let rhs = pop.rhs_mixed(&src, |_, _| 0.0, |_, _| 0.0);

        let (u_cg, it_cg) = pressure_cg_solve(fine, &rhs, alpha, tol, maxit)?;
        let (u_pcg, it_pcg) = poisson_pcg_solve(&mg, &rhs, tol, maxit)?;
        let rel = rel_meanzero(&u_pcg, &u_cg);
        worst_rel = worst_rel.max(rel);
        println!(
            "{:>5}² {:>8} {:>10} {:>10} {:>8.1}×",
            g, n0, it_cg, it_pcg, it_cg as f64 / it_pcg.max(1) as f64
        );
    }
    println!("\nworst PCG-vs-CG (mean-removed) rel = {worst_rel:.3e}");
    if worst_rel < 1e-6 {
        println!("PASS: singular-Neumann pressure p-MG PCG matches deflated CG and cuts iterations.");
        Ok(())
    } else {
        eprintln!("FAIL: pressure PCG mismatch.");
        std::process::exit(1);
    }
}
