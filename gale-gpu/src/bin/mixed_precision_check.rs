//! Validate + measure the mixed-precision V-cycle (FP32 gradient intermediates) vs the FP64 path.
//! The outer CG + reductions + deflation stay FP64, so the SOLUTION must still match the FP64 solve
//! to ~the outer tolerance (the f32 gx/gy perturbation is absorbed by the FP64 outer iteration);
//! iteration count may rise slightly (the preconditioner is inexact — expected, per the research).
//! Reports the realized matrix-free speedup of the gx/gy-f32 increment.
//!
//! Run: cargo oxide run --bin mixed-precision-check

use gale::dg::{Mesh2d, PMultigrid, Poisson};
use gale_gpu::GpuPoissonMg;
use std::time::Instant;

fn broadband(fine: &Mesh2d) -> Vec<f64> {
    let n0 = fine.n_elements() * fine.refq.n_nodes();
    let mut s: Vec<f64> = (0..n0)
        .map(|i| {
            let h = (i as u64).wrapping_mul(2862933555777941757).wrapping_add(3037000493);
            ((h >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect();
    let m = s.iter().sum::<f64>() / n0 as f64;
    s.iter_mut().for_each(|v| *v -= m);
    s
}

fn time_solve(reps: usize, mut f: impl FnMut() -> Result<(Vec<f64>, usize), Box<dyn std::error::Error>>)
    -> Result<(f64, usize, Vec<f64>), Box<dyn std::error::Error>>
{
    let (_w, _) = f()?;
    let mut ts = Vec::new();
    let (mut it, mut sol) = (0, Vec::new());
    for _ in 0..reps {
        let t0 = Instant::now();
        let (s, n) = f()?;
        ts.push(t0.elapsed().as_secs_f64() * 1e3);
        it = n;
        sol = s;
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok((ts[ts.len() / 2], it, sol))
}

fn rel(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum::<f64>().max(1e-300);
    (num / den).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (p, alpha, tol, maxit, reps, xr) = (4usize, 5.0, 1e-10, 100_000usize, 2usize, [0.0, 1.0]);
    let lambda = 100.0;
    println!("=== Mixed-precision V-cycle (FP32 gx/gy) vs FP64 — accuracy + speedup (p={p}) ===\n");
    println!("{:>9} {:>6} {:>11} {:>16} {:>9} {:>9} {:>8}", "op", "grid", "ndof", "‖mix−f64‖/‖f64‖", "f64 ms", "mix ms", "speedup");

    for &g in &[16usize] {
        let mesh = Mesh2d::rectangular(p, g, g, xr, xr);
        let n0 = mesh.n_elements() * mesh.refq.n_nodes();
        let tags = mesh.boundary_tags();
        let prhs = Poisson::with_bc(&mesh, alpha, 0.0, tags.clone()).rhs_mixed(&broadband(&mesh), |_, _| 0.0, |_, _| 0.0);
        let vrhs = Poisson::with_reaction(&mesh, alpha, lambda).rhs(&broadband(&mesh), |_, _| 0.0);
        // One host hierarchy build per (op, size): build the f64 handle, time it, then CONVERT to
        // mixed (reusing the same uploaded MgConst — no rebuild) and time that.
        for (name, rhs, build) in [
            ("pressure", &prhs, PMultigrid::with_bc(p, g, g, xr, xr, alpha, 0.0, tags.clone())),
            ("velocity", &vrhs, PMultigrid::with_reaction(p, g, g, xr, xr, alpha, lambda)),
        ] {
            let h = GpuPoissonMg::new(build)?;
            let (ms64, it64, sol64) = time_solve(reps, || h.solve(rhs, tol, maxit))?;
            let hm = h.with_mixed_precision(true); // reuses the hierarchy
            let (msmix, itmix, solmix) = time_solve(reps, || hm.solve(rhs, tol, maxit))?;
            println!("{:>9} {:>5}² {:>11} {:>16.3e} {:>7.2}({}) {:>7.2}({}) {:>7.2}×",
                     name, g, n0, rel(&solmix, &sol64), ms64, it64, msmix, itmix, ms64 / msmix);
        }
    }
    println!("\n(‖mix−f64‖ should be ~the solve tol — the FP64 outer CG absorbs the FP32 preconditioner.\n \
              iters in parens; speedup is the realized gx/gy-f32 matvec win.)");
    Ok(())
}
