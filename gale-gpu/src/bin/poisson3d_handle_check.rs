//! Validates the **3D persistent solver handle** [`gale_gpu::GpuPoisson3d`] (P4): bit-for-bit
//! vs the one-shot 3D solvers, and that repeated solves amortize the ~0.3 s setup.
//!
//! Three solve modes vs their one-shot equivalents:
//!   1. Dirichlet velocity Helmholtz  (`neumann_tags = &[]`, reaction = λ)  vs helmholtz3d_cg_solve
//!   2. per-region (one Neumann tag)  (reaction = λ)                         vs helmholtz3d_cg_solve_tags
//!   3. pure-Neumann pressure, deflated (all tags, reaction = 0)             vs pressure3d_cg_solve
//!
//! Run: cargo oxide run --bin poisson3d-handle-check

use gale::dg::{Mesh3d, Poisson3d};
use gale_gpu::{
    helmholtz3d_cg_solve, helmholtz3d_cg_solve_tags, pressure3d_cg_solve, GpuPoisson3d,
};
use std::time::Instant;

fn rel(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum::<f64>().max(1e-300);
    (num / den).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== gale_gpu::GpuPoisson3d persistent handle vs one-shot solvers ===\n");
    let (alpha, lambda, tol, maxit) = (5.0, 100.0, 1e-10, 20000);
    let mesh = Mesh3d::rectangular(3, 4, 4, 4, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
    let ndof = mesh.n_elements() * mesh.refh.n_nodes();

    let t_build = Instant::now();
    let handle = GpuPoisson3d::new(&mesh, alpha)?;
    println!("handle build (ctx+module+upload) = {:.1} ms\n", t_build.elapsed().as_secs_f64() * 1e3);

    let pois = Poisson3d::with_reaction(&mesh, alpha, lambda);
    let f: Vec<f64> = (0..ndof).map(|i| ((i % 7) as f64 - 3.0) * 0.1).collect();
    let rhs = pois.rhs(&f, |_, _, _| 0.0);

    let mut ok = true;
    let (xh, _) = handle.solve(&rhs, lambda, &[], false, tol, maxit)?;
    let (xr, _) = helmholtz3d_cg_solve(&mesh, &rhs, alpha, lambda, tol, maxit)?;
    let r1 = rel(&xh, &xr);
    println!("1. Dirichlet Helmholtz   : rel vs helmholtz3d_cg_solve      = {r1:.3e}  {}", pass(r1, &mut ok));

    let (xh, _) = handle.solve(&rhs, lambda, &[5], false, tol, maxit)?;
    let (xr, _) = helmholtz3d_cg_solve_tags(&mesh, &rhs, alpha, lambda, &[5], tol, maxit)?;
    let r2 = rel(&xh, &xr);
    println!("2. Per-region (tag 5 Neu): rel vs helmholtz3d_cg_solve_tags = {r2:.3e}  {}", pass(r2, &mut ok));

    let pp = Poisson3d::with_bc(&mesh, alpha, 0.0, mesh.boundary_tags());
    let fp: Vec<f64> = (0..ndof).map(|i| ((i % 5) as f64 - 2.0) * 0.1).collect();
    let rhsp = pp.rhs_mixed(&fp, |_, _, _| 0.0, |_, _, _| 0.0);
    let (xh, _) = handle.solve(&rhsp, 0.0, &mesh.boundary_tags(), true, tol, maxit)?;
    let (xr, _) = pressure3d_cg_solve(&mesh, &rhsp, alpha, tol, maxit)?;
    let r3 = rel(&xh, &xr);
    println!("3. Pressure (deflated)   : rel vs pressure3d_cg_solve       = {r3:.3e}  {}", pass(r3, &mut ok));

    let n = 8;
    let t = Instant::now();
    for _ in 0..n {
        let _ = handle.solve(&rhs, lambda, &[], false, tol, maxit)?;
    }
    let handle_ms = t.elapsed().as_secs_f64() * 1e3 / n as f64;
    let t = Instant::now();
    for _ in 0..n {
        let _ = helmholtz3d_cg_solve(&mesh, &rhs, alpha, lambda, tol, maxit)?;
    }
    let oneshot_ms = t.elapsed().as_secs_f64() * 1e3 / n as f64;
    println!(
        "\namortization ({n} solves): handle {handle_ms:.1} ms/solve  vs  one-shot {oneshot_ms:.1} ms/solve  ⇒  {:.1}× faster",
        oneshot_ms / handle_ms
    );

    if ok {
        println!("\nPASS: GpuPoisson3d handle matches the one-shot solvers and amortizes setup.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GpuPoisson3d handle mismatch.");
        std::process::exit(1);
    }
}

fn pass(r: f64, ok: &mut bool) -> &'static str {
    if r < 1e-9 {
        "OK"
    } else {
        *ok = false;
        "FAIL"
    }
}
