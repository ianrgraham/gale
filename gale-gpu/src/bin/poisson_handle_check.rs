//! Validates the **persistent solver handle** [`gale_gpu::GpuPoisson`] (P4): correctness
//! vs the one-shot solvers, and that repeated solves pay the ~0.3 s setup ONCE.
//!
//! Three solve modes are checked against their one-shot equivalents:
//!   1. Dirichlet velocity Helmholtz  (`neumann_tags = &[]`, reaction = λ)   vs helmholtz_cg_solve
//!   2. per-region (one Neumann tag)  (reaction = λ)                          vs helmholtz_cg_solve_tags
//!   3. pure-Neumann pressure, deflated (all tags, reaction = 0)              vs pressure_cg_solve
//! Then it times N handle solves vs N one-shot solves to show the setup amortization.
//!
//! Run: cargo oxide run --bin poisson-handle-check

use gale::dg::{Mesh2d, Poisson};
use gale_gpu::{helmholtz_cg_solve, helmholtz_cg_solve_tags, pressure_cg_solve, GpuPoisson};
use std::time::Instant;

fn rel(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum::<f64>().max(1e-300);
    (num / den).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== gale_gpu::GpuPoisson persistent handle vs one-shot solvers ===\n");
    let (alpha, lambda, tol, maxit) = (5.0, 100.0, 1e-10, 20000);
    let mesh = Mesh2d::rectangular(4, 24, 16, [0.0, 3.0], [0.0, 2.0]);
    let ndof = mesh.n_elements() * mesh.refq.n_nodes();

    let t_build = Instant::now();
    let handle = GpuPoisson::new(&mesh, alpha)?;
    let build_ms = t_build.elapsed().as_secs_f64() * 1e3;
    println!("handle build (ctx+module+upload) = {build_ms:.1} ms\n");

    let pois = Poisson::with_reaction(&mesh, alpha, lambda);
    let rhs = pois.rhs(&(0..ndof).map(|i| ((i % 7) as f64 - 3.0) * 0.1).collect::<Vec<_>>(), |_, _| 0.0);

    let mut ok = true;
    // 1. Dirichlet velocity Helmholtz.
    let (xh, _) = handle.solve(&rhs, lambda, &[], false, tol, maxit)?;
    let (xr, _) = helmholtz_cg_solve(&mesh, &rhs, alpha, lambda, tol, maxit)?;
    let r1 = rel(&xh, &xr);
    println!("1. Dirichlet Helmholtz   : rel vs helmholtz_cg_solve      = {r1:.3e}  {}", pass(r1, &mut ok));

    // 2. Per-region (tag 1 = right boundary Neumann).
    let (xh, _) = handle.solve(&rhs, lambda, &[1], false, tol, maxit)?;
    let (xr, _) = helmholtz_cg_solve_tags(&mesh, &rhs, alpha, lambda, &[1], tol, maxit)?;
    let r2 = rel(&xh, &xr);
    println!("2. Per-region (tag 1 Neu): rel vs helmholtz_cg_solve_tags = {r2:.3e}  {}", pass(r2, &mut ok));

    // 3. Pure-Neumann pressure (deflated). Both return mean-zero solutions.
    let pp = Poisson::with_bc(&mesh, alpha, 0.0, mesh.boundary_tags());
    let rhsp = pp.rhs_mixed(&(0..ndof).map(|i| ((i % 5) as f64 - 2.0) * 0.1).collect::<Vec<_>>(), |_, _| 0.0, |_, _| 0.0);
    let (xh, _) = handle.solve(&rhsp, 0.0, &mesh.boundary_tags(), true, tol, maxit)?;
    let (xr, _) = pressure_cg_solve(&mesh, &rhsp, alpha, tol, maxit)?;
    let r3 = rel(&xh, &xr);
    println!("3. Pressure (deflated)   : rel vs pressure_cg_solve       = {r3:.3e}  {}", pass(r3, &mut ok));

    // Amortization: N handle solves (setup once) vs N one-shot solves (setup each).
    let n = 10;
    let t = Instant::now();
    for _ in 0..n {
        let _ = handle.solve(&rhs, lambda, &[], false, tol, maxit)?;
    }
    let handle_ms = t.elapsed().as_secs_f64() * 1e3 / n as f64;
    let t = Instant::now();
    for _ in 0..n {
        let _ = helmholtz_cg_solve(&mesh, &rhs, alpha, lambda, tol, maxit)?;
    }
    let oneshot_ms = t.elapsed().as_secs_f64() * 1e3 / n as f64;
    println!(
        "\namortization ({n} solves): handle {handle_ms:.1} ms/solve  vs  one-shot {oneshot_ms:.1} ms/solve  ⇒  {:.1}× faster",
        oneshot_ms / handle_ms
    );

    if ok {
        println!("\nPASS: GpuPoisson handle matches the one-shot solvers and amortizes setup.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GpuPoisson handle mismatch.");
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
