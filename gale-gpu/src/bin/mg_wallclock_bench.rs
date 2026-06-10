//! **Wall-clock** profile of the p-MG-PCG default vs the persistent-CG fallback for the
//! two flow elliptic solves, across mesh refinement. The PCG validation bins report only
//! the *iteration-count* win (pressure 136× fewer iters at 64², velocity 25×); this bin
//! closes the gap by timing the actual `solve()` wall-clock on persistent handles (setup
//! amortized), so we can see whether the iteration win survives the MG V-cycle's
//! host-readback dots (the per-dot sync the on-device-scalar CG avoids) — i.e. whether
//! MG-PCG is the right *wall-clock* default, and where the next bottleneck is.
//!
//! Run: cargo oxide run --bin mg-wallclock-bench

use gale::dg::{Mesh2d, PMultigrid, Poisson};
use gale_gpu::{GpuPoisson, GpuPoissonMg};
use std::time::Instant;

/// Median of repeated timed `solve()` calls (ms), after a warmup. Returns (ms, iters).
fn time_solve(reps: usize, mut f: impl FnMut() -> Result<usize, Box<dyn std::error::Error>>)
    -> Result<(f64, usize), Box<dyn std::error::Error>>
{
    let _ = f()?; // warmup (JIT/caches)
    let mut ts = Vec::with_capacity(reps);
    let mut iters = 0;
    for _ in 0..reps {
        let t0 = Instant::now();
        iters = f()?;
        ts.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok((ts[ts.len() / 2], iters))
}

/// Deterministic broadband mean-zero source over the mesh (full spectral content ⇒ CG
/// iterations grow with 1/h, exposing the multigrid win — a single eigenmode is trivial).
fn broadband(fine: &Mesh2d) -> Vec<f64> {
    let nn = fine.refq.n_nodes();
    let n0 = fine.n_elements() * nn;
    let mut src = vec![0.0; n0];
    for i in 0..n0 {
        let h = (i as u64).wrapping_mul(2862933555777941757).wrapping_add(3037000493);
        src[i] = ((h >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
    }
    let mean = src.iter().sum::<f64>() / n0 as f64;
    src.iter_mut().for_each(|v| *v -= mean);
    src
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (alpha, tol, maxit, reps) = (5.0, 1e-10, 100_000usize, 3usize);
    let xr = [0.0, 1.0];
    let lambda = 100.0; // velocity Helmholtz reaction λ = 1/(νΔt), the ve-check value
    // (order, grids): large grids at low order (cheap CG reference); high order capped at 64²
    // since the unpreconditioned-CG reference grows expensive (and that's the regime MG wins).
    let configs: [(usize, &[usize]); 4] =
        [(2, &[32, 64, 128]), (4, &[32, 64, 128]), (6, &[32, 64]), (8, &[32, 64])];

    println!("=== Wall-clock: p-MG-PCG vs persistent CG, ms/solve, p-sweep (Titan V) ===");
    let hdr = || println!("{:>3} {:>6} {:>9} {:>11} {:>7} {:>11} {:>7} {:>9} {:>9}",
                          "p", "grid", "ndof", "CG ms", "it", "MG ms", "it", "wall×", "iter×");

    // ---- Pressure: singular pure-Neumann, deflated ----------------------------------
    println!("\nPRESSURE  (singular pure-Neumann, deflated)");
    hdr();
    for &(p, grids) in &configs {
        for &g in grids {
            let mesh = Mesh2d::rectangular(p, g, g, xr, xr);
            let tags = mesh.boundary_tags();
            let n0 = mesh.n_elements() * mesh.refq.n_nodes();
            let rhs = Poisson::with_bc(&mesh, alpha, 0.0, tags.clone()).rhs_mixed(&broadband(&mesh), |_, _| 0.0, |_, _| 0.0);

            let cg = GpuPoisson::new(&mesh, alpha)?;
            let mg = GpuPoissonMg::new(PMultigrid::with_bc(p, g, g, xr, xr, alpha, 0.0, tags.clone()))?;
            let (cg_ms, cg_it) = time_solve(reps, || Ok(cg.solve(&rhs, 0.0, &tags, true, tol, maxit)?.1))?;
            let (mg_ms, mg_it) = time_solve(reps, || Ok(mg.solve(&rhs, tol, maxit)?.1))?;
            println!("{:>3} {:>5}² {:>9} {:>11.2} {:>7} {:>11.2} {:>7} {:>8.1}× {:>8.1}×",
                     p, g, n0, cg_ms, cg_it, mg_ms, mg_it, cg_ms / mg_ms, cg_it as f64 / mg_it.max(1) as f64);
        }
    }

    // ---- Velocity Helmholtz: all-Dirichlet, non-singular -----------------------------
    println!("\nVELOCITY  (Helmholtz λ={lambda}, all-Dirichlet)");
    hdr();
    for &(p, grids) in &configs {
        for &g in grids {
            let mesh = Mesh2d::rectangular(p, g, g, xr, xr);
            let n0 = mesh.n_elements() * mesh.refq.n_nodes();
            // A smooth RHS for the (mass-dominated) Helmholtz, assembled with all-Dirichlet data.
            let rhs = Poisson::with_reaction(&mesh, alpha, lambda).rhs(&broadband(&mesh), |_, _| 0.0);

            let cg = GpuPoisson::new(&mesh, alpha)?;
            let mg = GpuPoissonMg::new(PMultigrid::with_reaction(p, g, g, xr, xr, alpha, lambda))?;
            let (cg_ms, cg_it) = time_solve(reps, || Ok(cg.solve(&rhs, lambda, &[], false, tol, maxit)?.1))?;
            let (mg_ms, mg_it) = time_solve(reps, || Ok(mg.solve(&rhs, tol, maxit)?.1))?;
            println!("{:>3} {:>5}² {:>9} {:>11.2} {:>7} {:>11.2} {:>7} {:>8.1}× {:>8.1}×",
                     p, g, n0, cg_ms, cg_it, mg_ms, mg_it, cg_ms / mg_ms, cg_it as f64 / mg_it.max(1) as f64);
        }
    }

    // ---- Opt-in LARGE sweep (MG-only) — find the launch-bound → bandwidth-bound crossover.
    // Set MG_BIG=1. CG is omitted: at these sizes it needs ~1e4–5e4 iters (minutes/solve), which
    // is itself the point (it doesn't scale). `~mem` is the working-set estimate ≈ 35·ndof·f64
    // (≈35 ndof-sized device vectors across MG levels + scratch + PCG); validate vs nvidia-smi.
    if std::env::var("MG_BIG").is_ok() {
        println!("\nLARGE p=4 (MG-only; CG impractical here)  — GPU is 12 GB/Titan V");
        println!("{:>6} {:>11} {:>11} {:>7} {:>10}", "grid", "ndof", "MG ms", "it", "~mem MB");
        for &g in &[256usize, 512, 1024] {
            let mesh = Mesh2d::rectangular(4, g, g, xr, xr);
            let tags = mesh.boundary_tags();
            let n0 = mesh.n_elements() * mesh.refq.n_nodes();
            let rhs = Poisson::with_bc(&mesh, alpha, 0.0, tags.clone()).rhs_mixed(&broadband(&mesh), |_, _| 0.0, |_, _| 0.0);
            let mg = GpuPoissonMg::new(PMultigrid::with_bc(4, g, g, xr, xr, alpha, 0.0, tags))?;
            let (mg_ms, mg_it) = time_solve(reps, || Ok(mg.solve(&rhs, tol, maxit)?.1))?;
            let mem_mb = 35.0 * n0 as f64 * 8.0 / 1.0e6;
            println!("{:>5}² {:>11} {:>11.2} {:>7} {:>10.0}", g, n0, mg_ms, mg_it, mem_mb);
        }
    }

    println!("\n(wall× = CG ms / MG ms — the real speedup; iter× = iteration-count ratio.\n \
              MG iters should stay ~flat across p AND grid if the smoother is p-robust.\n \
              MG_BIG=1 adds a large-size MG-only sweep for the launch/bandwidth crossover.)");
    Ok(())
}
