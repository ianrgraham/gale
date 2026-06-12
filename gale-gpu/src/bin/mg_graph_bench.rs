//! **CUDA-graph A/B** for the p-MG-PCG V-cycle. The MG solve is launch-bound (~83%
//! `cuLaunchKernel` per Nsight), so the V-cycle's ~500-launch sequence is captured once into a
//! CUDA graph and replayed with a single `cuGraphLaunch` (see `GpuPoissonMg::with_cuda_graph`).
//! This bin is the head-to-head: same hierarchy, graph OFF vs ON, median wall-clock per solve on
//! persistent handles (setup amortized), plus a correctness check that the two solutions agree to
//! solver tolerance (NOT bit-exact: the graph path's coarse CG runs a fixed iteration count to
//! stay host-branch-free, so it may take a different number of coarse iters).
//!
//! Run: cargo oxide run --bin mg-graph-bench   (MG_BIG=1 adds a large-size MG-only sweep)

use gale::dg::{Mesh2d, PMultigrid, Poisson};
use gale_gpu::GpuPoissonMg;
use std::time::Instant;

/// Median of repeated timed `solve()` calls (ms), after a warmup. Returns (ms, iters, solution).
fn time_solve(
    reps: usize,
    mut f: impl FnMut() -> Result<(Vec<f64>, usize), Box<dyn std::error::Error>>,
) -> Result<(f64, usize, Vec<f64>), Box<dyn std::error::Error>> {
    let (x, _) = f()?; // warmup (JIT/caches)
    let mut ts = Vec::with_capacity(reps);
    let mut iters = 0;
    for _ in 0..reps {
        let t0 = Instant::now();
        iters = f()?.1;
        ts.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok((ts[ts.len() / 2], iters, x))
}

/// Deterministic broadband mean-zero source (full spectral content ⇒ CG iters grow with 1/h).
fn broadband(fine: &Mesh2d) -> Vec<f64> {
    let n0 = fine.n_elements() * fine.refq.n_nodes();
    let mut src = vec![0.0; n0];
    for i in 0..n0 {
        let h = (i as u64).wrapping_mul(2862933555777941757).wrapping_add(3037000493);
        src[i] = ((h >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
    }
    let mean = src.iter().sum::<f64>() / n0 as f64;
    src.iter_mut().for_each(|v| *v -= mean);
    src
}

/// Max relative L2 difference between two solutions (agreement to solver tolerance, not bit-exact).
fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
    let den: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
    num / den
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (alpha, tol, maxit, reps) = (5.0, 1e-10, 100_000usize, 5usize);
    let xr = [0.0, 1.0];
    let lambda = 100.0; // velocity Helmholtz reaction λ = 1/(νΔt)
    // Launch overhead dominates at small/medium grids; sweep where the win should be biggest first.
    let configs: [(usize, &[usize]); 4] =
        [(2, &[32, 64, 128]), (4, &[32, 64, 128]), (6, &[32, 64]), (8, &[32, 64])];

    println!("=== CUDA-graph A/B: p-MG-PCG V-cycle, ms/solve (Titan V) ===");
    let hdr = || println!("{:>3} {:>6} {:>10} {:>11} {:>6} {:>11} {:>6} {:>8} {:>10}",
                          "p", "grid", "ndof", "off ms", "it", "on ms", "it", "graph×", "rel diff");

    // ---- Pressure: singular pure-Neumann, deflated ----------------------------------
    println!("\nPRESSURE  (singular pure-Neumann, deflated)");
    hdr();
    for &(p, grids) in &configs {
        for &g in grids {
            let mesh = Mesh2d::rectangular(p, g, g, xr, xr);
            let tags = mesh.boundary_tags();
            let n0 = mesh.n_elements() * mesh.refq.n_nodes();
            let rhs = Poisson::with_bc(&mesh, alpha, 0.0, tags.clone())
                .rhs_mixed(&broadband(&mesh), |_, _| 0.0, |_, _| 0.0);

            let off = GpuPoissonMg::new(PMultigrid::with_bc(p, g, g, xr, xr, alpha, 0.0, tags.clone()))?;
            let on = GpuPoissonMg::new(PMultigrid::with_bc(p, g, g, xr, xr, alpha, 0.0, tags))?
                .with_cuda_graph(true)?;
            let (off_ms, off_it, x_off) = time_solve(reps, || Ok(off.solve(&rhs, tol, maxit)?))?;
            let (on_ms, on_it, x_on) = time_solve(reps, || Ok(on.solve(&rhs, tol, maxit)?))?;
            println!("{:>3} {:>5}² {:>10} {:>11.3} {:>6} {:>11.3} {:>6} {:>7.2}× {:>10.1e}",
                     p, g, n0, off_ms, off_it, on_ms, on_it, off_ms / on_ms, rel_l2(&x_off, &x_on));
        }
    }

    // ---- Velocity Helmholtz: all-Dirichlet, non-singular -----------------------------
    println!("\nVELOCITY  (Helmholtz λ={lambda}, all-Dirichlet)");
    hdr();
    for &(p, grids) in &configs {
        for &g in grids {
            let mesh = Mesh2d::rectangular(p, g, g, xr, xr);
            let n0 = mesh.n_elements() * mesh.refq.n_nodes();
            let rhs = Poisson::with_reaction(&mesh, alpha, lambda).rhs(&broadband(&mesh), |_, _| 0.0);

            let off = GpuPoissonMg::new(PMultigrid::with_reaction(p, g, g, xr, xr, alpha, lambda))?;
            let on = GpuPoissonMg::new(PMultigrid::with_reaction(p, g, g, xr, xr, alpha, lambda))?
                .with_cuda_graph(true)?;
            let (off_ms, off_it, x_off) = time_solve(reps, || Ok(off.solve(&rhs, tol, maxit)?))?;
            let (on_ms, on_it, x_on) = time_solve(reps, || Ok(on.solve(&rhs, tol, maxit)?))?;
            println!("{:>3} {:>5}² {:>10} {:>11.3} {:>6} {:>11.3} {:>6} {:>7.2}× {:>10.1e}",
                     p, g, n0, off_ms, off_it, on_ms, on_it, off_ms / on_ms, rel_l2(&x_off, &x_on));
        }
    }

    // ---- Opt-in LARGE sweep (MG_BIG=1) — confirm the win shrinks as kernels dominate launch cost.
    if std::env::var("MG_BIG").is_ok() {
        println!("\nLARGE p=4 PRESSURE (graph A/B at scale)");
        hdr();
        for &g in &[256usize, 512, 1024] {
            let mesh = Mesh2d::rectangular(4, g, g, xr, xr);
            let tags = mesh.boundary_tags();
            let n0 = mesh.n_elements() * mesh.refq.n_nodes();
            let rhs = Poisson::with_bc(&mesh, alpha, 0.0, tags.clone())
                .rhs_mixed(&broadband(&mesh), |_, _| 0.0, |_, _| 0.0);
            let off = GpuPoissonMg::new(PMultigrid::with_bc(4, g, g, xr, xr, alpha, 0.0, tags.clone()))?;
            let on = GpuPoissonMg::new(PMultigrid::with_bc(4, g, g, xr, xr, alpha, 0.0, tags))?
                .with_cuda_graph(true)?;
            let (off_ms, off_it, x_off) = time_solve(reps, || Ok(off.solve(&rhs, tol, maxit)?))?;
            let (on_ms, on_it, x_on) = time_solve(reps, || Ok(on.solve(&rhs, tol, maxit)?))?;
            println!("{:>3} {:>5}² {:>10} {:>11.3} {:>6} {:>11.3} {:>6} {:>7.2}× {:>10.1e}",
                     4, g, n0, off_ms, off_it, on_ms, on_it, off_ms / on_ms, rel_l2(&x_off, &x_on));
        }
    }

    println!("\n(graph× = off ms / on ms — the launch-overhead win. rel diff = ‖x_on−x_off‖/‖x_off‖,\n \
              should be ~solver tol since the graph path's coarse CG runs a fixed iter count.\n \
              Win should be largest at small grids/high p, shrinking as kernel work hides launch cost.)");
    Ok(())
}
