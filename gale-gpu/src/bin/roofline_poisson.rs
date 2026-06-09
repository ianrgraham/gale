//! Roofline microbenchmark for the 2D GPU SIPG-Poisson kernels (the CG bottleneck).
//!
//! For each polynomial order it times every kernel in isolation (CUDA events, device
//! side) and does the ideal-traffic byte/FLOP accounting, placing each on the **Titan V**
//! roofline (FP64 peak ≈ 6.9 TFLOP/s, HBM2 ≈ 652.8 GB/s → ridge AI ≈ 10.6 FLOP/byte).
//! It then times a full `helmholtz_cg_solve` wall-clock and compares to the summed
//! device-event time of the kernels a CG iteration issues — the gap is the per-iteration
//! host-sync + launch overhead (the dot-product `to_host_vec` syncs).
//!
//! Numbers are *ideal* traffic (each global array counted once); achieved GB/s below peak
//! flags either under-utilization (tiny blocks / single-block reductions) or extra traffic.
//!
//! Run: cargo oxide run --bin roofline-poisson

use gale::dg::{Mesh2d, Poisson};
use gale_gpu::{bench_poisson_kernels, helmholtz_cg_solve};
use std::time::Instant;

// Titan V (Volta GV100) FP64 peak and HBM2 bandwidth.
const PEAK_GFLOPS: f64 = 6900.0;
const PEAK_GBPS: f64 = 652.8;

/// Ideal traffic (bytes) and FLOPs per launch for a kernel over `ndof` nodes at `n1`.
fn cost(name: &str, ndof: usize, n1: u32) -> (f64, f64) {
    let nd = ndof as f64;
    let n1 = n1 as f64;
    match name {
        // gradient: read u,rx,ry,sx,sy,d (~6) + write gx,gy (2) ≈ 8 doubles/node;
        // 2 contractions of length n1 (4 flop each) + 6 metric flop.
        "gradient" => (nd * 8.0 * 8.0, nd * (4.0 * n1 + 6.0)),
        // operator: read u,gx,gy,rx,ry,sx,sy,jw,d (~9) + write out (1) ≈ 10 doubles/node;
        // final contraction 4·n1 + PR/PS setup + face work (~15 amortized).
        "operator" => (nd * 10.0 * 8.0, nd * (4.0 * n1 + 15.0)),
        // axpy/xpby: read x,y + write y = 3 doubles/node; 1 mul + 1 add.
        "axpy" | "xpby" => (nd * 3.0 * 8.0, nd * 2.0),
        // dot: read a,b = 2 doubles/node; 1 mul + 1 add.
        "dot_partial" => (nd * 2.0 * 8.0, nd * 2.0),
        _ => (0.0, 0.0),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Roofline: 2D GPU SIPG-Poisson kernels (Titan V: {PEAK_GFLOPS:.0} GFLOP/s FP64, {PEAK_GBPS:.1} GB/s) ===");
    println!("    ridge-point AI = {:.2} FLOP/byte  (below ⇒ memory-bound)\n", PEAK_GFLOPS / PEAK_GBPS);

    let alpha = 5.0;
    let reps = 200;
    let grid = 64; // 64×64 elements; ndof = (p+1)²·4096

    for &p in &[2usize, 4, 6, 8] {
        let mesh = Mesh2d::rectangular(p, grid, grid, [0.0, 1.0], [0.0, 1.0]);
        let b = bench_poisson_kernels(&mesh, alpha, reps)?;
        println!(
            "--- p={p}  ({} elems, nn={}, ndof={}) ---",
            b.ne, b.nn, b.ndof
        );
        println!(
            "  {:<12} {:>9} {:>10} {:>10} {:>8} {:>7}",
            "kernel", "us/call", "GFLOP/s", "GB/s", "%BW", "AI"
        );
        let mut k_ms = std::collections::HashMap::new();
        for k in &b.kernels {
            let (bytes, flops) = cost(k.name, b.ndof, b.n1);
            let s = k.ms * 1e-3;
            let gflops = flops / s / 1e9;
            let gbps = bytes / s / 1e9;
            let ai = flops / bytes;
            println!(
                "  {:<12} {:>9.2} {:>10.1} {:>10.1} {:>7.1}% {:>7.2}",
                k.name, k.ms * 1e3, gflops, gbps, 100.0 * gbps / PEAK_GBPS, ai
            );
            k_ms.insert(k.name.to_string(), k.ms);
        }

        // Full-solve overhead: a CG iteration issues gradient+operator + 2 dots
        // (pap, rs_new) + 2 axpy + 1 xpby. Compare predicted device time to wall clock.
        let lambda = 100.0;
        let pois = Poisson::with_reaction(&mesh, alpha, lambda);
        let rhs = pois.rhs(&vec![1.0; b.ndof], |_, _| 0.0);
        // warmup + measured solve
        let _ = helmholtz_cg_solve(&mesh, &rhs, alpha, lambda, 1e-8, 5000)?;
        let t0 = Instant::now();
        let (_x, iters) = helmholtz_cg_solve(&mesh, &rhs, alpha, lambda, 1e-8, 5000)?;
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
        let g = |n: &str| k_ms.get(n).copied().unwrap_or(0.0);
        let dev_per_iter = g("gradient") + g("operator") + 2.0 * g("dot_partial") + 2.0 * g("axpy") + g("xpby");
        let wall_per_iter = wall_ms / iters as f64;
        let overhead = wall_per_iter - dev_per_iter;
        println!(
            "  CG solve: {iters} iters, {wall_ms:.2} ms wall ({:.1} us/iter)",
            wall_per_iter * 1e3
        );
        println!(
            "    device kernels/iter = {:.1} us;  host sync+launch overhead = {:.1} us/iter ({:.0}% of iter)\n",
            dev_per_iter * 1e3,
            overhead * 1e3,
            100.0 * overhead / wall_per_iter
        );
    }

    Ok(())
}
