//! Minimal single-solve target for **Nsight profiling** (nsys timeline / ncu per-kernel). One
//! warmup + one profiled MG-PCG solve of the pressure (or velocity) operator, so a trace is a
//! clean single solve rather than the full `mg-wallclock-bench` sweep. Configure via env:
//!   `MG_GRID` (default 64), `MG_P` (4), `MG_OP` (`pressure` | `velocity`).
//!
//! Timeline + per-kernel/API (sync) totals — no special permissions:
//!   cargo oxide build mg-profile
//!   nsys profile -o /tmp/mg target/release/mg-profile
//!   nsys stats --report cuda_gpu_kern_sum,cuda_api_sum /tmp/mg.nsys-rep
//!
//! Per-kernel HW counters / roofline (needs the profiling-counter permission — run as root, or
//! set the driver param `NVreg_RestrictProfilingToAdminUsers=0`). Bound the work with `-c/-s`
//! (the warmup solve runs first) and a kernel filter, else ncu replays every launch:
//!   sudo ncu --set full -k 'regex:gradient|operator|dot_partial' -c 12 -o /tmp/mg_ncu target/release/mg-profile

use gale::dg::{Mesh2d, PMultigrid, Poisson};
use gale_gpu::GpuPoissonMg;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// Deterministic broadband mean-zero source (full spectral content).
fn broadband(n: usize) -> Vec<f64> {
    let mut s: Vec<f64> = (0..n)
        .map(|i| {
            let h = (i as u64).wrapping_mul(2862933555777941757).wrapping_add(3037000493);
            ((h >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect();
    let m = s.iter().sum::<f64>() / n as f64;
    s.iter_mut().for_each(|v| *v -= m);
    s
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (p, g) = (env_usize("MG_P", 4), env_usize("MG_GRID", 64));
    let op = std::env::var("MG_OP").unwrap_or_else(|_| "pressure".into());
    let (alpha, tol, maxit, xr) = (5.0, 1e-10, 100_000usize, [0.0, 1.0]);
    let lambda = 100.0;

    let mesh = Mesh2d::rectangular(p, g, g, xr, xr);
    let n0 = mesh.n_elements() * mesh.refq.n_nodes();
    let src = broadband(n0);
    // MG_GRAPH=1 captures the V-cycle into a CUDA graph and replays it (launch-bound remediation).
    let use_graph = std::env::var("MG_GRAPH").ok().as_deref() == Some("1");
    let (mg, rhs) = if op == "velocity" {
        let rhs = Poisson::with_reaction(&mesh, alpha, lambda).rhs(&src, |_, _| 0.0);
        (GpuPoissonMg::new(PMultigrid::with_reaction(p, g, g, xr, xr, alpha, lambda))?.with_cuda_graph(use_graph)?, rhs)
    } else {
        let tags = mesh.boundary_tags();
        let rhs = Poisson::with_bc(&mesh, alpha, 0.0, tags.clone()).rhs_mixed(&src, |_, _| 0.0, |_, _| 0.0);
        (GpuPoissonMg::new(PMultigrid::with_bc(p, g, g, xr, xr, alpha, 0.0, tags))?.with_cuda_graph(use_graph)?, rhs)
    };
    eprintln!("MG_GRAPH={}", use_graph);

    let _warmup = mg.solve(&rhs, tol, maxit)?; // JIT/caches — first solve, skip in analysis
    // MG_REPS repeated solves: 1 for a clean single-solve trace; many to sustain a GPU-busy
    // window (so `watch nvidia-smi` / live sampling can confirm the process is on-device).
    let reps = env_usize("MG_REPS", 1);
    let mut it = 0;
    for _ in 0..reps {
        it = mg.solve(&rhs, tol, maxit)?.1;
    }
    println!("mg-profile: op={op} p={p} grid={g}² ndof={n0} iters={it} reps={reps}");
    Ok(())
}
