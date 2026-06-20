//! Single-size SIPG matvec target for Nsight Compute (clean `operator`/`gradient` launch stream).
//! Reuses `bench_poisson_kernels` at one mesh so `ncu --kernel-name regex:operator_TID` profiles a
//! single uniform operator kernel. Env: MG_P (default 4), MG_GRID (default 64), MG_REPS (default 200).
//!   sudo env MG_P=4 MG_GRID=64 /usr/local/cuda/bin/ncu --kernel-name regex:operator_TID \
//!     --launch-skip 5 --launch-count 2 --section WarpStateStats --section SchedulerStats \
//!     --section Occupancy target/release/op_profile
use gale::dg::Mesh2d;
use gale_gpu::bench_poisson_kernels;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p: usize = std::env::var("MG_P").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let g: usize = std::env::var("MG_GRID").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    let reps: u32 = std::env::var("MG_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    let mesh = Mesh2d::rectangular(p, g, g, [0.0, 1.0], [0.0, 1.0]);
    let b = bench_poisson_kernels(&mesh, 5.0, reps)?;
    println!("op_profile: p={p} grid={g}² ndof={}", b.ndof);
    for k in &b.kernels {
        println!("  {:<24} {:8.2} µs/call", k.name, k.ms * 1e3);
    }
    Ok(())
}
