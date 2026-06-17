//! Perf gate for the GPU-resident adapt cycle (`GpuAdaptiveMesh::adapt`) at PRODUCTION cap — the scale
//! where the old single-thread compaction would have been catastrophic. Default base 256×256, l_max=2
//! ⇒ cap = 256²·(1+4+16) ≈ 1.38M slots; every per-slot kernel (mark / block-aggregated compaction /
//! pos2slot / balance-flag / connectivity) scans the full cap, and the free-list compaction scatters
//! ~1.3M entries (the heavy case the block-aggregated atomics must handle in parallel).
//!
//! Env: AMR_NX (base nx=ny, default 256), AMR_STEPS (timed adapts, default 10), AMR_LMAX (default 2).
//! Run (timing):   cargo oxide run --bin amr-adapt-perf
//! Profile (ncu):  ncu --launch-count 40 --kernel-name regex:'amr_' --set basic \
//!                   --export /tmp/prof_amr -f ./target/release/amr-adapt-perf

use gale::dg::RefineQuad;
use gale_gpu::amr_mesh::GpuAmrMesh;
use gale_gpu::operators::amr::GpuAdaptiveMesh;
use std::time::Instant;

const P: usize = 3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let nx: usize = std::env::var("AMR_NX").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let steps: usize = std::env::var("AMR_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let l_max: usize = std::env::var("AMR_LMAX").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let refq = RefineQuad::new(P);
    let nn = (P + 1) * (P + 1);

    let host = GpuAmrMesh::new(nx, ny_of(nx), l_max);
    let cap = host.cap;
    let mem_mb = (cap * nn * 8 * 2) as f64 / 1e6; // field + fscratch dominate
    println!("=== AMR resident-adapt perf — base {nx}×{nx}, l_max={l_max}, p={P} ===");
    println!("  cap = {cap} slots ({} active at start), field≈{:.0} MB", nx * nx, mem_mb);

    let mut dev = GpuAdaptiveMesh::from_host(&host, &refq)?;
    let field = vec![0.0f64; cap * nn];
    let mut dfield = dev.upload_field(&field)?;

    // Flag: refine a band of base cells (exercises refine+remap+apply); the rest 0. After the first
    // adapt these become internal, so the structure reaches a steady state and the full-cap per-slot
    // kernels (mark/compact/pos2slot/connectivity) dominate every subsequent adapt — the steady cost.
    let mut flag = vec![0i32; cap];
    let mut nflag = 0;
    for cy in 0..nx {
        for cx in 0..nx {
            if (cx + cy) % 8 == 0 {
                if let Some(s) = host.slot_at(0, cx as u32, cy as u32) {
                    flag[s] = 1;
                    nflag += 1;
                }
            }
        }
    }
    println!("  refine flag set on {nflag} base cells");

    let zero = vec![0i32; cap];

    // JIT warm with a no-op adapt (excluded from timing).
    dev.adapt(&zero, &mut dfield)?;

    // (1) A real WORK adapt: refine 8192 base cells + full field remap (the cost when the mesh changes).
    let t = Instant::now();
    let pw = dev.adapt(&flag, &mut dfield)?;
    let work_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("  WORK adapt (refine {nflag} cells + remap, {pw} balance passes): {work_ms:.3} ms");

    // (2) Steady no-op adapts: flag resolves to no structural change, so only the full-cap per-slot
    // kernels run (mark / compaction / pos2slot / balance-flag / connectivity) — the adapt-cycle floor.
    let t = Instant::now();
    for _ in 0..steps {
        dev.adapt(&zero, &mut dfield)?;
    }
    let noop = t.elapsed().as_secs_f64() * 1e3 / steps as f64;
    println!("  NO-OP adapt (full-cap scans only): {noop:.3} ms/adapt over {steps}");
    println!("  per-slot scan throughput ≈ {:.0} M slots/ms (kernels touch all {cap} slots)", cap as f64 / noop / 1e3);
    Ok(())
}

fn ny_of(nx: usize) -> usize {
    nx
}
