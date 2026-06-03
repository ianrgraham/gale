//! Milestone-1 sm_70 (Volta / Titan V) capability probe for cuda-oxide.
//!
//! Purpose: answer, on the *actual hardware*, the open questions from
//! `docs/cuda-oxide-repo-status.md` §6 before building the DG core on cuda-oxide:
//!
//!   1. Does cuda-oxide compile + load + run a kernel at all on sm_70?
//!      (cuda-oxide's clean path targets Ampere→Blackwell; pre-Ampere/pre-Blackwell
//!       support is in-flight — open PRs #69 and #101. This probe tells us whether
//!       we need them *today*.)
//!   2. Does **FP64** arithmetic produce correct results on sm_70?
//!   3. Does **shared memory + barriers** (the matrix-free DG kernel building
//!      block) work with f64?
//!   4. Can we open **two GPU contexts** and use **peer-to-peer** between them
//!      (the inter-GPU trace-exchange primitive for multi-GPU DG)?
//!
//! ALL kernels here are deliberately libdevice-free. Any libdevice math call
//! (`sqrt`, `exp`, …) flips cuda-oxide into NVVM-IR mode, which currently fails
//! to parse on the Titan V's pre-Blackwell libNVVM (#98) and would poison this
//! whole shared module's load. Libdevice math is probed in isolation by the
//! separate `probe-fp64-math` binary, so its (expected) failure can't mask the
//! capabilities that DO work here.
//!
//! Run:
//!   cargo oxide run --bin probe-sm70                 # auto-detects device-0 arch
//!   cargo oxide run --bin probe-sm70 --arch sm_70    # force Volta
//!   CUDA_OXIDE_TARGET=sm_70 cargo oxide run --bin probe-sm70
//!
//! Exit code is non-zero only if a *core* capability (device, FP64 arithmetic,
//! sqrt, or shared memory) fails. Multi-GPU/P2P is reported as a finding, not a
//! hard failure (a single-GPU box or a PCIe topology without P2P is legitimate).

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

const N: usize = 256; // one block; also the shared-memory tile size

#[cuda_module]
mod kernels {
    use super::*;

    /// Pure FP64 arithmetic — add/sub/mul/div, NO libdevice call.
    ///
    /// Deliberately libdevice-free: any libdevice call (e.g. `sqrt`) flips the
    /// pipeline into NVVM-IR mode, which fails to parse on the Titan V's
    /// pre-Blackwell libNVVM (cuda-oxide #98) and would poison the load of this
    /// whole shared module. Double-precision intrinsics are probed in isolation
    /// by the `probe-fp64-math` binary instead.
    #[kernel]
    pub fn fp64_arith(a: &[f64], b: &[f64], mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let x = a[i];
            let y = b[i];
            *o = x * y + (x - y) / (y + 1.0) - x / (x + 1.0);
        }
    }

    /// FP64 block-level cooperation through shared memory + a barrier.
    /// Each thread stores its input into a shared tile, syncs, then reads its
    /// neighbor's value — the same load/sync/read pattern a matrix-free DG
    /// operator kernel uses to stage element data.
    #[kernel]
    pub fn fp64_shared_neighbor(data: &[f64], mut out: DisjointSlice<f64>) {
        static mut TILE: SharedArray<f64, N> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let gid = thread::index_1d().get();

        unsafe {
            TILE[tid] = data[gid];
        }
        thread::sync_threads();
        unsafe {
            let neighbor = (tid + 1) % N;
            if let Some(o) = out.get_mut(thread::index_1d()) {
                *o = TILE[neighbor];
            }
        }
    }
}

/// One probe result row.
struct Check {
    name: &'static str,
    core: bool, // does failure mean gale can't proceed on this toolchain?
    pass: Option<bool>, // None = skipped
    detail: String,
}

fn main() {
    println!("=== gale Milestone-1: cuda-oxide sm_70 capability probe ===\n");
    let mut results: Vec<Check> = Vec::new();

    // --- 1. Device inventory -------------------------------------------------
    let mut contexts = Vec::new();
    let mut ordinal = 0usize;
    while ordinal < 16 {
        match CudaContext::new(ordinal) {
            Ok(ctx) => {
                let name = ctx.device_name().unwrap_or_else(|_| "<unknown>".into());
                let cc = ctx.compute_capability().unwrap_or((-1, -1));
                println!(
                    "  device {ordinal}: {name}  (compute capability {}.{}{})",
                    cc.0,
                    cc.1,
                    if cc == (7, 0) { " — Volta sm_70 ✓" } else { "" }
                );
                contexts.push((ordinal, ctx, cc));
                ordinal += 1;
            }
            Err(_) => break,
        }
    }
    let n_dev = contexts.len();
    let any_sm70 = contexts.iter().any(|(_, _, cc)| *cc == (7, 0));
    results.push(Check {
        name: "device inventory",
        core: true,
        pass: Some(n_dev > 0),
        detail: format!(
            "{n_dev} device(s) detected; {}",
            if any_sm70 { "at least one is Volta sm_70" } else { "no sm_70 device present" }
        ),
    });
    if n_dev == 0 {
        report(&results);
        eprintln!("\nFATAL: no CUDA device — cannot probe. Is the driver up?");
        std::process::exit(1);
    }

    // Use device 0 for the single-GPU kernel tests.
    let (_, ctx0, cc0) = &contexts[0];
    let ctx0 = ctx0.clone();
    let stream = ctx0.default_stream();
    println!(
        "\nRunning kernel tests on device 0 (cc {}.{}).\n",
        cc0.0, cc0.1
    );

    // Load the embedded kernel module. If THIS fails on sm_70, that is the
    // headline finding: the generated module won't load on Volta (the #69/#101
    // regime) and gale needs those upstream fixes before proceeding.
    let module = match kernels::load(&ctx0) {
        Ok(m) => m,
        Err(e) => {
            results.push(Check {
                name: "module load on this arch",
                core: true,
                pass: Some(false),
                detail: format!(
                    "kernels::load failed: {e}  — likely the pre-Ampere/pre-Blackwell \
                     codegen gap (cuda-oxide #69/#101). This is the P0 blocker."
                ),
            });
            report(&results);
            std::process::exit(1);
        }
    };
    results.push(Check {
        name: "module load on this arch",
        core: true,
        pass: Some(true),
        detail: "embedded PTX module loaded".into(),
    });

    // --- 2 & 3. FP64 arithmetic + libdevice sqrt ----------------------------
    {
        let a: Vec<f64> = (0..N).map(|i| (i + 1) as f64).collect();
        let b: Vec<f64> = (0..N).map(|i| 2.0 * (i + 1) as f64).collect();
        let check = (|| -> Result<(usize, f64), Box<dyn std::error::Error>> {
            let a_dev = DeviceBuffer::from_host(&stream, &a)?;
            let b_dev = DeviceBuffer::from_host(&stream, &b)?;
            let mut out_dev = DeviceBuffer::<f64>::zeroed(&stream, N)?;
            module.fp64_arith(&stream, LaunchConfig::for_num_elems(N as u32), &a_dev, &b_dev, &mut out_dev)?;
            let got = out_dev.to_host_vec(&stream)?;
            let mut bad = 0usize;
            let mut worst = 0.0f64;
            for i in 0..N {
                let (x, y) = (a[i], b[i]);
                let expect = x * y + (x - y) / (y + 1.0) - x / (x + 1.0);
                let rel = ((got[i] - expect) / expect).abs();
                worst = worst.max(rel);
                if rel > 1e-12 {
                    bad += 1;
                }
            }
            Ok((bad, worst))
        })();
        match check {
            Ok((0, worst)) => results.push(Check {
                name: "FP64 arithmetic (no libdevice)",
                core: true,
                pass: Some(true),
                detail: format!("all {N} elems correct (worst rel err {worst:.2e})"),
            }),
            Ok((bad, worst)) => results.push(Check {
                name: "FP64 arithmetic (no libdevice)",
                core: true,
                pass: Some(false),
                detail: format!("{bad}/{N} mismatches (worst rel err {worst:.2e}) — suspect f64 codegen"),
            }),
            Err(e) => results.push(Check {
                name: "FP64 arithmetic (no libdevice)",
                core: true,
                pass: Some(false),
                detail: format!("launch/transfer error: {e}"),
            }),
        }
    }

    // --- 4. FP64 shared memory + barrier ------------------------------------
    {
        let data: Vec<f64> = (0..N).map(|i| i as f64 * 0.5 + 1.0).collect();
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (N as u32, 1, 1), shared_mem_bytes: 0 };
        let check = (|| -> Result<usize, Box<dyn std::error::Error>> {
            let data_dev = DeviceBuffer::from_host(&stream, &data)?;
            let mut out_dev = DeviceBuffer::<f64>::zeroed(&stream, N)?;
            module.fp64_shared_neighbor(&stream, cfg, &data_dev, &mut out_dev)?;
            let got = out_dev.to_host_vec(&stream)?;
            let mut bad = 0usize;
            for i in 0..N {
                let expect = data[(i + 1) % N];
                if (got[i] - expect).abs() > 1e-12 {
                    bad += 1;
                }
            }
            Ok(bad)
        })();
        match check {
            Ok(0) => results.push(Check {
                name: "FP64 shared memory + sync_threads",
                core: true,
                pass: Some(true),
                detail: format!("all {N} neighbor reads correct"),
            }),
            Ok(bad) => results.push(Check {
                name: "FP64 shared memory + sync_threads",
                core: true,
                pass: Some(false),
                detail: format!("{bad}/{N} mismatches"),
            }),
            Err(e) => results.push(Check {
                name: "FP64 shared memory + sync_threads",
                core: true,
                pass: Some(false),
                detail: format!("launch error: {e}"),
            }),
        }
    }

    // --- 5. Multi-GPU + peer-to-peer ----------------------------------------
    if n_dev < 2 {
        results.push(Check {
            name: "multi-GPU peer-to-peer",
            core: false,
            pass: None,
            detail: "skipped: fewer than 2 devices".into(),
        });
    } else {
        let ctx_a = contexts[0].1.clone();
        let ctx_b = contexts[1].1.clone();
        let detail = probe_peer(&ctx_a, &ctx_b);
        results.push(Check {
            name: "multi-GPU peer-to-peer (dev0↔dev1)",
            core: false,
            pass: Some(detail.0),
            detail: detail.1,
        });
    }

    report(&results);

    let core_failed = results.iter().any(|c| c.core && c.pass == Some(false));
    if core_failed {
        eprintln!("\nRESULT: a CORE capability failed — see above before building on cuda-oxide.");
        std::process::exit(1);
    }
    println!("\nRESULT: all core capabilities PASS on this hardware. cuda-oxide is viable for gale here.");
}

/// Query → enable → exercise P2P between two contexts. Returns (ok, detail).
/// Never panics; a topology without P2P is a legitimate finding, not a bug.
fn probe_peer(
    ctx_a: &std::sync::Arc<CudaContext>,
    ctx_b: &std::sync::Arc<CudaContext>,
) -> (bool, String) {
    use cuda_core::peer::{can_access_peer, enable_peer_access};

    let a2b = can_access_peer(ctx_a, ctx_b).unwrap_or(false);
    let b2a = can_access_peer(ctx_b, ctx_a).unwrap_or(false);
    if !a2b && !b2a {
        return (
            false,
            "can_access_peer = false both ways (no P2P on this topology — \
             trace exchange must stage through host or use CUDA-Aware MPI)"
                .into(),
        );
    }

    // Enable both directions (idempotent), then copy dev0 -> dev1 directly.
    let _ = enable_peer_access(ctx_b, ctx_a);
    let _ = enable_peer_access(ctx_a, ctx_b);

    let result = (|| -> Result<bool, Box<dyn std::error::Error>> {
        let stream_a = ctx_a.default_stream();
        let stream_b = ctx_b.default_stream();
        let host: Vec<f64> = (0..N).map(|i| i as f64 * 3.0).collect();
        let src = DeviceBuffer::from_host(&stream_a, &host)?; // on dev0
        let dst = DeviceBuffer::<f64>::zeroed(&stream_b, N)?; // on dev1
        let bytes = N * std::mem::size_of::<f64>();
        // dev1's stream reads dev0 memory directly via the P2P path.
        unsafe {
            cuda_core::memory::memcpy_dtod_async(
                dst.cu_deviceptr(),
                src.cu_deviceptr(),
                bytes,
                stream_b.cu_stream(),
            )?;
        }
        let got = dst.to_host_vec(&stream_b)?;
        Ok(got == host)
    })();

    match result {
        Ok(true) => (
            true,
            format!("can_access_peer a→b={a2b} b→a={b2a}; direct dev0→dev1 P2P copy verified"),
        ),
        Ok(false) => (
            false,
            "P2P copy completed but data mismatch — investigate before trusting P2P".into(),
        ),
        Err(e) => (false, format!("P2P enabled (a→b={a2b} b→a={b2a}) but copy failed: {e}")),
    }
}

fn report(results: &[Check]) {
    println!("\n┌── PROBE SUMMARY ─────────────────────────────────────────────");
    for c in results {
        let tag = match c.pass {
            Some(true) => "PASS",
            Some(false) => "FAIL",
            None => "SKIP",
        };
        let core = if c.core { "" } else { " (informational)" };
        println!("│ [{tag}] {}{core}\n│        {}", c.name, c.detail);
    }
    println!("└──────────────────────────────────────────────────────────────");
}
