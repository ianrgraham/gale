//! Isolated FP64 transcendental probe for cuda-oxide on the local GPU.
//!
//! Tests `f64::exp`, `f64::ln`, and `f64::powf` — the double-precision libdevice
//! functions gale needs for viscoelastic constitutive models (the
//! log-conformation representation evolves `exp`/`log` of the conformation
//! tensor; FENE-P needs `powf`/division). These map to `__nv_exp` / `__nv_log` /
//! `__nv_pow`.
//!
//! IMPORTANT — loader choice. Any libdevice call makes cuda-oxide emit **NVVM IR**
//! (not PTX) and skip `llc`; the consumer must then run libNVVM + libdevice →
//! LTOIR → nvJitLink → cubin. That pipeline lives in `cuda_host::ltoir`, reached
//! via the **file-based `load_kernel_module` loader** (the `manual_launch_libdevice`
//! pattern). The embedded `#[cuda_module]` / `kernels::load` path does NOT handle
//! the NVVM-IR/libdevice case and fails with `nvvmCompileProgram: "parse expected
//! type"`. So this probe uses a top-level `#[kernel]` + `load_kernel_module`.
//!
//! On the Volta Titan V (sm_70) this additionally requires cuda-oxide PR #101
//! (typed NVVM IR for pre-Blackwell) — issue #98. With #101's backend pinned (see
//! Cargo.toml + .cargo/config.toml), this probe is the regression gate: green
//! means libdevice f64 math works on our hardware.
//!
//! Run:
//!   cargo oxide run --bin probe-fp64-math
//!   cargo oxide run --bin probe-fp64-math --arch sm_70
//!
//! Note: `f64::atan`/`atan2` are NOT tested — not yet mapped (cuda-oxide #77/#78).

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::{cuda_launch, load_kernel_module};

const N: usize = 256;

/// out = ln(exp(x)) * y^p  — should equal x * y^p to within libdevice tolerance.
///
/// NOTE: writes via `get_unchecked_mut` after an explicit bounds check rather
/// than `get_mut`. `get_mut` returns `Option<&mut f64>`, whose `{ i8, i8* }`
/// aggregate makes #101 emit a `double*` into an `i8*` slot without a bitcast —
/// illegal in typed-pointer NVVM IR (libNVVM: "'%vNN' defined with type
/// 'double*'"). The direct-write form sidesteps that codegen gap.
#[kernel]
pub fn fp64_transcendental(x: &[f64], y: &[f64], p: f64, mut out: DisjointSlice<f64>) {
    let i = thread::index_1d().get();
    if i < out.len() {
        let a = x[i];
        let b = y[i];
        unsafe {
            *out.get_unchecked_mut(i) = a.exp().ln() * b.powf(p);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== gale FP64 transcendental probe (exp / ln / powf) ===\n");

    let ctx = CudaContext::new(0)?;
    let cc = ctx.compute_capability().unwrap_or((-1, -1));
    println!(
        "device 0: {}  (cc {}.{})\n",
        ctx.device_name().unwrap_or_else(|_| "<unknown>".into()),
        cc.0,
        cc.1
    );
    let stream = ctx.default_stream();

    // File-based loader: builds the cubin from the emitted NVVM IR via
    // libNVVM + libdevice + nvJitLink (cuda_host::ltoir). This is the path that
    // handles libdevice; the embedded `kernels::load` path does not.
    let module = match load_kernel_module(&ctx, "probe_fp64_math") {
        Ok(m) => m,
        Err(e) => {
            eprintln!("FAIL: load_kernel_module failed: {e}");
            eprintln!("(On sm_70 a libNVVM 'parse expected type' here means PR #101's typed");
            eprintln!(" NVVM IR still isn't accepted by this libNVVM — report it on #101.)");
            std::process::exit(1);
        }
    };

    // Keep arguments modest so exp() doesn't overflow f64.
    let x: Vec<f64> = (0..N).map(|i| (i as f64) / 64.0).collect(); // 0 .. ~4
    let y: Vec<f64> = (0..N).map(|i| 1.0 + (i as f64) / 32.0).collect(); // >= 1
    let p = 1.5_f64;

    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let y_dev = DeviceBuffer::from_host(&stream, &y)?;
    let mut out_dev = DeviceBuffer::<f64>::zeroed(&stream, N)?;

    cuda_launch! {
        kernel: fp64_transcendental,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(N as u32),
        args: [slice(x_dev), slice(y_dev), p, slice_mut(out_dev)]
    }?;

    let got = out_dev.to_host_vec(&stream)?;
    let mut bad = 0usize;
    let mut worst = 0.0f64;
    for i in 0..N {
        let expect = x[i].exp().ln() * y[i].powf(p);
        let rel = ((got[i] - expect) / expect.max(1e-300)).abs();
        worst = worst.max(rel);
        if rel > 1e-9 {
            bad += 1;
        }
    }

    if bad == 0 {
        println!("PASS: f64 exp/ln/powf correct on cc {}.{} (worst rel err {worst:.2e})", cc.0, cc.1);
        println!("→ log-conformation math is usable on this hardware.");
        Ok(())
    } else {
        eprintln!("FAIL: {bad}/{N} mismatches (worst rel err {worst:.2e}) — libdevice result suspect");
        std::process::exit(1);
    }
}
