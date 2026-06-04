//! Probe / regression gate for cuda-oxide libm (inverse-trig & hyperbolic) device math.
//!
//! `f64::exp/ln/sin/cos/sqrt/powf` work because they are `core::intrinsics`-backed and
//! the backend maps them to `__nv_*`. But `atan2/atan/asin/acos/sinh/cosh/tanh/hypot/cbrt`
//! are libm extern-symbol calls (not intrinsics), which the backend did not redirect →
//! unresolved on device (cuda-oxide #77/#78; gale's log-conformation kernels worked
//! around `atan2`). This probe exercises them through the embedded `#[cuda_module]` path
//! and checks against the host `std` result.
//!
//! Run: cargo oxide run --bin probe-libm

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice};
use cuda_host::cuda_module;

const N: usize = 256;

#[cuda_module]
mod kernels {
    use super::*;

    /// A battery of libm f64 functions the backend must map to libdevice `__nv_*`.
    #[kernel]
    pub fn libm_battery(a: &[f64], b: &[f64], mut out: DisjointSlice<f64>) {
        let i = thread::index_1d();
        let ii = i.get();
        let x = a[ii];
        let y = b[ii];
        // Mix several libm calls so each must resolve: atan2, atan, asin, acos,
        // sinh, cosh, tanh, hypot, cbrt.
        let v = x.atan2(y)
            + x.atan()
            + (0.5 * x.tanh()).asin()
            + (0.5 * y.tanh()).acos()
            + x.sinh()
            + y.cosh()
            + x.tanh()
            + x.hypot(y)
            + x.cbrt()
            + x.abs();
        // NOTE: `x.signum()` is intentionally NOT included — it currently triggers a
        // libNVVM "parse expected value token" error (a separate NVVM-IR codegen bug,
        // not a libdevice-mapping gap: signum lowers to a NaN-check + copysign + select).
        // gale's log-conformation kernels use a manual `if x >= 0 { 1 } else { -1 }`.
        if let Some(o) = out.get_mut(i) {
            *o = v;
        }
    }
}

fn host_ref(x: f64, y: f64) -> f64 {
    x.atan2(y)
        + x.atan()
        + (0.5 * x.tanh()).asin()
        + (0.5 * y.tanh()).acos()
        + x.sinh()
        + y.cosh()
        + x.tanh()
        + x.hypot(y)
        + x.cbrt()
        + x.abs()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== cuda-oxide libm device-math probe (atan2/atan/asin/acos/sinh/cosh/tanh/hypot/cbrt) ===\n");
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let a: Vec<f64> = (0..N).map(|i| -1.0 + (i as f64) / 128.0).collect(); // ~[-1, 1]
    let b: Vec<f64> = (0..N).map(|i| 0.5 + (i as f64) / 256.0).collect();
    let a_dev = DeviceBuffer::from_host(&stream, &a)?;
    let b_dev = DeviceBuffer::from_host(&stream, &b)?;
    let mut out_dev = DeviceBuffer::<f64>::zeroed(&stream, N)?;

    let module = kernels::load(&ctx)?;
    module.libm_battery(&stream, LaunchConfig::for_num_elems(N as u32), &a_dev, &b_dev, &mut out_dev)?;
    let got = out_dev.to_host_vec(&stream)?;

    let mut worst = 0.0f64;
    let mut bad = 0;
    for i in 0..N {
        let expect = host_ref(a[i], b[i]);
        let rel = ((got[i] - expect) / expect.abs().max(1e-300)).abs();
        worst = worst.max(rel);
        if rel > 1e-9 {
            bad += 1;
        }
    }
    if bad == 0 {
        println!("PASS: all libm f64 functions map to libdevice correctly (worst rel {worst:.2e})");
        Ok(())
    } else {
        eprintln!("FAIL: {bad}/{N} mismatches (worst rel {worst:.2e})");
        std::process::exit(1);
    }
}
