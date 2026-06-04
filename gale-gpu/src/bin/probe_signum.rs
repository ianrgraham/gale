//! Regression gate for the `f64::signum` NVVM-IR codegen bug: signum lowers to a
//! NaN-check + `copysign` + select, and the NaN constant used to be emitted as the
//! bare `nan` keyword which libNVVM rejects ("parse expected value token"). Fixed by
//! emitting NaN in IEEE hex form (cuda-oxide `dialect-llvm` `format_float_literal`).
//! Green here means NaN constants — hence `signum` and any NaN-producing device code —
//! work on this hardware.
//!
//! Run: cargo oxide run --bin probe-signum

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;
    #[kernel]
    pub fn sgn(a: &[f64], mut out: DisjointSlice<f64>) {
        let i = thread::index_1d();
        let x = a[i.get()];
        if let Some(o) = out.get_mut(i) {
            *o = x.signum();
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let s = ctx.default_stream();
    let input = [-2.0f64, 0.0, 3.0, -0.0];
    let expect = [-1.0f64, 1.0, 1.0, -1.0]; // f64::signum = copysign(1.0, x)
    let a = DeviceBuffer::from_host(&s, &input)?;
    let mut o = DeviceBuffer::<f64>::zeroed(&s, 4)?;
    kernels::load(&ctx)?.sgn(&s, LaunchConfig::for_num_elems(4), &a, &mut o)?;
    let got = o.to_host_vec(&s)?;
    if got == expect {
        println!("PASS: f64::signum (NaN-constant codegen) works: {got:?}");
        Ok(())
    } else {
        eprintln!("FAIL: signum got {got:?}, expected {expect:?}");
        std::process::exit(1);
    }
}
