//! Probe: can ONE generic `#[kernel]` over a custom `Scalar` trait give us both the f64
//! (across-the-board, opt-out) path AND the f32 (mixed-precision storage) path — read-widen to
//! f64, accumulate in f64, write-narrow to the storage type — instantiated via turbofish
//! `module.gcontract::<f64>` / `::<f32>`? If yes, the FP32 V-cycle does NOT fork the kernel suite:
//! the same generic operator/smoother/transfer kernels serve both precisions, and a plain f64
//! instantiation preserves the exact current behaviour (mixed precision stays opt-out-able).
//!
//! Run: cargo oxide run --bin probe-generic-kernel

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice, DynamicSharedArray};
use cuda_host::cuda_module;

/// Storage scalar with widen/narrow to the f64 accumulation type. f64 ⇒ identity (the across-the-
/// board path is bit-for-bit the current code); f32 ⇒ the mixed-precision storage path.
pub trait Scalar: Copy {
    fn to_f64(self) -> f64;
    fn from_f64(x: f64) -> Self;
}
impl Scalar for f64 {
    fn to_f64(self) -> f64 {
        self
    }
    fn from_f64(x: f64) -> Self {
        x
    }
}
impl Scalar for f32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(x: f64) -> Self {
        x as f32
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Generic over the STORAGE type `T`; always accumulates in f64 (the matvec idiom). Reads `T`
    /// field into f64 shared, contracts against an f64 matrix, writes `T`.
    #[kernel]
    pub fn gcontract<T: Scalar>(field: &[T], mat: &[f64], n: u32, scale: f64, mut out: DisjointSlice<T>) {
        let sm = DynamicSharedArray::<f64>::get();
        let n = n as usize;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        unsafe {
            *sm.add(m) = field[e * n + m].to_f64();
        }
        thread::sync_threads();
        let mut acc = 0.0f64;
        let mut k = 0usize;
        while k < n {
            acc += mat[m * n + k] * unsafe { *sm.add(k) };
            k += 1;
        }
        acc *= scale;
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = T::from_f64(acc);
        }
    }
}

fn cpu_ref(field: &[f64], mat: &[f64], ne: usize, n: usize, scale: f64) -> Vec<f64> {
    let mut out = vec![0.0; ne * n];
    for e in 0..ne {
        for m in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += mat[m * n + k] * field[e * n + k];
            }
            out[e * n + m] = acc * scale;
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let (ne, n, scale) = (64usize, 8usize, 2.5f64);
    let field: Vec<f64> = (0..ne * n).map(|i| ((i % 13) as f64 - 6.0) * 0.1).collect();
    let mat: Vec<f64> = (0..n * n).map(|i| ((i % 7) as f64 - 3.0) * 0.05).collect();
    let want = cpu_ref(&field, &mat, ne, n, scale);
    let mat_dev = DeviceBuffer::from_host(&stream, &mat)?;
    let cfg = LaunchConfig { grid_dim: (ne as u32, 1, 1), block_dim: (n as u32, 1, 1), shared_mem_bytes: (n * 8) as u32 };
    let module = kernels::load(&ctx)?;

    println!("=== generic #[kernel] probe (one kernel ⇒ both f64 and f32 instantiations) ===");

    // f64 instantiation — the across-the-board / opt-out path (must be bit-exact).
    let f64_dev = DeviceBuffer::from_host(&stream, &field)?;
    let mut o64 = DeviceBuffer::<f64>::zeroed(&stream, ne * n)?;
    module.gcontract::<f64>(&stream, cfg, &f64_dev, &mat_dev, n as u32, scale, &mut o64)?;
    let g64 = o64.to_host_vec(&stream)?;
    let e64 = g64.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);

    // f32 instantiation — the mixed-precision storage path (f32 in/out, f64 accumulate).
    let field32: Vec<f32> = field.iter().map(|&v| v as f32).collect();
    let f32_dev = DeviceBuffer::from_host(&stream, &field32)?;
    let mut o32 = DeviceBuffer::<f32>::zeroed(&stream, ne * n)?;
    module.gcontract::<f32>(&stream, cfg, &f32_dev, &mat_dev, n as u32, scale, &mut o32)?;
    let g32 = o32.to_host_vec(&stream)?;
    let e32 = g32.iter().zip(&want).map(|(a, b)| (*a as f64 - b).abs()).fold(0.0f64, f64::max);

    println!("  gcontract::<f64>  max|gpu − cpu| = {e64:.3e}   (expect ~0: bit-exact opt-out path)");
    println!("  gcontract::<f32>  max|gpu − cpu| = {e32:.3e}   (expect ~f32 epsilon)");
    if e64 < 1e-13 && e32 < 1e-5 {
        println!("PASS: one generic kernel serves both precisions ⇒ FP64 stays first-class (opt-out preserved).");
        Ok(())
    } else {
        eprintln!("FAIL: generic instantiation mismatch.");
        std::process::exit(1);
    }
}
