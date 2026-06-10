//! Codegen probe for the mixed-precision matvec idiom (FP32 storage / FP64 accumulate).
//! `vecadd` already proves `&[f32]` / `DisjointSlice<f32>` work; the OPEN question for the
//! NVVM-text backend is a SINGLE kernel that mixes `&[f32]` and `&[f64]` args, casts `f32 as f64`,
//! accumulates in f64, and writes `f64 as f32` — plus an f64 contraction out of f64 shared fed by
//! f32 global loads (the exact shape the SIPG matvec would take). If this compiles + matches the
//! CPU f64 reference to ~f32 epsilon, the FP32-storage matvec rewrite is feasible.
//!
//! Run: cargo oxide run --bin probe-mixed-precision

use cuda_device::{kernel, thread, DisjointSlice, DynamicSharedArray};
use cuda_host::cuda_module;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

#[cuda_module]
mod kernels {
    use super::*;

    /// Per-element f32→f64→f32: read an f32 field tile into f64 shared, contract against an f64
    /// matrix (mimics the gradient's tensor product), accumulate in f64, write f32. Mixes `&[f32]`
    /// (field) + `&[f64]` (matrix) + a single f64 scalar arg in one kernel.
    #[kernel]
    pub fn mixed_contract(field: &[f32], mat: &[f64], n: u32, scale: f64, mut out: DisjointSlice<f32>) {
        let sm = DynamicSharedArray::<f64>::get(); // f64 shared, fed by f32 global loads
        let n = n as usize;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        unsafe {
            *sm.add(m) = field[e * n + m] as f64; // f32 → f64 on load (the storage→accumulate cast)
        }
        thread::sync_threads();
        let mut acc = 0.0f64;
        let mut k = 0usize;
        while k < n {
            acc += mat[m * n + k] * unsafe { *sm.add(k) }; // f64 accumulate
            k += 1;
        }
        acc *= scale;
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = acc as f32; // f64 → f32 on store
        }
    }

    // Well-occupied (256-thread) streaming kernels: out = a·x, read 1 + write 1 per element. The
    // textbook bandwidth test — measures the f32-vs-f64 STORAGE ratio the hardware delivers, free
    // of the contraction's occupancy confound.
    #[kernel]
    pub fn stream_f64(x: &[f64], a: f64, mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = a * x[i];
        }
    }
    #[kernel]
    pub fn stream_f32(x: &[f32], a: f32, mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = a * x[i];
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let (ne, n, scale) = (64usize, 8usize, 2.5f64);

    // f32 field, f64 contraction matrix.
    let field: Vec<f32> = (0..ne * n).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let mat: Vec<f64> = (0..n * n).map(|i| ((i % 7) as f64 - 3.0) * 0.05).collect();

    let field_dev = DeviceBuffer::from_host(&stream, &field)?;
    let mat_dev = DeviceBuffer::from_host(&stream, &mat)?;
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, ne * n)?;

    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig {
        grid_dim: (ne as u32, 1, 1),
        block_dim: (n as u32, 1, 1),
        shared_mem_bytes: (n * 8) as u32,
    };
    module.mixed_contract(&stream, cfg, &field_dev, &mat_dev, n as u32, scale, &mut out_dev)?;
    let gpu = out_dev.to_host_vec(&stream)?;

    // CPU f64 reference, cast to f32 (what the GPU should produce bit-approx).
    let mut max_err = 0.0f32;
    for e in 0..ne {
        for m in 0..n {
            let mut acc = 0.0f64;
            for k in 0..n {
                acc += mat[m * n + k] * field[e * n + k] as f64;
            }
            let want = (acc * scale) as f32;
            max_err = max_err.max((gpu[e * n + m] - want).abs());
        }
    }
    println!("=== mixed-precision codegen probe (f32 storage / f64 accumulate) ===");
    println!("ne={ne} n={n}  max|gpu − cpu(f64→f32)| = {max_err:.3e}");
    if max_err >= 1e-5 {
        eprintln!("FAIL: mismatch beyond f32 rounding — investigate codegen.");
        std::process::exit(1);
    }
    println!("PASS: NVVM-text backend codegens mixed f32/f64 + as-casts + f64 shared correctly.\n");

    // --- hardware bandwidth: f32-storage vs f64-storage on a well-occupied streaming kernel ----
    // Confirms the storage ratio the GPU actually delivers (the matvec's f32-able traffic is
    // u/gx/gy/out). The REALIZED matvec win — with its mixed traffic + neighbor gather — must be
    // measured on the real kernel after the library rewrite; this is the hardware upper bound.
    let (n, reps) = (26_214_400usize, 100u32); // ~1024²/p=4 dof (209 MB f64), grid-stride 256-thread
    let x64: Vec<f64> = (0..n).map(|i| (i % 17) as f64 * 0.01).collect();
    let x32: Vec<f32> = x64.iter().map(|&v| v as f32).collect();
    let x64_dev = DeviceBuffer::from_host(&stream, &x64)?;
    let x32_dev = DeviceBuffer::from_host(&stream, &x32)?;
    let mut o64 = DeviceBuffer::<f64>::zeroed(&stream, n)?;
    let mut o32 = DeviceBuffer::<f32>::zeroed(&stream, n)?;
    let scfg = LaunchConfig::for_num_elems(n as u32);

    let f = cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT;
    macro_rules! timed {
        ($launch:expr) => {{
            $launch?; // warmup
            let (s, e) = (ctx.new_event(Some(f))?, ctx.new_event(Some(f))?);
            s.record(&stream)?;
            for _ in 0..reps {
                $launch?;
            }
            e.record(&stream)?;
            e.synchronize()?;
            s.elapsed_ms(&e)? as f64 / reps as f64
        }};
    }
    let ms64: f64 = timed!(module.stream_f64(&stream, scfg, &x64_dev, scale, &mut o64));
    let ms32: f64 = timed!(module.stream_f32(&stream, scfg, &x32_dev, scale as f32, &mut o32));
    let nf = n as f64;
    let gbps = |ms: f64, bytes: f64| bytes / (ms * 1e-3) / 1e9;
    println!("hardware bandwidth (out = a·x, {n} elems, Titan V peak 652.8 GB/s):");
    println!("  f64 storage (16 B/elem): {:8.3} ms   {:6.1} GB/s", ms64, gbps(ms64, nf * 16.0));
    println!("  f32 storage ( 8 B/elem): {:8.3} ms   {:6.1} GB/s", ms32, gbps(ms32, nf * 8.0));
    println!("  speedup f64→f32: {:.2}×   (≈2× ⇒ the matvec's f32-able traffic scales with bytes)", ms64 / ms32);
    Ok(())
}
