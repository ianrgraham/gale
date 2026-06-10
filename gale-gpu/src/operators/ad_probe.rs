//! TEMPORARY probe for the cuda-oxide std::autodiff integration. Mirrors gale's
//! intended shape: inside a `#[cuda_module]`, a `#[autodiff_forward]`-annotated
//! **device core** taking slices (indexing by thread internally), reachable from a
//! real `#[kernel]` (so panic/bounds-check handling is set up and the primal is
//! collected). The backend records a spec and emits `sq_core` as a device fn; the
//! Enzyme pipeline stage synthesizes the `d_sq`/`d_sq_primal` launchable kernels.
//! Delete after validating the integration on gale's own `implicit_relax`.

use cuda_device::{device, kernel, thread, DisjointSlice};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;
    use core::autodiff::autodiff_forward;

    /// `o[i] = k * x[i]^2`, so `d(o[i])/dk = x[i]^2`. Uses **raw pointers** (which
    /// std::autodiff supports as `Dual`) — `<[T]>::get_mut` is mistranslated by
    /// cuda-oxide to write a discarded stack copy, and `o[i] =` is an unsupported
    /// 2-level `Deref->Index` write. `#[device]` rewrites the `index_1d()` stub
    /// (the bare item is an `unreachable!(&str)` the translator can't lower).
    /// Offsets via integer arithmetic to avoid `offset`/`add` precondition panics.
    #[device]
    #[autodiff_forward(d_sq, Const, Dual, Dual, Const)]
    pub fn sq_core(x: *const f64, k: f64, o: *mut f64, n: usize) {
        let i = thread::index_1d().get();
        if i < n {
            unsafe {
                let xi = *((x as usize).wrapping_add(i * 8) as *const f64);
                *((o as usize).wrapping_add(i * 8) as *mut f64) = k * xi * xi;
            }
        }
    }

    /// Primal kernel: makes `sq_core` device-reachable (panic handling) and gives a
    /// host-launchable primal. Writes `o[i] = k * x[i]^2`.
    #[kernel]
    pub fn sq_primal_k(x: &[f64], k: f64, mut o: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = o.get_mut(idx) {
            *slot = k * x[i] * x[i];
        }
    }
}

/// Host wrapper: launches the pipeline-synthesized `d_sq` kernel **through the
/// normal cuda-host bundle loader** (`kernels::load` → `load_function` by name),
/// returning `(o, d_o/dk)` where `o[i] = k·x[i]²`. This is the template for gale's
/// real differentiable host wrappers — the differentiated kernel loads and launches
/// exactly like any other kernel in the crate's (now Cubin-payload) bundle.
pub fn ad_probe_grad(x: &[f64], k: f64) -> Result<(Vec<f64>, Vec<f64>), Box<dyn std::error::Error>> {
    use cuda_core::{CudaContext, DeviceBuffer};
    use std::ffi::c_void;

    let n = x.len();
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let xb = DeviceBuffer::from_host(&stream, x)?;
    let ob = DeviceBuffer::<f64>::zeroed(&stream, n)?;
    let dob = DeviceBuffer::<f64>::zeroed(&stream, n)?;

    let module = kernels::load(&ctx)?;
    let func = module.as_cuda_module().load_function("d_sq")?;

    // d_sq(x_ptr, k, o_ptr, do_ptr, n) — raw-pointer + scalar ABI.
    let mut xp = xb.cu_deviceptr();
    let mut kk = k;
    let mut op = ob.cu_deviceptr();
    let mut dop = dob.cu_deviceptr();
    let mut nn = n as u64;
    let mut args: Vec<*mut c_void> = vec![
        &mut xp as *mut _ as *mut c_void,
        &mut kk as *mut _ as *mut c_void,
        &mut op as *mut _ as *mut c_void,
        &mut dop as *mut _ as *mut c_void,
        &mut nn as *mut _ as *mut c_void,
    ];
    unsafe {
        cuda_core::launch_kernel_on_stream(
            &func,
            (1, 1, 1),
            (n as u32, 1, 1),
            0,
            &stream,
            &mut args,
        )?;
    }
    Ok((ob.to_host_vec(&stream)?, dob.to_host_vec(&stream)?))
}
