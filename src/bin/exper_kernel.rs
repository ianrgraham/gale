
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn test(a: &[f64], x: &[f64], y: &[f64], mut b: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(b_elem) = b.get_mut(idx) {
            *b_elem = a[i] * x[i] + y[i]
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 64;
    let a_host: Vec<f64> = (0..N).map(|i| (i * 2) as f64).collect();
    let x_host: Vec<f64> = (0..N).map(|i| i as f64).collect();
    let y_host: Vec<f64> = (0..N).map(|i| -50.0 * (i as f64)).collect();

    let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
    let x_dev = DeviceBuffer::from_host(&stream, &x_host).unwrap();
    let y_dev = DeviceBuffer::from_host(&stream, &y_host).unwrap();
    let mut b_dev = DeviceBuffer::<f64>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    module.test(
        &stream,
        LaunchConfig::for_num_elems(N as u32),
        &a_dev,
        &x_dev,
        &y_dev,
        &mut b_dev
    ).expect("Kernel launch failed");

    let b_host = b_dev.to_host_vec(&stream).unwrap();

    for (idx, b) in b_host.iter().enumerate() {
        println!("{idx}: {b}")
    }
}