//! Validates the std::autodiff → cargo-oxide → cuda-host path end-to-end: the
//! pipeline-synthesized `d_sq` kernel is loaded **through the normal embedded
//! bundle loader** and launched by name, and its forward-mode tangent matches the
//! analytic gradient on the GPU.
//!
//! Run: cargo oxide run --bin ad-probe-check

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 16usize;
    let k = 3.0;
    let x: Vec<f64> = (0..n).map(|i| 0.5 + 0.1 * i as f64).collect();

    let (o, dodk) = gale_gpu::operators::ad_probe::ad_probe_grad(&x, k)?;

    let mut max_p = 0.0f64;
    let mut max_d = 0.0f64;
    for i in 0..n {
        let op = k * x[i] * x[i]; // o = k x²
        let od = x[i] * x[i]; // d o / dk = x²
        max_p = max_p.max((o[i] - op).abs());
        max_d = max_d.max((dodk[i] - od).abs());
    }
    println!("primal max err = {max_p:.2e}   tangent max err = {max_d:.2e}");
    if max_p < 1e-9 && max_d < 1e-9 {
        println!(
            "PASS: #[autodiff_forward] kernel built by `cargo oxide`, loaded via cuda-host, \
             correct gradient on hardware."
        );
        Ok(())
    } else {
        Err("autodiff gradient mismatch".into())
    }
}
