//! Stage 3a validation: the GPU per-element Persson–Peraire smoothness indicator (`smoothness_se_gpu`)
//! vs the host `gale::dg::SmoothnessIndicator::indicator`, element by element, on a field with both
//! smooth and sharp regions (so Se spans its range). Must match to round-off. This is the first
//! device kernel of the GPU-resident AMR port (docs/plan-stage3-gpu-resident-amr.md).
//! Run: cargo oxide run --bin amr-indicator-check

use gale::dg::{Mesh2d, SmoothnessIndicator};
use gale_gpu::smoothness_se_gpu;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 3usize;
    let (nx, ny) = (12usize, 12usize);
    let xr = [0.0, 1.0];
    let yr = [0.0, 1.0];
    let mesh = Mesh2d::rectangular(p, nx, ny, xr, yr);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let two_pi = std::f64::consts::TAU;
    println!("=== Stage 3a: GPU smoothness indicator vs host — {nx}×{ny} p={p} ===");

    // A field that is smooth in most of the domain but has a sharp localized feature (so some
    // elements have high modal energy -> nonzero Se, others near zero).
    let mut field = vec![0.0; ndof];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            let (x, y) = (el.geom.x[k], el.geom.y[k]);
            let smooth = (two_pi * x).sin() * (two_pi * y).cos();
            let strand = (-((x - 0.5).powi(2) + (y - 0.5).powi(2)) / 0.004).exp(); // sharp blob
            field[e * nn + k] = smooth + 5.0 * strand;
        }
    }

    let si = SmoothnessIndicator::new(p);
    let host: Vec<f64> = (0..ne)
        .map(|e| si.indicator(&field[e * nn..(e + 1) * nn]))
        .collect();
    let dev = smoothness_se_gpu(&mesh, &field, &si)?;

    let mut max_abs = 0.0f64;
    let mut hmax = 0.0f64;
    for e in 0..ne {
        max_abs = max_abs.max((dev[e] - host[e]).abs());
        hmax = hmax.max(host[e]);
    }
    let n_active = host.iter().filter(|&&s| s > 1e-3).count();
    println!("  max |Se_gpu - Se_host| = {max_abs:.3e}   (host Se range up to {hmax:.3e}, {n_active}/{ne} elems Se>1e-3)");
    if max_abs < 1e-12 {
        println!("OK: GPU smoothness indicator matches the host to round-off.");
        Ok(())
    } else {
        eprintln!("FAIL: GPU indicator diverges from host.");
        std::process::exit(1);
    }
}
