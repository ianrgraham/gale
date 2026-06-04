//! Validation harness for [`gale_gpu::GpuStokes`] — the GPU unsteady-Stokes solver
//! (build-order step 2). Integrates the **decaying Taylor–Green / Stokes vortex** (an
//! exact solution: `u = −cos(πx)sin(πy)e^{−2π²νt}`, `v = sin(πx)cos(πy)e^{−2π²νt}`,
//! `p ≡ 0`, `f ≡ 0`) and checks the GPU trajectory against both the analytic solution
//! and the validated CPU `gale::dg::Stokes` loop with identical setup.
//!
//! Run: cargo oxide run --bin stokes-check

use gale::dg::{Mesh2d, Stokes};
use std::f64::consts::PI;

fn nodal(mesh: &Mesh2d, f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    let nn = mesh.refq.n_nodes();
    let mut v = vec![0.0; mesh.n_elements() * nn];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            v[e * nn + k] = f(el.geom.x[k], el.geom.y[k]);
        }
    }
    v
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let nu = 1.0;
    let p = 4;
    let alpha = 5.0;
    let mesh = Mesh2d::rectangular(p, 4, 4, [0.0, 1.0], [0.0, 1.0]);
    let decay = |t: f64| (-2.0 * PI * PI * nu * t).exp();
    let eu = |x: f64, y: f64, t: f64| -(PI * x).cos() * (PI * y).sin() * decay(t);
    let ev = |x: f64, y: f64, t: f64| (PI * x).sin() * (PI * y).cos() * decay(t);
    let zero = |_: f64, _: f64, _: f64| 0.0;
    let t_end = 0.1;
    let nsteps = 10usize;
    let dt = t_end / nsteps as f64;
    println!("=== gale_gpu::GpuStokes decaying vortex, {nsteps} steps (p={p}, ν={nu}) ===\n");

    // GPU run.
    let gst = gale_gpu::GpuStokes::new(&mesh, alpha, nu, dt);
    let mut gux = nodal(&mesh, |x, y| eu(x, y, 0.0));
    let mut guy = nodal(&mesh, |x, y| ev(x, y, 0.0));
    // CPU reference run (identical setup).
    let cst = Stokes::new(&mesh, alpha, nu, dt);
    let mut cux = gux.clone();
    let mut cuy = guy.clone();

    let mut t = 0.0;
    for _ in 0..nsteps {
        t += dt;
        let (nx, ny) = gst.step(&gux, &guy, t, eu, ev, zero, zero)?;
        gux = nx;
        guy = ny;
        let (mx, my) = cst.step(&cux, &cuy, t, eu, ev, zero, zero);
        cux = mx;
        cuy = my;
    }

    // GPU vs analytic.
    let exu = nodal(&mesh, |x, y| eu(x, y, t_end));
    let exv = nodal(&mesh, |x, y| ev(x, y, t_end));
    let ex: Vec<f64> = gux.iter().zip(&exu).map(|(a, b)| a - b).collect();
    let ey: Vec<f64> = guy.iter().zip(&exv).map(|(a, b)| a - b).collect();
    let err_exact = (gst.l2_norm(&ex).powi(2) + gst.l2_norm(&ey).powi(2)).sqrt();

    // GPU vs CPU Stokes.
    let dx: Vec<f64> = gux.iter().zip(&cux).map(|(a, b)| a - b).collect();
    let dy: Vec<f64> = guy.iter().zip(&cuy).map(|(a, b)| a - b).collect();
    let cnorm = (cst.l2_norm(&cux).powi(2) + cst.l2_norm(&cuy).powi(2)).sqrt().max(1e-300);
    let rel_cpu = (gst.l2_norm(&dx).powi(2) + gst.l2_norm(&dy).powi(2)).sqrt() / cnorm;

    println!("‖u_gpu − u_exact‖   = {err_exact:.3e}   (vortex decay error)");
    println!("‖u_gpu − u_cpu‖/‖u‖ = {rel_cpu:.3e}   (GPU vs validated CPU Stokes)");
    if rel_cpu < 1e-7 && err_exact < 5e-3 {
        println!("\nPASS: GPU Stokes matches the CPU solver and recovers the decaying vortex.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU Stokes mismatch (rel_cpu={rel_cpu:.3e}, err_exact={err_exact:.3e}).");
        std::process::exit(1);
    }
}
