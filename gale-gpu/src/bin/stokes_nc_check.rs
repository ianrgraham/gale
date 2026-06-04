//! Validation harness for [`gale_gpu::GpuStokes`] on a **2:1 non-conforming (adaptive)**
//! mesh — AMR-with-GPU-flow. Integrates the decaying Taylor–Green/Stokes vortex on a
//! refined mesh (so the GPU elliptic solves route through the mortar NC path) and checks
//! the GPU trajectory against the validated CPU `gale::dg::Stokes` on the same mesh.
//!
//! Run: cargo oxide run --bin stokes-nc-check

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
    // Two refined cells → CoarseToFine / FineToCoarse mortar interfaces.
    let mesh = Mesh2d::cartesian_refined(p, 4, 4, [0.0, 1.0], [0.0, 1.0], &[(1, 1), (2, 2)]);
    let decay = move |t: f64| (-2.0 * PI * PI * nu * t).exp();
    let eu = move |x: f64, y: f64, t: f64| -(PI * x).cos() * (PI * y).sin() * decay(t);
    let ev = move |x: f64, y: f64, t: f64| (PI * x).sin() * (PI * y).cos() * decay(t);
    let zero = |_: f64, _: f64, _: f64| 0.0;
    let t_end = 0.1;
    let nsteps = 10usize;
    let dt = t_end / nsteps as f64;
    println!("=== GpuStokes on a 2:1 refined mesh ({} elements), {nsteps} steps ===\n", mesh.n_elements());

    let gst = gale_gpu::GpuStokes::new(&mesh, alpha, nu, dt);
    let cst = Stokes::new(&mesh, alpha, nu, dt);
    let mut gux = nodal(&mesh, |x, y| eu(x, y, 0.0));
    let mut guy = nodal(&mesh, |x, y| ev(x, y, 0.0));
    let (mut cux, mut cuy) = (gux.clone(), guy.clone());
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

    // GPU vs CPU on the refined mesh.
    let dx: Vec<f64> = gux.iter().zip(&cux).map(|(a, b)| a - b).collect();
    let dy: Vec<f64> = guy.iter().zip(&cuy).map(|(a, b)| a - b).collect();
    let cnorm = (cst.l2_norm(&cux).powi(2) + cst.l2_norm(&cuy).powi(2)).sqrt().max(1e-300);
    let rel_cpu = (gst.l2_norm(&dx).powi(2) + gst.l2_norm(&dy).powi(2)).sqrt() / cnorm;
    // vs analytic vortex (recovery on the refined mesh).
    let exu = nodal(&mesh, |x, y| eu(x, y, t_end));
    let exv = nodal(&mesh, |x, y| ev(x, y, t_end));
    let ex: Vec<f64> = gux.iter().zip(&exu).map(|(a, b)| a - b).collect();
    let ey: Vec<f64> = guy.iter().zip(&exv).map(|(a, b)| a - b).collect();
    let err_exact = (gst.l2_norm(&ex).powi(2) + gst.l2_norm(&ey).powi(2)).sqrt();

    println!("GPU vs CPU Stokes (refined) = {rel_cpu:.3e}");
    println!("GPU vs analytic vortex      = {err_exact:.3e}");
    if rel_cpu < 1e-7 && err_exact < 2e-2 {
        println!("\nPASS: GpuStokes runs on a 2:1 non-conforming mesh, matching CPU Stokes.");
        Ok(())
    } else {
        eprintln!("\nFAIL: NC Stokes mismatch (rel_cpu={rel_cpu:.3e}, err_exact={err_exact:.3e}).");
        std::process::exit(1);
    }
}
