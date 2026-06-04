//! Validation harness for the GPU incompressible **Navier–Stokes** path
//! ([`gale_gpu::GpuStokes::step_ns`] + the [`gale_gpu::GpuDualSplitting`] framework
//! integrator) — build-order step 3. Integrates the **Taylor–Green vortex** (an exact
//! NS solution) and checks against the validated CPU `gale::dg::Stokes::step_ns` and
//! the analytic solution, for both the nodal and split-form-DG convection schemes;
//! plus the framework run via `Simulation` must reproduce the direct loop.
//!
//! Run: cargo oxide run --bin ns-check

use gale::dg::{ConvectionScheme, Mesh2d, Stokes};
use gale::sim::{Simulation, State};
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
    let decay = move |t: f64| (-2.0 * PI * PI * nu * t).exp();
    let eu = move |x: f64, y: f64, t: f64| -(PI * x).cos() * (PI * y).sin() * decay(t);
    let ev = move |x: f64, y: f64, t: f64| (PI * x).sin() * (PI * y).cos() * decay(t);
    let zero = |_: f64, _: f64, _: f64| 0.0;
    let t_end = 0.1;
    let nsteps = 10usize;
    let dt = t_end / nsteps as f64;
    println!("=== GPU Navier–Stokes (Taylor–Green), {nsteps} steps (p={p}, ν={nu}) ===\n");

    let mut ok = true;
    for scheme in [ConvectionScheme::Nodal, ConvectionScheme::SplitFormDg] {
        // Direct GPU NS loop + CPU reference (identical setup).
        let mut gst = gale_gpu::GpuStokes::new(&mesh, alpha, nu, dt);
        gst.convection_scheme = scheme;
        let mut cst = Stokes::new(&mesh, alpha, nu, dt);
        cst.convection_scheme = scheme;
        let mut gux = nodal(&mesh, |x, y| eu(x, y, 0.0));
        let mut guy = nodal(&mesh, |x, y| ev(x, y, 0.0));
        let (mut cux, mut cuy) = (gux.clone(), guy.clone());
        let mut t = 0.0;
        for _ in 0..nsteps {
            t += dt;
            let (nx, ny) = gst.step_ns(&gux, &guy, t, eu, ev, zero, zero)?;
            gux = nx;
            guy = ny;
            let (mx, my) = cst.step_ns(&cux, &cuy, t, eu, ev, zero, zero);
            cux = mx;
            cuy = my;
        }
        let exu = nodal(&mesh, |x, y| eu(x, y, t_end));
        let exv = nodal(&mesh, |x, y| ev(x, y, t_end));
        let ex: Vec<f64> = gux.iter().zip(&exu).map(|(a, b)| a - b).collect();
        let ey: Vec<f64> = guy.iter().zip(&exv).map(|(a, b)| a - b).collect();
        let err_exact = (gst.l2_norm(&ex).powi(2) + gst.l2_norm(&ey).powi(2)).sqrt();
        let dx: Vec<f64> = gux.iter().zip(&cux).map(|(a, b)| a - b).collect();
        let dy: Vec<f64> = guy.iter().zip(&cuy).map(|(a, b)| a - b).collect();
        let cn = (cst.l2_norm(&cux).powi(2) + cst.l2_norm(&cuy).powi(2)).sqrt().max(1e-300);
        let rel_cpu = (gst.l2_norm(&dx).powi(2) + gst.l2_norm(&dy).powi(2)).sqrt() / cn;
        let pass = rel_cpu < 1e-7 && err_exact < 5e-3;
        ok &= pass;
        println!(
            "{scheme:?}: vs CPU = {rel_cpu:.3e}   vs exact = {err_exact:.3e}   {}",
            if pass { "OK" } else { "FAIL" }
        );
    }

    // Framework: Simulation + GpuDualSplitting (nodal) must reproduce the direct loop.
    let mut st = State::new(mesh.clone());
    let vid = st.add_field_from(
        "velocity",
        &[
            Box::new(move |x: f64, y: f64| eu(x, y, 0.0)) as Box<dyn Fn(f64, f64) -> f64>,
            Box::new(move |x: f64, y: f64| ev(x, y, 0.0)),
        ],
    );
    let mut sim = Simulation::new(st);
    sim.set_integrator(gale_gpu::GpuDualSplitting::new(vid, dt, nu, alpha).boundary(eu, ev));
    sim.run(nsteps as u64);
    let fux = sim.state.field("velocity").component(0).to_vec();
    let fuy = sim.state.field("velocity").component(1).to_vec();

    let mut gst = gale_gpu::GpuStokes::new(&mesh, alpha, nu, dt);
    gst.convection_scheme = ConvectionScheme::Nodal;
    let mut dux = nodal(&mesh, |x, y| eu(x, y, 0.0));
    let mut duy = nodal(&mesh, |x, y| ev(x, y, 0.0));
    let mut t = 0.0;
    for _ in 0..nsteps {
        t += dt;
        let (nx, ny) = gst.step_ns(&dux, &duy, t, eu, ev, zero, zero)?;
        dux = nx;
        duy = ny;
    }
    let ddx: Vec<f64> = fux.iter().zip(&dux).map(|(a, b)| a - b).collect();
    let ddy: Vec<f64> = fuy.iter().zip(&duy).map(|(a, b)| a - b).collect();
    let dn = (gst.l2_norm(&dux).powi(2) + gst.l2_norm(&duy).powi(2)).sqrt().max(1e-300);
    let rel_fw = (gst.l2_norm(&ddx).powi(2) + gst.l2_norm(&ddy).powi(2)).sqrt() / dn;
    let fw_ok = rel_fw < 1e-12;
    ok &= fw_ok;
    println!("framework GpuDualSplitting vs direct = {rel_fw:.3e}   {}", if fw_ok { "OK" } else { "FAIL" });

    if ok {
        println!("\nPASS: GPU Navier–Stokes matches the CPU solver, the analytic vortex, and the framework path.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU NS mismatch.");
        std::process::exit(1);
    }
}
