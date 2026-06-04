//! Framework GPU flow: assemble a `gale::sim::Simulation` driven by the GPU
//! unsteady-Stokes integrator [`gale_gpu::GpuStokesIntegrator`] and run it through the
//! same HOOMD-style API as the CPU path — each step's pressure + viscous solves on the
//! GPU. Validates the framework trajectory against (a) the direct [`gale_gpu::GpuStokes`]
//! loop (framework fidelity) and (b) the analytic decaying vortex.
//!
//! Run: cargo oxide run --bin sim-stokes-check

use gale::dg::Mesh2d;
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
    let t_end = 0.1;
    let nsteps = 10u64;
    let dt = t_end / nsteps as f64;
    println!("=== GPU-framework Stokes (Simulation + GpuStokesIntegrator), {nsteps} steps ===\n");

    // Exact decaying Taylor–Green/Stokes vortex (move-captures nu ⇒ 'static, Copy).
    let eu = move |x: f64, y: f64, t: f64| -(PI * x).cos() * (PI * y).sin() * (-2.0 * PI * PI * nu * t).exp();
    let ev = move |x: f64, y: f64, t: f64| (PI * x).sin() * (PI * y).cos() * (-2.0 * PI * PI * nu * t).exp();

    // Framework run: velocity field + GpuStokesIntegrator through Simulation::run.
    let mut st = State::new(mesh.clone());
    let vid = st.add_field_from(
        "velocity",
        &[
            Box::new(move |x: f64, y: f64| eu(x, y, 0.0)) as Box<dyn Fn(f64, f64) -> f64>,
            Box::new(move |x: f64, y: f64| ev(x, y, 0.0)),
        ],
    );
    let mut sim = Simulation::new(st);
    sim.set_integrator(gale_gpu::GpuStokesIntegrator::new(vid, dt, nu, alpha).boundary(eu, ev));
    sim.run(nsteps);
    let fux = sim.state.field("velocity").component(0).to_vec();
    let fuy = sim.state.field("velocity").component(1).to_vec();

    // Direct GpuStokes loop (identical setup) — framework must reproduce it exactly.
    let gst = gale_gpu::GpuStokes::new(&mesh, alpha, nu, dt);
    let mut dux = nodal(&mesh, |x, y| eu(x, y, 0.0));
    let mut duy = nodal(&mesh, |x, y| ev(x, y, 0.0));
    let zero = |_: f64, _: f64, _: f64| 0.0;
    let mut t = 0.0;
    for _ in 0..nsteps {
        t += dt;
        let (nx, ny) = gst.step(&dux, &duy, t, eu, ev, zero, zero)?;
        dux = nx;
        duy = ny;
    }

    // Framework vs direct loop.
    let ddx: Vec<f64> = fux.iter().zip(&dux).map(|(a, b)| a - b).collect();
    let ddy: Vec<f64> = fuy.iter().zip(&duy).map(|(a, b)| a - b).collect();
    let dnorm = (gst.l2_norm(&dux).powi(2) + gst.l2_norm(&duy).powi(2)).sqrt().max(1e-300);
    let rel_direct = (gst.l2_norm(&ddx).powi(2) + gst.l2_norm(&ddy).powi(2)).sqrt() / dnorm;

    // Framework vs analytic.
    let exu = nodal(&mesh, |x, y| eu(x, y, t_end));
    let exv = nodal(&mesh, |x, y| ev(x, y, t_end));
    let ex: Vec<f64> = fux.iter().zip(&exu).map(|(a, b)| a - b).collect();
    let ey: Vec<f64> = fuy.iter().zip(&exv).map(|(a, b)| a - b).collect();
    let err_exact = (gst.l2_norm(&ex).powi(2) + gst.l2_norm(&ey).powi(2)).sqrt();

    println!("‖u_fw − u_direct‖/‖u‖ = {rel_direct:.3e}   (framework vs direct GpuStokes)");
    println!("‖u_fw − u_exact‖      = {err_exact:.3e}   (vortex decay error)");
    if rel_direct < 1e-12 && err_exact < 5e-3 {
        println!("\nPASS: GPU Stokes runs through the Simulation framework and matches the direct loop.");
        Ok(())
    } else {
        eprintln!("\nFAIL: framework mismatch (rel_direct={rel_direct:.3e}, err_exact={err_exact:.3e}).");
        std::process::exit(1);
    }
}
