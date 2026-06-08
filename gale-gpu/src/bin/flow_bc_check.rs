//! Validation harness for the GPU **per-region boundary conditions** —
//! [`gale_gpu::GpuStokes::with_bcs`] / `step_bc`. Mirrors the CPU
//! `uniform_flow_through_outflow` test: a channel with uniform inflow (west) +
//! matching moving walls (top/bottom) + traction-free outflow (east). The exact
//! steady state `u = (1,0)`, `v = 0`, `p = 0` (harmonic, divergence-free) must be
//! preserved as it passes through the outflow. Checks the GPU trajectory against both
//! the exact field and the validated CPU `gale::dg::Stokes::with_bcs` loop.
//!
//! Run: cargo oxide run --bin flow-bc-check

use gale::dg::{BoundaryConditions, FlowBc, Mesh2d, Stokes};

/// Build the channel BCs fresh (the registry holds boxed closures, so it isn't Clone —
/// the GPU and CPU runs each get their own).
fn channel_bcs() -> BoundaryConditions {
    BoundaryConditions::no_slip()
        .set(3, FlowBc::velocity(|_, _, _| (1.0, 0.0))) // west inflow
        .set(0, FlowBc::velocity(|_, _, _| (1.0, 0.0))) // bottom moving wall
        .set(2, FlowBc::velocity(|_, _, _| (1.0, 0.0))) // top moving wall
        .set(1, FlowBc::Outflow) // east outflow
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let nu = 1.0;
    let p = 4;
    let alpha = 5.0;
    let mesh = Mesh2d::rectangular(p, 5, 3, [0.0, 2.0], [0.0, 1.0]);
    let nn = mesh.refq.n_nodes();
    let ndof = mesh.n_elements() * nn;
    let dt = 0.05;
    let nsteps = 20usize;
    let zero = |_: f64, _: f64, _: f64| 0.0;
    println!("=== gale_gpu::GpuStokes::with_bcs channel: inflow + moving walls + outflow ===");
    println!("    {nsteps} steps (p={p}, ν={nu}), exact steady state u=(1,0)\n");

    // GPU run.
    let gbcs = channel_bcs();
    let gst = gale_gpu::GpuStokes::with_bcs(&mesh, alpha, nu, dt, &gbcs);
    let mut gux = vec![1.0; ndof];
    let mut guy = vec![0.0; ndof];
    // CPU reference run (identical setup).
    let cbcs = channel_bcs();
    let cst = Stokes::with_bcs(&mesh, alpha, nu, dt, &cbcs);
    let mut cux = vec![1.0; ndof];
    let mut cuy = vec![0.0; ndof];

    let mut t = 0.0;
    for _ in 0..nsteps {
        t += dt;
        let (nx, ny) = gst.step_bc(&gux, &guy, t, &gbcs, zero, zero)?;
        gux = nx;
        guy = ny;
        let (mx, my) = cst.step_bc(&cux, &cuy, t, &cbcs, zero, zero);
        cux = mx;
        cuy = my;
    }

    // GPU vs exact uniform flow (1, 0).
    let one = vec![1.0; ndof];
    let eu: Vec<f64> = gux.iter().map(|v| v - 1.0).collect();
    let err_exact = (gst.l2_norm(&eu).powi(2) + gst.l2_norm(&guy).powi(2)).sqrt() / gst.l2_norm(&one);

    // GPU vs CPU.
    let dx: Vec<f64> = gux.iter().zip(&cux).map(|(a, b)| a - b).collect();
    let dy: Vec<f64> = guy.iter().zip(&cuy).map(|(a, b)| a - b).collect();
    let cnorm = (cst.l2_norm(&cux).powi(2) + cst.l2_norm(&cuy).powi(2)).sqrt().max(1e-300);
    let rel_cpu = (gst.l2_norm(&dx).powi(2) + gst.l2_norm(&dy).powi(2)).sqrt() / cnorm;

    println!("‖u_gpu − (1,0)‖/‖u‖  = {err_exact:.3e}   (outflow preserves uniform flow)");
    println!("‖u_gpu − u_cpu‖/‖u‖  = {rel_cpu:.3e}   (GPU vs validated CPU with_bcs)");
    if rel_cpu < 1e-7 && err_exact < 1e-6 {
        println!("\nPASS: GPU per-region BCs match the CPU solver and the outflow is natural.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU BC mismatch (rel_cpu={rel_cpu:.3e}, err_exact={err_exact:.3e}).");
        std::process::exit(1);
    }
}
