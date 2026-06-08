//! Validation harness for the GPU **per-region boundary conditions** —
//! [`gale_gpu::GpuStokes::with_bcs`] / `step_bc`. Two scenarios whose exact steady
//! state is the uniform flow `u = (1,0)`, `v = 0`, `p = 0` (harmonic, divergence-free):
//!
//! 1. **Outflow** — uniform inflow (west) + matching moving walls (top/bottom) +
//!    traction-free outflow (east). A reflecting/Dirichlet-0 outflow would distort it.
//! 2. **Free-slip** — periodic-x channel with symmetry (slip) top/bottom walls. No-slip
//!    walls would force `u_x → 0`; slip walls (normal-component Dirichlet, tangential
//!    Neumann) preserve it.
//!
//! Each is checked against the exact field and the validated CPU `Stokes::with_bcs`.
//!
//! Run: cargo oxide run --bin flow-bc-check

use gale::dg::{BoundaryConditions, FlowBc, Mesh2d, Stokes};

/// Run one scenario both on GPU and CPU; return whether GPU matches CPU and the exact
/// uniform flow. `bcs` is a builder (the registry holds boxed closures ⇒ not Clone).
fn run(
    label: &str,
    mesh: &Mesh2d,
    bcs: impl Fn() -> BoundaryConditions,
) -> Result<bool, Box<dyn std::error::Error>> {
    let (nu, alpha, dt, nsteps) = (1.0, 5.0, 0.05, 20usize);
    let nn = mesh.refq.n_nodes();
    let ndof = mesh.n_elements() * nn;
    let zero = |_: f64, _: f64, _: f64| 0.0;

    let gbcs = bcs();
    let gst = gale_gpu::GpuStokes::with_bcs(mesh, alpha, nu, dt, &gbcs);
    let mut gux = vec![1.0; ndof];
    let mut guy = vec![0.0; ndof];
    let cbcs = bcs();
    let cst = Stokes::with_bcs(mesh, alpha, nu, dt, &cbcs);
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

    let one = vec![1.0; ndof];
    let eu: Vec<f64> = gux.iter().map(|v| v - 1.0).collect();
    let err_exact = (gst.l2_norm(&eu).powi(2) + gst.l2_norm(&guy).powi(2)).sqrt() / gst.l2_norm(&one);
    let dx: Vec<f64> = gux.iter().zip(&cux).map(|(a, b)| a - b).collect();
    let dy: Vec<f64> = guy.iter().zip(&cuy).map(|(a, b)| a - b).collect();
    let cnorm = (cst.l2_norm(&cux).powi(2) + cst.l2_norm(&cuy).powi(2)).sqrt().max(1e-300);
    let rel_cpu = (gst.l2_norm(&dx).powi(2) + gst.l2_norm(&dy).powi(2)).sqrt() / cnorm;
    let pass = rel_cpu < 1e-7 && err_exact < 1e-6;
    println!(
        "{label}: ‖u_gpu−(1,0)‖/‖u‖ = {err_exact:.3e}   ‖u_gpu−u_cpu‖/‖u‖ = {rel_cpu:.3e}   {}",
        if pass { "OK" } else { "FAIL" }
    );
    Ok(pass)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== gale_gpu::GpuStokes::with_bcs per-region BCs vs CPU (uniform flow preserved) ===\n");
    let mut ok = true;

    // Scenario 1: inflow (west=3) + moving walls (bottom=0, top=2) + outflow (east=1).
    let outflow_mesh = Mesh2d::rectangular(4, 5, 3, [0.0, 2.0], [0.0, 1.0]);
    ok &= run("outflow ", &outflow_mesh, || {
        BoundaryConditions::no_slip()
            .set(3, FlowBc::velocity(|_, _, _| (1.0, 0.0)))
            .set(0, FlowBc::velocity(|_, _, _| (1.0, 0.0)))
            .set(2, FlowBc::velocity(|_, _, _| (1.0, 0.0)))
            .set(1, FlowBc::Outflow)
    })?;

    // Scenario 2: periodic-x channel with free-slip (symmetry) walls (tags 0, 2).
    let slip_mesh = Mesh2d::channel_x(4, 4, 3, [0.0, 2.0], [0.0, 1.0]);
    ok &= run("free-slip", &slip_mesh, || {
        BoundaryConditions::no_slip().set(0, FlowBc::Symmetry).set(2, FlowBc::Symmetry)
    })?;

    if ok {
        println!("\nPASS: GPU per-region BCs (outflow + symmetry) match the CPU oracle.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU per-region BC mismatch.");
        std::process::exit(1);
    }
}
